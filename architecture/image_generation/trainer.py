
# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
#     os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))))
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torchvision
import time
import numpy as np
from architecture.utils.inception_score import InceptionScore
from architecture.image_generation.model import NetD, NetG
from architecture.utils.utils import gen_image_grid, weights_init, generate_figure
from easydict import EasyDict as edict

cifar10_label_names = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat",
                       4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}


class VAE_DFGAN(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        if type(cfg) is dict:
            cfg = edict(cfg)
        self.cfg = cfg
        self.save_hyperparameters(self.cfg)

        self.encoder = NetD(cfg.MODEL.NF, cfg.IM_SIZE,
                            cfg.MODEL.Z_DIM, cfg.MODEL.EF_DIM, disc=False)
        self.encoder.apply(weights_init)
        self.decoder = NetG(cfg.MODEL.NF, cfg.IM_SIZE, cfg.MODEL.Z_DIM, cfg.MODEL.EF_DIM)
        self.decoder.apply(weights_init)
        self.kl_loss = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        self.start = time.perf_counter()

    def on_pretrain_routine_start(self):
        self.logger.experiment.add_graph(
            self.encoder, (torch.ones(1, 3, self.cfg.IM_SIZE, self.cfg.IM_SIZE).cuda(),
                           torch.ones(1, self.cfg.MODEL.EF_DIM).cuda()))

        self.logger.experiment.add_graph(self.decoder, (torch.ones(1, self.cfg.MODEL.Z_DIM).cuda(),
                                                        torch.ones(1, self.cfg.MODEL.EF_DIM).cuda()))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y=None):
        # in lightning, forward defines the prediction/inference actions
        return self.decoder(x, y)

    def training_step(self, batch, batch_idx):
        x = batch["img"]
        y = batch["qa_embedding"]
        self.eval_y = y
        self.eval_text = batch["text"]

        mu, logvar = self.encoder(x, y)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z, y)

        recon_loss = F.mse_loss(
            recon_x.view(-1, self.cfg.IM_SIZE ** 2 * 3),
            x.view(-1, self.cfg.IM_SIZE ** 2 * 3),
            reduction="sum")

        kl_loss = self.kl_loss(mu, logvar) / x.view(-1, self.cfg.IM_SIZE ** 2 * 3).data.shape[0]
        loss = recon_loss + kl_loss
        self.log('Loss/KL', kl_loss, on_step=False, on_epoch=True)
        self.log('Loss/Recon', recon_loss, on_step=False, on_epoch=True)
        self.log('Loss/VAE', loss, on_step=False, on_epoch=True)
        return loss

    def on_epoch_end(self):
        elapsed_time = time.perf_counter() - self.start
        self.start = time.perf_counter()
        self.print(
            f"\nEpoch {self.current_epoch} finished in {round(elapsed_time, 2)}s")

        if self.current_epoch % self.trainer.check_val_every_n_epoch == 0:
            noise = torch.randn((self.eval_y.size(0), self.cfg.MODEL.Z_DIM)).type_as(self.eval_y)
            recon_x = self.forward(noise, self.eval_y)
            val_images = []
            for img, text in zip(recon_x, self.eval_text):
                val_images.append(generate_figure(img, text))
            self.logger.experiment.add_images(
                f"Train/Epoch_{self.current_epoch}", torch.stack(val_images),
                global_step=self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.cfg.TRAIN.VAE_LR, betas=(0, 0.999))
        return optimizer


class DFGAN(pl.LightningModule):

    def __init__(self, cfg, pretrained_encoder=None, vqa_model=None):
        super().__init__()
        self.automatic_optimization = False
        if type(cfg) is dict:
            cfg = edict(cfg)
        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.vqa_model = vqa_model

        if pretrained_encoder:
            self.generator = pretrained_encoder
        else:
            self.generator = NetG(cfg.MODEL.NF, cfg.IM_SIZE, cfg.MODEL.Z_DIM, cfg.MODEL.EF_DIM)
        self.generator.apply(weights_init)
        self.discriminator = NetD(cfg.MODEL.NF, cfg.IM_SIZE, cfg.MODEL.Z_DIM, cfg.MODEL.EF_DIM, disc=True)
        self.discriminator.apply(weights_init)
        self.opt_g, self.opt_d = self.configure_optimizers()

        self.start = time.perf_counter()
        self.inception = InceptionScore()
        self.real_acc = pl.metrics.Accuracy()
        self.fake_acc = pl.metrics.Accuracy()
        if self.vqa_model:
            self.vqa_model.eval()
            self.train_vqa_acc = pl.metrics.Accuracy()
            self.val_vqa_acc = pl.metrics.Accuracy()
            self.test_vqa_acc = pl.metrics.Accuracy()

    def on_pretrain_routine_start(self):
        self.logger.experiment.add_graph(
            self.discriminator, (torch.ones(1, 3, self.cfg.IM_SIZE, self.cfg.IM_SIZE).cuda(),
                                 torch.ones(1, self.cfg.MODEL.EF_DIM).cuda()))

        self.logger.experiment.add_graph(self.generator, (torch.ones(1, self.cfg.MODEL.Z_DIM).cuda(),
                                                          torch.ones(1, self.cfg.MODEL.EF_DIM).cuda()))

    def forward(self, x, y=None):
        # in lightning, forward defines the prediction/inference actions

        return self.generator(x, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
      #  (opt_g, opt_d) = self.optimizers()
        real_img = batch["img"]
        text_embed = batch["qa_embedding"]

        self.eval_y = text_embed
        self.eval_text = batch["text"]

        batch_size = real_img.size(0)

        # 1. Discriminator loss

        # Prediction on the real data
        real_pred = self.discriminator(real_img, text_embed)
        d_acc_real = self.real_acc(
            torch.sigmoid(real_pred),
            torch.ones(batch_size, 1, dtype=torch.int32).cuda())
        d_loss_real = torch.nn.ReLU()(1.0 - real_pred).mean()
     #   d_loss_real = F.binary_cross_entropy_with_logits(real_pred, real)
        # Prediction on the mismatched/wrong labeled data
        wrong_pred = self.discriminator(real_img, text_embed.roll(1))
        d_loss_wrong = torch.nn.ReLU()(1.0 + wrong_pred).mean()
       # d_loss_wrong = F.binary_cross_entropy_with_logits(wrong_pred, fake)
        # Forward pass
        noise = torch.randn(batch_size, 100).type_as(real_img)
        fake_img = self.generator(noise, text_embed)

        # Prediction on the generated images
        fake_pred = self.discriminator(fake_img.detach(), text_embed)
        d_acc_fake = self.fake_acc(torch.sigmoid(fake_pred),
                                   torch.zeros(batch_size, dtype=torch.int32).cuda())
        d_loss_fake = torch.nn.ReLU()(1.0 + fake_pred).mean()
        #d_loss_fake = F.binary_cross_entropy_with_logits(fake_pred, fake)

        d_loss = d_loss_real + (d_loss_fake + d_loss_wrong) / 2.0
        self.opt_d.zero_grad()
        self.opt_g.zero_grad()
        self.manual_backward(d_loss, self.opt_d)
       # self.manual_optimizer_step(self.opt_d)
        self.opt_d.step()

        self.log("Loss/Real", d_loss_real, on_step=False, on_epoch=True)
        self.log("Loss/Fake", d_loss_fake, on_step=False, on_epoch=True)
        self.log("Loss/Wrong", d_loss_wrong, on_step=False, on_epoch=True)
        self.log("Loss/Discriminator", d_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("Acc/Real", d_acc_real, on_step=False, on_epoch=True, prog_bar=True)
        self.log("Acc/Fake", d_acc_fake, on_step=False, on_epoch=True, prog_bar=True)

        # 2. Matching aware gradient penalty on the discriminator
        interpolated = real_img.requires_grad_()
        sent_inter = text_embed.requires_grad_()
        out = self.discriminator(real_img, text_embed)
        grads = torch.autograd.grad(outputs=out,
                                    inputs=(interpolated, sent_inter),
                                    grad_outputs=torch.ones(out.size()).type_as(real_img),
                                    retain_graph=True,
                                    create_graph=True,
                                    only_inputs=True)
        grad0 = grads[0].view(grads[0].size(0), -1)
        grad1 = grads[1].view(grads[1].size(0), -1)
        grad = torch.cat((grad0, grad1), dim=1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm) ** 6) * 2
        self.opt_d.zero_grad()
        self.opt_g.zero_grad()
        self.manual_backward(d_loss_gp, self.opt_d)
       # self.manual_optimizer_step(self.opt_d)
        self.opt_d.step()
        self.log("Loss/GradientPenalty", d_loss_gp, on_step=False, on_epoch=True)

        # 3. Generator loss
        fake_pred = self.discriminator(fake_img, text_embed)
        g_loss = - fake_pred.mean()
    #    g_loss = F.binary_cross_entropy_with_logits(fake_pred, real)
        self.opt_g.zero_grad()
        self.opt_d.zero_grad()
        self.manual_backward(g_loss, self.opt_g)
     #   self.manual_optimizer_step(self.opt_g)

        if self.vqa_model:
            if self.cfg.TRAIN.VQA_LAMBDA == 0:
                with torch.no_grad():
                    vqa_pred = self.vqa_model(self.vqa_model.preprocess_img(fake_img), batch["q_embedding"])
            else:
                vqa_pred = self.vqa_model(self.vqa_model.preprocess_img(fake_img), batch["q_embedding"])
                vqa_loss = F.cross_entropy(vqa_pred, batch["target"]) * self.cfg.TRAIN.VQA_LAMBDA
                self.manual_backward(vqa_loss, self.opt_g)
            self.train_vqa_acc(F.softmax(vqa_pred, dim=1), batch["target"])
            self.log('VQA/Train', self.train_vqa_acc, on_step=False, on_epoch=True)

        self.opt_g.step()
        self.log("Loss/Generator", g_loss, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        text_embed = batch["qa_embedding"]

        batch_size = self.cfg.TRAIN.BATCH_SIZE
        # Generate images
        noise = torch.randn(batch_size, self.cfg.MODEL.Z_DIM).type_as(text_embed)

        fake_img = self.forward(noise, text_embed)

        incep_mean, incep_std = self.inception.compute_score(fake_img, num_splits=1)

        self.log("Inception/Val", incep_mean, on_step=False, on_epoch=True)

        if self.vqa_model:
            with torch.no_grad():
                vqa_pred = self.vqa_model(self.vqa_model.preprocess_img(fake_img), batch["q_embedding"])
                self.val_vqa_acc(F.softmax(vqa_pred, dim=1), batch["target"])
                self.log('VQA/Val', self.val_vqa_acc, on_step=False, on_epoch=True, prog_bar=True)
        # if not self.trainer.running_sanity_check and batch_idx == 0:
        #     grid = gen_image_grid(fake_img, batch["text"])
        #    # grid = torchvision.utils.make_grid(fake_x, normalize=True)
        #     self.logger.experiment.add_image(f"Val epoch {self.current_epoch}",
        #                                      grid, global_step=self.current_epoch)

    def test_step(self, batch, batch_idx):
        text_embed = batch["qa_embedding"]
        batch_size = self.cfg.TRAIN.BATCH_SIZE
        # Generate images
        noise = torch.randn(batch_size, self.cfg.MODEL.Z_DIM).type_as(text_embed)

        fake_img = self.forward(noise, text_embed)

        incep_mean, incep_std = self.inception.compute_score(fake_img, num_splits=1)

        self.log("Inception/Test", incep_mean, on_step=False, on_epoch=True)

        if self.vqa_model:
            with torch.no_grad():
                vqa_pred = self.vqa_model(self.vqa_model.preprocess_img(fake_img), batch["q_embedding"])
                self.test_vqa_acc(F.softmax(vqa_pred, dim=1), batch["target"])
                self.log('VQA/Test', self.test_vqa_acc, on_epoch=True, on_step=False)

        val_images = []
        for img, text in zip(fake_img, batch["text"]):
            val_images.append(generate_figure(img, text))
        self.logger.experiment.add_images(
            f"Test/Batch_{batch_idx}", torch.stack(val_images),
            global_step=self.current_epoch)
        #    # grid = torchvision.utils.make_grid(fake_x, normalize=True)
        #     self.logger.experiment.add_image(f"Val epoch {self.current_epoch}",
        #                                      grid, global_step=self.current_epoch)

    def on_epoch_end(self):
        elapsed_time = time.perf_counter() - self.start
        self.start = time.perf_counter()
        self.print(
            f"\nEpoch {self.current_epoch} finished in {round(elapsed_time, 2)}s")

        batch_size = self.cfg.TRAIN.BATCH_SIZE

        if not self.trainer.running_sanity_check:
            noise = torch.randn(batch_size, self.cfg.MODEL.Z_DIM).type_as(self.eval_y)
            fake_img = self.forward(noise, self.eval_y)
            val_images = []
            for img, text in zip(fake_img, self.eval_text):
                val_images.append(generate_figure(img, text))
            self.logger.experiment.add_images(
                f"Train/Epoch_{self.current_epoch}", torch.stack(val_images),
                global_step=self.current_epoch)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(),
                                 lr=self.cfg.TRAIN.G_LR, betas=(0, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(),
                                 lr=self.cfg.TRAIN.D_LR, betas=(0, 0.999))
        return opt_g, opt_d

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("loss", None)
        return items
