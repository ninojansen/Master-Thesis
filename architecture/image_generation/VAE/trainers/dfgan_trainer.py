
# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
#     os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))))
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torchvision
import time
from architecture.utils.inception_score import InceptionScore
from architecture.image_generation.VAE.models.dfgan_model import NetD, NetG

cifar10_label_names = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat",
                       4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}


class DFGAN_VAE(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = NetD(cfg.MODEL.NF, cfg.IM_SIZE, cfg.MODEL.Z_DIM, cfg.MODEL.EF_DIM, disc=False)
        self.decoder = NetG(cfg.MODEL.NF, cfg.IM_SIZE, cfg.MODEL.Z_DIM, cfg.MODEL.EF_DIM)
        self.kl_loss = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        self.start = time.perf_counter()

        self.eval_y = None
        self.eval_size = 50

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y=None):
        # in lightning, forward defines the prediction/inference actions
        return self.decoder(x, y)

    def process_data(self, batch, dataset_name):
        if dataset_name == "easy_vqa":
            images, questions, answers, combined = batch
            return images, combined
        elif dataset_name == "cifar10":
            images, target = batch
            target = F.one_hot(target, num_classes=10).float()
            if self.eval_y == None:
                self.eval_y = F.one_hot(torch.randint(0, 10, (self.eval_size,))).type_as(target)
            return images, target

    def training_step(self, batch, batch_idx):
        x, y = self.process_data(batch, self.cfg.DATASET_NAME)

        mu, logvar = self.encoder(x, y)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z, y)

        recon_loss = F.mse_loss(
            recon_x.view(-1, self.cfg.IM_SIZE ** 2 * 3),
            x.view(-1, self.cfg.IM_SIZE ** 2 * 3),
            reduction="sum")

        kl_loss = self.kl_loss(mu, logvar)

        loss = recon_loss + kl_loss
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def on_epoch_end(self):
        elapsed_time = time.perf_counter() - self.start
        self.start = time.perf_counter()
        self.print(
            f"\nEpoch {self.current_epoch} finished in {round(elapsed_time, 2)}s")

        if self.current_epoch % self.trainer.check_val_every_n_epoch == 0:
            noise = torch.randn((self.eval_size, self.cfg.MODEL.Z_DIM)).cuda()
            recon_x = self.forward(noise, self.eval_y)
            grid = torchvision.utils.make_grid(recon_x, normalize=True)
            self.logger.experiment.add_image(f"Epoch {self.current_epoch}", grid, global_step=self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.cfg.TRAIN.VAE_LR, betas=(0.5, 0.999))
        return optimizer


class DFGAN(pl.LightningModule):

    def __init__(self, cfg, pretrained_encoder=None):
        super().__init__()
        self.cfg = cfg
        if pretrained_encoder:
            self.generator = pretrained_encoder
        else:
            self.generator = NetG(cfg.MODEL.NF, cfg.IM_SIZE, cfg.MODEL.Z_DIM, cfg.MODEL.EF_DIM)
        self.discriminator = NetD(cfg.MODEL.NF, cfg.IM_SIZE, cfg.MODEL.Z_DIM, cfg.MODEL.EF_DIM, disc=True)

        self.start = time.perf_counter()
        self.inception = InceptionScore()
        self.eval_y = None
        self.eval_size = 50

    def forward(self, x, y=None):
        # in lightning, forward defines the prediction/inference actions

        return self.generator(x, y)

    def process_data(self, batch, dataset_name):
        if dataset_name == "easy_vqa":
            images, questions, answers, combined = batch
            return images, combined, combined
        elif dataset_name == "cifar10":
            images, target = batch
            target_np = target.cpu().numpy()
            target = F.one_hot(target, num_classes=10).float()

            raw_str = [cifar10_label_names[idx] for idx in target_np]
            if self.eval_y == None:
                self.eval_y = F.one_hot(torch.randint(0, 10, (self.eval_size,))).type_as(target)
            return images, target, raw_str

    def training_step(self, batch, batch_idx, optimizer_idx):
        (opt_g, opt_d) = self.optimizers()
        x, y, raw_str = self.process_data(batch, self.cfg.DATASET_NAME)

        batch_size = x.size(0)

        real = torch.ones(batch_size, 1).type_as(x)
        fake = torch.zeros(batch_size, 1).type_as(x)

        # 1. Discriminator loss

        # Prediction on the real data
        real_pred = self.discriminator(x, y)

        d_loss_real = F.binary_cross_entropy_with_logits(real_pred, real)
        # Prediction on the mismatched/wrong labeled data
        wrong_pred = self.discriminator(x, y.roll(1))
        d_loss_wrong = F.binary_cross_entropy_with_logits(wrong_pred, fake)
        # Forward pass
        noise = torch.randn(batch_size, 100).type_as(x)
        fake_x = self.generator(noise, y)

        # Prediction on the generated images
        fake_pred = self.discriminator(fake_x.detach(), y)

        d_loss_fake = F.binary_cross_entropy_with_logits(fake_pred, fake)

        d_loss = d_loss_real + (d_loss_fake + d_loss_wrong) / 2.0
        opt_d.zero_grad()
        opt_g.zero_grad()
        self.manual_backward(d_loss, opt_d)
        self.manual_optimizer_step(opt_d)

        self.log("d_loss", d_loss, prog_bar=True)

        # 2. Matching aware gradient penalty on the discriminator
        interpolated = x.requires_grad_()
        sent_inter = y.requires_grad_()
        out = self.discriminator(x, y)
        grads = torch.autograd.grad(outputs=out,
                                    inputs=(interpolated, sent_inter),
                                    grad_outputs=torch.ones(out.size()).type_as(x),
                                    retain_graph=True,
                                    create_graph=True,
                                    only_inputs=True)
        grad0 = grads[0].view(grads[0].size(0), -1)
        grad1 = grads[1].view(grads[1].size(0), -1)
        grad = torch.cat((grad0, grad1), dim=1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm) ** 6) * 2
        opt_d.zero_grad()
        opt_g.zero_grad()
        self.manual_backward(d_loss_gp, opt_d)
        self.manual_optimizer_step(opt_d)
        self.log("d_loss_gp", d_loss_gp)

        # 3. Generator loss
        fake_pred = self.discriminator(fake_x, y)
        g_loss = F.binary_cross_entropy_with_logits(fake_pred, real)
        opt_g.zero_grad()
        opt_d.zero_grad()
        self.manual_backward(g_loss, opt_g)
        self.manual_optimizer_step(opt_g)
        self.log("g_loss", g_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y, raw_str = self.process_data(batch, self.cfg.DATASET_NAME)

        batch_size = self.cfg.TRAIN.BATCH_SIZE
        # Generate images
        z = torch.randn(batch_size, self.cfg.MODEL.Z_DIM).type_as(x)

        fake_x = self.forward(z, y)

        incep_mean, incep_std = self.inception.compute_score(fake_x, num_splits=1)

        self.log("Inception score (val)", incep_mean)

        if not self.trainer.running_sanity_check and batch_idx == 0:
            grid = torchvision.utils.make_grid(fake_x, normalize=True)
            self.logger.experiment.add_image(f"Val epoch {self.current_epoch}",
                                             grid, global_step=self.current_epoch)

    def test_step(self, batch, batch_idx):
        x, y, raw_str = self.process_data(batch, self.cfg.DATASET_NAME)

        batch_size = self.cfg.TRAIN.BATCH_SIZE
        # Generate images
        z = torch.randn(batch_size, self.cfg.MODEL.Z_DIM).type_as(x)

        fake_x = self.forward(z, y)

        incep_mean, incep_std = self.inception.compute_score(fake_x, num_splits=1)

        self.log("Inception score (test)", incep_mean)

    def on_epoch_end(self):
        elapsed_time = time.perf_counter() - self.start
        self.start = time.perf_counter()
        self.print(
            f"\nEpoch {self.current_epoch} finished in {round(elapsed_time, 2)}s")

        if self.current_epoch % self.trainer.check_val_every_n_epoch == 0:
            noise = torch.randn((self.eval_size, self.cfg.MODEL.Z_DIM)).cuda()
            recon_x = self.forward(noise, self.eval_y)
            grid = torchvision.utils.make_grid(recon_x, normalize=True)
            self.logger.experiment.add_image(f"Epoch {self.current_epoch}", grid, global_step=self.current_epoch)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(),
                                 lr=self.cfg.TRAIN.G_LR, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(),
                                 lr=self.cfg.TRAIN.D_LR, betas=(0.5, 0.999))
        return opt_g, opt_d

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("loss", None)
        return items
