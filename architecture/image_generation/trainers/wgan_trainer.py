
# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
#     os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))))
from matplotlib.pyplot import text, ylabel
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torchvision
import time
from architecture.utils.inception_score import InceptionScore
from architecture.image_generation.models.dfgan_model import NetD, NetG
from architecture.utils.utils import gen_image_grid, weights_init
from easydict import EasyDict as edict
from torch.autograd import Variable
from torch import autograd


class VAE_WGAN(pl.LightningModule):

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

        kl_loss = self.kl_loss(mu, logvar)

        loss = recon_loss + kl_loss
        self.log('vae_loss', loss, on_step=True, on_epoch=True)
        return loss

    def on_epoch_end(self):
        elapsed_time = time.perf_counter() - self.start
        self.start = time.perf_counter()
        self.print(
            f"\nEpoch {self.current_epoch} finished in {round(elapsed_time, 2)}s")

        if self.current_epoch % self.trainer.check_val_every_n_epoch == 0:
            noise = torch.randn((self.eval_y.size(0), self.cfg.MODEL.Z_DIM)).type_as(self.eval_y)
            recon_x = self.forward(noise, self.eval_y)
         #   grid = torchvision.utils.make_grid(recon_x, normalize=True)
            grid = gen_image_grid(recon_x.detach(), self.eval_text)
            self.logger.experiment.add_image(f"Epoch {self.current_epoch}", grid, global_step=self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.cfg.TRAIN.VAE_LR, betas=(0, 0.999))
        return optimizer


class WGAN(pl.LightningModule):

    def __init__(self, cfg, pretrained_encoder=None):
        super().__init__()
        if type(cfg) is dict:
            cfg = edict(cfg)
        self.cfg = cfg
        self.save_hyperparameters(self.cfg)

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

        self.critic_iter = 5
        self.lambda_term = 10
        self.weight_cliping_limit = 0.01
      #  self.eval_y = None

    def forward(self, x, y=None):
        # in lightning, forward defines the prediction/inference actions

        return self.generator(x, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
      #  (opt_g, opt_d) = self.optimizers()

        x = batch["img"]
        y = batch["qa_embedding"]
        self.eval_y = y
        self.eval_text = batch["text"]
        batch_size = x.size(0)

        one = torch.FloatTensor([1]).type_as(x)
        mone = one * -1
        # Train discriminator critic iter times per generator loop
        if (batch_idx + 1) % self.critic_iter == 0:
            # 3. Generator loss
           # self.opt_d.zero_grad()
            self.opt_g.zero_grad()
            for p in self.discriminator.parameters():
                p.requires_grad = False  # to avoid computation

            noise = torch.randn(batch_size, 100).type_as(x)
            fake_x = self.generator(noise, y)
            fake_pred = self.discriminator(fake_x, y)
            g_loss = fake_pred.mean()
            #g_loss = - fake_pred.mean()
            g_loss.backward()
            g_cost = -g_loss
            self.opt_g.step()
            self.log("g_loss", g_cost, on_step=False, on_epoch=True, prog_bar=True)

        for p in self.discriminator.parameters():
            p.requires_grad = True
        self.opt_d.zero_grad()

        # 1. Discriminator loss

        # Prediction on the real data
        real_pred = self.discriminator(x, y)
        d_loss_real = real_pred.mean()
        d_loss_real.backward()
        # Prediction on the mismatched/wrong labeled data
        # wrong_pred = self.discriminator(x, y.roll(1))
        # d_loss_wrong = torch.nn.ReLU()(1.0 + wrong_pred).mean()
        # Forward pass
        noise = torch.randn(batch_size, 100).type_as(x)
        fake_x = self.generator(noise, y)

        # Prediction on the generated images
        fake_pred = self.discriminator(fake_x.detach(), y)
        d_loss_fake = fake_pred.mean()
        d_loss_fake.backward()

        gradient_penalty = self.calculate_gradient_penalty(x, y)
        gradient_penalty.backward()
        d_loss = d_loss_fake - d_loss_real
     #   d_loss = d_loss_fake - d_loss_real + gradient_penalty
        Wasserstein_D = d_loss_real - d_loss_fake
       # self.manual_optimizer_step(self.opt_d)
        self.opt_d.step()

        self.log("d_loss_real", d_loss_real, on_step=False, on_epoch=True)
        self.log("d_loss_fake", d_loss_fake, on_step=False, on_epoch=True)
       # self.log("d_loss_wrong", d_loss_wrong, on_step=False, on_epoch=True)
        self.log("d_loss", d_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("wasserstein_d", Wasserstein_D, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x = batch["img"]
        y = batch["qa_embedding"]

        batch_size = self.cfg.TRAIN.BATCH_SIZE
        # Generate images
        z = torch.randn(batch_size, self.cfg.MODEL.Z_DIM).type_as(x)

        fake_x = self.forward(z, y)

        incep_mean, incep_std = self.inception.compute_score(fake_x, num_splits=1)

        self.log("Inception score (val)", incep_mean, on_step=False, on_epoch=True)

        if not self.trainer.running_sanity_check and batch_idx == 0:
            grid = gen_image_grid(fake_x, batch["text"])
           # grid = torchvision.utils.make_grid(fake_x, normalize=True)
            self.logger.experiment.add_image(f"Val epoch {self.current_epoch}",
                                             grid, global_step=self.current_epoch)

    def test_step(self, batch, batch_idx):
        x = batch["img"]
        y = batch["qa_embedding"]

        batch_size = self.cfg.TRAIN.BATCH_SIZE
        # Generate images
        z = torch.randn(batch_size, self.cfg.MODEL.Z_DIM).type_as(x)

        fake_x = self.forward(z, y)

        incep_mean, incep_std = self.inception.compute_score(fake_x, num_splits=1)

        self.log("Inception score (test)", incep_mean, on_step=False, on_epoch=True)

    def on_epoch_end(self):
        elapsed_time = time.perf_counter() - self.start
        self.start = time.perf_counter()
        self.print(
            f"\nEpoch {self.current_epoch} finished in {round(elapsed_time, 2)}s")

        if self.current_epoch % self.trainer.check_val_every_n_epoch == 0:
            noise = torch.randn((self.eval_y.size(0), self.cfg.MODEL.Z_DIM)).type_as(self.eval_y)
            recon_x = self.forward(noise, self.eval_y)
            grid = gen_image_grid(recon_x, self.eval_text)
          #  grid = torchvision.utils.make_grid(recon_x, normalize=True)
            self.logger.experiment.add_image(f"Epoch {self.current_epoch}", grid, global_step=self.current_epoch)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(),
                                 lr=self.cfg.TRAIN.G_LR, betas=(0.0, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(),
                                 lr=self.cfg.TRAIN.D_LR, betas=(0.0, 0.999))

        return opt_g, opt_d

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("loss", None)
        return items

    # def calculate_gradient_penalty(self, x, y):
    #     interpolated = x.requires_grad_()
    #     sent_inter = y.requires_grad_()
    #     out = self.discriminator(x, y)
    #     grads = torch.autograd.grad(outputs=out,
    #                                 inputs=(interpolated, sent_inter),
    #                                 grad_outputs=torch.ones(out.size()).type_as(x),
    #                                 retain_graph=True,
    #                                 create_graph=True,
    #                                 only_inputs=True)
    #     grad0 = grads[0].view(grads[0].size(0), -1)
    #     grad1 = grads[1].view(grads[1].size(0), -1)
    #     grad = torch.cat((grad0, grad1), dim=1)
    #     grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
    #     grad_penalty = torch.mean((grad_l2norm) ** 6) * 2
    #     return grad_penalty

    def calculate_gradient_penalty(self, x, y):
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

        return d_loss_gp
