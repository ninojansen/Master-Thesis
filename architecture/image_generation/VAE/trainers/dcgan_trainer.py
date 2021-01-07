
import pytorch_lightning as pl
from torch.utils import data
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import torch.nn.functional as F
from torch import nn
import torch
import torchvision
import os
from architecture.image_generation.VAE.models.dcgan_model import *
import time
from architecture.utils.inception_score import InceptionScore


class VAE(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg.MODEL.NF, cfg.IM_SIZE, cfg.MODEL.Z_DIM, conditional=True, ef_dim=cfg.MODEL.EF_DIM)
        self.decoder = Decoder(cfg.MODEL.NF, cfg.IM_SIZE, cfg.MODEL.Z_DIM, conditional=True, ef_dim=cfg.MODEL.EF_DIM)
        self.kl_loss = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        self.start = time.perf_counter()

        self.eval_y = None
        self.eval_size = 50

    def forward(self, x, y=None):
        # in lightning, forward defines the prediction/inference actions
        return self.decoder(x, y)

    def process_data(self, batch, dataset_name):
        if dataset_name == "easy_vqa":
            images, questions, answers, combined = batch
            return images, combined
        elif dataset_name == "cifar10":
            images, target = batch
            target = F.one_hot(target, num_classes=10)
            if self.eval_y == None:
                self.eval_y = F.one_hot(torch.randint(0, 10, (self.eval_size,))).type_as(target)
            return images, target

    def training_step(self, batch, batch_idx):
        x, y = self.process_data(batch, self.cfg.DATASET_NAME)

        mu, logvar = self.encoder(x, y)
        z = reparameterize(mu, logvar)
        recon_x = self.decoder(z, y)

        recon_loss = F.mse_loss(
            recon_x.view(-1, self.cfg.IM_SIZE ** 2 * 2),
            x.view(-1, self.cfg.IM_SIZE ** 2 * 2),
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

        noise = torch.randn((self.eval_size, self.cfg.MODEL.Z_DIM)).cuda()
        recon_x = self.forward(noise, self.eval_y)
        grid = torchvision.utils.make_grid(recon_x, normalize=True)
        self.logger.experiment.add_image(f"Epoch {self.current_epoch}", grid, global_step=self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.cfg.TRAIN.VAE_LR, betas=(0.5, 0.999))
        return optimizer


class DCGAN(pl.LightningModule):

    def __init__(self, cfg, pretrained_decoder=None):
        super().__init__()
        self.cfg = cfg
        if pretrained_decoder:
            self.generator = pretrained_decoder
        else:
            self.generator = Decoder(cfg.MODEL.NF, cfg.IM_SIZE, cfg.MODEL.Z_DIM,
                                     conditional=True, ef_dim=cfg.MODEL.EF_DIM)

        self.discriminator = Encoder(cfg.MODEL.NF, cfg.IM_SIZE, cfg.MODEL.Z_DIM,
                                     conditional=True, ef_dim=cfg.MODEL.EF_DIM, disc=True)

        self.adversarial_loss = lambda y_hat, y: F.binary_cross_entropy(y_hat, y)
        self.start = time.perf_counter()

        self.inception = InceptionScore()
        self.eval_y = None
        self.eval_size = 50

    def forward(self, z, y=None):
        # in lightning, forward defines the prediction/inference actions
        return self.generator(z, y)

    def process_data(self, batch, dataset_name):
        if dataset_name == "easy_vqa":
            images, questions, answers, combined = batch
            return images, combined
        elif dataset_name == "cifar10":
            images, target = batch
            target = F.one_hot(target, num_classes=10)
            if self.eval_y == None:
                self.eval_y = F.one_hot(torch.randint(0, 10, (self.eval_size,))).type_as(target)
            return images, target

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = self.process_data(batch, self.cfg.DATASET_NAME)

        z = torch.randn(x.shape[0], self.cfg.MODEL.Z_DIM).type_as(x)

        # Train generator
        if optimizer_idx == 0:
            self.generated_x = self(z, y)
            valid = torch.ones(x.size(0), 1).type_as(x)
            g_loss = self.adversarial_loss(self.discriminator(self(z, y), y), valid)
            self.log("g_loss", g_loss, prog_bar=True, on_step=True)
            return g_loss
        # Train discriminator
        if optimizer_idx == 1:
            valid = torch.ones(x.size(0), 1).type_as(x)
            fake = torch.zeros(x.size(0), 1).type_as(x)

            real_loss = self.adversarial_loss(self.discriminator(x, y), valid)
            fake_loss = self.adversarial_loss(self.discriminator(self(z, y).detach(), y), fake)

            d_loss = (real_loss + fake_loss) / 2
            self.log("d_loss", d_loss, prog_bar=True, on_step=True)
            return d_loss

    def validation_step(self, batch, batch_idx):
        x, y = self.process_data(batch, self.cfg.DATASET_NAME)

        batch_size = self.cfg.TRAIN.BATCH_SIZE
        # Generate images
        z = torch.randn(batch_size, self.cfg.MODEL.Z_DIM).type_as(x)

        generated_x = self.forward(z, y)

        incep_mean, incep_std = self.inception.compute_score(generated_x, num_splits=1)

        self.log("Inception score (val)", incep_mean)

        if not self.trainer.running_sanity_check and batch_idx == 0:
            grid = torchvision.utils.make_grid(generated_x, normalize=True)
            self.logger.experiment.add_image(f"Val epoch {self.current_epoch}",
                                             grid, global_step=self.current_epoch)

    def test_step(self, batch, batch_idx):
        x, y = self.process_data(batch, self.cfg.DATASET_NAME)

        batch_size = self.cfg.TRAIN.BATCH_SIZE
        # Generate images
        z = torch.randn(batch_size, self.cfg.MODEL.Z_DIM).type_as(x)

        generated_x = self.forward(z, y)

        incep_mean, incep_std = self.inception.compute_score(generated_x, num_splits=1)

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
            self.logger.experiment.add_image(f"Train Epoch {self.current_epoch}", grid, global_step=self.current_epoch)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.parameters(),
                                 lr=self.cfg.TRAIN.G_LR, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.parameters(),
                                 lr=self.cfg.TRAIN.D_LR, betas=(0.5, 0.999))
        return opt_g, opt_d
