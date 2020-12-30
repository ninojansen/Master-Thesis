
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
from models import Encoder, Decoder, reparameterize
import time


class VAE(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg.MODEL.NF, z_dim=cfg.MODEL.Z_DIM, img_size=cfg.IM_SIZE)
        self.decoder = Decoder(cfg.MODEL.NF, z_dim=cfg.MODEL.Z_DIM, img_size=cfg.IM_SIZE)
        self.kl_loss = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        self.start = time.perf_counter()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.decoder(x)

    def process_data(self, batch, dataset_name):
        if dataset_name == "easy_vqa":
            images, questions, answers, combined = batch
            return images, combined
        elif dataset_name == "cifar10":
            images, target = batch
            return images, target

    def training_step(self, batch, batch_idx):
        x, y = self.process_data(batch, self.cfg.DATASET_NAME)

        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        x_hat = self.decoder(z)

        recon_loss = F.mse_loss(x_hat, x)
        kl_loss = self.kl_loss(mu, logvar)

        loss = recon_loss + kl_loss
        self.log('train_loss', loss)
        return loss

    def on_epoch_end(self):
        elapsed_time = time.perf_counter() - self.start
        self.start = time.perf_counter()
        self.print(
            f"\nEpoch {self.current_epoch} finished in {round(elapsed_time, 2)}s")

        noise = torch.rand((25, self.cfg.MODEL.Z_DIM)).cuda()
        x_hat = self.forward(noise)
        grid = torchvision.utils.make_grid(x_hat, normalize=True)
        self.logger.experiment.add_image(f"Epoch {self.current_epoch}", grid, global_step=self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.cfg.TRAIN.LR, betas=(0.5, 0.999))
        return optimizer
