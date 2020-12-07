from collections import OrderedDict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
import random
from pytorch_lightning.callbacks import GPUStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from scipy.stats import entropy
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from easydict import EasyDict as edict
import time
from models import INCEPTION_V3, NetD, NetG
from misc.utils import image_grid, compute_inception_score
import matplotlib.pyplot as plt
import io
from PIL import Image


class DFGAN(pl.LightningModule):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        #self.save_hyperparameters(self.cfg.TREE.BASE_SIZE, self.cfg.TRAIN.NF, self.cfg.TEXT.EMBEDDING_DIM, )
        self.cfg = cfg
        self.save_hyperparameters(self.cfg)

        # networks
        self.cfg = edict(cfg)
        self.generator = self.init_generator()
        self.discriminator = self.init_discriminator()
        self.inception_model = INCEPTION_V3()
        self.predictions = []
        self.start = time.perf_counter()

    def init_generator(self):
        generator = NetG(self.cfg.TRAIN.NF, 100, self.cfg.TEXT.EMBEDDING_DIM, self.cfg.TREE.BASE_SIZE)
        return generator

    def init_discriminator(self):
        discriminator = NetD(self.cfg.TRAIN.NF, self.cfg.TEXT.EMBEDDING_DIM,  self.cfg.TREE.BASE_SIZE)
        return discriminator

    def forward(self, z, embeds):
        return self.generator(z, embeds)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_images, caption_embeds, captions_str, class_ids, keys = batch
        self.last_embeds = caption_embeds
        self.last_str = captions_str

        real_images = real_images[0]
        batch_size = self.cfg.TRAIN.BATCH_SIZE
        # ignore optimizer_idx
        (opt_g, opt_d) = self.configure_optimizers()

        # Calculate D errors on real data and mismatched data
        real_features = self.discriminator(real_images)
        output = self.discriminator.COND_DNET(real_features, caption_embeds)
        d_loss_real = torch.nn.ReLU()(1.0 - output).mean()

        output = self.discriminator.COND_DNET(real_features[:(batch_size - 1)], caption_embeds[1:batch_size])
        d_loss_mismatch = torch.nn.ReLU()(1.0 + output).mean()

        # Generate images
        z = torch.randn(batch_size, 100).type_as(real_images)
        fake_images = self.forward(z, caption_embeds)

        # Calculate D error on generated images
        fake_features = self.discriminator(fake_images.detach())
        d_loss_fake = self.discriminator.COND_DNET(fake_features, caption_embeds)
        d_loss_fake = torch.nn.ReLU()(1.0 + d_loss_fake).mean()

        d_loss = d_loss_real + (d_loss_fake + d_loss_mismatch)/2.0

        # Update the discriminator with the regular loss
        opt_d.zero_grad()
        opt_g.zero_grad()
        self.manual_backward(d_loss, opt_d)
        self.manual_optimizer_step(opt_d)

        # Update the discriminator loss with MA-GP
        interpolated = (real_images.data).requires_grad_()
        sent_inter = (caption_embeds.data).requires_grad_()
        features = self.discriminator(interpolated)
        out = self.discriminator.COND_DNET(features, sent_inter)
        grads = torch.autograd.grad(outputs=out,
                                    inputs=(interpolated, sent_inter),
                                    grad_outputs=torch.ones(out.size()).cuda(),
                                    retain_graph=True,
                                    create_graph=True,
                                    only_inputs=True)
        grad0 = grads[0].view(grads[0].size(0), -1)
        grad1 = grads[1].view(grads[1].size(0), -1)
        grad = torch.cat((grad0, grad1), dim=1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm) ** 6)
        d_loss_gp = 2.0 * d_loss_gp
        opt_d.zero_grad()
        opt_g.zero_grad()
        self.manual_backward(d_loss_gp, opt_d)
        self.manual_optimizer_step(opt_d)

        # Update the Generator through adversarial loss
        features = self.discriminator(fake_images)
        output = self.discriminator.COND_DNET(features, caption_embeds)
        g_loss = - output.mean()
        opt_g.zero_grad()
        opt_d.zero_grad()
        self.manual_backward(g_loss, opt_g)
        self.manual_optimizer_step(opt_g)
        # log losses
        self.log('d_loss', d_loss+d_loss_gp, prog_bar=True)
        self.log('g_loss', g_loss, prog_bar=True)

        self.inception_model.eval()
      #  pred = self.inception_model(fake_images[-1].unsqueeze(0).detach())
     #   pred = self.inception_model(fake_images.detach()).data.cpu().numpy()
        self.predictions.append(self.inception_model(fake_images.detach()[:5]).data.cpu().numpy())

    def test_step(self, batch, batch_idx):
        real_images, caption_embeds, captions_str, class_ids, keys = batch

        batch_size = self.cfg.TRAIN.BATCH_SIZE
        # Generate images
        z = torch.randn(batch_size, 100).type_as(caption_embeds)
        fake_images = self.forward(z, caption_embeds)

        self.inception_model.eval()
        pred = self.inception_model(fake_images.detach()).data.cpu().numpy()
        incep_mean, incep_std = compute_inception_score(pred, num_splits=1)

        self.log("test_incep_mean", incep_mean)

    def configure_optimizers(self,):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.cfg.TRAIN.GENERATOR_LR, betas=(0.0, 0.9))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.0, 0.9))
        return opt_g, opt_d

    def on_epoch_end(self):
        z = torch.randn(self.cfg.TRAIN.BATCH_SIZE, 100).type_as(self.last_embeds)
        with torch.no_grad():
            fake_images = self.forward(z, self.last_embeds)

       # grid_xx = self.image_grid(np.swapaxes(fake_images.data.cpu().numpy(), 1, 3), self.last_str)
        grid = image_grid(fake_images, self.last_str)
        #grid = torchvision.utils.make_grid(fake_images, normalize=True)
        self.logger.experiment.add_image(f"Epoch {self.current_epoch}", grid,
                                         global_step=self.current_epoch, dataformats="HWC")

        pred = np.asfarray(self.predictions)
        pred = np.reshape(pred, (pred.shape[0] * pred.shape[1], 1000))
        incep_mean, incep_std = compute_inception_score(pred, num_splits=10)

        self.log("val_incep_mean", incep_mean)
       # self.logger.experiment.add_scalar("Inception Mean", incep_mean, self.current_epoch)
        self.predictions = []
        elapsed_time = time.perf_counter() - self.start
        self.start = time.perf_counter()
        self.logger.experiment.add_scalar("Elapsed Time per epoch", elapsed_time, self.current_epoch)
        self.print(f"Epoch {self.current_epoch} took {elapsed_time} seconds val_incep_={incep_mean}")

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items.pop("loss", None)
        return items
