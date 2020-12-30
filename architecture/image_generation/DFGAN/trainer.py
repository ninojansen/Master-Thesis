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
import math
from sentence_transformers import SentenceTransformer


class DFGAN(pl.LightningModule):
    def __init__(self, cfg, train_img_interval=5, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(self.cfg)

        # networks
        self.train_img_interval = train_img_interval
        self.cfg = edict(cfg)
        self.generator = self.init_generator()
        self.discriminator = self.init_discriminator()
        self.inception_model = INCEPTION_V3()
        self.predictions = []
        self.start = time.perf_counter()
        self.opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.cfg.TRAIN.GENERATOR_LR, betas=(0.0, 0.9))
        self.opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.0, 0.9))
        if self.cfg.DATASET_NAME == "easyVQA":
            self.text_encoder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

    def init_generator(self):
        generator = NetG(self.cfg.TRAIN.NF, 100, self.cfg.TEXT.EMBEDDING_DIM, self.cfg.TREE.BASE_SIZE)
        return generator

    def init_discriminator(self):
        discriminator = NetD(self.cfg.TRAIN.NF, self.cfg.TEXT.EMBEDDING_DIM, self.cfg.TREE.BASE_SIZE)
        return discriminator

    def forward(self, z, embeds):
        return self.generator(z, embeds)

    def process_data(self, data):
        if self.cfg.DATASET_NAME == "CUB200":
            real_images, caption_embeds, captions_str, class_ids, keys = data

            return real_images, caption_embeds, captions_str
        elif self.cfg.DATASET_NAME == "easyVQA":
            images, questions, answers, combined = data
            embeddings = torch.from_numpy(self.text_encoder.encode(combined)).type_as(images[0].data)
            return images, embeddings, combined
        else:
            raise(f"Dataset {self.cfg.DATASET_NAME} not supported.")

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_images, embeddings, raw_strings = self.process_data(batch)

        # self.last_embeds = caption_embeds
        # self.last_str = captions_str

        real_images = real_images[0]
        batch_size = self.cfg.TRAIN.BATCH_SIZE
        # ignore optimizer_idx
      #  (opt_g, opt_d) = self.configure_optimizers()
        opt_g = self.opt_g
        opt_d = self.opt_d
        # Calculate D errors on real data and mismatched data
        real_features = self.discriminator(real_images)
        output = self.discriminator.COND_DNET(real_features, embeddings)
        d_loss_real = torch.nn.ReLU()(1.0 - output).mean()

        output = self.discriminator.COND_DNET(real_features[:(batch_size - 1)], embeddings[1:batch_size])
        d_loss_mismatch = torch.nn.ReLU()(1.0 + output).mean()

        # Generate images
        z = torch.randn(batch_size, 100).type_as(real_images)
        fake_images = self.forward(z, embeddings)

        if self.current_epoch % self.train_img_interval == 0 and batch_idx == 0:
            grid = image_grid(fake_images, raw_strings)
            self.logger.experiment.add_image(f"Train {self.current_epoch}",
                                             grid, global_step=self.current_epoch, dataformats="HWC")
        # Calculate D error on generated images
        fake_features = self.discriminator(fake_images.detach())
        d_loss_fake = self.discriminator.COND_DNET(fake_features, embeddings)
        d_loss_fake = torch.nn.ReLU()(1.0 + d_loss_fake).mean()

        d_loss = d_loss_real + (d_loss_fake + d_loss_mismatch) / 2.0

        # Update the discriminator with the regular loss
        opt_d.zero_grad()
        opt_g.zero_grad()
        self.manual_backward(d_loss, opt_d)
        self.manual_optimizer_step(opt_d)

        # Update the discriminator loss with MA-GP
        interpolated = (real_images.data).requires_grad_()
        sent_inter = (embeddings.data).requires_grad_()
        features = self.discriminator(interpolated)
        out = self.discriminator.COND_DNET(features, sent_inter)
        grads = torch.autograd.grad(outputs=out,
                                    inputs=(interpolated, sent_inter),
                                    grad_outputs=torch.ones(out.size()).type_as(out),
                                    retain_graph=True,
                                    create_graph=True,
                                    only_inputs=True)
        grad0 = grads[0].view(grads[0].size(0), -1)
        grad1 = grads[1].view(grads[1].size(0), -1)
        grad = torch.cat((grad0, grad1), dim=1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm) ** 6)
        d_loss_gp2 = 2.0 * d_loss_gp
        opt_d.zero_grad()
        opt_g.zero_grad()
        self.manual_backward(d_loss_gp2, opt_d)
        self.manual_optimizer_step(opt_d)

        # # Update the Generator through adversarial loss
        features = self.discriminator(fake_images)
        output = self.discriminator.COND_DNET(features, embeddings)
        g_loss = - output.mean()
        opt_g.zero_grad()
        opt_d.zero_grad()
        self.manual_backward(g_loss, opt_g)
        self.manual_optimizer_step(opt_g)
        # log losses
        self.log('Discriminator loss', d_loss)
        self.log('Generator loss', g_loss)

    def test_step(self, batch, batch_idx):
        real_images, embeddings, raw_strings = self.process_data(batch)

        batch_size = self.cfg.TRAIN.BATCH_SIZE
        # Generate images
        z = torch.randn(batch_size, 100).type_as(embeddings)
        self.inception_model.eval()
        with torch.no_grad():
            fake_images = self.forward(z, embeddings)
            noise_images = self.forward(z, embeddings + torch.randn_like(embeddings))
            pred_fake = self.inception_model(fake_images.detach()).data.cpu().numpy()
            pred_noise = self.inception_model(noise_images.detach()).data.cpu().numpy()

        incep_mean, incep_std = compute_inception_score(pred_fake, num_splits=1)
        incep_noise_mean, _ = compute_inception_score(pred_noise, num_splits=1)

        self.log("Inception score (test)", incep_mean)
        self.log("Inception noise score (test)", incep_noise_mean)

    def validation_step(self, batch, batch_idx):
        real_images, embeddings, raw_strings = self.process_data(batch)

        batch_size = self.cfg.TRAIN.BATCH_SIZE

        # Generate images
        z = torch.randn(batch_size, 100).type_as(embeddings)
        self.inception_model.eval()
        with torch.no_grad():
            fake_images = self.forward(z, embeddings)
            noise_images = self.forward(z, embeddings + torch.randn_like(embeddings))
            pred_fake = self.inception_model(fake_images.detach()).data.cpu().numpy()
            pred_noise = self.inception_model(noise_images.detach()).data.cpu().numpy()

        incep_mean, incep_std = compute_inception_score(pred_fake, num_splits=1)
        incep_noise_mean, _ = compute_inception_score(pred_noise, num_splits=1)

        if not self.trainer.running_sanity_check and batch_idx == 0:
            grid = image_grid(fake_images, raw_strings)
            self.logger.experiment.add_image(f"Val epoch {self.current_epoch}",
                                             grid, global_step=self.current_epoch, dataformats="HWC")

        return incep_mean, incep_noise_mean

    def validation_epoch_end(self, validation_step_outputs):
        if len(validation_step_outputs) > 1:
            incep, incep_noise = np.mean(validation_step_outputs[0]), np.mean(validation_step_outputs[1])
            self.logger.experiment.add_scalar("Inception score (val)", incep, self.current_epoch)
            self.logger.experiment.add_scalar("Inception noise score (val)", incep_noise, self.current_epoch)
            self.print(f"\nEpoch {self.current_epoch} validation: incep={incep}, incep_noise={incep_noise}\n")

    def configure_optimizers(self,):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.cfg.TRAIN.GENERATOR_LR, betas=(0.0, 0.9))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.0, 0.9))
        return opt_g, opt_d

    def on_epoch_end(self):
        elapsed_time = time.perf_counter() - self.start
        self.start = time.perf_counter()
        self.print(
            f"\nEpoch {self.current_epoch} finished in {round(elapsed_time, 2)}s")

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items.pop("loss", None)
        return items
