from collections import OrderedDict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from pytorch_lightning.callbacks import GPUStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from scipy.stats import entropy
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from datasets import TextDataset
from light_datasets import CUB200DataModule
from misc.config import cfg, cfg_from_file
from models import NetD, NetG, INCEPTION_V3


class DFGAN(pl.LightningModule):
    def __init__(
        self,
        **kwargs
    ):

        super().__init__()
        #self.save_hyperparameters(cfg.TREE.BASE_SIZE, cfg.TRAIN.NF, cfg.TEXT.EMBEDDING_DIM, )

        # networks
        self.generator = self.init_generator()
        self.discriminator = self.init_discriminator()
        self.inception_model = INCEPTION_V3()
        self.predictions = []

    def init_generator(self):
        generator = NetG(cfg.TRAIN.NF, 100, cfg.TEXT.EMBEDDING_DIM, cfg.TREE.BASE_SIZE)
        return generator

    def init_discriminator(self):
        discriminator = NetD(cfg.TRAIN.NF, cfg.TEXT.EMBEDDING_DIM,  cfg.TREE.BASE_SIZE)
        return discriminator

    def forward(self, z, embeds):
        return self.generator(z, embeds)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_images, caption_embeds, captions_str, class_ids, keys = batch
        self.last_embeds = caption_embeds
        real_images = real_images[0]
        batch_size = cfg.TRAIN.BATCH_SIZE
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
        self.log('d_loss', d_loss+d_loss_gp, on_step=True, on_epoch=True, prog_bar=True)
        self.log('g_loss', g_loss, on_step=True, on_epoch=True, prog_bar=True)

        self.inception_model.eval()
        pred = self.inception_model(fake_images[-1].unsqueeze(0).detach())

        self.predictions.append(pred.data.cpu().numpy())

    def test_step(self, batch, batch_idx):
        real_images, caption_embeds, captions_str, class_ids, keys = batch

        batch_size = cfg.TRAIN.BATCH_SIZE
        # Generate images
        z = torch.randn(batch_size, 100).type_as(caption_embeds)
        fake_images = self.forward(z, caption_embeds)

        self.inception_model.eval()
        pred = self.inception_model(fake_images.detach()).data.cpu().numpy()
        incep_mean, incep_std = self.compute_inception_score(pred, num_splits=1)
        self.log("test_incep_mean", incep_mean)

    def configure_optimizers(self,):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=cfg.TRAIN.GENERATOR_LR, betas=(0.0, 0.9))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.0, 0.9))
        return opt_g, opt_d

    def on_epoch_end(self):
        z = torch.randn(cfg.TRAIN.BATCH_SIZE, 100).type_as(self.last_embeds)
        with torch.no_grad():
            fake_images = self.forward(z, self.last_embeds)

        grid = torchvision.utils.make_grid(fake_images, normalize=True)
        str_title = 'gen_{:04d}_images'.format(self.current_epoch)
        self.logger.experiment.add_image(str_title, grid, global_step=self.current_epoch)

        self.predictions = np.asfarray(self.predictions)

        incep_mean, incep_std = self.compute_inception_score(np.asarray(self.predictions), num_splits=1)

        self.logger.experiment.add_scalar("Inception Mean", incep_mean, self.current_epoch)
        self.predictions = []

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items.pop("loss", None)
        return items

    def compute_inception_score(self, preds, num_splits=1):
        N = preds.shape[0]
        # Now compute the mean kl-div
        split_scores = []

        for k in range(num_splits):
            part = preds[k * (N // num_splits): (k+1) * (N // num_splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        return np.mean(split_scores), np.std(split_scores)


cfg_from_file("./cfg/bird.yml")
# imsize = cfg.TREE.BASE_SIZE
# batch_size = cfg.TRAIN.BATCH_SIZE
# image_transform = transforms.Compose([
#     transforms.Resize(int(imsize * 76 / 64)),
#     transforms.RandomCrop(imsize),
#     transforms.RandomHorizontalFlip()])
# dataset = TextDataset(cfg.DATA_DIR, 'train',
#                       base_size=cfg.TREE.BASE_SIZE,
#                       transform=image_transform, encoder_type=cfg.TEXT.ENCODER)
# assert dataset
# dataloader = torch.utils.data.DataLoader(
#     dataset, batch_size=batch_size, drop_last=True,
#     shuffle=True, num_workers=4, pin_memory=True)

CUB200 = CUB200DataModule()
gpu_stats = GPUStatsMonitor()
epochs = cfg.TRAIN.MAX_EPOCH
epochs = 2
trainer = pl.Trainer(automatic_optimization=False, gpus=1, max_epochs=epochs, callbacks=[
                     gpu_stats], precision=16, limit_train_batches=0.05, limit_test_batches=0.05)
model = DFGAN()

trainer.fit(model, CUB200)
result = trainer.test(ckpt_path=None)
print(result)
