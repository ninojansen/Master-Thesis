from __future__ import print_function
import argparse
import torch
from torch.nn.modules.batchnorm import BatchNorm2d
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Based on https://github.com/Zeleni9/pytorch-wgan


class Encoder(nn.Module):
    def __init__(self, nf, img_size, z_dim, ef_dim, disc=False):
        super(Encoder, self).__init__()
        self.disc = disc
        self.ef_dim = ef_dim

        # self.encode_img = nn.Sequential(
        #     nn.Conv2d(3, nf * 4, 3, 2, 1,),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(nf * 4, nf * 8, 3, 2, 1,),
        #     nn.InstanceNorm2d(nf * 8, affine=True),
        #     nn.LeakyReLU(0.2, inplace=True),

        #     nn.Conv2d(nf * 8, nf * 16, 3, 2, 1,),
        #     nn.InstanceNorm2d(nf * 16, affine=True),
        #     nn.LeakyReLU(0.2, inplace=True),

        #     nn.Conv2d(nf * 16, nf * 16, 3, 2, 1,),
        #     nn.InstanceNorm2d(nf * 16, affine=True),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # nn.Conv2d(nf * 2, nf * 4, 3, 2, 1,),
        #     # nn.InstanceNorm2d(nf * 4, affine=True),
        #     # nn.LeakyReLU(0.2, inplace=True),

        #     # nn.Conv2d(nf * 4, nf * 8, 3, 2, 1,),
        #     # nn.InstanceNorm2d(nf * 8, affine=True),
        #     # nn.LeakyReLU(0.2, inplace=True),
        # )
        self.encode_img = nn.Sequential(
            # 64x64
            nn.Conv2d(3, nf * 4, 3, 2, 1),
            nn.InstanceNorm2d(nf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32
            nn.Conv2d(nf * 4, nf * 8, 3, 2, 1,),
            nn.InstanceNorm2d(nf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16
            nn.Conv2d(nf * 8, nf * 16, 3, 2, 1,),
            nn.InstanceNorm2d(nf * 16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8
            nn.Conv2d(nf * 16, nf * 32, 3, 2, 1,),
            nn.InstanceNorm2d(nf * 32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x4
        )

        h_dim = 4 * 4 * nf * 8 + ef_dim
        #h_dim = int(img_size / 8) * int(img_size / 8) * nf * 16 + ef_dim

        self.joint_conv = nn.Sequential(
            nn.Conv2d(nf * 32 + ef_dim, nf * 2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        if self.disc:
            self.out = nn.Sequential(nn.Conv2d(nf * 2, 1, 3, 1, 0))
        else:
            self.fc11 = nn.Linear(h_dim, z_dim)
            self.fc12 = nn.Linear(h_dim, z_dim)

    def forward(self, x, y):

        x = self.encode_img(x)
        # Flatten
     #   x = x.view(x.size(0), -1)
      #  x = torch.cat((x, c), 1)

        y = y.view(-1, self.ef_dim, 1, 1)
        y = y.repeat(1, 1, 4, 4)
        h_c_code = torch.cat((x, y), 1)
        x = self.joint_conv(h_c_code)

        if self.disc:
            return self.out(x)
        else:
            x = x.view(x.size(0), -1)
            mu = self.fc11(x)
            mu = F.relu(mu)
            logvar = self.fc12(x)
            logvar = F.relu(logvar)
            return mu, logvar


class Decoder(nn.Module):
    def __init__(self, nf, img_size, z_dim, ef_dim):
        super().__init__()
        self.nf = nf
        self.img_size = img_size
        self.ef_dim = ef_dim

        h_dim = z_dim + ef_dim
        self.fc1 = nn.Sequential(
            nn.Linear(h_dim, nf * 32 * 4 * 4, bias=False),
            nn.ReLU((True)))

        self.decode_img = nn.Sequential(
            nn.Conv2d(nf * 32, nf * 32, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(nf * 32, nf * 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(nf * 16),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(nf * 16, nf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(nf * 8, nf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(nf * 4, 3, 3, 1, 1),
            nn.Tanh(),
        )
        # self.decode_img = nn.Sequential(

        #     nn.ConvTranspose2d(nf * 32, nf * 32, 4, 1, 1),
        #     nn.BatchNorm2d(nf * 32),
        #     nn.LeakyReLU(0.2),
        #     nn.ConvTranspose2d(nf * 32, nf * 16, 4, 2, 1),
        #     nn.BatchNorm2d(nf * 16),
        #     nn.LeakyReLU(0.2),
        #     #8x8
        #     nn.ConvTranspose2d(nf * 16, nf * 8, 4, 2, 1),
        #     nn.BatchNorm2d(nf * 8),
        #     nn.LeakyReLU(0.2),
        #     # 16x16
        #     nn.ConvTranspose2d(nf * 8, nf * 4, 4, 2, 1),
        #     nn.BatchNorm2d(nf * 4),
        #     nn.LeakyReLU(0.2),
        #     # 32x32
        #     nn.ConvTranspose2d(nf * 4, 3, 4, 2, 1),
        #     # 64x64
        #     nn.Tanh()
        # )

    def forward(self, x, c):
        x = torch.cat((x, c), 1)
        x = self.fc1(x)
        x = x.view(x.size(0), self.nf * 32, 4, 4)

        x = self.decode_img(x)
        return x


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
