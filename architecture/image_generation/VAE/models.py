from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


class Encoder(nn.Module):
    def __init__(self, ndf, img_size, z_dim, conditional=False, ef_dim=None, disc=False):
        super(Encoder, self).__init__()
        self.disc = disc
        self.conditional = conditional
        self.ef_dim = ef_dim
        self.encode_img = nn.Sequential(
            nn.Conv2d(3, ndf, 3, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        h_dim = int(img_size / 8) * int(img_size / 8) * ndf * 4 + ef_dim if conditional else 0
        if self.disc:
            self.fc11 = nn.Linear(h_dim, 1)
        else:
            self.fc11 = nn.Linear(h_dim, z_dim)
            self.fc12 = nn.Linear(h_dim, z_dim)

    def forward(self, x, c=None):

        x = self.encode_img(x)
        # Flatten
        x = x.view(x.size(0), -1)
        if self.conditional:
            x = torch.cat((x, c), 1)
        if self.disc:
            pred = self.fc11(x)
            return F.sigmoid(pred)
        else:
            mu = self.fc11(x)
            mu = F.relu(mu)
            logvar = self.fc12(x)
            logvar = F.relu(logvar)
            return mu, logvar


class Decoder(nn.Module):
    def __init__(self, ngf, img_size, z_dim, conditional=False, ef_dim=None):
        super().__init__()
        self.ngf = ngf
        self.img_size = img_size
        self.conditional = conditional
        self.ef_dim = ef_dim

        h_dim = z_dim + ef_dim if conditional else 0
        self.fc1 = nn.Sequential(
            nn.Linear(h_dim, ngf * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(ngf * 8 * 4 * 4),
            nn.LeakyReLU(0.2))

        self.decode_img = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 4, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf * 4, ngf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf * 2, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf, 3, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x, c=None):
        if self.conditional:
            x = torch.cat((x, c), 1)
        x = self.fc1(x)
        x = x.view(x.size(0), self.ngf * 8, 4, 4)

        x = self.decode_img(x)
        return x


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
