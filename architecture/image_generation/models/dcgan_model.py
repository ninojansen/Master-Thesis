from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


class Encoder(nn.Module):
    def __init__(self, nf, img_size, z_dim, conditional=False, ef_dim=None, disc=False):
        super(Encoder, self).__init__()
        self.disc = disc
        self.conditional = conditional
        self.ef_dim = ef_dim
        self.encode_img = nn.Sequential(
            nn.Conv2d(3, nf, 3, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(nf, nf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(nf * 2, nf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        h_dim = int(img_size / 8) * int(img_size / 8) * nf * 4 + ef_dim if conditional else 0
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
            return torch.sigmoid(pred)
        else:
            mu = self.fc11(x)
            mu = F.relu(mu)
            logvar = self.fc12(x)
            logvar = F.relu(logvar)
            return mu, logvar


class Decoder(nn.Module):
    def __init__(self, nf, img_size, z_dim, conditional=False, ef_dim=None):
        super().__init__()
        self.nf = nf
        self.img_size = img_size
        self.conditional = conditional
        self.ef_dim = ef_dim

        h_dim = z_dim + ef_dim if conditional else 0
        self.fc1 = nn.Sequential(
            nn.Linear(h_dim, nf * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(nf * 8 * 4 * 4),
            nn.LeakyReLU(0.2))

        self.decode_img = nn.Sequential(
            nn.Conv2d(nf * 8, nf * 4, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(nf * 4, nf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(nf, 3, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x, c=None):
        if self.conditional:
            x = torch.cat((x, c), 1)
        x = self.fc1(x)
        x = x.view(x.size(0), self.nf * 8, 4, 4)

        x = self.decode_img(x)
        return x


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
