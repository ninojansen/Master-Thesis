from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


class Encoder(nn.Module):
    def __init__(self, ndf, img_size, z_dim=30):
        super(Encoder, self).__init__()
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
        h_dim = int(img_size / 8) * int(img_size / 8) * ndf * 4
        self.fc11 = nn.Linear(h_dim, z_dim)
        self.fc12 = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        x = self.encode_img(x)
        # Flatten
        x = x.view(x.size(0), -1)

        mu = self.fc11(x)
        mu = F.relu(mu)
        logvar = self.fc12(x)
        logvar = F.relu(logvar)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, ngf, z_dim, img_size):
        super().__init__()
        self.ngf = ngf
        self.img_size = img_size
        self.fc1 = nn.Sequential(
            nn.Linear(z_dim, ngf * 8 * 4 * 4, bias=False),
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

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(x.size(0), self.ngf * 8, 4, 4)

        x = self.decode_img(x)
        return x


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
