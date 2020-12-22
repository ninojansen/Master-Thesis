import numpy as np
import torch
import torchvision.datasets as dset
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from misc.config import cfg


class CVAE(nn.Module):
    def __init__(self, hidden_dim=512, latent_dim=2):
        super().__init__()
        input_dim = cfg.TREE.BASE_SIZE*cfg.TREE.BASE_SIZE+cfg.TEXT.EMBEDDING_DIM
        self.encoder = Encoder(input_dim,
                               hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, cfg.TREE.BASE_SIZE*cfg.TREE.BASE_SIZE)

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.mu = torch.nn.Linear(hidden_dim, latent_dim)
        self.var = torch.nn.Linear(hidden_dim, latent_dim)

    def forward(self, x_embed, x_im):
        hidden = F.relu(self.fc1(torch.cat([x_embed, nn.Flatten(x_im)])))
        mean = self.mu(hidden)
        log_var = self.var(hidden)
        return mean, log_var


class Decoder(nn.Module):
    ''' This the decoder part of VAE

    '''

    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()

        self.latent_to_hidden = nn.Linear(latent_dim + n_classes, hidden_dim)
        self.hidden_to_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is of shape [batch_size, latent_dim + num_classes]
        x = F.relu(self.latent_to_hidden(x))
        # x is of shape [batch_size, hidden_dim]
        generated_x = F.sigmoid(self.hidden_to_out(x))
        # x is of shape [batch_size, output_dim]

        return generated_x
