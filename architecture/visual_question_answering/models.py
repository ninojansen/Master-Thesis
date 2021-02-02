
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules import activation
# https://victorzhou.com/blog/easy-vqa/


class SimpleVQA(nn.Module):
    def __init__(self, im_dim, ef_dim, n_answers):
        super(SimpleVQA, self).__init__()
        self.n_answers = n_answers

        self.image = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(im_dim, 32), nn.Tanh())

        self.question = nn.Sequential(nn.Linear(ef_dim, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())

        self.out = nn.Sequential(nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, n_answers), nn.LogSoftmax(dim=1))

    def forward(self, x, y):
        x = self.image(x)
        y = self.question(y)

        z = x * y

        z = self.out(z)
        return z
