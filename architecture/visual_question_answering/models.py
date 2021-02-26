
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules import activation
# https://victorzhou.com/blog/easy-vqa/


class SimpleVQA(nn.Module):
    def __init__(self, im_size, ef_dim, n_answers, pretrained_img=False, im_dim=None):
        super(SimpleVQA, self).__init__()
        self.n_answers = n_answers
        self.pretrained_img = pretrained_img
        self.im_dim = im_dim
        if not pretrained_img:
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.conv3 = nn.Conv2d(16, 32, 5)
            self.pool = nn.MaxPool2d(2, 2)
        if pretrained_img:
            self.fc1 = nn.Linear(im_dim, 32)
        else:
            self.fc1 = nn.Linear(32 * 4 * 4, 32)

        self.question = nn.Sequential(nn.Linear(ef_dim, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())

        self.out = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, n_answers))

    def forward(self, x, y):
        if not self.pretrained_img:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 32 * 4 * 4)

        x = F.relu(self.fc1(x))

        y = self.question(y)

        z = x * y

        z = self.out(z)
        return z


class AbstractVQA(nn.Module):
    def __init__(self, ef_dim, n_answers, im_dim=None):
        super(AbstractVQA, self).__init__()
        self.n_answers = n_answers
        self.pretrained_img = pretrained_img
        self.im_dim = im_dim
        self.fc1 = nn.Linear(im_dim, 32)
        self.question = nn.Sequential(nn.Linear(ef_dim, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())

        self.out = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, n_answers))

    def forward(self, x, y):
        x = F.relu(self.fc1(x))

        y = self.question(y)

        z = x * y

        z = self.out(z)
        return z


if __name__ == "__main__":
    model = SimpleVQA(64, 34, 13)
    print(model)
