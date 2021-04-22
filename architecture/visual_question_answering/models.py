
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules import activation
import os
import sys

from torch.nn.modules.pooling import MaxPool2d
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from architecture.utils.utils import weights_init, model_summary
# https://victorzhou.com/blog/easy-vqa/


class SimpleVQA(nn.Module):
    def __init__(self, im_size, ef_dim, n_answers, n_hidden):
        super(SimpleVQA, self).__init__()
        self.n_answers = n_answers
        self.im_size = im_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, 5, stride=1, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))

        self.conv3 = nn.Sequential(nn.Conv2d(16, 32, 5, stride=1, padding=2), nn.BatchNorm2d(32), nn.ReLU(),
                                   nn.MaxPool2d(2, 2))
        self.conv4 = nn.Sequential(nn.Conv2d(32, 64, 5, stride=1, padding=2), nn.BatchNorm2d(64), nn.ReLU(),
                                   nn.MaxPool2d(2, 2))

        self.im_dim = int(((im_size / 2**4)**2)) * 64
        self.fc1 = nn.Linear(self.im_dim, n_hidden)

        self.question = nn.Sequential(nn.Linear(ef_dim, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_hidden), nn.ReLU())

        self.out = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(n_hidden, n_answers))

    def forward(self, x, y):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(-1, self.im_dim)

        x = F.relu(self.fc1(x))

        y = self.question(y)

        z = x * y

        z = self.out(z)
        return z


class PretrainedVQA(nn.Module):
    def __init__(self, ef_dim, n_answers, n_hidden, im_dim=None):
        super(PretrainedVQA, self).__init__()
        self.n_answers = n_answers
        self.im_dim = im_dim
        self.n_hidden = n_hidden

        self.fc1 = nn.Linear(im_dim, n_hidden)
        self.question = nn.Sequential(nn.Linear(ef_dim, n_hidden), nn.ReLU())
        self.out = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(n_hidden, n_answers))
        self.apply(weights_init)

    def forward(self, img, question):
        x = F.relu(self.fc1(img))
     #   x = F.dropout(x, 0.5)

        y = self.question(question)
       # y =F.dropout(y, 0.5)
        z = x * y

        z = self.out(z)
        return z

# https://github.com/hengyuan-hu/bottom-up-attention-vqa/blob/master/attention.py

# https://github.com/MILVLG/bottom-up-attention.pytorch


class AttentionVQA(nn.Module):
    def __init__(self, ef_dim, v_dim, n_hidden, n_answers):
        super(AttentionVQA, self).__init__()
        self.att_non_linear = GatedTanh(v_dim + ef_dim, n_hidden)
        self.att_linear = nn.Linear(n_hidden, v_dim)

        self.ef_non_linear = GatedTanh(ef_dim, n_hidden)
        self.v_non_linear = GatedTanh(v_dim, n_hidden)

        self.joint_non_linear = GatedTanh(n_hidden, n_hidden)
        self.out_linear = nn.Linear(n_hidden, n_answers)
      #  self.apply(weights_init)

    def forward(self, v_emb, q_emb):
        v = v_emb.permute(0, 2, 1)
        q = q_emb.unsqueeze(1).repeat(1, v.size(1), 1)
        vq = torch.cat((v, q), dim=2)

        a = F.softmax(self.att_linear(self.att_non_linear(vq)), dim=1)

        v_hat = torch.sum(torch.mul(a, v), dim=1)

        h = self.ef_non_linear(q_emb) * self.v_non_linear(v_hat)
        h = F.dropout(h, 0.5)
        out = self.out_linear(self.joint_non_linear(h))
        return out


class GatedTanh(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(GatedTanh, self).__init__()

        self.fc_yhat = nn.Linear(in_dim, out_dim)
        self.fc_g = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        yhat = torch.tanh(self.fc_yhat(x))
        g = torch.sigmoid(self.fc_g(x))
        y = yhat * g
        return y


if __name__ == "__main__":
    model = PretrainedVQA(32, 33, 512, im_dim=4096)
    model_summary(model)
