import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict


class NetG(nn.Module):
    def __init__(self, ngf=64, nz=100, n_emb=256, img_size=256):
        super(NetG, self).__init__()
        self.ngf = ngf
        self.img_size = img_size
        # layer1输入的是一个100x1x1的随机噪声, 输出尺寸(ngf*8)x4x4
        self.fc = nn.Linear(nz, ngf*8*4*4)
        self.block0 = UpBlock(ngf * 8, ngf * 8, n_emb)  # 4x4
        self.block1 = UpBlock(ngf * 8, ngf * 8, n_emb)  # 4x4
        self.block2 = UpBlock(ngf * 8, ngf * 8, n_emb)  # 8x8
        self.block3 = UpBlock(ngf * 8, ngf * 8, n_emb)  # 16x16
        self.block4 = UpBlock(ngf * 8, ngf * 4, n_emb)  # 32x32
        self.block5 = UpBlock(ngf * 4, ngf * 2, n_emb)  # 64x64
        self.block6 = UpBlock(ngf * 2, ngf * 1, n_emb)  # 128x128

        self.conv_img = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x, c):

        out = self.fc(x)
        out = out.view(x.size(0), 8*self.ngf, 4, 4)
        out = self.block0(out, c)
        if self.img_size >= 256:
            out = F.interpolate(out, scale_factor=2)
            out = self.block1(out, c)
        if self.img_size >= 128:
            out = F.interpolate(out, scale_factor=2)
            out = self.block2(out, c)
        if self.img_size >= 64:
            out = F.interpolate(out, scale_factor=2)
            out = self.block3(out, c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block4(out, c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block5(out, c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block6(out, c)

        out = self.conv_img(out)

        return out


class UpBlock(nn.Module):
    # UPBlock
    def __init__(self, in_ch, out_ch, n_emb=256):
        super(UpBlock, self).__init__()

        self.learnable_sc = in_ch != out_ch
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.affine0 = affine(in_ch, n_emb)
        self.affine1 = affine(in_ch, n_emb)
        self.affine2 = affine(out_ch, n_emb)
        self.affine3 = affine(out_ch, n_emb)
        self.gamma = nn.Parameter(torch.zeros(1))
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)

    def forward(self, x, y=None):
        return self.shortcut(x) + self.gamma * self.deep_fusion(x, y)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def deep_fusion(self, x, y=None):
        # DFBlock
        h = self.affine0(x, y)
        h = nn.LeakyReLU(0.2, inplace=True)(h)
        h = self.affine1(h, y)
        h = nn.LeakyReLU(0.2, inplace=True)(h)
        h = self.c1(h)

        h = self.affine2(h, y)
        h = nn.LeakyReLU(0.2, inplace=True)(h)
        h = self.affine3(h, y)
        h = nn.LeakyReLU(0.2, inplace=True)(h)
        h = self.c2(h)
        return h


class affine(nn.Module):

    def __init__(self, num_features, n_emb=256):
        super(affine, self).__init__()

        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(n_emb, n_emb)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(n_emb, num_features)),
        ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(n_emb, n_emb)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(n_emb, num_features)),
        ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None):
        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias


class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, n_emb=256):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.n_emb = n_emb
        self.joint_conv = nn.Sequential(
            nn.Conv2d(ndf * 16+n_emb, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False),
        )

    def forward(self, out, y):

        y = y.view(-1, self.n_emb, 1, 1)
        y = y.repeat(1, 1, 4, 4)
        h_c_code = torch.cat((out, y), 1)
        out = self.joint_conv(h_c_code)
        return out


class NetD(nn.Module):
    def __init__(self, ndf, n_emb=256, img_size=256):
        super(NetD, self).__init__()
        self.img_size = img_size
        self.conv_img = nn.Conv2d(3, ndf, 3, 1, 1)  # 128
        self.block0 = DownBlock(ndf * 1, ndf * 2)  # 64
        self.block1 = DownBlock(ndf * 2, ndf * 4)  # 32
        self.block2 = DownBlock(ndf * 4, ndf * 8)  # 16
        self.block3 = DownBlock(ndf * 8, ndf * 16)  # 8
        self.block4 = DownBlock(ndf * 16, ndf * 16)  # 4
        self.block5 = DownBlock(ndf * 16, ndf * 16)  # 4

        self.COND_DNET = D_GET_LOGITS(ndf, n_emb)

    def forward(self, x):

        out = self.conv_img(x)
        out = self.block0(out)
        out = self.block1(out)
        out = self.block2(out)
        if self.img_size >= 64:
            out = self.block3(out)
        if self.img_size >= 128:
            out = self.block4(out)
        if self.img_size >= 256:
            out = self.block5(out)

        return out


class DownBlock(nn.Module):
    def __init__(self, fin, fout, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(fout, fout, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_s = nn.Conv2d(fin, fout, 1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, c=None):
        return self.shortcut(x)+self.gamma*self.residual(x)

    def shortcut(self, x):
        if self.learned_shortcut:
            x = self.conv_s(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        return self.conv_r(x)
