
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from scipy.stats import truncnorm


class NetG(nn.Module):
    def __init__(self, nf, img_size, z_dim, ef_dim):
        super(NetG, self).__init__()
        self.nf = nf
        self.img_size = img_size
        self.z_dim = z_dim
        # layer1输入的是一个100x1x1的随机噪声, 输出尺寸(ngf*8)x4x4

        self.fc = nn.Linear(z_dim, self.nf * 8 * 4 * 4)

        if img_size == 32:
            self.block0 = G_Block(nf * 8, nf * 8, ef_dim)  # 16x16
            self.block1 = G_Block(nf * 8, nf * 4, ef_dim)  # 32x32
            self.block2 = G_Block(nf * 4, nf * 2, ef_dim)  # 64x64
            self.block3 = G_Block(nf * 2, nf * 1, ef_dim)  # 128x128

        elif img_size == 64:
            self.block0 = G_Block(nf * 8, nf * 8, ef_dim)  # 8x8
            self.block1 = G_Block(nf * 8, nf * 8, ef_dim)  # 16x16
            self.block2 = G_Block(nf * 8, nf * 4, ef_dim)  # 32x32
            self.block3 = G_Block(nf * 4, nf * 2, ef_dim)  # 64x64
            self.block4 = G_Block(nf * 2, nf * 1, ef_dim)  # 128x128
        elif img_size == 128:
            self.block0 = G_Block(nf * 8, nf * 8, ef_dim)  # 4x4
            self.block1 = G_Block(nf * 8, nf * 8, ef_dim)  # 8x8
            self.block2 = G_Block(nf * 8, nf * 8, ef_dim)  # 16x16
            self.block3 = G_Block(nf * 8, nf * 4, ef_dim)  # 32x32
            self.block4 = G_Block(nf * 4, nf * 2, ef_dim)  # 64x64
            self.block5 = G_Block(nf * 2, nf * 1, ef_dim)  # 128x128
        else:
            self.block0 = G_Block(nf * 8, nf * 8, ef_dim)  # 4x4
            self.block1 = G_Block(nf * 8, nf * 8, ef_dim)  # 4x4
            self.block2 = G_Block(nf * 8, nf * 8, ef_dim)  # 8x8
            self.block3 = G_Block(nf * 8, nf * 8, ef_dim)  # 16x16
            self.block4 = G_Block(nf * 8, nf * 4, ef_dim)  # 32x32
            self.block5 = G_Block(nf * 4, nf * 2, ef_dim)  # 64x64
            self.block6 = G_Block(nf * 2, nf * 1, ef_dim)  # 128x128

        self.conv_img = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, 3, 3, 1, 1),
            nn.Tanh(),
        )

    # def truncate_z(self, z, truncation=0.5, a=-1, b=1):
    #     return torch.clamp(z, a, b) * truncation

    def forward(self, x, c):
        # x: noise vector
        # c: sentence vector
        out = self.fc(x)
        out = out.view(x.size(0), 8 * self.nf, 4, 4)
        out = self.block0(out, c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block1(out, c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block2(out, c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block3(out, c)

        if self.img_size >= 64:
            out = F.interpolate(out, scale_factor=2)
            out = self.block4(out, c)
        if self.img_size >= 128:
            out = F.interpolate(out, scale_factor=2)
            out = self.block5(out, c)
        if self.img_size >= 256:

            out = F.interpolate(out, scale_factor=2)
            out = self.block6(out, c)

        out = self.conv_img(out)

        return out


class G_Block(nn.Module):
    def __init__(self, in_ch, out_ch, ef_dim):
        super(G_Block, self).__init__()

        self.learnable_sc = in_ch != out_ch
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.affine0 = affine(in_ch, ef_dim)
        self.affine1 = affine(in_ch, ef_dim)
        self.affine2 = affine(out_ch, ef_dim)
        self.affine3 = affine(out_ch, ef_dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)

    def forward(self, x, y=None):
        return self.shortcut(x) + self.gamma * self.residual(x, y)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, x, y=None):
        h = self.affine0(x, y)
        h = nn.LeakyReLU(0.2, inplace=True)(h)
        h = self.affine1(h, y)
        h = nn.LeakyReLU(0.2, inplace=True)(h)
        h = self.c1(h)

        h = self.affine2(h, y)
        h = nn.LeakyReLU(0.2, inplace=True)(h)
        h = self.affine3(h, y)
        h = nn.LeakyReLU(0.2, inplace=True)(h)
        return self.c2(h)


class affine(nn.Module):

    def __init__(self, num_features, ef_dim):
        super(affine, self).__init__()

        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(ef_dim, ef_dim)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(ef_dim, num_features)),
        ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(ef_dim, ef_dim)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(ef_dim, num_features)),
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


class NetD(nn.Module):
    def __init__(self, nf, img_size, z_dim, ef_dim, disc=False):
        super(NetD, self).__init__()
        self.ef_dim = ef_dim
        self.disc = disc
        self.img_size = img_size
        self.conv_img = nn.Conv2d(3, nf, 3, 1, 1)  # 128

        self.block0 = resD(nf * 1, nf * 2)  # 64
        self.block1 = resD(nf * 2, nf * 4)  # 32
        self.block2 = resD(nf * 4, nf * 8)  # 16
        n_channels = nf * 16
        if img_size == 32:
            n_channels = nf * 8
        if img_size >= 64:
            self.block3 = resD(nf * 8, nf * 16)  # 8
            n_channels = nf * 16
        if img_size >= 128:
            self.block4 = resD(nf * 16, nf * 16)  # 4
            n_channels = nf * 16
        if img_size >= 256:
            self.block5 = resD(nf * 16, nf * 16)  # 4
            n_channels = nf * 16

        self.joint_conv = nn.Sequential(
            nn.Conv2d(n_channels + ef_dim, nf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        if self.disc:
            self.logits = nn.Conv2d(nf * 2, 1, 4, 1, 0, bias=False)
        else:
            h_dim = 4 * 4 * nf * 2
            self.fc11 = nn.Linear(h_dim, z_dim)
            self.fc12 = nn.Linear(h_dim, z_dim)

    def forward(self, x, y):

        x = self.conv_img(x)

        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        if self.img_size >= 64:
            x = self.block3(x)
        if self.img_size >= 128:
            x = self.block4(x)
        if self.img_size >= 256:
            x = self.block5(x)

        y = y.view(-1, self.ef_dim, 1, 1)
        y = y.repeat(1, 1, 4, 4)
        h_c_code = torch.cat((x, y), 1)
        x = self.joint_conv(h_c_code)

        if self.disc:
            x = self.logits(x)
            x = x.view(-1, 1)
            return x
        else:
            x = x.view(x.size(0), -1)
            mu = self.fc11(x)
            logvar = self.fc12(x)
            return mu, logvar


class resD(nn.Module):
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
        return self.shortcut(x) + self.gamma * self.residual(x)

    def shortcut(self, x):
        if self.learned_shortcut:
            x = self.conv_s(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        return self.conv_r(x)
