from collections import OrderedDict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision

from scipy.stats import entropy
from torch import nn


from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.model_zoo as model_zoo
from torchvision import models


class InceptionScore:

    def __init__(self):
        self.inception_model = INCEPTION_V3().cuda()

    def compute_score(self, data, num_splits=1, precomputed=False):
        if not precomputed:
            self.inception_model.eval()
            data = self.inception_model(data)
        data = data.cpu().numpy()
        N = data.shape[0]
        # Now compute the mean kl-div
        split_scores = []

        for k in range(num_splits):
            part = data[k * (N // num_splits): (k + 1) * (N // num_splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        return np.mean(split_scores), np.std(split_scores)


class INCEPTION_V3(nn.Module):
    def __init__(self):
        super(INCEPTION_V3, self).__init__()
        self.model = models.inception_v3(pretrained=True, progress=True)
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        # print(next(model.parameters()).data)
        state_dict = \
            model_zoo.load_url(url, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)
        for param in self.model.parameters():
            param.requires_grad = False
     #   print('Load pretrained model from ', url)
        # print(next(self.model.parameters()).data)
        # print(self.model)

    def forward(self, input):
        # [-1.0, 1.0] --> [0, 1.0]
        x = input * 0.5 + 0.5
        # mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        # --> mean = 0, std = 1
        x[:, 0, :] = (x[:, 0, :] - 0.485) / 0.229
        x[:, 1, :] = (x[:, 1, :] - 0.456) / 0.224
        x[:, 2, :] = (x[:, 2, :] - 0.406) / 0.225
        #
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True)(x)
        # 299 x 299 x 3
        x = self.model(x)
        x = nn.Softmax(dim=0)(x)
        return x