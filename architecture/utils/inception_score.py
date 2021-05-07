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
        self.inception_model = models.inception_v3(pretrained=True, transform_input=False).cuda()
        self.inception_model.eval()

        self.act = None

    def compute_score(self, splits=10):
        # Now compute the mean kl-div
        split_scores = []
        N = self.act.shape[0]
        for k in range(splits):
            part = self.act[k * (N // splits): (k + 1) * (N // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        self.act = None
        return np.mean(split_scores), np.std(split_scores)

    def compute_statistics(self, batch):
        real_features = self.inception_model
        batch = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)(batch)
        pred = self.inception_model(batch)
        pred = F.softmax(pred, dim=0).data.cpu().numpy()

        if isinstance(self.act, np.ndarray):
            self.act = np.vstack([self.act, pred])
        else:
            self.act = pred
