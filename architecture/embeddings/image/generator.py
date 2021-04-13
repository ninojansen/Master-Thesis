
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
import numpy as np
import torch
import torch.nn.functional as F
from architecture.embeddings.image.frcnn import FRCNN
from torch import nn
from torchvision import models
from tqdm import tqdm


class ImageEmbeddingGenerator():

    def __init__(self, data_dir, model_name):

        if model_name == "vgg16":
            self.model = models.vgg16(pretrained=True)
            self.model = nn.Sequential(*list(self.model.features.children())[:-1])
            self.extension = "vgg16"
            self.flatten = False
            self.dim = 512

        elif model_name == "vgg16_flat":
            self.model = models.vgg16(pretrained=True)
            self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-3])
            self.extension = "vgg16_flat"
            self.flatten = True
            self.dim = 4096

        elif model_name == "resnet18":
            self.model = models.resnet18(pretrained=True)
            self.model = nn.Sequential(*(list(self.model.children())[:-2]))
            self.extension = "resnet18"
            self.flatten = False
            self.dim = 512
        elif model_name == "resnet152":
            self.model = models.resnet152(pretrained=True)
            self.model = nn.Sequential(*(list(self.model.children())[:-2]))
            self.extension = "resnet152"
            self.flatten = False
            self.dim = 2049
        elif model_name == "frcnn":
            self.model = FRCNN()
            self.extension = "frcnn"
            self.flatten = False
            self.dim = 2048
        else:
            raise NotImplementedError
        self.data_dir = data_dir
        self.model.cuda()
        self.model.eval()
        self.norm_mean = torch.as_tensor([0.485, 0.456, 0.406]).cuda()[None, :, None, None]
        self.norm_std = torch.as_tensor([0.229, 0.224, 0.225]).cuda()[None, :, None, None]

    def generate_embeddings(self, dataloader, outdir):
        for batch in tqdm(dataloader):
            images = batch["img"].cuda()
            with torch.no_grad():
                features = self.process_batch(images)
                for i, path in enumerate(batch["img_path"]):
                    np.save(os.path.join(outdir, path.replace(".png", ".npy")), features[i].cpu())

    def process_batch(self, images, transform=False):
        with torch.no_grad():
            if transform:
                images = F.interpolate(images, size=224)
                images.sub_(self.norm_mean).div_(self.norm_std)
            if self.extension == "frcnn":
                features = torch.stack([self.model(x.unsqueeze(0)) for x in images]).squeeze(1)
                #[print(x.shape) for x in images]
            else:
                features = self.model(images)
                # if self.extension == "resnet152":
                #     features = F.avg_pool2d()
                # L2-Norm
                features = F.normalize(features, p=2, dim=1)
                if self.flatten:
                    pass
                else:
                    features = features.view(
                        features.shape[0],
                        features.shape[1],
                        features.shape[2] * features.shape[3])
        return features


# https://github.com/airsplay/py-bottom-up-attention
