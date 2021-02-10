
# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
#     os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))))
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from torch import nn
from torch.optim import optimizer
import torchvision
import time
from architecture.utils.inception_score import InceptionScore
from architecture.visual_question_answering.models import SimpleVQA
from easydict import EasyDict as edict
from architecture.utils.utils import weights_init


class VQA(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        if type(cfg) is dict:
            cfg = edict(cfg)
        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.lr = cfg.TRAIN.LR

        self.model = SimpleVQA(64, self.cfg.MODEL.EF_DIM, self.cfg.MODEL.N_ANSWERS,
                               pretrained_img=False, im_dim=self.cfg.MODEL.IM_DIM, )
        self.model.apply(weights_init)
        self.start = time.perf_counter()
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
        self.criterion = nn.CrossEntropyLoss()
        self.opt = self.configure_optimizers()

    def forward(self, x, y):
        # in lightning, forward defines the prediction/inference actions
        return self.model(x, y)

    def training_step(self, batch, batch_idx):
      #  (opt_g, opt_d) = self.optimizers()
     #   self.opt.zero_grad()
        #y_pred = self(batch["img_embedding"], batch["q_embedding"])
        y_pred = self(batch["img"], batch["q_embedding"])
        loss = self.criterion(y_pred, batch["target"])
      #  loss.backward()
       # self.opt.step()
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_acc(y_pred, batch["target"])
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
       # y_pred = self(batch["img_embedding"], batch["q_embedding"])
        y_pred = self(batch["img"], batch["q_embedding"])

        self.valid_acc(y_pred, batch["target"])
        self.log('val_acc', self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
      #  y_pred = self(batch["img_embedding"], batch["q_embedding"])
        y_pred = self(batch["img"], batch["q_embedding"])

        self.test_acc(y_pred, batch["target"])
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_epoch_end(self):
        elapsed_time = time.perf_counter() - self.start
        self.start = time.perf_counter()
        self.print(
            f"\nEpoch {self.current_epoch} finished in {round(elapsed_time, 2)}s")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, eps=1e-07)
      #  opt = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return opt

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("loss", None)
        return items
