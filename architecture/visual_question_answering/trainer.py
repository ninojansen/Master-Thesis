
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
from architecture.visual_question_answering.models import SimpleVQA, PretrainedVQA, AttentionVQA
from easydict import EasyDict as edict
from architecture.utils.utils import weights_init
from torchvision import datasets, models, transforms
from architecture.embeddings.image.generator import ImageEmbeddingGenerator


class VQA(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        if type(cfg) is dict:
            cfg = edict(cfg)
        self.cfg = cfg
        self.lr = cfg.TRAIN.LR
        self.save_hyperparameters(self.cfg)

        if cfg.MODEL.CNN_TYPE != "cnn":
            self.embedding_generator = ImageEmbeddingGenerator(cfg.DATA_DIR, cfg.MODEL.CNN_TYPE)

        if cfg.MODEL.ATTENTION:
            self.model = AttentionVQA(self.cfg.MODEL.EF_DIM, self.embedding_generator.dim,
                                      self.cfg.MODEL.N_HIDDEN, self.cfg.MODEL.N_ANSWERS)
            if cfg.MODEL.CNN_TYPE == "frcnn":
                k = 6
            else:
                k = 49
            self.example_input_array = (
                torch.ones(1, self.embedding_generator.dim, k),
                torch.ones(1, self.cfg.MODEL.EF_DIM))
        else:
            if cfg.MODEL.CNN_TYPE == "cnn":
                self.model = SimpleVQA(self.cfg.IM_SIZE, self.cfg.MODEL.EF_DIM, self.cfg.MODEL.N_ANSWERS,
                                       self.cfg.MODEL.N_HIDDEN)
                self.example_input_array = (
                    torch.ones(1, 3, self.cfg.IM_SIZE, self.cfg.IM_SIZE),
                    torch.ones(1, self.cfg.MODEL.EF_DIM))
            else:
                self.model = PretrainedVQA(self.cfg.MODEL.EF_DIM, self.cfg.MODEL.N_ANSWERS,
                                           self.cfg.MODEL.N_HIDDEN, im_dim=self.embedding_generator.dim)
                self.example_input_array = (
                    torch.ones(1, self.embedding_generator.dim),
                    torch.ones(1, self.cfg.MODEL.EF_DIM))
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
        img = batch["img"]

        if self.cfg.MODEL.CNN_TYPE != "cnn":
            # Get image features if not default cnn
            if len(batch["img_embedding"].shape) == 1:
                img = self.preprocess_img(batch["img"])
            else:
                img = batch["img_embedding"]

        y_pred = self(img, batch["q_embedding"])
        loss = self.criterion(y_pred, batch["target"])
      #  loss.backward()
       # self.opt.step()
        self.log("Loss/CrossEntropy", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_acc(F.softmax(y_pred, dim=1), batch["target"])
        self.log('Acc/Train', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
       # y_pred = self(batch["img_embedding"], batch["q_embedding"])
        img = batch["img"]
        if self.cfg.MODEL.CNN_TYPE != "cnn":
            if len(batch["img_embedding"].shape) == 1:
                img = self.preprocess_img(batch["img"])
            else:
                img = batch["img_embedding"]
        y_pred = self(img, batch["q_embedding"])

        self.valid_acc(F.softmax(y_pred, dim=1), batch["target"])
        self.log('Acc/Val', self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
      #  y_pred = self(batch["img_embedding"], batch["q_embedding"])
        y_pred = self(self.preprocess_img(batch["img"]), batch["q_embedding"])

        self.test_acc(F.softmax(y_pred, dim=1), batch["target"])
        self.log('Acc/Test', self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_epoch_end(self):
        elapsed_time = time.perf_counter() - self.start
        self.start = time.perf_counter()
        self.print(
            f"\nEpoch {self.current_epoch} finished in {round(elapsed_time, 2)}s")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0, 0.999))
       # opt = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return opt

    def preprocess_img(self, images):
        if self.cfg.MODEL.CNN_TYPE != "cnn":
            images = self.embedding_generator.process_batch(images, transform=True)
        return images

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("loss", None)
        return items
