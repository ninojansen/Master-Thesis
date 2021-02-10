
# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
#     os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))))
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from torch import nn
import torchvision
import numpy as np
import time
from architecture.utils.inception_score import InceptionScore
from architecture.visual_question_answering.models import SimpleVQA


class Cycle(pl.LightningModule):

    def __init__(self, cfg, vqa_model, ig_model, answer_map):
        super().__init__()
        self.cfg = cfg
        self.vqa_model = vqa_model
        self.ig_model = ig_model
        self.answer_map = answer_map
        self.start = time.perf_counter()

        self.criterion = nn.CrossEntropyLoss()
        self.train_vqa_acc = pl.metrics.Accuracy()
        self.train_cns_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
        self.opt = self.configure_optimizers()

    def forward(self, x, y):
        # in lightning, forward defines the prediction/inference actions
        return self.vqa_model(x, y)

    def training_step(self, batch, batch_idx):

        # 1. (I, Q) -> A' through VQA model
        answer1 = self.vqa_model(batch["img"], batch["q_embedding"])

        # (A', A) loss
        vqa_loss = self.criterion(answer1, batch["target"])
        vqa_loss.backward()
        self.train_vqa_acc(answer1, batch["target"])
        # 2. (Q, A) -> I' through IG model
        noise = torch.randn(answer1.size(0), 100).type_as(answer1)

        # Combine answer and question embeddings for IG
        answer1_idx = torch.argmax(answer1, dim=1)
        answer1_embedding = torch.from_numpy(np.stack([self.answer_map[idx][1]
                                                       for idx in answer1_idx])).type_as(answer1)
        qa_embedding = torch.cat((batch["q_embedding"], answer1_embedding), dim=1)

        # Forward pass through IG
        with torch.no_grad():
            gen_img = self.ig_model(noise, qa_embedding)
            # (I, I') loss TODO Figure out what to do with this loss
            image_consistency_loss = F.mse_loss(gen_img, batch["img"], reduction="sum")

        # 3. (I', Q) -> A'' through VQA model
        answer2 = self.vqa_model(gen_img, batch["q_embedding"])
        answer2_idx = torch.argmax(answer1, dim=1)

        # (A'', A) loss
        consistency_loss = self.criterion(answer2, batch["target"])
        consistency_loss.backward()
        self.train_cns_acc(answer2, batch["target"])

        self.opt.step()
        self.log("vqa_loss", vqa_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("consistency_loss", consistency_loss, on_step=False, on_epoch=True, prog_bar=True)
        #self.log("image_consistency_loss", image_consistency_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("vqa_acc", self.train_vqa_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("cs_acc", self.train_cns_acc, on_step=False, on_epoch=True, prog_bar=True)
      #  answer1_text = self.answer_map[answer1_idx][0]
        #answer2_text = self.answer_map[answer2_idx][0]
        return

    def validation_step(self, batch, batch_idx):
        # 1. (I, Q) -> A' through VQA model
        answer1 = self.vqa_model(batch["img"], batch["q_embedding"])

        # (A', A) acc
        self.val_acc(answer1, batch["target"])
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # 1. (I, Q) -> A' through VQA model
        answer1 = self.vqa_model(batch["img"], batch["q_embedding"])

        # (A', A) acc
        self.test_acc(answer1, batch["target"])
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_epoch_end(self):
        elapsed_time = time.perf_counter() - self.start
        self.start = time.perf_counter()
        self.print(
            f"\nEpoch {self.current_epoch} finished in {round(elapsed_time, 2)}s")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.vqa_model.parameters(), lr=self.cfg.TRAIN.LR)
        return opt

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("loss", None)
        return items
