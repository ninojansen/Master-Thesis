
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

    def forward(self, x, y):
        # in lightning, forward defines the prediction/inference actions
        return self.vqa_model(x, y)

    def training_step(self, batch, batch_idx):
      #  (opt_g, opt_d) = self.optimizers()
        # TODO Fix embedding sizes
        answer1 = self.vqa_model(batch["img"], batch["q_embedding"])
        noise = torch.randn(answer1.size(0), 100).type_as(answer1)

        answer1_idx = torch.argmax(answer1, dim=1)
        answer_embedding = [self.answer_map[idx][1] for idx in answer1_idx]
        answer_embedding = torch.from_numpy(np.stack(answer_embedding)).type_as(answer1)

        qa_embedding = torch.cat((batch["q_embedding"], answer_embedding), dim=1)
        gen_img = self.ig_model(noise, qa_embedding)
        answer2 = self.vqa_model(gen_img, batch["q_embedding"])
        answer2_idx = torch.argmax(answer1, dim=1)

        vqa_loss = F.nll_loss(answer1, batch["target"])
        consistency_loss = F.nll_loss(answer2, batch["target"])
        image_consistency_loss = F.mse_loss(gen_img, batch["img"], reduction="sum")
        self.log("vqa_loss", vqa_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("consistency_loss", consistency_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("image_consistency_loss", image_consistency_loss, on_step=True, on_epoch=True, prog_bar=True)
      #  answer1_text = self.answer_map[answer1_idx][0]
        #answer2_text = self.answer_map[answer2_idx][0]
        return
    # def validation_step(self, batch, batch_idx):
    #     y_pred = self(batch["img"], batch["q_embedding"])

    #     self.valid_acc(y_pred, batch["target"])
    #     self.log('val_acc', self.valid_acc, on_step=True, on_epoch=True, prog_bar=True)

    # def test_step(self, batch, batch_idx):
    #     y_pred = self(batch["img"], batch["q_embedding"])

    #     self.test_acc(y_pred, batch["target"])
    #     self.log('test_acc', self.test_acc, on_step=True, on_epoch=True, prog_bar=True)

    def on_epoch_end(self):
        elapsed_time = time.perf_counter() - self.start
        self.start = time.perf_counter()
        self.print(
            f"\nEpoch {self.current_epoch} finished in {round(elapsed_time, 2)}s")

    def configure_optimizers(self):
        return
       # opt = torch.optim.Adam(self.parameters(), lr=self.cfg.TRAIN.LR)
      #  opt = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return opt

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("loss", None)
        return items
