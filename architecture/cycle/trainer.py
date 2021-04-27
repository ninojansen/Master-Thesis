
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
from architecture.embeddings.image.generator import ImageEmbeddingGenerator


class Cycle(pl.LightningModule):

    def __init__(self, cfg, vqa_model, ig_model, answer_map):
        super().__init__()
        self.automatic_optimization = False
        self.cfg = cfg
        self.vqa_model = vqa_model
        self.ig_model = ig_model
        self.answer_map = answer_map
        self.start = time.perf_counter()
        self.opt = self.configure_optimizers()

        self.embedding_generator = ImageEmbeddingGenerator(cfg.DATA_DIR, "vgg16_flat")

        self.metrics = {"Train/Acc/VQA": pl.metrics.Accuracy().cuda(),
                        "Train/Acc/Consistency": pl.metrics.Accuracy().cuda()}
        self.vqa_opt = self.vqa_model.configure_optimizers()

    def forward(self, x, y):
        # in lightning, forward defines the prediction/inference actions
        return self.vqa_model(x, y)

    def image_consistency_loss(self, real, generated):
        real_features = self.embedding_generator.process_batch(real, transform=True)
        generated_features = self.embedding_generator.process_batch(generated, transform=True)

        return F.cosine_embedding_loss(real_features, generated_features, torch.ones(1).cuda())

    def training_step(self, batch, batch_idx):
        if self.cfg.TRAIN.TYPE == "finetune_vqa":
            #  1. (I, Q) -> A' through VQA model
            img = batch["img"]
            if len(batch["img_embedding"].shape) == 1:
                img = self.preprocess_img(batch["img"])
            else:
                img = batch["img_embedding"]
            answer1 = self.vqa_model(img, batch["q_embedding"])

            # (A', A) loss
            vqa_loss = self.vqa_model.criterion(answer1, batch["target"])

            # 2. (Q, A) -> I' through IG model
            noise = torch.randn(answer1.size(0), 100).type_as(answer1)

            # Combine answer and question embeddings for IG
            answer1_idx = torch.argmax(answer1, dim=1)
            answer1_embedding = torch.from_numpy(
                np.stack([self.answer_map[int(idx.cpu())][1] for idx in answer1_idx])).type_as(answer1)

            qa_embedding = torch.cat((batch["q_embedding"], answer1_embedding), dim=1)

            # Forward pass through IG
            with torch.no_grad():
                gen_img = self.ig_model(noise, qa_embedding)
                # (I, I') loss TODO Figure out what to do with this loss
                image_consistency_loss = self.image_consistency_loss(batch["img"], gen_img)

            # 3. (I', Q) -> A'' through VQA model
            answer2 = self.vqa_model(self.vqa_model.preprocess_img(gen_img), batch["q_embedding"])
            # (A'', A) loss
            answer_consistency_loss = self.vqa_model.criterion(answer2, batch["target"])

            total_loss = vqa_loss + self.cfg.TRAIN.LA * answer_consistency_loss + self.cfg.TRAIN.LC * image_consistency_loss
            total_loss.backward()
            self.vqa_opt.step()

            self.metrics["Train/Acc/VQA"](F.softmax(answer1, dim=1), batch["target"])
            self.metrics["Train/Acc/Consistency"](F.softmax(answer1, dim=1), batch["target"])

            self.log("Train/Acc/VQA", self.metrics["Train/Acc/VQA"], on_step=False, on_epoch=True)
            self.log("Train/Acc/Consistency", self.metrics["Train/Acc/Consistency"], on_step=False, on_epoch=True)

            self.log("Loss/Total", total_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("Loss/VQA", vqa_loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log("Loss/Answer_Consistency", answer_consistency_loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log("Loss/Image_Consistency", image_consistency_loss, on_step=False, on_epoch=True, prog_bar=False)

    #     # 1. (I, Q) -> A' through VQA model
    #     answer1 = self.vqa_model(batch["img"], batch["q_embedding"])

    #     # (A', A) loss
    #     vqa_loss = self.criterion(answer1, batch["target"])
    #     vqa_loss.backward()
    #     self.train_vqa_acc(answer1, batch["target"])
    #     # 2. (Q, A) -> I' through IG model
    #     noise = torch.randn(answer1.size(0), 100).type_as(answer1)

    #     # Combine answer and question embeddings for IG
    #     answer1_idx = torch.argmax(answer1, dim=1)
    #     answer1_embedding = torch.from_numpy(np.stack([self.answer_map[idx][1]
    #                                                    for idx in answer1_idx])).type_as(answer1)
    #     qa_embedding = torch.cat((batch["q_embedding"], answer1_embedding), dim=1)

    #     # Forward pass through IG
    #     with torch.no_grad():
    #         gen_img = self.ig_model(noise, qa_embedding)
    #         # (I, I') loss TODO Figure out what to do with this loss
    #         image_consistency_loss = F.mse_loss(gen_img, batch["img"], reduction="sum")

    #     # 3. (I', Q) -> A'' through VQA model
    #     answer2 = self.vqa_model(gen_img, batch["q_embedding"])
    #     answer2_idx = torch.argmax(answer1, dim=1)

    #     # (A'', A) loss
    #     consistency_loss = self.criterion(answer2, batch["target"])
    #     consistency_loss.backward()
    #     self.train_cns_acc(answer2, batch["target"])

    #     self.opt.step()
    #     self.log("vqa_loss", vqa_loss, on_step=False, on_epoch=True, prog_bar=True)
    #     self.log("consistency_loss", consistency_loss, on_step=False, on_epoch=True, prog_bar=True)
    #     #self.log("image_consistency_loss", image_consistency_loss, on_step=False, on_epoch=True, prog_bar=False)
    #     self.log("vqa_acc", self.train_vqa_acc, on_step=False, on_epoch=True, prog_bar=True)
    #     self.log("cs_acc", self.train_cns_acc, on_step=False, on_epoch=True, prog_bar=True)
    #   #  answer1_text = self.answer_map[answer1_idx][0]
    #     #answer2_text = self.answer_map[answer2_idx][0]
        return

    def validation_step(self, batch, batch_idx):
        pass
        # 1. (I, Q) -> A' through VQA model
        #answer1 = self.vqa_model(batch["img"], batch["q_embedding"])

        # (A', A) acc
        #self.val_acc(answer1, batch["target"])
       # self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

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
