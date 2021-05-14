
# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
#     os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))))
from numpy.lib.type_check import imag
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
from architecture.utils.utils import gen_image_grid, weights_init, generate_figure
from architecture.embeddings.text.generator import TextEmbeddingGenerator
import matplotlib.pyplot as plt


class FinetuneIG(pl.LightningModule):

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

        self.metrics = {"Acc/Train_VQA": pl.metrics.Accuracy().cuda(),
                        "Acc/Val_VQA": pl.metrics.Accuracy().cuda()}
        self.ig_opt = self.configure_optimizers()

    def forward(self, x, y):
        # in lightning, forward defines the prediction/inference actions
        return self.ig_model(x, y)

    def image_consistency_loss(self, real, generated):
        real_features = self.embedding_generator.process_batch(real, transform=True)
        generated_features = self.embedding_generator.process_batch(generated, transform=True)
        return F.cosine_embedding_loss(real_features, generated_features, torch.ones(1).cuda())

    def training_step(self, batch, batch_idx):
        # 1. (Q, A) -> I through IG model
        real_img = batch["img"]
        batch_size = real_img.size(0)
        noise = torch.randn(batch_size, 100).type_as(real_img)
        gen_img = self.ig_model(noise, batch["qa_embedding"])

        # 2. (I, Q) -> A' through VQA model
        answer = self.vqa_model(self.vqa_model.preprocess_img(gen_img), batch["q_embedding"])
        vqa_loss = self.vqa_model.criterion(answer, batch["target"])

        answer_idx = torch.argmax(answer, dim=1)
        answer_embedding = torch.from_numpy(
            np.stack([self.answer_map[int(idx.cpu())][1] for idx in answer_idx])).type_as(answer)
        cycle_qa_embedding = torch.cat((batch["q_embedding"], answer_embedding), dim=1)

        # 3. (Q, A') -> I'
        cycle_img = self.ig_model(noise, cycle_qa_embedding)

        self.log("Loss/VQA", vqa_loss)
        # (I', I) loss
        if self.cfg.TRAIN.LOSS == "full":
            image_consistency_loss = self.image_consistency_loss(gen_img, cycle_img)
            total_loss = vqa_loss + image_consistency_loss
            self.log("Loss/Image_Consistency", image_consistency_loss)
        else:
            total_loss = vqa_loss

        total_loss.backward()
        self.ig_opt.step()

        self.metrics["Acc/Train_VQA"](F.softmax(answer, dim=1), batch["target"])
        self.log("Acc/Train_VQA", self.metrics["Acc/Train_VQA"])
        self.log("Loss/Total", total_loss)
        return

    def validation_step(self, batch, batch_idx):
        text_embed = batch["qa_embedding"]

        batch_size = self.cfg.TRAIN.BATCH_SIZE
        # Generate images
        noise = torch.randn(batch_size, self.ig_model.cfg.MODEL.Z_DIM).type_as(text_embed)

        fake_img = self.forward(noise, text_embed)

      #  incep_mean, incep_std = self.inception.compute_score(fake_img, num_splits=1)

        # if not self.trainer.running_sanity_check:
        self.ig_model.inception.compute_statistics(fake_img)
        self.ig_model.fid.compute_statistics(batch["img"], fake_img)

        if self.vqa_model:
            with torch.no_grad():
                vqa_pred = self.vqa_model(self.vqa_model.preprocess_img(fake_img), batch["q_embedding"])
                self.metrics["Acc/Val_VQA"](F.softmax(vqa_pred, dim=1), batch["target"])
                self.log("Acc/Val_VQA", self.metrics["Acc/Val_VQA"], prog_bar=True)

        if not self.trainer.running_sanity_check and batch_idx == 0:
            val_images = []
            for img, text in zip(fake_img, batch["text"]):
                val_images.append(generate_figure(img, text))
            self.logger.experiment.add_images(
                f"Val/Epoch_{self.current_epoch}", torch.stack(val_images),
                global_step=self.current_epoch)

    def on_validation_epoch_end(self):
        fid_score = self.ig_model.fid.compute_fid()
        is_mean, is_std = self.ig_model.inception.compute_score()
        self.log("FID/Val", fid_score)
        self.log("Inception/Val_Mean", is_mean)
        self.log("Inception/Val_Std", is_std)

    def test_step(self, batch, batch_idx):
        pass

    def on_epoch_end(self):
        elapsed_time = time.perf_counter() - self.start
        self.start = time.perf_counter()
        self.print(
            f"\nEpoch {self.current_epoch} finished in {round(elapsed_time, 2)}s")

    def configure_optimizers(self):
        opt_g, _ = self.ig_model.configure_optimizers()
        return opt_g

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("loss", None)
        return items


class FinetuneVQA(pl.LightningModule):

    def __init__(self, cfg, vqa_model=None, ig_model=None, answer_map=None):
        super().__init__()
        self.automatic_optimization = False
        self.cfg = cfg
        self.vqa_model = vqa_model
        self.ig_model = ig_model
        self.answer_map = answer_map
        self.save_hyperparameters(self.cfg)
        self.start = time.perf_counter()

        self.embedding_generator = ImageEmbeddingGenerator(cfg.DATA_DIR, "vgg16_flat")
        self.text_embedding_generator = TextEmbeddingGenerator(
            ef_type=self.vqa_model.cfg.MODEL.EF_TYPE, data_dir=cfg.DATA_DIR)

        self.metrics = {"Acc/Train_VQA": pl.metrics.Accuracy().cuda(),
                        "Acc/Train_Consistency": pl.metrics.Accuracy().cuda(),
                        "Acc/Val": pl.metrics.Accuracy().cuda()}
        self.test_metrics = self.vqa_model.test_metrics
        self.vqa_opt = self.configure_optimizers()

    def forward(self, x, y):
        # in lightning, forward defines the prediction/inference actions
        return self.vqa_model(x, y)

    def display_images(self, images):
        grid = torchvision.utils.make_grid(images, normalize=True)
        plt.imshow(grid.permute(1, 2, 0))
        plt.show()

    def image_consistency_loss(self, real, generated):
        real_features = self.embedding_generator.process_batch(real, transform=True)
        generated_features = self.embedding_generator.process_batch(generated, transform=True)
        return F.cosine_embedding_loss(real_features, generated_features, torch.ones(1).cuda())

    def training_step(self, batch, batch_idx):
        #  1. (I, Q) -> A' through VQA model
        img = batch["img"]
        if len(batch["img_embedding"].shape) == 1:
            img = self.preprocess_img(batch["img"])
        else:
            img = batch["img_embedding"]
        answer1 = self.vqa_model(img, batch["q_embedding"])

        # (A', A) loss
        if self.cfg.TRAIN.LOSS in ["vqa_only", "full", "full_coeff"]:
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
            if self.cfg.TRAIN.LOSS in ["full", "full_coeff"]:
                image_consistency_loss = self.image_consistency_loss(batch["img"], gen_img)

                if self.cfg.TRAIN.GATING:
                    if image_consistency_loss >= 0.20:
                        return
        # 3. (I', Q) -> A'' through VQA model
        answer2 = self.vqa_model(self.vqa_model.preprocess_img(gen_img), batch["q_embedding"])
        # (A'', A) loss
        answer_consistency_loss = self.vqa_model.criterion(answer2, batch["target"])

        if self.cfg.TRAIN.LOSS == "vqa_only":
            total_loss = vqa_loss + self.cfg.TRAIN.LA * answer_consistency_loss
            self.log("Loss/VQA", vqa_loss)
        elif self.cfg.TRAIN.LOSS == "cns_only":
            total_loss = answer_consistency_loss
        elif self.cfg.TRAIN.LOSS == "full":
            total_loss = vqa_loss + self.cfg.TRAIN.LA * answer_consistency_loss + self.cfg.TRAIN.LC * image_consistency_loss
            self.log("Loss/VQA", vqa_loss, on_epoch=True)
            self.log("Loss/Image_Consistency", image_consistency_loss, on_epoch=True)
        elif self.cfg.TRAIN.LOSS == "full_coeff":
            total_loss = vqa_loss + answer_consistency_loss / image_consistency_loss
            self.log("Loss/VQA", vqa_loss, on_epoch=True)
            self.log("Loss/Image_Consistency", image_consistency_loss, on_epoch=True)
        else:
            total_loss = answer_consistency_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vqa_model.parameters(), 0.25)
        self.vqa_opt.step()

        self.metrics["Acc/Train_VQA"](F.softmax(answer1, dim=1), batch["target"])
        self.metrics["Acc/Train_Consistency"](F.softmax(answer1, dim=1), batch["target"])

        self.log("Acc/Train_VQA", self.metrics["Acc/Train_VQA"], on_epoch=True)
        self.log("Acc/Train_Consistency", self.metrics["Acc/Train_Consistency"], on_epoch=True)
        self.log("Loss/Total", total_loss, on_epoch=True)
        self.log("Loss/Answer_Consistency", answer_consistency_loss, on_epoch=True)
        return

    def validation_step(self, batch, batch_idx):
        img = batch["img"]
        if self.vqa_model.cfg.MODEL.CNN_TYPE != "cnn":
            if len(batch["img_embedding"].shape) == 1:
                img = self.vqa_model.preprocess_img(batch["img"])
            else:
                img = batch["img_embedding"]
        y_pred = self.vqa_model(img, batch["q_embedding"])

        self.metrics["Acc/Val"](F.softmax(y_pred, dim=1), batch["target"])
        self.log('Acc/Val', self.metrics["Acc/Val"], prog_bar=True)

    def test_step(self, batch, batch_idx):
      #  y_pred = self(batch["img_embedding"], batch["q_embedding"])

        q_embedding = self.text_embedding_generator.process_batch(batch["question"]).cuda()
        y_pred = self(self.vqa_model.preprocess_img(batch["img"]), q_embedding.cuda())

        bool_pred = [index for index, element in enumerate(batch["question_json"]['bool']) if element]
        open_pred = [index for index, element in enumerate(batch["question_json"]['bool']) if not element]
        size_pred = [index for index, element in enumerate(batch["question_json"]['type']) if element == "size"]
        color_pred = [index for index, element in enumerate(
            batch["question_json"]['type']) if element == "color"]
        location_pred = [index for index, element in enumerate(
            batch["question_json"]['type']) if element == "location"]
        shape_pred = [index for index, element in enumerate(
            batch["question_json"]['type']) if element == "shape"]
        count_pred = [index for index, element in enumerate(
            batch["question_json"]['type']) if element == "count"]

        spec1_pred = [index for index, element in enumerate(
            batch["question_json"]['specificity']) if element == 1]
        spec2_pred = [index for index, element in enumerate(
            batch["question_json"]['specificity']) if element == 2]
        spec3_pred = [index for index, element in enumerate(
            batch["question_json"]['specificity']) if element == 3]

        self.test_metrics["Test/Acc/General"](F.softmax(y_pred, dim=1), batch["target"])
        if len(bool_pred) > 0:
            self.test_metrics["Test/Acc/Bool"](F.softmax(y_pred[bool_pred], dim=1), batch["target"][bool_pred])
        if len(size_pred) > 0:
            self.test_metrics["Test/Acc/Open"](F.softmax(y_pred[open_pred], dim=1), batch["target"][open_pred])
        if len(size_pred) > 0:
            self.test_metrics["Test/Acc/Size"](F.softmax(y_pred[size_pred], dim=1), batch["target"][size_pred])
        if len(color_pred) > 0:
            self.test_metrics["Test/Acc/Color"](F.softmax(y_pred[color_pred], dim=1), batch["target"][color_pred])
        if len(location_pred) > 0:
            self.test_metrics["Test/Acc/Location"](F.softmax(y_pred[location_pred],
                                                             dim=1), batch["target"][location_pred])
        if len(count_pred) > 0:
            self.test_metrics["Test/Acc/Count"](F.softmax(y_pred[count_pred], dim=1), batch["target"][count_pred])
        if len(shape_pred) > 0:
            self.test_metrics["Test/Acc/Shape"](F.softmax(y_pred[shape_pred], dim=1), batch["target"][shape_pred])
        if len(spec1_pred) > 0:
            self.test_metrics["Test/Acc/Spec1"](F.softmax(y_pred[spec1_pred], dim=1), batch["target"][spec1_pred])
        if len(spec2_pred) > 0:
            self.test_metrics["Test/Acc/Spec2"](F.softmax(y_pred[spec2_pred], dim=1), batch["target"][spec2_pred])
        if len(spec3_pred) > 0:
            self.test_metrics["Test/Acc/Spec3"](F.softmax(y_pred[spec3_pred], dim=1), batch["target"][spec3_pred])

        for name, metric in self.test_metrics.items():
            self.log(name, metric, on_step=False, on_epoch=True)

    def on_epoch_end(self):
        elapsed_time = time.perf_counter() - self.start
        self.start = time.perf_counter()
        self.print(
            f"\nEpoch {self.current_epoch} finished in {round(elapsed_time, 2)}s")

    def configure_optimizers(self):
        return self.vqa_model.configure_optimizers()

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("loss", None)
        return items
