import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from architecture.datasets.easy_vqa import EasyVQADataModule

data_dir = "/home/nino/Documents/Datasets/ExtEasyVQA/"
datamodule = EasyVQADataModule(data_dir=data_dir, num_workers=12, batch_size=48, iterator="question")

datamodule.setup("fit")


questions = []
for batch in datamodule.train_dataloader():
    for q, a, type, specificity, bool in zip(
            batch["question_json"]["question"],
            batch["question_json"]["answer"],
            batch["question_json"]["type"],
            batch["question_json"]["specificity"],
            batch["question_json"]["bool"]):
        questions.append((q, a, type, specificity, bool))


n_bool = 0
n_open = 0

n_size = 0
n_count = 0
n_loc = 0
n_color = 0
n_shape = 0

n_spec1 = 0
n_spec2 = 0
n_spec3 = 0
