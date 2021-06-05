import os
import random
import json
import re

path = "/home/nino/Documents/Datasets/ExtEasyVQA"

im_path = os.path.join(path, "train", "images")
q_path = os.path.join(path, "train", "questions.json")

n_samples = 20


samples = random.sample(os.listdir(im_path), n_samples)

q = []
with open(q_path, 'r') as file:
    questions = json.load(file)
    for img in samples:
        id = re.search(r'\d+', img).group()
        q = random.sample(questions[id], 1)[0]
        print(id, q["question"], q["answer"])
