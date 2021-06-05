from re import L
from attributes import Color, Shape, Location, Size
# from images import create_image
# from questions import create_questions
import random
import json
import os
import shutil
from PIL import Image, ImageDraw, ImageFilter
from random import randint, uniform
import glob
import numpy as np
import itertools
from tqdm import tqdm
import math
import copy

# TODO
# 1. Make JSON of question types etc
# 2. Add a texture to the background
# 3. Add multiple objects


# 1. Fix that no questions arent overepresent3ed
# 2. Determine test question difficulty
# 3. Add more variation to questions, spelling errors etc


class DataGenerator:
    def __init__(self, data_dir, im_size, amount):
        self.data_dir = data_dir
        self.NUM_TRAIN = amount * 0.6
        self.NUM_VAL = amount * 0.2
        self.NUM_TEST = amount * 0.2
        self.init_folders()
        self.IM_SIZE = im_size
        self.IM_DRAW_SCALE = 2
        self.IM_DRAW_SIZE = self.IM_SIZE * self.IM_DRAW_SCALE
        self.SMALL_SHAPE_SIZE = int(self.IM_DRAW_SIZE * 0.15)
        self.MEDIUM_SHAPE_SIZE = int(self.IM_DRAW_SIZE * 0.25)
        self.LARGE_SHAPE_SIZE = int(self.IM_DRAW_SIZE * 0.35)
        self.SD_SHAPE = 4

        self.N_SHAPE_QUESTIONS = 5
        self.N_LOC_QUESTIONS = 3
        self.N_COUNT_QUESTIONS = 2
        self.BOOL_SPREAD = 0.5

        self.TRIANGLE_ANGLE_1 = 0
        self.TRIANGLE_ANGLE_2 = -math.pi / 3

        self.MAX_EXTRA = [0, 1]
        self.MAX_EXTRA_COEFF = [0.5, 0.5]
        #self.MAX_EXTRA = [0, 1, 2]
        #self.MAX_EXTRA_COEFF = [0.55, 0.35, 0.10]
        self.RELATIVE_POSITIONS = ["to the left of", "to the right of", "above", "below"]

        self.num2words_dict = {0: "zero", 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
                               6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}
    #    self.create_img(os.path.join(self.data_dir, "diag"), None)
       # self.generate()

    def init_folders(self):
        for root, dirs, files in os.walk(self.data_dir):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))

        self.train_dir = os.path.join(self.data_dir, "train")
        self.val_dir = os.path.join(self.data_dir, "val")
        self.test_dir = os.path.join(self.data_dir, "test")
        self.train_img_dir = os.path.join(self.train_dir, "images")
        self.val_img_dir = os.path.join(self.val_dir, "images")
        self.test_img_dir = os.path.join(self.test_dir, "images")
        os.makedirs(self.train_img_dir)
        os.makedirs(self.val_img_dir)
        os.makedirs(self.test_img_dir)

    def num2words(self, word):
        return self.num2words_dict[word]

    def create_img(self, attr_list):
        # attr(0)=Shape attr(1)=Color attr(2)=Size attr(3)=Location
        arrn = np.random.normal(loc=225, scale=0, size=(self.IM_DRAW_SIZE, self.IM_DRAW_SIZE))
        im = Image.fromarray(arrn)
        im = im.convert("RGB")

        # im = Image.new('RGB', (self.IM_DRAW_SIZE, self.IM_DRAW_SIZE),
        #                (randint(230, 255), randint(230, 255), randint(230, 255)))
        draw = ImageDraw.Draw(im)
        for attr in attr_list:
            self.draw_shape(draw, attr[0], attr[1], attr[2], attr[3])
        del draw
        im = im.resize((self.IM_SIZE, self.IM_SIZE), resample=Image.BILINEAR)
        return im

    def generate(self):
        attribute_combinations = list(itertools.product(Shape, Color, Size, Location))
        print(f"{len(attribute_combinations)} possible attribute combinations")
        random.shuffle(attribute_combinations)
        train_split, val_split, test_split = np.split(
            attribute_combinations, [int(.6 * len(attribute_combinations)),
                                     int(.8 * len(attribute_combinations))])

        train_split = np.vstack((train_split, train_split[np.random.choice(
            len(train_split),
            size=max(0, math.ceil(self.NUM_TRAIN - len(train_split))),
            replace=True), :]))

        val_split = np.vstack((val_split, val_split[np.random.choice(
            len(val_split),
            size=max(0, math.ceil(self.NUM_VAL - len(val_split))),
            replace=True), :]))

        test_split = np.vstack((test_split, test_split[np.random.choice(
            len(test_split),
            size=max(0, math.ceil(self.NUM_TEST - len(test_split))),
            replace=True), :]))
        # print(len(train_split))
        # print(len(val_split))
        # print(len(test_split))
        train_questions = self.create_data(train_split, "train")
        val_questions = self.create_data(val_split, "val")
        test_questions = self.create_data(test_split, "test")

        n_train_qs = 0
        n_val_qs = 0
        n_test_qs = 0
        all_questions = []
        for q_list in train_questions.values():
            n_train_qs += len(q_list)
            all_questions += q_list
        for q_list in val_questions.values():
            n_val_qs += len(q_list)
            all_questions += q_list
        for q_list in test_questions.values():
            n_test_qs += len(q_list)
            all_questions += q_list

        all_answers = ["yes", "no", self.num2words(0)]
        for item in self.MAX_EXTRA:
            all_answers.append(self.num2words(item + 1))
        # for item in self.RELATIVE_POSITIONS:
        #     all_answers.append(item)
        for item in Shape:
            all_answers.append(self._name(item))
        for item in Color:
            all_answers.append(self._name(item))
        for item in Size:
            all_answers.append(self._name(item))
        for item in Location:
            all_answers.append(self._name(item))

        #all_answers = list(set([q["answer"] for q in all_questions]))

        with open(os.path.join(self.train_dir, "questions.json"), 'w') as file:
            json.dump(train_questions, file)
        with open(os.path.join(self.val_dir, "questions.json"), 'w') as file:
            json.dump(val_questions, file)
        with open(os.path.join(self.test_dir, "questions.json"), 'w') as file:
            json.dump(test_questions, file)

        with open(os.path.join(self.data_dir, 'answers.txt'), 'w') as file:
            for answer in all_answers:
                file.write(f'{answer}\n')

        print(f'Generated {len(train_questions)} train images and {n_train_qs} train questions.')
        print(f'Generated {len(val_questions)} val images and {n_val_qs} val questions.')
        print(f'Generated {len(test_questions)} test images and {n_test_qs} test questions.')
        print(f'{len(all_questions)} total questions.')
        print(f'{len(all_answers)} total possible answers.')

    def create_data(self, attributes, split):
        all_questions = {}

        question_id = 0
        for i, attr in enumerate(tqdm(attributes)):
            # Ensure color + shape combo isnt equal to ensure no ambiguity
            count = 0
            max_extra = np.random.choice(self.MAX_EXTRA, 1,
                                         p=self.MAX_EXTRA_COEFF)
            all_attr = [attr]
            while count != max_extra:
                extra_attr = attributes[random.randrange(0, len(attributes))]
                found = True
                for tmp_attr in all_attr:
                    if tmp_attr[0] == extra_attr[0] or tmp_attr[1] == extra_attr[1] or tmp_attr[2] == extra_attr[2]:
                        # Same color/shape combo exists
                        found = False
                        break
                if found:
                    count += 1
                    all_attr.append(extra_attr)

            img = self.create_img(all_attr)
            img.save(os.path.join(self.data_dir, split, "images", f'{split}_{i}.png'), 'png')
            new_questions = self.create_questions(all_attr, i)
            for question in new_questions:
                question["id"] = question_id
                question_id += 1
            all_questions[i] = new_questions
        return all_questions

    def create_questions(self, attr_list, image_id):

        shape_questions = self.shape_questions(attr_list, image_id)
        # Split the questions into yes/no and open questions
        full_shape_bool = [q for q in shape_questions if q["bool"]]
        shape_open = [q for q in shape_questions if not q["bool"]]
        # Resample to balance yes/no answers
        #   Split yes/no
        shape_bool = [q for q in full_shape_bool if q["answer"] == "yes"]
        no_bool = [q for q in full_shape_bool if q["answer"] == "no"]
        for _ in range(len(shape_bool)):
            q = no_bool[random.randrange(0, len(no_bool))]
            shape_bool.append(q)
            no_bool.remove(q)

        # Determine the number of yes/no and open questions to sample
        probs = np.random.uniform(0, 1, self.N_SHAPE_QUESTIONS)
        n_bool = len([x for x in probs if x < self.BOOL_SPREAD])
        n_open = len([x for x in probs if x >= self.BOOL_SPREAD])

        # Sample the questions
        shape_sampled = random.sample(shape_bool, n_bool) + random.sample(shape_open, n_open)

        location_questions = self.location_questions(attr_list, image_id)

        full_location_bool = [q for q in location_questions if q["bool"]]
        location_open = [q for q in location_questions if not q["bool"]]

        location_bool = [q for q in full_location_bool if q["answer"] == "yes"]
        no_bool = [q for q in full_location_bool if q["answer"] == "no"]
        for _ in range(len(location_bool)):
            q = no_bool[random.randrange(0, len(no_bool))]
            location_bool.append(q)
            no_bool.remove(q)

        probs = np.random.uniform(0, 1, self.N_LOC_QUESTIONS)
        n_bool = len([x for x in probs if x < self.BOOL_SPREAD])
        n_open = len([x for x in probs if x >= self.BOOL_SPREAD])

        location_sampled = random.sample(location_bool, n_bool) + random.sample(location_open, n_open)

        count_questions = self.count_questions(attr_list, image_id)

        full_count_bool = [q for q in count_questions if q["bool"]]
        count_open = [q for q in count_questions if not q["bool"]]

        count_bool = [q for q in full_count_bool if q["answer"] == "yes"]
        no_bool = [q for q in full_count_bool if q["answer"] == "no"]
        for _ in range(len(count_bool)):
            if len(no_bool):
                q = no_bool[random.randrange(0, len(no_bool))]
                count_bool.append(q)
                no_bool.remove(q)

        probs = np.random.uniform(0, 1, self.N_COUNT_QUESTIONS)
        n_bool = len([x for x in probs if x < self.BOOL_SPREAD])
        n_open = len([x for x in probs if x >= self.BOOL_SPREAD])

        count_sampled = random.sample(count_bool, n_bool) + random.sample(count_open, n_open)

        questions = shape_sampled + location_sampled + count_sampled

        return questions

    def relate_positions(self, loc1, loc2, relative):
        if "left" in relative:
            # x_min
            if loc1.value[0] < loc2.value[0]:
                return "yes"
        elif "right" in relative:
            # x_min
            if loc1.value[0] > loc2.value[0]:
                return "yes"
        elif "above" in relative:
            # y_min
            if loc1.value[2] < loc2.value[2]:
                return "yes"
        elif "below" in relative:
            if loc1.value[2] > loc2.value[2]:
                return "yes"
        return "no"

    def shape_questions(self, attr_list, image_id):
        questions = []

        n_shapes = len(attr_list)

        for attr in attr_list:
            # Color
            questions.append({"question": f"Which color is the {self._name(attr[0])}?", "answer": self._name(
                attr[1]), "type": "color", "specificity": 1, "bool": False, "image_id": image_id, "attr": [attr[0], attr[1]]})
            questions.append({"question": f"Which color can the {self._name(attr[0])} be considered?", "answer": self._name(
                attr[1]), "type": "color", "specificity": 1, "bool": False, "image_id": image_id, "attr": [attr[0], attr[1]]})

            for c in Color:
                answer = "no"
                if attr[1] == c:
                    answer = "yes"
                questions.append(
                    {"question": f"Is the {self._name(attr[0])} {self._name(c)}?", "answer": answer, "type": "color",
                     "specificity": 1, "bool": True, "image_id": image_id, "attr": [attr[0], c]})
            # Size
            questions.append({"question": f"Which size is the {self._name(attr[0])}?", "answer": self._name(
                attr[2]), "type": "size", "specificity": 1, "bool": False, "image_id": image_id, "attr": [attr[0], attr[2]]})
            questions.append({"question": f"How large is the {self._name(attr[0])}?", "answer": self._name(
                attr[2]), "type": "size", "specificity": 1, "bool": False, "image_id": image_id, "attr": [attr[0], attr[2]]})
            for s in Size:
                answer = "no"
                if attr[2] == s:
                    answer = "yes"
                questions.append(
                    {"question": f"Is the {self._name(attr[0])} {self._name(s)}?", "answer": answer, "type": "size",
                     "specificity": 2, "bool": True, "image_id": image_id, "attr": [attr[0], s]})

            # Color + Size
            questions.append({"question": f"Which color is the {self._name(attr[2])} {self._name(attr[0])}?", "answer": self._name(
                attr[1]), "type": "color", "specificity": 2, "bool": False, "image_id": image_id, "attr": [attr[0], attr[1], attr[2]]})
            questions.append(
                {"question": f"Which color can the {self._name(attr[2])} {self._name(attr[0])} be considered?",
                 "answer": self._name(attr[1]),
                 "type": "color", "specificity": 2,
                 "bool": False, "image_id": image_id, "attr": [attr[0], attr[1], attr[2]]})

            questions.append({"question": f"Which size is the {self._name(attr[1])} {self._name(attr[0])}?", "answer": self._name(
                attr[2]), "type": "size", "specificity": 2, "bool": False, "image_id": image_id, "attr": [attr[0], attr[1], attr[2]]})
            questions.append({"question": f"How large is the {self._name(attr[1])} {self._name(attr[0])}?", "answer": self._name(
                attr[2]), "type": "size", "specificity": 2, "bool": False, "image_id": image_id, "attr": [attr[0], attr[1], attr[2]]})
            if n_shapes == 1:
                # Non-binary
                # Color
                questions.append({"question": f"Which color can the shape be considered?", "answer": self._name(
                    attr[1]), "type": "color", "specificity": 0, "bool": False, "image_id": image_id, "attr": [attr[1]]})
                questions.append({"question": f"Which color is the shape?", "answer": self._name(
                    attr[1]), "type": "color", "specificity": 0, "bool": False, "image_id": image_id, "attr": [attr[1]]})
                # Size
                questions.append({"question": f"Which size can the shape be considered?", "answer": self._name(
                    attr[2]), "type": "size", "specificity": 0, "bool": False, "image_id": image_id, "attr": [attr[2]]})
                questions.append({"question": f"How large is the shape?", "answer": self._name(
                    attr[2]), "type": "size", "specificity": 0, "bool": False, "image_id": image_id, "attr": [attr[2]]})

        for s in Shape:
            answer = "no"
            for attr in attr_list:
                if attr[0] == s:
                    answer = "yes"
                    break
            # Existance questions
            questions.append({"question": f"Is there a {self._name(s)} present?", "answer": answer,
                              "type": "shape", "specificity": 1, "bool": True, "image_id": image_id, "attr": [s]})
            questions.append({"question": f"Is there a {self._name(s)}?", "answer": answer,
                              "type": "shape", "specificity": 1, "bool": True, "image_id": image_id, "attr": [s]})
            questions.append({"question": f"Does the image contain a {self._name(s)}?", "answer": answer,
                              "type": "shape", "specificity": 1, "bool": True, "image_id": image_id, "attr": [s]})

            for size in Size:
                # Binary
                answer = "no"
                for attr in attr_list:
                    if attr[0] == s and attr[2] == size:
                        answer = "yes"
                        break
                # Size + Shape
                questions.append(
                    {"question": f"Is there a {self._name(size)} {self._name(s)} present?", "answer": answer,
                     "type": "size", "specificity": 2, "bool": True, "image_id": image_id, "attr": [size, s]})
                questions.append(
                    {"question": f"Is there a {self._name(size)} {self._name(s)}?", "answer": answer, "type": "size",
                     "specificity": 2, "bool": True, "image_id": image_id, "attr": [size, s]})
                questions.append(
                    {"question": f"Does the image contain a {self._name(size)} {self._name(s)}?", "answer": answer,
                     "type": "size", "specificity": 2, "bool": True, "image_id": image_id, "attr": [size, s]})
                # Size
                answer = "no"
                for attr in attr_list:
                    if attr[2] == size:
                        answer = "yes"
                        break
                    questions.append(
                        {"question": f"Is there a {self._name(size)} shape present?", "answer": answer, "type": "size",
                         "specificity": 1, "bool": True, "image_id": image_id, "attr": [size]})
                    questions.append(
                        {"question": f"Does the image contain a {self._name(size)} shape?", "answer": answer,
                         "type": "size", "specificity": 1, "bool": True, "image_id": image_id, "attr": [size]})

                for c in Color:
                    answer = "no"
                    for attr in attr_list:
                        if attr[0] == s and attr[1] == c:
                            answer = "yes"
                            break

                    # Color + Shape
                    questions.append(
                        {"question": f"Is there a {self._name(c)} {self._name(s)} present?", "answer": answer,
                         "type": "color", "specificity": 2, "bool": True, "image_id": image_id, "attr": [c, s]})
                    questions.append(
                        {"question": f"Is there a {self._name(c)} {self._name(s)}?", "answer": answer, "type": "color",
                         "specificity": 2, "bool": True, "image_id": image_id, "attr": [c, s]})
                    questions.append(
                        {"question": f"Does the image contain a {self._name(c)} {self._name(s)}?", "answer": answer,
                         "type": "color", "specificity": 2, "bool": True, "image_id": image_id, "attr": [c, s]})

                    answer = "no"
                    for attr in attr_list:
                        if attr[1] == c:
                            answer = "yes"
                            break
                    # Color
                    questions.append(
                        {"question": f"Is there a {self._name(c)} shape present?", "answer": answer, "type": "color",
                         "specificity": 1, "bool": True, "image_id": image_id, "attr": [c]})
                    questions.append(
                        {"question": f"Does the image contain a {self._name(c)} shape?", "answer": answer,
                         "type": "color", "specificity": 1, "bool": True, "image_id": image_id, "attr": [c]})

                    # Color + Size
                    answer = "no"
                    for attr in attr_list:
                        if attr[1] == c and attr[2] == size:
                            answer = "yes"
                            break
                    questions.append(
                        {"question": f"Is there a {self._name(c)} {self._name(size)} shape present?", "answer": answer,
                         "type": "shape", "specificity": 2, "bool": True, "image_id": image_id, "attr": [c, size]})
                    questions.append(
                        {"question": f"Does the image contain a {self._name(c)} {self._name(size)} shape?",
                         "answer": answer, "type": "shape", "specificity": 2, "bool": True, "image_id": image_id,
                         "attr": [c, size]})

                    # Color + Shape + Size
                    answer = "no"
                    for attr in attr_list:
                        if attr[0] == s and attr[1] == c and attr[2] == size:
                            answer = "yes"
                            break
                    questions.append(
                        {"question": f"Is there a {self._name(size)} {self._name(c)} {self._name(s)} present?",
                            "answer": answer, "type": "shape", "specificity": 3,
                            "bool": True, "image_id": image_id, "attr": [size, c, s]})
                    questions.append(
                        {"question": f"Is there a {self._name(size)} {self._name(c)} {self._name(s)}?",
                         "answer": answer, "type": "shape", "specificity": 3, "bool": True, "image_id": image_id,
                         "attr": [size, c, s]})
                    questions.append(
                        {"question": f"Does the image contain a {self._name(size)} {self._name(c)} {self._name(s)}?",
                            "answer": answer, "type": "shape", "specificity": 3,
                            "bool": True, "image_id": image_id, "attr": [size, c, s]})

        return questions

    def location_questions(self, attr_list, image_id):
        # TODO Add size here
        questions = []
        n_shapes = len(attr_list)

        if n_shapes > 1:
            answer = "no"
            for attr1 in attr_list:
                found = False
                for attr2 in attr_list:
                    if attr1.all() != attr2.all():
                        if attr1[3] == attr2[3]:
                            answer = "yes"
                            found = True
                            break
                if found:
                    break
            questions.append({"question": f"Are there any shapes overlapping?", "answer": answer,
                              "type": "location", "specificity": 0, "bool": True, "image_id": image_id, "attr": []})

            # for attr1 in attr_list:
            #     found = False
            #     for attr2 in attr_list:
            #         if attr1.all() != attr2.all():
            #             if attr1[3] == attr2[3]:
            #                 answer = "yes"
            #                 found = True
            #                 break
            #         questions.append(
            #             {"question": f"Where is the {self._name(attr1[0])} in relation to the {self._name(attr2[0])}?",
            #             "answer": answer, "type": "location", "specificity": 0, "bool": True, "image_id": image_id})

        for attr in attr_list:
            # Shape + Color
            questions.append(
                {"question": f"Where is the {self._name(attr[1])} {self._name(attr[0])} located?",
                    "answer": self._name(attr[3]),
                 "type": "location", "specificity": 2,
                    "bool": False, "image_id": image_id, "attr": [attr[0], attr[1], attr[3]]})
            questions.append(
                {"question": f"Where can you find the {self._name(attr[1])} {self._name(attr[0])}?",
                    "answer": self._name(attr[3]),
                 "type": "location", "specificity": 2,
                    "bool": False, "image_id": image_id, "attr": [attr[0], attr[1], attr[3]]})
            questions.append(
                {"question": f"In which part is the {self._name(attr[1])} {self._name(attr[0])}?",
                    "answer": self._name(attr[3]),
                 "type": "location", "specificity": 2,
                    "bool": False, "image_id": image_id, "attr": [attr[0], attr[1], attr[3]]})
            questions.append(
                {"question": f"In which part is the {self._name(attr[1])} {self._name(attr[0])} placed?",
                    "answer": self._name(attr[3]),
                 "type": "location", "specificity": 2,
                    "bool": False, "image_id": image_id, "attr": [attr[0], attr[1], attr[3]]})
            questions.append(
                {"question": f"Where is the {self._name(attr[1])} {self._name(attr[0])} placed?",
                    "answer": self._name(attr[3]),
                 "type": "location", "specificity": 2,
                    "bool": False, "image_id": image_id, "attr": [attr[0], attr[1], attr[3]]})

            # Shape + Size
            questions.append(
                {"question": f"Where is the {self._name(attr[2])} {self._name(attr[0])} located?",
                 "answer": self._name(attr[3]),
                 "type": "location", "specificity": 2,
                 "bool": False, "image_id": image_id, "attr": [attr[0], attr[2], attr[3]]})
            questions.append(
                {"question": f"Where can you find the {self._name(attr[2])} {self._name(attr[0])}?",
                    "answer": self._name(attr[3]),
                 "type": "location", "specificity": 2,
                    "bool": False, "image_id": image_id, "attr": [attr[0], attr[2], attr[3]]})
            questions.append(
                {"question": f"In which part is the {self._name(attr[2])} {self._name(attr[0])}?",
                    "answer": self._name(attr[3]),
                 "type": "location", "specificity": 2,
                    "bool": False, "image_id": image_id, "attr": [attr[0], attr[2], attr[3]]})
            questions.append(
                {"question": f"In which part is the {self._name(attr[2])} {self._name(attr[0])} placed?",
                    "answer": self._name(attr[3]),
                 "type": "location", "specificity": 2,
                    "bool": False, "image_id": image_id, "attr": [attr[0], attr[2], attr[3]]})
            questions.append(
                {"question": f"Where is the {self._name(attr[2])} {self._name(attr[0])} placed?",
                    "answer": self._name(attr[3]),
                 "type": "location", "specificity": 2,
                    "bool": False, "image_id": image_id, "attr": [attr[0], attr[2], attr[3]]})
            # Shape + Size + Color
            questions.append(
                {"question": f"Where is the {self._name(attr[1])} {self._name(attr[2])} {self._name(attr[0])} located?",
                 "answer": self._name(attr[3]),
                 "type": "location", "specificity": 3,
                 "bool": False, "image_id": image_id, "attr": [attr[0], attr[1], attr[2], attr[3]]})
            questions.append(
                {"question": f"Where can you find the {self._name(attr[1])} {self._name(attr[2])} {self._name(attr[0])}?",
                    "answer": self._name(attr[3]),
                 "type": "location", "specificity": 3,
                    "bool": False, "image_id": image_id, "attr": [attr[0], attr[1], attr[2], attr[3]]})
            questions.append(
                {"question": f"In which part is the {self._name(attr[1])} {self._name(attr[2])} {self._name(attr[0])}?",
                    "answer": self._name(attr[3]),
                 "type": "location", "specificity": 3,
                    "bool": False, "image_id": image_id, "attr": [attr[0], attr[1], attr[2], attr[3]]})
            questions.append(
                {"question": f"In which part is the {self._name(attr[1])} {self._name(attr[2])} {self._name(attr[0])} placed?",
                    "answer": self._name(attr[3]),
                 "type": "location", "specificity": 3,
                    "bool": False, "image_id": image_id, "attr": [attr[0], attr[1], attr[2], attr[3]]})
            questions.append(
                {"question": f"Where is the {self._name(attr[1])} {self._name(attr[2])} {self._name(attr[0])} placed?",
                    "answer": self._name(attr[3]),
                 "type": "location", "specificity": 3,
                    "bool": False, "image_id": image_id, "attr": [attr[0], attr[1], attr[2], attr[3]]})
            if n_shapes == 1:
                questions.append(
                    {"question": f"Where is the shape located?",
                        "answer": self._name(attr[3]),
                     "type": "location", "specificity": 0,
                        "bool": False, "image_id": image_id, "attr": [attr[3]]})
                questions.append(
                    {"question": f"Where is the shape placed?",
                        "answer": self._name(attr[3]),
                     "type": "location", "specificity": 0,
                        "bool": False, "image_id": image_id, "attr": [attr[3]]})
                questions.append(
                    {"question": f"Where can you find the shape?",
                        "answer": self._name(attr[3]),
                     "type": "location", "specificity": 0,
                        "bool": False, "image_id": image_id, "attr": [attr[3]]})
                questions.append(
                    {"question": f"In which part is the shape?",
                        "answer": self._name(attr[3]),
                     "type": "location", "specificity": 0,
                        "bool": False, "image_id": image_id, "attr": [attr[3]]})
                questions.append(
                    {"question": f"In which part is the shape placed?",
                        "answer": self._name(attr[3]),
                     "type": "location", "specificity": 0,
                        "bool": False, "image_id": image_id, "attr": [attr[3]]})

        for l in Location:
            for attr in attr_list:
                answer = "no"
                if attr[3] == l:
                    answer = "yes"
            # Color + Shape
            questions.append(
                {"question": f"Is there a {self._name(attr[1])} {self._name(attr[0])} in the {self._name(l)}?",
                    "answer": answer,
                 "type": "location", "specificity": 3,
                    "bool": True, "image_id": image_id, "attr": [attr[0], attr[1], l]})
            questions.append(
                {"question": f"Is there a {self._name(attr[1])} {self._name(attr[0])} present in the {self._name(l)}?",
                    "answer": answer,
                 "type": "location", "specificity": 3,
                    "bool": True, "image_id": image_id, "attr": [attr[0], attr[1], l]})

            questions.append(
                {"question": f"Does the image contain a {self._name(attr[1])} {self._name(attr[0])} in the {self._name(l)}?",
                    "answer": answer,
                 "type": "location", "specificity": 3,
                    "bool": True, "image_id": image_id, "attr": [attr[0], attr[1], l]})
            # Shape
            questions.append(
                {"question": f"Is there a {self._name(attr[0])} in the {self._name(l)}?",
                 "answer": answer,
                 "type": "location", "specificity": 2,
                 "bool": True, "image_id": image_id, "attr": [attr[0], l]})
            questions.append(
                {"question": f"Is there a {self._name(attr[0])} present in the {self._name(l)}?",
                 "answer": answer,
                 "type": "location", "specificity": 2,
                 "bool": True, "image_id": image_id, "attr": [attr[0], l]})

            questions.append(
                {"question": f"Does the image contain a {self._name(attr[0])} in the {self._name(l)}?",
                 "answer": answer,
                 "type": "location", "specificity": 2,
                 "bool": True, "image_id": image_id, "attr": [attr[0], l]})
            # General
            questions.append(
                {"question": f"Is there a shape in the {self._name(l)}?",
                 "answer": answer,
                 "type": "location", "specificity": 1,
                 "bool": True, "image_id": image_id, "attr": [l]})
            questions.append(
                {"question": f"Is there a shape present in the {self._name(l)}?",
                 "answer": answer,
                 "type": "location", "specificity": 1,
                 "bool": True, "image_id": image_id, "attr": [l]})

            questions.append(
                {"question": f"Does the image contain a shape in the {self._name(l)}?",
                 "answer": answer,
                 "type": "location", "specificity": 1,
                 "bool": True, "image_id": image_id, "attr": [l]})

        if n_shapes > 1:
            # Relative positions between shapes
            for attr1 in attr_list:
                for attr2 in attr_list:
                    if attr1.all() != attr2.all():
                        for position in self.RELATIVE_POSITIONS:
                            # Shape / Shape
                            questions.append(
                                {"question": f"Is there a {self._name(attr1[0])} {position} the {self._name(attr2[0])}?",
                                 "answer": self.relate_positions(attr1[3],
                                                                 attr2[3],
                                                                 position),
                                 "type": "location", "specificity": 3,
                                 "bool": True, "image_id": image_id, "attr": [attr1[0], attr2[0]]})

                            # Color + Shape / Color + Shape
                            questions.append(
                                {"question": f"Is there a {self._name(attr1[1])} {self._name(attr1[0])} {position} the {self._name(attr2[1])} {self._name(attr2[0])}?",
                                 "answer": self.relate_positions(attr1[3],
                                                                 attr2[3],
                                                                 position),
                                 "type": "location", "specificity": 3,
                                 "bool": True, "image_id": image_id, "attr": [attr1[0], attr1[1], attr2[0], attr2[1]]})
                            # Shape / Color + Shape
                            questions.append(
                                {"question": f"Is there a {self._name(attr1[0])} {position} the {self._name(attr2[1])} {self._name(attr2[0])}?",
                                 "answer": self.relate_positions(attr1[3],
                                                                 attr2[3],
                                                                 position),
                                 "type": "location", "specificity": 3,
                                 "bool": True, "image_id": image_id, "attr": [attr1[0], attr2[0], attr2[1]]})
                            # Color + Shape / Color + Shape
                            questions.append(
                                {"question": f"Is there a {self._name(attr1[1])} {self._name(attr1[0])} {position} the {self._name(attr2[0])}?",
                                 "answer": self.relate_positions(attr1[3],
                                                                 attr2[3],
                                                                 position),
                                 "type": "location", "specificity": 3,
                                 "bool": True, "image_id": image_id, "attr": [attr1[0], attr1[1], attr2[0], ]})

            for attr1 in attr_list:
                for position in self.RELATIVE_POSITIONS:
                    answer = "no"
                    for attr2 in attr_list:
                        if attr1.all() != attr2.all():
                            if self.relate_positions(attr2[3],
                                                     attr1[3],
                                                     position) == "yes":
                                answer = "yes"
                                break
                    # Shape
                    questions.append(
                        {"question": f"Is there a shape {position} the {self._name(attr1[1])} {self._name(attr1[0])}?",
                         "answer": answer, "type": "location", "specificity": 3,
                         "bool": True, "image_id": image_id, "attr": [attr1[0], attr1[1]]})
                    # Shape + Color
                    questions.append(
                        {"question": f"Is there a shape {position} the {self._name(attr1[0])}?", "answer": answer,
                         "type": "location", "specificity": 2,
                         "bool": True, "image_id": image_id, "attr": [attr1[0]]})

        return questions

    def count_questions(self, attr_list, image_id):
        questions = []

        n_shapes = len(attr_list)

        questions.append({"question": f"How many shapes are in the image?", "answer": self.num2words(n_shapes),
                          "type": "count", "specificity": 0, "bool": False, "image_id": image_id, "attr": []})
        questions.append({"question": f"How many objects are in the image?", "answer": self.num2words(n_shapes),
                          "type": "count", "specificity": 0, "bool": False, "image_id": image_id, "attr": []})

        for l in Location:
            answer = 0
            # Regular
            for attr in attr_list:
                # Make sure that bottom matches bottom_left/right too
                if self._name(attr[3]) in self._name(l):
                    answer += 1
            questions.append({"question": f"How many shapes are in the {self._name(l)}?", "answer": self.num2words(
                answer), "type": "count", "specificity": 1, "bool": False, "image_id": image_id, "attr": [l]})
            questions.append({"question": f"How many shapes are located in the {self._name(l)}?", "answer": self.num2words(
                answer), "type": "count", "specificity": 1, "bool": False, "image_id": image_id, "attr": [l]})
            answer = "no"
            for attr in attr_list:
                # Make sure that bottom matches bottom_left/right too
                if self._name(attr[3]) in self._name(l):
                    answer = "yes"
                    break
            questions.append({"question": f"Are there any shapes in the {self._name(l)}?", "answer": answer,
                              "type": "count", "specificity": 1, "bool": True, "image_id": image_id, "attr": [l]})
            questions.append(
                {"question": f"Are there any shapes located in the {self._name(l)}?", "answer": answer, "type": "count",
                 "specificity": 1, "bool": True, "image_id": image_id, "attr": [l]})

        for s in Shape:
            answer = 0
            for attr in attr_list:
                if attr[0] == s:
                    answer += 1
            questions.append({"question": f"How many {self._name(s)}s are in the image?", "answer": self.num2words(
                answer), "type": "count", "specificity": 1, "bool": False, "image_id": image_id, "attr": [s]})
            if answer > 0:
                answer = "yes"
            else:
                answer = "no"
            questions.append({"question": f"Are there any {self._name(s)}s?", "answer": answer,
                              "type": "count", "specificity": 1, "bool": True, "image_id": image_id, "attr": [s]})
            questions.append({"question": f"Are there any {self._name(s)}s in the image?", "answer": answer,
                              "type": "count", "specificity": 1, "bool": True, "image_id": image_id, "attr": [s]})

            # Dropped these because there are always different colores/sized shapes
            # for c in Color:
            #     # Color + Shape
            #     answer = 0
            #     for attr in attr_list:
            #         if attr[0] == attr[0] and attr[1] == attr[1]:
            #             answer += 1
            #     questions.append({"question": f"How many {self._name(c)} {self._name(s)}s are in the image?",
            #                     "answer": answer, "type": "count", "bool": False, "image_id": image_id})
            #     if answer > 0:
            #         answer = "yes"
            #     else:
            #         answer = "no"
            #     questions.append({"question": f"Are there any {self._name(c)} {self._name(s)}s?",
            #                     "answer": answer, "type": "count", "bool": True, "image_id": image_id})
            #     questions.append({"question": f"Are there any {self._name(c)} {self._name(s)}s in the image?",
            #                     "answer": answer, "type": "count", "bool": True, "image_id": image_id})
            #     for size in Size:
            #         # Color + Size + Shape
            # for size in Size:
            #     # Size + Shape
            #     answer = 0
            #     for attr in attr_list:
            #             if attr[0] == attr[0] and attr[1] == attr[1]:
            #                 answer += 1
            #         questions.append({"question": f"How many {self._name(c)} {self._name(s)}s are in the image?",
            #                         "answer": answer, "type": "count", "bool": False, "image_id": image_id})
            #         if answer > 0:
            #             answer = "yes"
            #         else:
            #             answer = "no"
            #         questions.append({"question": f"Are there any {self._name(c)} {self._name(s)}s?",
            #                         "answer": answer, "type": "count", "bool": True, "image_id": image_id})
            #         questions.append({"question": f"Are there any {self._name(c)} {self._name(s)}s in the image?",
            #                         "answer": answer, "type": "count", "bool": True, "image_id": image_id})

        for c in Color:
            answer = 0
            for attr in attr_list:
                if attr[1] == c:
                    answer += 1
            questions.append({"question": f"How many {self._name(c)} shapes are in the image?", "answer": self.num2words(
                answer), "type": "count", "specificity": 1, "bool": False, "image_id": image_id, "attr": [c]})
            questions.append({"question": f"How many {self._name(c)} objects are in the image?", "answer": self.num2words(
                answer), "type": "count", "specificity": 1, "bool": False, "image_id": image_id, "attr": [c]})
            if answer > 0:
                answer = "yes"
            else:
                answer = "no"
            questions.append({"question": f"Are there any {self._name(c)} shapes?", "answer": answer,
                              "type": "count", "specificity": 1, "bool": True, "image_id": image_id, "attr": [c]})
            questions.append({"question": f"Are there any {self._name(c)} shapes in the image?", "answer": answer,
                              "type": "count", "specificity": 1, "bool": True, "image_id": image_id, "attr": [c]})

        for s in Size:
            answer = 0
            for attr in attr_list:
                if attr[2] == s:
                    answer += 1
            questions.append({"question": f"How many {self._name(s)} shapes are in the image?", "answer": self.num2words(
                answer), "type": "count", "specificity": 1, "bool": False, "image_id": image_id, "attr": [s]})
            questions.append({"question": f"How many {self._name(s)} objects are in the image?", "answer": self.num2words(
                answer), "type": "count", "specificity": 1, "bool": False, "image_id": image_id, "attr": [s]})

            if answer > 0:
                answer = "yes"
            else:
                answer = "no"
            questions.append({"question": f"Are there any {self._name(s)} shapes?", "answer": answer,
                              "type": "count", "specificity": 1, "bool": True, "image_id": image_id, "attr": [s]})
            questions.append({"question": f"Are there {self._name(s)} shapes in the image?", "answer": answer,
                              "type": "count", "specificity": 1, "bool": True, "image_id": image_id, "attr": [s]})
        return questions

    def create_questions_old(self, attr, image_id):
        # TODO
        # Equal Yes/No
        # More variation
        attr = attr[0]
        shape = attr[0]
        color = attr[1]
        size = attr[2]
        location = attr[3]
        shape_name = shape.name.lower()
        color_name = color.name.lower().replace("_", " ")
        size_name = size.name.lower()
        location_name = location.name.lower().replace("_", " ")

        questions = [
            # Shape questions
            (f'what shape is in the image?', shape_name),
            (f'what shape is present?', shape_name),
            (f'what shape does the image contain?', shape_name),
            (f'what is the {color_name} shape?', shape_name),
            # Color questions
            (f'what color is the {shape_name}?', color_name),
            (f'what is the color of the {shape_name}?', color_name),
            (f'what color is the shape?', color_name),
            (f'what is the color of the shape?', color_name),
            # Size questions
            (f"what is the size of the {color_name} shape?", size_name),
            (f"what size is the shape?", size_name),
            (f"how large is the shape?", size_name),
            (f"what size is the shape in the image?", size_name),
            (f"what size can the shape be considered?", size_name),
            # Location questions
            (f"in which part of the image is the shape located?", location_name),
            (f"where in the image is the shape?", location_name),
            (f"where can you find the shape in the image?", location_name),
            (f"where is the shape placed?", location_name)
        ]

        yes_questions = []
        no_questions = []
        for s in Shape:
            cur_shape_name = s.name.lower()
            if s is shape:
                pos_answer = 'yes'
                res_list = yes_questions
            else:
                pos_answer = 'no'
                res_list = no_questions
            res_list.append((f'is there a {cur_shape_name}?', pos_answer))
            res_list.append((f'is there a {cur_shape_name} in the image?', pos_answer))
            res_list.append((f'does the image contain a {cur_shape_name}?', pos_answer))
            res_list.append((f'is a {cur_shape_name} present?', pos_answer))

            # neg_answer = 'no' if s is shape else 'yes'
            # yes_no_questions.append((f'is there not a {cur_shape_name}?', neg_answer))
            # yes_no_questions.append((f'is there not a {cur_shape_name} in the image?', neg_answer))
            # yes_no_questions.append((f'does the image not contain a {cur_shape_name}?', neg_answer))
            # yes_no_questions.append((f'is no {cur_shape_name} present?', neg_answer))

        for c in Color:
            cur_color_name = c.name.lower().replace("_", " ")
            if s is color:
                pos_answer = 'yes'
                res_list = yes_questions
            else:
                pos_answer = 'no'
                res_list = no_questions
            res_list.append((f'is there a {cur_color_name} shape?', pos_answer))
            res_list.append((f'is there a {cur_color_name} shape in the image?', pos_answer))
            res_list.append((f'does the image contain a {cur_color_name} shape?', pos_answer))
            res_list.append((f'is a {cur_color_name} shape present?', pos_answer))

            # neg_answer = 'no' if c is color else 'yes'
            # yes_no_questions.append((f'is there not a {cur_color_name} shape?', neg_answer))
            # yes_no_questions.append((f'is there not a {cur_color_name} shape in the image?', neg_answer))
            # yes_no_questions.append((f'does the image not contain a {cur_color_name} shape?', neg_answer))
            # yes_no_questions.append((f'is no {cur_color_name} shape present?', neg_answer))

        for s in Size:
            cur_size_name = s.name.lower()
            if s is size:
                pos_answer = 'yes'
                res_list = yes_questions
            else:
                pos_answer = 'no'
                res_list = no_questions
            res_list.append((f'is there a {cur_size_name} sized shape?', pos_answer))
            res_list.append((f'is there a {cur_size_name} shape in the image?', pos_answer))
            res_list.append((f'does the image contain a {cur_size_name} shape?', pos_answer))
            res_list.append((f'is a {cur_shape_name} shape present?', pos_answer))

            # neg_answer = 'no' if s is size else 'yes'
            # yes_no_questions.append((f'is there not a {cur_size_name} sized shape?', neg_answer))
            # yes_no_questions.append((f'is there not a {cur_size_name} in the image?', neg_answer))
            # yes_no_questions.append((f'does the image not contain a {cur_size_name} shape?', neg_answer))
            # yes_no_questions.append((f'is no {cur_shape_name} shape present?', neg_answer))

        for l in Location:
            cur_loc_name = self._name(l)
            if s is location:
                pos_answer = 'yes'
                res_list = yes_questions
            else:
                pos_answer = 'no'
                res_list = no_questions
            res_list.append((f'is there a shape located in the {cur_loc_name}?', pos_answer))
            res_list.append((f'is the shape located in the {cur_loc_name}?', pos_answer))
            res_list.append((f'does the image contain a shape located in the {cur_loc_name}?', pos_answer))
            res_list.append((f'is a shape present in the {cur_loc_name}?', pos_answer))

            # neg_answer = 'no' if s is location else 'yes'
            # yes_no_questions.append((f'is there not a shape located in the {cur_loc_name}?', neg_answer))
            # yes_no_questions.append((f'is the shape not located in the {cur_loc_name}?', neg_answer))
            # yes_no_questions.append((f'does the image not contain a shape located in the {cur_loc_name}?', neg_answer))
            # yes_no_questions.append((f'is no shape present in the {cur_loc_name}?', neg_answer))

        questions = random.sample(questions, 6)
        yes_no_questions = random.sample(yes_questions, 3) + random.sample(no_questions, 3)
        all_questions = questions + yes_no_questions
        return list(map(lambda x: x + (image_id,), all_questions))

    def _name(self, attr):
        if type(attr) == Size and attr == Size.MEDIUM:
            return "medium sized"
        return attr.name.lower().replace("_", " ")

    def draw_shape(self, draw, shape, color, size, location):
     #   shape = Shape.TRIANGLE
     #   location = Location.BOTTOM_LEFT
       # size = Size.LARGE
        if size is Size.SMALL:
            size_n = self.SMALL_SHAPE_SIZE
        elif size is Size.MEDIUM:
            size_n = self.MEDIUM_SHAPE_SIZE
        else:
            size_n = self.LARGE_SHAPE_SIZE

        r_color = (color.value[0] + int(np.random.normal(scale=10)),
                   color.value[1] + int(np.random.normal(scale=10)),
                   color.value[2] + int(np.random.normal(scale=10)))

        x_min = location.value[0] * self.IM_DRAW_SIZE
        x_max = location.value[1] * self.IM_DRAW_SIZE
        y_min = location.value[2] * self.IM_DRAW_SIZE
        y_max = int(location.value[3] * self.IM_DRAW_SIZE * 0.9)

        if shape is Shape.RECTANGLE:
            w = 0
            h = 1
            min_diff = randint(0, self.SD_SHAPE)
            while True:
                w = randint(1, size_n * 2)
                h = randint(1, size_n * 2)
                if abs(w * h - size_n * size_n) < self.SD_SHAPE * 2 and abs(w - h) > min_diff:
                    break

            if (x_max - w) < 0 or (x_max - w) < x_min:
                x_max += abs(x_max - w) + self.SD_SHAPE

            if (y_max - h) < 0 or (y_max - h) < y_min:
                y_max += abs(y_max - h) + self.SD_SHAPE
            x0 = randint(x_min, max(x_min, x_max - w))
            y0 = randint(y_min, max(y_min + self.SD_SHAPE, y_max - h))
          #  x0 = randint(0, self.IM_DRAW_SIZE - w)
           # y0 = randint(0, self.IM_DRAW_SIZE - h)
            x1 = x0 + w
            y1 = y0 + h
            draw.rectangle([(x0, y0), (x1, y1)], fill=r_color)

        elif shape is Shape.CIRCLE:
            d = int(np.random.normal(size_n, scale=self.SD_SHAPE))

            if (x_max - d) < 0 or (x_max - d) < x_min:
                x_max += abs(x_max - d) + self.SD_SHAPE

            if (y_max - d) < 0 or (y_max - d) < y_min:
                y_max += abs(y_max - d) + self.SD_SHAPE
            x0 = randint(x_min, x_max - d)
            y0 = randint(y_min, y_max - d)
          #  x0 = randint(0, self.IM_DRAW_SIZE - d)
           # y0 = randint(0, self.IM_DRAW_SIZE - d)
            x1 = x0 + d
            y1 = y0 + d
            draw.ellipse([(x0, y0), (x1, y1)], fill=color.value)

        elif shape is Shape.TRIANGLE:
            s = int(np.random.normal(size_n, scale=self.SD_SHAPE))
            x = randint(x_min, x_max - s)
            y = randint(y_min + s, y_max)
            # y = randint(math.ceil(s * math.sin(math.pi / 3)), y_max)
            # TODO make sure triangle stays within scene
           # self.TRIANGLE_ANGLE_2 = uniform(0, math.pi) * -1
            self.TRIANGLE_ANGLE_1 = np.random.normal(loc=0, scale=0.25)
            self.TRIANGLE_ANGLE_2 = -math.pi / np.random.normal(loc=3, scale=0.25)
            draw.polygon([
                (x, y),
                (x + s * math.cos(self.TRIANGLE_ANGLE_1), y + s * math.sin(self.TRIANGLE_ANGLE_1)),
                (x + s * math.cos(self.TRIANGLE_ANGLE_2), y + s * math.sin(self.TRIANGLE_ANGLE_2)),
            ], fill=color.value)

        # else:
        #     raise Exception('Invalid shape!')
if __name__ == "__main__":
   # data_dir = "/data/s2965690/datasets/ExtEasyVQA/"
    data_dir = "/home/nino/Documents/Datasets/TestExtEasyVQA/"
    generator = DataGenerator(data_dir, 128, 15000)
    generator.generate()
    print()
    # attr(0)=Shape attr(1)=Color attr(2)=Size attr(3)=Location

    shape = Shape.CIRCLE
    size = Size.MEDIUM
    location = Location.CENTRE

    # for color in Color:
    #     print(generator._name(color))
    #     attr_list = [(shape, color, size, location)]
    #     im = generator.create_img(attr_list)
    #     im.save(f"{generator._name(color)}.png")
