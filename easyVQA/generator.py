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

        self.TRIANGLE_ANGLE_1 = 0
        self.TRIANGLE_ANGLE_2 = -math.pi / 3

    #    self.create_img(os.path.join(self.data_dir, "diag"), None)
        self.generate()

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

    def create_img(self, attr):
        # attr(0)=Shape attr(1)=Color attr(2)=Size attr(3)=Location
        arrn = np.random.normal(loc=255, scale=10, size=(self.IM_DRAW_SIZE, self.IM_DRAW_SIZE))
        im = Image.fromarray(arrn)
        im = im.convert("RGB")
        draw = ImageDraw.Draw(im)
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

        all_questions = train_questions + val_questions + test_questions
        all_answers = list(set(map(lambda q: q[1], all_questions)))

        with open(os.path.join(self.train_dir, "questions.json"), 'w') as file:
            json.dump(train_questions, file)
        with open(os.path.join(self.val_dir, "questions.json"), 'w') as file:
            json.dump(val_questions, file)
        with open(os.path.join(self.test_dir, "questions.json"), 'w') as file:
            json.dump(test_questions, file)

        with open(os.path.join(self.data_dir, 'answers.txt'), 'w') as file:
            for answer in all_answers:
                file.write(f'{answer}\n')

        print(f'Generated {self.NUM_TRAIN} train images and {len(train_questions)} train questions.')
        print(f'Generated {self.NUM_VAL} val images and {len(val_questions)} val questions.')
        print(f'Generated {self.NUM_TEST} test images and {len(test_questions)} test questions.')
        print(f'{len(all_answers)} total possible answers.')

    def create_data(self, attributes, split):
        qs = []
        for i, attr in enumerate(tqdm(attributes)):
            img = self.create_img(attr)
            img.save(os.path.join(self.data_dir, split, "images", f'{split}_{i}.png'), 'png')
            new_qs = self.create_questions(attr, i)
            qs += new_qs
        return qs

    def create_questions(self, attr, image_id):
        # TODO
        # Equal Yes/No
        # More variation
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
            cur_loc_name = l.name.lower().replace("_", " ")
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
        y_max = location.value[3] * self.IM_DRAW_SIZE

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
    data_dir = "/home/nino/Documents/Datasets/ExtEasyVQA/"
    generator = DataGenerator(data_dir, 64, 10000)
