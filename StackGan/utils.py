import pickle
import numpy as np
import pandas as pd
from PIL import Image
import PIL
import tensorflow as tf
import os
import imageio
import glob


class DataLoader():
    def __init__(self, data_dir="/home/nino/Documents/Datasets/Birds"):
        self.data_dir = data_dir

        train_dir = data_dir + "/captions/train"
        test_dir = data_dir + "/captions/test"

        self.embeddings_path_train = train_dir + "/char-CNN-RNN-embeddings.pickle"
        self.embeddings_path_test = test_dir + "/char-CNN-RNN-embeddings.pickle"
        self.filename_path_train = train_dir + "/filenames.pickle"
        self.filename_path_test = test_dir + "/filenames.pickle"
        self.class_id_path_train = train_dir + "/class_info.pickle"
        self.class_id_path_test = test_dir + "/class_info.pickle"

    def load_class_ids_filenames(self, class_id_path, filename_path):
        with open(class_id_path, 'rb') as file:
            class_id = pickle.load(file, encoding='latin1')

        with open(filename_path, 'rb') as file:
            filename = pickle.load(file, encoding='latin1')

        return class_id, filename

    def load_text_embeddings(self, text_embeddings):
        with open(text_embeddings, 'rb') as file:
            embeds = pickle.load(file, encoding='latin1')
            embeds = np.array(embeds)

        return embeds

    def load_bbox(self, data_path):
        bbox_path = data_path + '/bounding_boxes.txt'
        image_path = data_path + '/images.txt'
        bbox_df = pd.read_csv(bbox_path, delim_whitespace=True,
                              header=None).astype(int)
        filename_df = pd.read_csv(
            image_path, delim_whitespace=True, header=None)

        filenames = filename_df[1].tolist()
        bbox_dict = {i[:-4]: [] for i in filenames[:2]}

        for i in range(0, len(filenames)):
            bbox = bbox_df.iloc[i][1:].tolist()
            dict_key = filenames[i][:-4]
            bbox_dict[dict_key] = bbox

        return bbox_dict

    def load_image(self, image_path, bounding_box, size):
        """Crops the image to the bounding box and then resizes it.
        """
        image = Image.open(image_path).convert('RGB')
        w, h = image.size
        if bounding_box is not None:
            r = int(np.maximum(bounding_box[2], bounding_box[3]) * 0.75)
            c_x = int((bounding_box[0] + bounding_box[2]) / 2)
            c_y = int((bounding_box[1] + bounding_box[3]) / 2)
            y1 = np.maximum(0, c_y - r)
            y2 = np.minimum(h, c_y + r)
            x1 = np.maximum(0, c_x - r)
            x2 = np.minimum(w, c_x + r)
            image = image.crop([x1, y1, x2, y2])

        image = image.resize(size, PIL.Image.BILINEAR)
        return image

    def preprocess_images(self, size=64):
        bbox_dict = self.load_bbox(self.data_dir)

        preprocessed_dir = os.path.join(self.data_dir, f"preprocessed_images_{size}")
        if not os.path.exists(preprocessed_dir):
            os.makedirs(preprocessed_dir)

        for root, dirs, files in os.walk(os.path.join(self.data_dir, "images")):
            dir_name = os.path.basename(os.path.normpath(root))
            for file in files:
                bbox = bbox_dict[os.path.join(
                    dir_name, file).replace(".jpg", "")]
                preprocessed_image_dir = os.path.join(
                    preprocessed_dir, dir_name)
                if not os.path.exists(preprocessed_image_dir):
                    os.makedirs(preprocessed_image_dir)

                image = self.load_image(os.path.join(
                    root, file), bbox, (size, size))

                image.save(os.path.join(preprocessed_image_dir, file))

    def load_data(self, train=True, batch_size=64, preprocessed=True, img_size=(64), small=False):
        """Loads the Dataset.
        """
        if train:
            filename_path = self.filename_path_train
            class_id_path = self.class_id_path_train
            embeddings_path = self.embeddings_path_train
        else:
            filename_path = self.filename_path_test
            class_id_path = self.class_id_path_test
            embeddings_path = self.embeddings_path_test

        class_id, filenames = self.load_class_ids_filenames(
            class_id_path, filename_path)
        embeddings = self.load_text_embeddings(embeddings_path)
        bbox_dict = self.load_bbox(self.data_dir)

        x, y, embeds = [], [], []

        size = 0
        for i, filename in enumerate(filenames):
            bbox = bbox_dict[filename]
            if small and size > 64:
                break
            try:
                if preprocessed:
                    image_path = f'{self.data_dir}/preprocessed_images_{img_size}/{filename}.jpg'
                    image = Image.open(image_path).convert('RGB')
                else:
                    image_path = f'{self.data_dir}/images/{filename}.jpg'
                    image = self.load_image(image_path, bbox, (img_size, img_size))
                e = embeddings[i, :, :]
                embed_index = np.random.randint(0, e.shape[0] - 1)
                embed = e[embed_index, :]

                x.append(np.array(image))
                y.append(class_id[i])
                embeds.append(embed)
                size += 1

            except Exception as e:
                print(f'{e}')

        x = np.array(x)
        x = (x - 127.5) / 127.5

        y = np.array(y)
        embeds = np.array(embeds)

        # dataset = tf.data.Dataset.from_tensor_slices(
        #     (x, y, embeds))

        return x, y, embeds


def make_gif(source_dir):
    anim_file = os.path.join(source_dir, 'training.gif')

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(os.path.join(source_dir, '*.png'))
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
