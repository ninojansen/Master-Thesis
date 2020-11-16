import tensorflow as tf


class Dataset:
    def __init__(self, batch_size=24):
        self.batch_size = batch_size

    # def load_data_cats_vs_dogs(self,):
    #     self.tf_dataset, self.info = tfds.load("cats_vs_dogs", split="train", with_info=True, as_supervised=True)

    #     self.n_samples = self.info.splits['train'].num_examples
    #     self.n_classes = self.info.features['label'].num_classes

    #     self.tf.dataset.map()

    def load_flowers(self,):
        flowers = tf.keras.utils.get_file(
            'flower_photos',
            'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
            untar=True)

        img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=lambda img: (img - 127.5) / 127.5,)

        self.tf_ds = tf.data.Dataset.from_generator(
            lambda: img_gen.flow_from_directory(
                flowers, target_size=(256, 256),
                class_mode='categorical', batch_size=self.batch_size, shuffle=True),
            output_types=(tf.float32, tf.float32),
            output_shapes=([None, 256, 256, 3],
                           [None, 5]))
        self.n_samples = 3670
        self.n_classes = 5
        self.n_batches = self.n_samples // self.batch_size

    # def load_data_cifar10(self,):
    #     (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    #     train_images = (train_images - 127.5) / 127.5

    #     class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
    #                    'dog', 'frog', 'horse', 'ship', 'truck']

    #     train_labels_encoded = tf.keras.utils.to_categorical(train_labels)

    #     return tf.data.Dataset.from_tensor_slices((train_images, train_labels, train_labels_encoded))

    def shuffle(self,):
        return self.tf_ds.shuffle(self.n_samples)
