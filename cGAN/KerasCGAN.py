import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Embedding, LeakyReLU, Conv2DTranspose, Conv2D, Reshape, Concatenate, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Progbar
from tensorflow.keras.optimizers import Adam
import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt
import argparse
import os
import glob


def make_discriminator(im_shape=(28, 28, 1), n_classes=10):
    input_img = Input(shape=(28, 28, 1))

    input_label = Input(shape=(1,))
    label_emb = Embedding(n_classes, 50, trainable=False)(input_label)
    label_emb = Dense(im_shape[0]*im_shape[1])(label_emb)
    label_emb = Reshape((im_shape[0], im_shape[1], 1))(label_emb)

    merged = Concatenate()([input_img, label_emb])

    conv1 = Conv2D(64, (3, 3), strides=(2, 2), padding="same")(merged)
    conv1 = LeakyReLU(alpha=0.2)(conv1)

    conv2 = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(conv1)
    conv2 = LeakyReLU(alpha=0.2)(conv2)

    flat = Flatten()(conv2)
    flat = Dropout(0.4)(flat)

    out = Dense(1, activation="sigmoid")(flat)

    opt = Adam(lr=0.002, beta_1=0.5)
    model = Model([input_img, input_label], out)
    model.compile(loss="binary_crossentropy",
                  optimizer=opt, metrics=['accuracy'])
    return model


def make_generator(noise_dim, n_classes=10):
    input_noise = Input(noise_dim, )
    img_emb = Dense(7*7*128)(input_noise)
    img_emb = LeakyReLU(alpha=0.2)(img_emb)
    img_emb = Reshape((7, 7, 128))(img_emb)

    input_label = Input(shape=(1,))
    label_emb = Embedding(n_classes, 50, trainable=False)(input_label)
    label_emb = Dense(7*7)(label_emb)
    label_emb = Reshape((7, 7, 1))(label_emb)

    merged = Concatenate()([img_emb, label_emb])

    gen1 = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(merged)
    gen1 = LeakyReLU(alpha=0.2)(gen1)

    gen2 = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')(gen1)
    gen2 = LeakyReLU(alpha=0.2)(gen2)

    out = Conv2D(1, (7, 7), activation='tanh', padding='same')(gen2)

    model = Model([input_noise, input_label], out)
    return model


def make_gan(gen_model, disc_model):
    disc_model.trainable = False
    gen_noise, gen_label = gen_model.input

    gen_output = gen_model.output

    gan_output = disc_model([gen_output, gen_label])

    model = Model([gen_noise, gen_label], gan_output)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


def save_examples(gen, epoch, output_dir):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).

    label_map = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress",
                 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}

    labels = tf.convert_to_tensor(list(label_map.keys()))

    noise = tf.random.normal([10, 100])
    images = gen([noise, labels], training=False)

    fig = plt.figure(figsize=(5, 2))

    for i in range(images.shape[0]):
        plt.subplot(5, 2, i+1)
        plt.imshow(images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.text(36, 20, label_map[i])
        plt.axis('off')

    plt.savefig(os.path.join(output_dir, "gen_epoch_{:04d}.png".format(epoch)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fashion MNIST CGAN')
    parser.add_argument('--epochs', default=10)
    parser.add_argument('--output_dir', default="./")
    args = parser.parse_args()

    (train_images, train_labels), (_, _) = tf.keras.datasets.fashion_mnist.load_data()

    BUFFER_SIZE = len(train_images)
    batch_size = 128
    noise_dim = 100
    n_classes = 10
    epochs = int(args.epochs)

    train_images = train_images.reshape(
        train_images.shape[0], 28, 28, 1).astype('float32')
    # Normalize the images to [-1, 1]
    train_images = (train_images - 127.5) / 127.5

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_images, train_labels)).shuffle(BUFFER_SIZE).batch(batch_size)

    discriminator = make_discriminator()
    discriminator.summary()

    generator = make_generator(noise_dim)
    generator.summary()

    gan = make_gan(generator, discriminator)

    for file in glob.glob(os.path.join(args.output_dir, "gen_epoch_*")):
        os.remove(file)

    for epoch in range(epochs):

        print("\nepoch {}/{}".format(epoch+1, epochs))
        save_examples(generator, epoch, args.output_dir)
        pb = Progbar(BUFFER_SIZE, stateful_metrics=["disc_loss", "gen_loss"])
        for image_batch, labels_batch in train_dataset:
            n_samples = len(labels_batch)
            noise = tf.random.normal([n_samples, 100])

            fake_images = generator([noise, labels_batch])

            d_real_loss, _ = discriminator.train_on_batch(
                [image_batch, labels_batch], tf.ones_like(labels_batch))

            d_fake_loss, _ = discriminator.train_on_batch(
                [fake_images, labels_batch], tf.zeros_like(labels_batch))

            noise = tf.random.normal([len(labels_batch), 100])
            gen_labels = np.random.randint(0, n_classes, n_samples)
            g_loss = gan.train_on_batch(
                [noise, gen_labels], tf.ones_like(labels_batch))

            pb.add(batch_size, values=[
                ('disc_loss', d_real_loss+d_fake_loss), ('gen_loss', g_loss)])

    generator.save("CGAN_gen")
    discriminator.save("CGAN_disc")

    print("Finished!")
