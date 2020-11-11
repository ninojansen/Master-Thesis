import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Embedding, LeakyReLU, Conv2DTranspose, Conv2D, Reshape, Concatenate, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Progbar
import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt


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

    model = Model([input_img, input_label], out)
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


def disc_loss(real_pred, fake_pred):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_pred), real_pred)
    fake_loss = cross_entropy(tf.zeros_like(fake_pred), fake_pred)
    total_loss = real_loss + fake_loss
    return total_loss


def gen_loss(fake_pred):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_pred), fake_pred)


@tf.function
def step(generator, discriminator, image_batch, labels_batch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        noise = tf.random.normal([len(labels_batch), 100])
        fake_images = generator([noise, labels_batch])

        fake_predictions = discriminator([fake_images, labels_batch])
        real_predictions = discriminator([image_batch, labels_batch])

        discriminator_loss = disc_loss(real_predictions, fake_predictions)
        generator_loss = gen_loss(fake_predictions)

        generator_optimizer = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        discriminator_optimizer = tf.keras.optimizers.Adam(
            lr=0.0002, beta_1=0.5)

        generator_gradients = gen_tape.gradient(
            generator_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(
            zip(generator_gradients, generator.trainable_variables))

        discriminator_gradients = disc_tape.gradient(
            discriminator_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, discriminator.trainable_variables))


def generate_examples(gen, epoch):
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

    plt.savefig('cGAN_imgs/image_at_epoch_{:04d}.png'.format(epoch))


(train_images, train_labels), (_, _) = tf.keras.datasets.fashion_mnist.load_data()

BUFFER_SIZE = len(train_images)
BATCH_SIZE = 128
NOISE_DIM = 100
EPOCHS = 10

train_images = train_images.reshape(
    train_images.shape[0], 28, 28, 1).astype('float32')
# Normalize the images to [-1, 1]
train_images = (train_images - 127.5) / 127.5

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_images, train_labels)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

discriminator = make_discriminator()
discriminator.summary()

generator = make_generator(NOISE_DIM)
generator.summary()

for epoch in range(EPOCHS):

    print("\nepoch {}/{}".format(epoch+1, EPOCHS))
    generate_examples(generator, epoch)
    pb = Progbar(BUFFER_SIZE, stateful_metrics=["disc_loss", "gen_loss"])
    for image_batch, labels_batch in train_dataset:
        disc_loss = step(
            generator, discriminator, image_batch, labels_batch)

        # pb.add(BATCH_SIZE, values=[
        #        ('disc_loss', disc_loss), ('gen_loss', gen_loss)])


generator.save("CGAN_gen")
discriminator.save("CGAN_disc")

print("Finished!")
