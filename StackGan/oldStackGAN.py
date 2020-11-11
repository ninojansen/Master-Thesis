from utils import DataLoader, make_gif
import os
import pickle
import random
import shutil
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
import tensorflow.keras.backend as K
from PIL import Image
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Activation, Add, BatchNormalization,
                                     Concatenate, Conv2D, Conv2DTranspose,
                                     Dense, Flatten, Lambda, LeakyReLU, ReLU,
                                     Reshape, UpSampling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Progbar


# Data source: https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view
# Embeddings source: https://drive.google.com/file/d/0B3y_msrWZaXLT1BZdVdycDY5TEE

# TODO https://www.tensorflow.org/datasets/overview use this to load dataset CALTEB Birds 2011
# TODO Make embeddings myself using an  LSTM/ https://www.analyticsvidhya.com/blog/2020/08/top-4-sentence-embedding-techniques-using-python/


class StackGAN():

    def __init__(
            self, epochs=10, batch_size=64, text_embedding_dim=1024, ca_dim=128, noise_dim=100, lr=0.0002,
            lambda_param=1, output_dir="./output", data_dir="/home/nino/Documents/Datasets/Birds"):
        self.epochs = epochs
        self.batch_size = batch_size
        self.ca_dim = ca_dim
        self.noise_dim = noise_dim
        self.text_embedding_dim = text_embedding_dim
        self.lr = lr
        self.lambda_param = lambda_param
        self.data_loader = DataLoader(data_dir=data_dir)
        self.output_dir = output_dir
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

        self.output_test_images = os.path.join(self.output_dir, "test_images")

        if not os.path.exists(self.output_test_images):
            os.makedirs(self.output_test_images)

        self.output_weights = os.path.join(self.output_dir, "weights")
        if not os.path.exists(self.output_weights):
            os.makedirs(self.output_weights)

        self.output_models = os.path.join(self.output_dir, "models")
        if not os.path.exists(self.output_models):
            os.makedirs(self.output_models)

        self.output_loss = os.path.join(self.output_dir, "loss")
        if not os.path.exists(self.output_loss):
            os.makedirs(self.output_loss)

        self.output_logs = os.path.join(self.output_dir, "logs")
        if not os.path.exists(self.output_logs):
            os.makedirs(self.output_logs)
        # Setup stage 1
        self.stage1_gen = self.build_stage1_generator()

        self.stage1_disc = self.build_stage1_discriminator()

        # Setup stage 2
        self.stage2_gen = self.build_stage2_generator()
        self.stage2_disc = self.build_stage2_discriminator()

        self.stage1_adversarial = self.build_stage1_adversarial(
            self.stage1_gen, self.stage1_disc)
        self.stage2_adversarial = self.build_stage2_adversarial(
            self.stage2_gen, self.stage2_disc)

        self.stage1_disc.compile(
            loss="binary_crossentropy", optimizer=Adam(lr=self.lr, beta_1=0.5, beta_2=0.999))
        self.stage1_adversarial.compile(loss=["binary_crossentropy", self.kl_loss], loss_weights=[
            1, 2], optimizer=Adam(lr=self.lr, beta_1=0.5, beta_2=0.999))

        self.stage2_disc.compile(
            loss="binary_crossentropy", optimizer=Adam(lr=self.lr, beta_1=0.5, beta_2=0.999))
        self.stage2_adversarial.compile(loss=['binary_crossentropy', self.kl_loss], loss_weights=[
                                        1, 1], optimizer=Adam(lr=self.lr, beta_1=0.5, beta_2=0.999))

    def ca_sampling(self, args):
        # Reparameziation trick following https://towardsdatascience.com/reparameterization-trick-126062cfd3c3
        mean, log_var = args
        epsilon = tf.random.normal(
            shape=(self.ca_dim,), mean=0.)
        return mean + tf.math.exp(log_var / 2) * epsilon

    def ca_block(self, x):
        # Conditioning Augmentation block

        mean = Dense(self.ca_dim)(x)
        mean = LeakyReLU(alpha=0.2)(mean)

        log_var = Dense(self.ca_dim)(x)
        log_var = LeakyReLU(alpha=0.2)(log_var)

        norm_dist = Concatenate()([mean, log_var])
        c = Lambda(self.ca_sampling)((mean, log_var))

        return c, norm_dist

    def upsampling_block(self, x, num_kernels):
        """An Upsample block with Upsampling2D, Conv2D, BatchNormalization and a ReLU activation.

        Args:
            x: The preceding layer as input.
            num_kernels: Number of kernels for the Conv2D layer.

        Returns:
            x: The final activation layer after the Upsampling block.
        """
        x = UpSampling2D(size=(2, 2), interpolation="nearest")(x)
        x = Conv2D(num_kernels, kernel_size=(3, 3), padding='same', strides=1, use_bias=False,
                   kernel_initializer='he_uniform')(x)
        x = BatchNormalization(gamma_initializer='ones',
                               beta_initializer='zeros')(x)
        x = ReLU()(x)
        return x

    def downsampling_block(self, x, num_kernels, kernel_size=(4, 4), strides=2, batch_norm=True, activation=True):
        """A ConvBlock with a Conv2D, BatchNormalization and LeakyReLU activation.

        Args:
            x: The preceding layer as input.
            num_kernels: Number of kernels for the Conv2D layer.

        Returns:
            x: The final activation layer after the ConvBlock block.
        """

        x = Conv2D(num_kernels, kernel_size=kernel_size, padding='same', strides=strides, use_bias=False,
                   kernel_initializer='he_uniform')(x)
        if batch_norm:
            x = BatchNormalization(gamma_initializer='ones',
                                   beta_initializer='zeros')(x)
        if activation:
            x = LeakyReLU(alpha=0.2)(x)
        return x

    def spatial_replication_block(self, input, repl_size=4, compress=True):
        if compress:
            input = Dense(128)(input)
            input = ReLU()(input)

        x = Reshape((1, 1, 128))(input)

        # x = Lambda(K.tile, arguments={
        #            'n': (1, repl_size, repl_size, 1)})(x)
        x = Lambda(lambda x: K.tile(
            x, (1, repl_size, repl_size, 1,)))(x)
        return x

    def residual_block(self, input):
        # https://towardsdatascience.com/hitchhikers-guide-to-residual-networks-resnet-in-keras-385ec01ec8ff
        x = Conv2D(640, kernel_size=(3, 3), padding='same', use_bias=False,
                   kernel_initializer='he_uniform')(input)
        x = BatchNormalization(gamma_initializer='ones',
                               beta_initializer='zeros')(x)
        x = ReLU()(x)

        x = Conv2D(640, kernel_size=(3, 3), padding='same', use_bias=False,
                   kernel_initializer='he_uniform')(input)
        x = BatchNormalization(gamma_initializer='ones',
                               beta_initializer='zeros')(x)

        x = Add()([x, input])
        x = ReLU()(x)

        return x

    def build_stage1_generator(self,):
        input_embed = Input(shape=(self.text_embedding_dim, ))
        input_noise = Input(shape=(self.noise_dim,))

        c, norm_dist = self.ca_block(input_embed)
        concat = Concatenate()([c, input_noise])

        x = Dense(16384, use_bias=False)(concat)
        x = ReLU()(x)
        x = Reshape((4, 4, 1024), input_shape=(16384,))(x)

        x = self.upsampling_block(x, 512)
        x = self.upsampling_block(x, 256)
        x = self.upsampling_block(x, 128)
        x = self.upsampling_block(x, 64)

        x = Conv2D(3, kernel_size=3, padding='same', strides=1, use_bias=False,
                   kernel_initializer='he_uniform')(x)
        img = Activation('tanh')(x)

        return Model(inputs=[input_embed, input_noise], outputs=[img, norm_dist])

    def build_stage1_discriminator(self,):
        input_text_embed = Input(shape=(self.text_embedding_dim,))
        compressed_embed = self.spatial_replication_block(
            input_text_embed, compress=True)

        input_img = Input(shape=(64, 64, 3))

        x = self.downsampling_block(input_img, 64, batch_norm=False)
        x = self.downsampling_block(x, 128)
        x = self.downsampling_block(x, 256)
        x = self.downsampling_block(x, 512)

        concat = Concatenate()([x, compressed_embed])

        joint_x = Conv2D(512, kernel_size=(1, 1), padding='same', strides=1, use_bias=False,
                         kernel_initializer='he_uniform')(concat)
        # joint_x = BatchNormalization(gamma_initializer='ones',
        #                              beta_initializer='zeros')(joint_x)
        joint_x = LeakyReLU(alpha=0.2)(joint_x)

        # Flatten and add a FC layer to predict.
        joint_x = Flatten()(joint_x)
        joint_x = Dense(1)(joint_x)
        pred = Activation('sigmoid')(joint_x)

        return Model(inputs=[input_text_embed, input_img], outputs=[pred])

    def build_stage2_generator(self,):
        input_embed = Input(shape=(self.text_embedding_dim,))
        input_img = Input(shape=(64, 64, 3))

        c, norm_dist = self.ca_block(input_embed)
        compressed_embed = self.spatial_replication_block(
            c,  repl_size=16, compress=False)
        x = self.downsampling_block(input_img, 256, batch_norm=False)
        x = self.downsampling_block(x, 512)

        concat = Concatenate()([x, compressed_embed])

        x = self.residual_block(concat)
        x = self.residual_block(x)
        x = self.residual_block(x)
        x = self.residual_block(x)
        # 16x16
        x = self.upsampling_block(x, 512)
        # 32x32
        x = self.upsampling_block(x, 256)
        # 64x64
        x = self.upsampling_block(x, 128)
        # 128x128
        x = self.upsampling_block(x, 64)
        # 256x256

        x = Conv2D(3, kernel_size=3, padding='same', strides=1, use_bias=False,
                   kernel_initializer='he_uniform')(x)
        img = Activation('tanh')(x)
        return Model(inputs=[input_embed, input_img], outputs=[img, norm_dist])

    def build_stage2_discriminator(self):
        input_text_embed = Input(shape=(self.text_embedding_dim,))
        compressed_embed = self.spatial_replication_block(
            input_text_embed, compress=True)

        input_img = Input(shape=(256, 256, 3))

        x = self.downsampling_block(input_img, 64, batch_norm=False)
        x = self.downsampling_block(x, 128)
        x = self.downsampling_block(x, 256)
        x = self.downsampling_block(x, 512)
        x = self.downsampling_block(x, 1024)
        x = self.downsampling_block(x, 1024)

        concat = Concatenate()([x, compressed_embed])

        joint_x = Conv2D(512, kernel_size=(1, 1), padding='same', strides=1, use_bias=False,
                         kernel_initializer='he_uniform')(concat)
        joint_x = BatchNormalization(gamma_initializer='ones',
                                     beta_initializer='zeros')(joint_x)
        joint_x = LeakyReLU(alpha=0.2)(joint_x)

        # Flatten and add a FC layer to predict.
        joint_x = Flatten()(joint_x)
        joint_x = Dense(1)(joint_x)
        pred = Activation('sigmoid')(joint_x)

        return Model(inputs=[input_text_embed, input_img], outputs=[pred])

    def build_stage1_adversarial(self, generator, discriminator):
        input_embed = Input(shape=(1024,))
        input_noise = Input(shape=(100,))

        gen_img, norm_dist = generator([input_embed, input_noise])

        discriminator.trainable = False

        pred = discriminator([input_embed, gen_img])

        return Model(inputs=[input_embed, input_noise], outputs=[pred, norm_dist])

    def build_stage2_adversarial(self, generator, discriminator):
        input_embed = Input(shape=(1024,))
        input_img = Input(shape=(64, 64, 3))

        gen_img, norm_dist = generator([input_embed, input_img])

        discriminator.trainable = False

        pred = discriminator([input_embed, gen_img])

        return Model(inputs=[input_embed, input_img], outputs=[pred, norm_dist])

    def kl_loss(self, y_true, y_pred):
        mean = y_pred[:, :self.ca_dim]
        ls = y_pred[:, self.ca_dim:]
        loss = -ls + 0.5 * (-1 + tf.math.exp(2.0 * ls) + tf.math.square(mean))
        loss = K.mean(loss)
        return loss

    def save_models(self):
        self.stage1_gen.save_weights(os.path.join(self.output_weights, "stage1_gen.h5"))
        self.stage1_disc.save_weights(os.path.join(self.output_weights, "stage1_disc.h5"))
        self.stage2_gen.save_weights(os.path.join(self.output_weights, "stage2_gen.h5"))
        self.stage2_disc.save_weights(os.path.join(self.output_weights, "stage2_disc.h5"))
        self.stage1_adversarial.save_weights(os.path.join(self.output_weights, "stage1_adversarial.h5"))
        self.stage2_adversarial.save_weights(os.path.join(self.output_weights, "stage2_adversarial.h5"))

        self.stage1_gen.save(os.path.join(self.output_models, "stage1_gen"))
        self.stage2_disc.save(os.path.join(self.output_models, "stage1_disc"))
        self.stage1_gen.save(os.path.join(self.output_models, "stage2_gen"))
        self.stage2_disc.save(os.path.join(self.output_models, "stage2_disc"))
        self.stage1_adversarial.save(os.path.join(self.output_models, "stage1_adversarial"))
        self.stage2_adversarial.save(os.path.join(self.output_models, "stage2_adversarial"))

    def train_step(self, image_batch, embeddings_batch, stage1=True):
        n_samples = tf.shape(image_batch)[0]

        noise_batch = tf.random.normal(shape=(n_samples, self.noise_dim))

        fake_images_64, _ = self.stage1_gen.predict([embeddings_batch, noise_batch])

        fake_images_256 = None

        positive_labels = tf.ones((n_samples, 1))
        negative_labels = tf.zeros((n_samples, 1))

       # labels += 0.05 * tf.random.uniform(labels.shape)

        if stage1:

            d_loss_pos = self.stage1_disc.train_on_batch([embeddings_batch, image_batch], positive_labels)

            d_loss_neg = self.stage1_disc.train_on_batch([embeddings_batch, fake_images_64], negative_labels)

            noise_batch = tf.random.normal(shape=(n_samples, self.noise_dim))

            g_labels = tf.zeros((n_samples, 1))

            g_loss = self.stage1_adversarial.train_on_batch([embeddings_batch, noise_batch],
                                                            [g_labels, g_labels])
            g_loss = K.sum(g_loss)

        else:
            # stage 2
            fake_images_256 = self.stage2_gen.predict([embeddings_batch, fake_images_64])

            d_loss_pos = self.stage1_disc.train_on_batch([embeddings_batch, image_batch], positive_labels)

            d_loss_neg = self.stage1_disc.train_on_batch([embeddings_batch, fake_images_256], negative_labels)

            g_labels = tf.ones((n_samples, 1))

            g_loss = self.stage2_adversarial.train_on_batch([embeddings_batch, fake_images_64],
                                                            [g_labels, g_labels])
            pass

        d_loss = d_loss_pos + d_loss_neg
        return d_loss, g_loss, fake_images_64, fake_images_256

    def train(self, display_progress=True, small=False):
        self.train_dataset, size = self.data_loader.load_data(
            train=True, batch_size=self.batch_size, preprocessed=True, img_size=64, small=small)

        stage1 = True
        changed_dataset = False

        # tensorboard = TensorBoard(log_dir="logs/".format(time.time()))
        # tensorboard.set_model(self.stage1_gen)
        # tensorboard.set_model(self.stage1_disc)
        # tensorboard.set_model(self.stage2_gen)
        # tensorboard.set_model(self.stage2_disc)

        epoch_g_loss = []
        epoch_d_loss = []
        print("Training Stage I")
        for epoch in range(1, self.epochs+1):

            batch_d_loss = []
            batch_g_loss = []
            example_images = []
            print("epoch {}/{}".format(epoch, self.epochs))
            start = time.time()

            pb = Progbar(size, stateful_metrics=["d_loss", "g_loss"])

            for image_batch, labels_batch, embeddings_batch in self.train_dataset:
                d_loss, g_loss, generated_images_64, generated_images_256 = self.train_step(
                    image_batch, embeddings_batch, stage1=stage1)

                batch_d_loss.append(d_loss)
                batch_g_loss.append(g_loss)

                if display_progress:
                    pb.add(self.batch_size, values=[
                        ('d_loss', d_loss), ('g_loss', g_loss)])

                if stage1:
                    rand_idx = random.randrange(tf.shape(image_batch)[0])
                    example_images.append((generated_images_64[rand_idx], np.ones((256, 256, 3))))
                else:
                    rand_idx = random.randrange(tf.shape(image_batch)[0])
                    example_images.append((generated_images_64[rand_idx], generated_images_256[rand_idx]))

            print(
                f"Total time for epoch {epoch}: {time.time()-start} seconds. d_loss={np.mean(batch_d_loss)} g_loss={np.mean(batch_g_loss)}")
            epoch_d_loss.append(np.mean(batch_d_loss))
            epoch_g_loss.append(np.mean(batch_g_loss))

            random.shuffle(example_images)
            idx = 1
            plt.clf()
            for image_64, image_256 in example_images[:8]:
                ax = plt.subplot(4, 4, idx)
                plt.imshow((image_64 + 1) / 2)
                ax.set_title("64x64", fontdict={'fontsize': 7})
                plt.axis('off')
                ax = plt.subplot(4, 4, idx+1)
                plt.imshow((image_256 + 1) / 2)
                ax.set_title("256x256", fontdict={'fontsize': 7})
                plt.axis('off')
                idx += 2
            plt.savefig(os.path.join(self.output_test_images, "gen_epoch_{:04d}.png".format(epoch)))

            if epoch % 100 == 0:
                # Half the learning rates every 100 epochs
                print("100th epoch; Halving the learning rate...")
                self.lr = self.lr / 2
                if stage1:
                    K.set_value(self.stage1_disc.optimizer.learning_rate, self.lr)
                    K.set_value(self.stage1_adversarial.optimizer.learning_rate, self.lr)
                else:
                    K.set_value(self.stage2_disc.optimizer.learning_rate, self.lr)
                    K.set_value(self.stage2_adversarial.optimizer.learning_rate, self.lr)

            if epoch > self.epochs / 2 and not changed_dataset:
                # Train stage 1 with stage 2 fixed for 600 epochs and stage 2 with stage 1 fixed for another
                # Taken as half the epochs for flexibility
                print("Training Stage II")
                #stage1 = False
                # self.train_dataset, _ = self.data_loader.load_data(
                #     train=True, batch_size=self.batch_size, preprocessed=True, img_size=256, small=small)
                changed_dataset = True
                # self.lr = 0.0002

            if epoch % 10 == 0:
                print("10th iteration; Saving model...")
                self.save_models()
                np.savetxt(os.path.join(self.output_loss, f"g_loss_{epoch}.txt"), np.array(epoch_g_loss))
                np.savetxt(os.path.join(self.output_loss, f"d_loss_{epoch}.txt"), np.array(epoch_d_loss))

        print("Finished training")
        make_gif(self.output_test_images)
        self.save_models()
        np.savetxt(os.path.join(self.output_loss, f"g_loss_final.txt"), np.array(epoch_g_loss))
        np.savetxt(os.path.join(self.output_loss, f"d_loss_final.txt"), np.array(epoch_d_loss))

    def validate(self, show_img=False):
        print("Executing model validation...")
        text_embed = tf.random.normal([1, self.text_embedding_dim])
        noise = tf.random.normal([1, self.noise_dim])

        print("Generating text conditioning variables...")

        print("Testing stage 1 generator...")
        self.stage1_gen.summary()
        print("Generating stage 1 image...")
        generated_image1, _ = self.stage1_gen(
            [text_embed, noise], training=False)

        if show_img:
            plt.imshow(generated_image1[0])
            plt.show()

        print("Testing stage 1 discriminator...")
        self.stage1_disc.summary()
        disc1_pred = self.stage1_disc(
            [text_embed, generated_image1], training=False)
        print("Stage 1 discriminator prediction={}".format(disc1_pred))
        print("Testing stage 2 generator,..")
        self.stage2_gen.summary()
        print("Generating stage 2 image...")
        generated_image2, _ = self.stage2_gen(
            [text_embed, generated_image1], training=False)

        if show_img:
            plt.imshow(generated_image2[0])
            plt.show()

        print("Testing stage 2 discriminator...")
        self.stage2_disc.summary()
        disc2_pred = self.stage2_disc(
            [text_embed, generated_image2], training=False)
        print("Stage 2 discriminator prediction={}".format(disc2_pred))

        print("Testing adversarial")
        self.stage1_adversarial.summary()
        print("Validation Succesful!")


stackGAN = StackGAN()
stackGAN.build_stage1_discriminator().summary()
