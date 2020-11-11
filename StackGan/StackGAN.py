from random import sample
from utils import DataLoader, make_gif
import os
import random
import shutil
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Activation, Add, BatchNormalization,
                                     Concatenate, Conv2D, Conv2DTranspose,
                                     Dense, Flatten, Lambda, LeakyReLU, ReLU,
                                     Reshape, UpSampling2D)
from tensorflow.keras.optimizers import Adam
import yaml
from tqdm import tqdm
import copy


class StackGAN():
    def __init__(self, cfg_file="/home/nino/Dropbox/Documents/Master/Thesis/StackGan/cfg.yml",
                 data_dir="/home/nino/Documents/Datasets/Birds", output_dir="./output"):
        self.load_cfg(cfg_file)
        self.data_loader = DataLoader(data_dir=data_dir)
        self.init_output_folders(output_dir)

        self.im_shape1 = (64, 64, 3)

        self.im_shape2 = (256, 256, 3)

    def load_cfg(self, cfg_file):
        with open(cfg_file, 'r') as stream:
            data_loaded = yaml.safe_load(stream)
            self.epochs = data_loaded['TRAIN']["MAX_EPOCH"]
            self.batch_size = data_loaded['TRAIN']["BATCH_SIZE"]
            self.discriminator_lr = data_loaded['TRAIN']["DISCRIMINATOR_LR"]
            self.generator_lr = data_loaded['TRAIN']["GENERATOR_LR"]
            self.lr_decay_epoch = data_loaded['TRAIN']["LR_DECAY_EPOCH"]
            self.kl_coeff = data_loaded["TRAIN"]["COEFF"]["KL"]

            self.noise_dim = data_loaded["Z_DIM"]
            self.embedding_dim = data_loaded["EMBEDDING_DIM"]

            self.ca_dim = data_loaded["GAN"]["EMBEDDING_DIM"]
            self.df_dim = data_loaded["GAN"]["DF_DIM"]
            self.gf_dim = data_loaded["GAN"]["GF_DIM"]

    def init_output_folders(self, output_dir):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        self.output_test_images = os.path.join(output_dir, "test_images")

        if not os.path.exists(self.output_test_images):
            os.makedirs(self.output_test_images)

        self.output_checkpoints = os.path.join(output_dir, "checkpoints")
        if not os.path.exists(self.output_checkpoints):
            os.makedirs(self.output_checkpoints)

        self.output_models = os.path.join(output_dir, "models")
        if not os.path.exists(self.output_models):
            os.makedirs(self.output_models)

        self.output_loss = os.path.join(output_dir, "loss")
        if not os.path.exists(self.output_loss):
            os.makedirs(self.output_loss)

    def init_stage1(self,):
        self.kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02)
        self.generator1 = self.build_stage1_generator()
        self.discriminator1 = self.build_stage1_discriminator()

        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.disc_optimizer = Adam(self.discriminator_lr, beta_1=0.5)
        self.gen_optimizer = Adam(self.generator_lr, beta_1=0.5)

        self.images, self.labels, self.embeds = self.data_loader.load_data(
            train=True, batch_size=self.batch_size, preprocessed=True, img_size=self.im_shape1[0], small=False)

        self.sample_size = len(self.images)

        # self.dataset = tf.data.Dataset.from_tensor_slices(
        #     (self.images, self.embeds, list(range(self.sample_size))))

        self.dataset = tf.data.Dataset.from_tensor_slices(
            (self.images, self.embeds))
        self.checkpoint_prefix = os.path.join(self.output_checkpoints, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.gen_optimizer,
                                              discriminator_optimizer=self.disc_optimizer,
                                              generator=self.generator1,
                                              discriminator=self.discriminator1)

    def ca_sampling(self, args):
        # Reparameziation trick following https://towardsdatascience.com/reparameterization-trick-126062cfd3c3
        mean, log_var = args
        epsilon = tf.random.normal(
            shape=(self.ca_dim,), mean=0.)
        return mean + tf.math.exp(log_var / 2) * epsilon

    def ca_block(self, embed):
        # Conditioning Augmentation block

        conditions = Dense(self.ca_dim*2)(embed)
        conditions = LeakyReLU(alpha=0.2)(conditions)

        mean = conditions[:, :self.ca_dim]
        log_sigma = conditions[:, self.ca_dim:]
        c = Lambda(self.ca_sampling)((mean, log_sigma))

        return c, conditions

    def upsampling_block(self, x, num_kernels, final=False):
        # Upsample using nearest neighbour interpolation followed by a convolotion layer
        x = UpSampling2D(size=(2, 2), interpolation="nearest")(x)
        x = Conv2D(num_kernels, kernel_size=(3, 3), padding='same', strides=1, use_bias=False,
                   kernel_initializer=self.kernel_initializer)(x)

        # x = Conv2DTranspose(num_kernels, kernel_size=(3, 3), strides=2,
        #                     padding='same', kernel_initializer=self.kernel_initializer)(x)
        if final:
            # Use tanh for the final layer to output (-1, 1) and skip batch_norm
            x = Activation('tanh')(x)
        else:
            x = BatchNormalization()(x)
            x = ReLU()(x)
        return x

    def downsampling_block(self, x, num_kernels, batch_norm=True, activation=True):
        x = Conv2D(num_kernels, kernel_size=(4, 4), padding='same', strides=2, use_bias=False,
                   kernel_initializer=self.kernel_initializer)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        if activation:
            x = LeakyReLU(alpha=0.2)(x)
        return x

    def spatial_replication_block(self, input, repl_size, compress=True):
        if compress:
            input = Dense(self.ca_dim)(input)
            input = ReLU()(input)
            #input = LeakyReLU(alpha=0.2)(input)

        x = Reshape((1, 1, self.ca_dim))(input)
        x = Lambda(lambda x: K.tile(
            x, (1, repl_size, repl_size, 1,)))(x)
        return x

    def build_stage1_generator(self,):
        s16 = int(self.im_shape1[0] / 16)

        # Inputs are text embedding and latent noise vector
        input_embed = Input(shape=(self.embedding_dim,))
        input_noise = Input(shape=(self.noise_dim,))

        # Perform conditioning augmentation
        c, norm_dist = self.ca_block(input_embed)
        # Merge the inputs
        concat = Concatenate()([c, input_noise])

        # Reshape to a small s16xs16 img
        x = Dense(s16 * s16 * self.gf_dim * 8, use_bias=False)(concat)
        x = Reshape((s16, s16, self.gf_dim * 8))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Upsample to im_size
        x = self.upsampling_block(x, self.gf_dim * 4)
        x = self.upsampling_block(x, self.gf_dim * 2)
        x = self.upsampling_block(x, self.gf_dim)
        out_img = self.upsampling_block(x, len(self.im_shape1), final=True)

        return Model(inputs=[input_embed, input_noise], outputs=[out_img, norm_dist])

    def build_stage1_discriminator(self,):
        s16 = int(self.im_shape1[0] / 16)
        input_embed = Input(shape=(self.embedding_dim,))
        compressed_embed = self.spatial_replication_block(input_embed, s16, compress=True)

        input_img = Input(shape=(64, 64, 3))

        x = self.downsampling_block(input_img, self.df_dim, batch_norm=False)
        x = self.downsampling_block(x, self.df_dim * 2)
        x = self.downsampling_block(x, self.df_dim * 4)
        downsampled_img = self.downsampling_block(x, self.df_dim * 8)

        concat = Concatenate(axis=3)([downsampled_img, compressed_embed])

        joint_x = Conv2D(self.df_dim * 8, kernel_size=(1, 1), padding='same', strides=1, use_bias=False,
                         kernel_initializer=self.kernel_initializer)(concat)

        joint_x = BatchNormalization()(joint_x)
        joint_x = LeakyReLU(alpha=0.2)(joint_x)

        joint_x = Flatten()(joint_x)

        joint_x = Dense(1)(joint_x)
        pred = Activation('sigmoid')(joint_x)

        return Model(inputs=[input_embed, input_img], outputs=[pred])

    def save_images(self, images, epoch):
        idx = 1
        plt.clf()
        for image in images[:16]:
            ax = plt.subplot(4, 4, idx)
            plt.imshow((image + 1) / 2)
            # ax.set_title("64x64", fontdict={'fontsize': 7})
            plt.axis('off')
            # ax = plt.subplot(4, 4, idx+1)
            # plt.imshow((image_256 + 1) / 2)
            # ax.set_title("256x256", fontdict={'fontsize': 7})
            # plt.axis('off')
            idx += 1
        plt.savefig(os.path.join(self.output_test_images, "gen_epoch_{:04d}.png".format(epoch)))

    def KL_loss(self, mu, log_sigma):
        loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mu))
        loss = tf.reduce_mean(loss)
        return loss

    def sample_wrong_images(self, n):
        fake_inds = []

        while len(fake_inds) < n:
            # Avoid collisions of the random sample with the current batch
            idx = random.randrange(self.sample_size)
            # if idx not in batch_inds and idx not in fake_inds:
            fake_inds.append(idx)

        sampled_images = self.images[fake_inds]

        return tf.convert_to_tensor(sampled_images)

    def generate_examples(self, n_cols, n_rows, epoch):
        inds = random.sample(range(self.sample_size), n_cols*n_rows)

        sample_embeds = self.embeds[inds]
        noise = tf.random.normal(shape=(n_cols*n_rows, self.noise_dim))

        example_images, _ = self.generator1([sample_embeds, noise], training=False)

        plt.clf()
        for i, image in enumerate(example_images):
            ax = plt.subplot(n_rows, n_cols, i+1)
            plt.imshow((image + 1) / 2)
            # ax.set_title("64x64", fontdict={'fontsize': 7})
            plt.axis('off')
            # ax = plt.subplot(4, 4, idx+1)
            # plt.imshow((image_256 + 1) / 2)
            # ax.set_title("256x256", fontdict={'fontsize': 7})
            # plt.axis('off')

        plt.savefig(os.path.join(self.output_test_images, "gen_epoch_{:04d}.png".format(epoch)))

    def save_images(self, images, epoch):
        idx = 1
        plt.clf()
        for image in images[:25]:
            ax = plt.subplot(5, 5, idx)
            plt.imshow((image + 1) / 2)
            # ax.set_title("64x64", fontdict={'fontsize': 7})
            plt.axis('off')
            # ax = plt.subplot(4, 4, idx+1)
            # plt.imshow((image_256 + 1) / 2)
            # ax.set_title("256x256", fontdict={'fontsize': 7})
            # plt.axis('off')
            idx += 1
        plt.savefig(os.path.join(self.output_test_images, "batch_epoch_{:04d}.png".format(epoch)))

    @tf.function()
    def train_step(self, real_images, wrong_images, embeddings):
        n_samples = tf.shape(real_images)[0]

        noise = tf.random.normal(shape=(n_samples, self.noise_dim))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images, norm_dist = self.generator1([embeddings, noise], training=True)

            real_pred = self.discriminator1([embeddings, real_images], training=True)

            wrong_pred = self.discriminator1([embeddings, wrong_images], training=True)

            fake_pred = self.discriminator1([embeddings, fake_images], training=True)

            d_loss_real = self.bce(tf.ones_like(real_pred), real_pred)
            d_loss_wrong = self.bce(tf.zeros_like(wrong_pred), wrong_pred)
            d_loss_fake = self.bce(tf.zeros_like(fake_pred), fake_pred)

            d_loss = d_loss_real + (d_loss_wrong + d_loss_fake) / 2

            g_loss = self.bce(tf.ones_like(fake_pred),
                              fake_pred) + self.KL_loss(norm_dist[:, : self.ca_dim],
                                                        norm_dist[:, self.ca_dim:])

        grad_disc = disc_tape.gradient(d_loss, self.discriminator1.trainable_variables)
        grad_gen = gen_tape.gradient(g_loss, self.generator1.trainable_variables)

        self.disc_optimizer.apply_gradients(zip(grad_disc, self.discriminator1.trainable_variables))
        self.gen_optimizer.apply_gradients(zip(grad_gen, self.generator1.trainable_variables))
        return d_loss, g_loss, fake_images

    def train(self, display_progress=True):
        self.init_stage1()

        epoch_d_loss = []
        epoch_g_loss = []
        for epoch in range(self.epochs):
            start = time.time()

            batch_d_loss = []
            batch_g_loss = []
            example_images = []
            pbar = tqdm(total=self.sample_size, postfix={"d_loss": 0, "g_loss": 0})

            for image_batch, embeddings_batch in self.dataset.shuffle(
                    self.sample_size).batch(
                    self.batch_size):
                wrong_image_batch = self.sample_wrong_images(tf.shape(image_batch)[0].numpy())
                #wrong_image_batch = tf.random.shuffle(image_batch)
                d_loss, g_loss, fake_images = self.train_step(image_batch, wrong_image_batch, embeddings_batch)

                batch_d_loss.append(d_loss)
                batch_g_loss.append(g_loss)

                # Update progress bar
                if display_progress:
                    pbar.set_postfix({"d_loss": d_loss.numpy(), "g_loss": g_loss.numpy()})
                    pbar.update(self.batch_size)

                rand_idx = random.randrange(tf.shape(fake_images)[0])
                example_images.append((fake_images[rand_idx]))

            epoch_d_loss.append(np.mean(batch_d_loss))
            epoch_g_loss.append(np.mean(batch_g_loss))

            pbar.set_description(
                f'Epoch {epoch} took {time.time()-start} sec, avg d_loss={epoch_d_loss[epoch]} g_loss={epoch_g_loss[epoch]}')
            pbar.close()

            if (epoch + 1) % 1 == 0:
                random.shuffle(example_images)
                self.save_images(example_images, epoch)
                self.generate_examples(5, 5, epoch)

            if (epoch + 1) % self.lr_decay_epoch == 0:
                print(f"{self.lr_decay_epoch}th epoch; Halving the learning rate")
                self.discriminator_lr /= 2
                self.generator_lr /= 2
                self.disc_optimizer = Adam(self.discriminator_lr, beta_1=0.5)
                self.gen_optimizer = Adam(self.generator_lr, beta_1=0.5)

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                print("15th iteration; Saving checkpoints")
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                np.savetxt(os.path.join(self.output_loss, f"g_loss_{epoch+1}.txt"), np.array(epoch_g_loss))
                np.savetxt(os.path.join(self.output_loss, f"d_loss_{epoch+1}.txt"), np.array(epoch_d_loss))

        # make_gif(self.output_test_images)
        self.generator1.save(os.path.join(self.output_models, "stage1_gen"))
        self.discriminator1.save(os.path.join(self.output_models, "stage1_disc"))
        np.savetxt(os.path.join(self.output_loss, f"g_loss_final.txt"), np.array(epoch_g_loss))
        np.savetxt(os.path.join(self.output_loss, f"d_loss_final.txt"), np.array(epoch_d_loss))
        print("Finished training!")
