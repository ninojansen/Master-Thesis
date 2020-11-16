from model import Discriminators, Generators
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

# TODO look into tensorboard viz
# Add CA embedding and kl loss
# Add Mismatched loss for d_net


class StackGANv2Trainer:
    def __init__(self, cfg, dataset, output_dir="./output"):
        self.cfg = cfg
        self.dataset = dataset

        self.init_output_folders(output_dir)

        self.g_net = Generators(branch_num=cfg.TREE.BRANCH_NUM, Ng=cfg.GAN.GF_DIM,
                                c_dim=self.cfg.GAN.EMBEDDING_DIM, z_dim=cfg.GAN.Z_DIM).model()

        discriminator_loader = Discriminators(Nd=self.cfg.GAN.DF_DIM, c_dim=self.cfg.GAN.EMBEDDING_DIM)
        self.d_nets = []
        if self.cfg.TREE.BRANCH_NUM > 0:
            self.d_nets.append(discriminator_loader.D_NET64())
        if self.cfg.TREE.BRANCH_NUM > 1:
            self.d_nets.append(discriminator_loader.D_NET128())
        if self.cfg.TREE.BRANCH_NUM > 2:
            self.d_nets.append(discriminator_loader.D_NET256())

        self.g_optimizer = Adam(lr=self.cfg.TRAIN.DISCRIMINATOR_LR, beta_1=0.5, beta_2=0.999)
        self.d_optimizers = [Adam(lr=self.cfg.TRAIN.DISCRIMINATOR_LR, beta_1=0.5, beta_2=0.999) for x in self.d_nets]
        self.bce = BinaryCrossentropy(from_logits=True)

        # self.checkpoint_prefix = os.path.join(self.output_checkpoints, "ckpt")
        # self.checkpoint = tf.train.Checkpoint(g_opt=self.,
        #                                       discriminator_optimizer=self.disc_optimizer,
        #                                       generator=self.generator1,

        #                                       discriminator=self.discriminator1)

        # self.g_net.summary()
        # [x.summary() for x in self.d_nets]

    def init_output_folders(self, output_dir):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        self.output_test_images = os.path.join(output_dir, "examples")

        if not os.path.exists(self.output_test_images):
            os.makedirs(self.output_test_images)

        # self.output_checkpoints = os.path.join(output_dir, "checkpoints")
        # if not os.path.exists(self.output_checkpoints):
        #     os.makedirs(self.output_checkpoints)

        self.output_models = os.path.join(output_dir, "models")
        if not os.path.exists(self.output_models):
            os.makedirs(self.output_models)

        # self.output_loss = os.path.join(output_dir, "loss")
        # if not os.path.exists(self.output_loss):
        #     os.makedirs(self.output_loss)

    @tf.function
    def loss_generator(self, d_pred, d_pred_uncond):
        return self.bce(tf.ones_like(d_pred),
                        d_pred) + self.cfg.TRAIN.COEFF.UNCOND_LOSS * self.bce(tf.ones_like(d_pred_uncond),
                                                                              d_pred_uncond)

    @tf.function
    def loss_discriminator(self, real, d_pred, d_pred_uncond):
        if real:
            return self.bce(tf.ones_like(d_pred),
                            d_pred) + self.cfg.TRAIN.COEFF.UNCOND_LOSS * self.bce(tf.ones_like(d_pred_uncond),
                                                                                  d_pred_uncond)
        else:
            return self.bce(tf.zeros_like(d_pred),
                            d_pred) + self.cfg.TRAIN.COEFF.UNCOND_LOSS * self.bce(tf.zeros_like(d_pred_uncond),
                                                                                  d_pred_uncond)

    @tf.function
    def train_step(self, image_batch_scales, labels_batch):
        it_batch_size = tf.shape(labels_batch)[0]
        z_batch = tf.random.normal(shape=(it_batch_size, self.cfg.GAN.Z_DIM))

        d0_loss, d1_loss, d2_loss, g_loss = 0, 0, 0, 0

        with tf.GradientTape(persistent=True) as tape:
            # Forward pass G0-Gn
            fake_images_64, fake_images_128, fake_images_256 = None, None, None
            if self.cfg.TREE.BRANCH_NUM == 1:
                fake_images_64 = self.g_net([labels_batch, z_batch], training=True)
            elif self.cfg.TREE.BRANCH_NUM == 2:
                fake_images_64, fake_images_128 = self.g_net([labels_batch, z_batch],  training=True)
            elif self.cfg.TREE.BRANCH_NUM > 2:
                fake_images_64, fake_images_128, fake_images_256 = self.g_net(
                    [labels_batch, z_batch],  training=True)

            fake_images = fake_images_64

            for i, d_net in enumerate(self.d_nets):
                if i == 1:
                    fake_images = fake_images_128
                elif i == 2:
                    fake_images = fake_images_256

                real_preds, real_preds_uncond = d_net([labels_batch, image_batch_scales[i]], training=True)
                fake_preds, fake_preds_uncond = d_net([labels_batch, fake_images], training=True)
                wrong_preds, wrong_preds_uncond = d_net(
                    [labels_batch, tf.random.shuffle(image_batch_scales[i])], training=True)

                if i == 0:
                    d0_loss = self.loss_discriminator(True, real_preds, real_preds_uncond) + 0.5 * (self.loss_discriminator(
                        False, fake_preds, fake_preds_uncond) + self.loss_discriminator(False, wrong_preds, wrong_preds_uncond))
                elif i == 1:
                    d1_loss = self.loss_discriminator(True, real_preds, real_preds_uncond) + 0.5 * (self.loss_discriminator(
                        False, fake_preds, fake_preds_uncond) + self.loss_discriminator(False, wrong_preds, wrong_preds_uncond))
                elif i == 2:
                    d2_loss = self.loss_discriminator(True, real_preds, real_preds_uncond) + 0.5 * (self.loss_discriminator(
                        False, fake_preds, fake_preds_uncond) + self.loss_discriminator(False, wrong_preds, wrong_preds_uncond))

                g_loss += self.loss_generator(fake_preds, fake_preds_uncond)

        for i, (d_net, d_opt) in enumerate(zip(self.d_nets, self.d_optimizers)):
            d_loss = d0_loss
            if i == 1:
                d_loss = d1_loss
            elif i == 2:
                d_loss = d2_loss
            d_grads = tape.gradient(d_loss, d_net.trainable_variables)
            d_opt.apply_gradients(zip(d_grads, d_net.trainable_variables))

        g_grads = tape.gradient(g_loss, self.g_net.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.g_net.trainable_variables))

        d_loss = (d0_loss + d1_loss + d2_loss) / self.cfg.TREE.BRANCH_NUM
        return d_loss, g_loss

    def generate_examples(self, labels_batch, epoch):
        label_map = ["dandelion", 'daisy', "tulips", 'sunflowers', 'roses']
        it_batch_size = tf.shape(labels_batch)[0]
        z_batch = tf.random.normal(shape=(it_batch_size, self.cfg.GAN.Z_DIM))

        fake_images_64, fake_images_128, fake_images_256 = None, None, None
        if self.cfg.TREE.BRANCH_NUM == 1:
            fake_images_64 = self.g_net([labels_batch, z_batch], training=True)
            plt.clf()
            for i, im_64 in enumerate(fake_images_64):
                ax = plt.subplot(4, 4, i+1)
                plt.imshow((im_64 + 1) / 2)
                ax.set_title(label_map[tf.argmax(labels_batch[i], axis=0)], fontdict={'fontsize': 7})
                plt.axis('off')
                if i == 15:
                    break

        elif self.cfg.TREE.BRANCH_NUM == 2:
            fake_images_64, fake_images_128 = self.g_net([labels_batch, z_batch],  training=True)
            plt.clf()

            i = 1
            for j, (im_64, im_128) in enumerate(zip(fake_images_64, fake_images_128)):
                ax = plt.subplot(4, 4, i)
                plt.imshow((im_64 + 1) / 2)
                ax.set_title(f'{label_map[tf.argmax(labels_batch[j], axis=0)]} 64x64', fontdict={'fontsize': 7})
                plt.axis('off')
                ax = plt.subplot(4, 4, i+1)
                plt.imshow((im_128 + 1) / 2)
                ax.set_title(f'{label_map[tf.argmax(labels_batch[j], axis=0)]} 128x128', fontdict={'fontsize': 7})
                plt.axis('off')

                i += 2
                if i > 16:
                    break

        elif self.cfg.TREE.BRANCH_NUM > 2:
            fake_images_64, fake_images_128, fake_images_256 = self.g_net(
                [labels_batch, z_batch],  training=True)
        plt.savefig(os.path.join(self.output_test_images, "gen_epoch_{:04d}.png".format(epoch)))

    def save_models(self,):
        self.g_net.save(os.path.join(self.output_models, "g_net"))
        for i, d_net in enumerate(self.d_nets):
            d_net.save(os.path.join(self.output_models, f"d{i}_net"))

    def train(self, hide_progress=False):

        epoch_d_loss, epoch_g_loss = [], []
        for epoch in range(self.cfg.TRAIN.MAX_EPOCH):
            start = time.time()

            if not hide_progress:
                pbar = tqdm(total=self.dataset.n_samples, postfix={"d_loss": 0, "g_loss": 0})
            else:
                pbar = None

            batch_d_loss, batch_g_loss = [], []
            for i, (image_batch, labels_batch) in enumerate(self.dataset.tf_ds):

                image_batch_scales = []

                if self.cfg.TREE.BRANCH_NUM > 0:
                    image_batch_scales.append(tf.image.resize(image_batch, [64, 64]))
                if self.cfg.TREE.BRANCH_NUM > 1:
                    image_batch_scales.append(tf.image.resize(image_batch, [128, 128]))
                if self.cfg.TREE.BRANCH_NUM > 2:
                    # Images are 256x256 by default
                    image_batch_scales.append(image_batch)

                d_loss, g_loss = self.train_step(
                    image_batch_scales, labels_batch)

                batch_d_loss.append(d_loss)
                batch_g_loss.append(g_loss)

                if not hide_progress:
                    pbar.set_postfix({"d_loss": d_loss.numpy(), "g_loss": g_loss.numpy()})
                    pbar.update(self.dataset.batch_size)

                if i > self.dataset.n_batches:
                    # Iterater over all the batches
                    if (epoch + 1) % 1 == 0:
                        self.generate_examples(labels_batch, epoch)
                    break

            epoch_d_loss.append(np.mean(batch_d_loss))
            epoch_g_loss.append(np.mean(batch_g_loss))

            if not hide_progress:
                pbar.set_description(
                    f'Epoch {epoch} took {time.time()-start} sec, avg d_loss={epoch_d_loss[epoch]} g_loss={epoch_g_loss[epoch]}')
                pbar.close()
            else:
                print(
                    f'Epoch {epoch} took {time.time()-start} sec, avg d_loss={epoch_d_loss[epoch]} g_loss={epoch_g_loss[epoch]}')

            if (epoch + 1) % 10 == 0:
                self.save_models()

        self.save_models()
