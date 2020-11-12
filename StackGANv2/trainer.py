from tensorflow.python.ops.variables import trainable_variables
from model import Discriminators, Generators
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tqdm import tqdm
import time


class StackGANv2Trainer:
    def __init__(self, cfg, dataset):
        self.cfg = cfg
        self.dataset = dataset

        self.g_net = Generators(cfg).model()

        self.n_samples = self.cfg.TRAIN.N_SAMPLES
        self.batch_size = self.cfg.TRAIN.BATCH_SIZE
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

        im_0, im_1, im_2 = self.g_net([tf.random.normal(shape=(100, 10)), tf.random.normal(shape=(100, 100))])

        # self.g_net.summary()
        # [x.summary() for x in self.d_nets]

    @tf.function
    def train_step(self, image_batch_scales, encoded_labels_batch, fake_images):
        it_batch_size = tf.shape(encoded_labels_batch)[0]
        x, y = self.d_nets[0]([encoded_labels_batch, fake_images])
        #######################################################
        # (1) Generate fake images
        ######################################################

        #
        # if self.cfg.TREE.BRANCH_NUM == 1:
        #     fake_images_64 = self.g_net([encoded_labels_batch, z_batch])
        # elif self.cfg.TREE.BRANCH_NUM == 2:
        #     fake_images_64, fake_images_128 = self.g_net([encoded_labels_batch, z_batch])
        # elif self.cfg.TREE.BRANCH_NUM > 2:
        #     fake_images_64, fake_images_128, fake_images_256 = self.g_net([encoded_labels_batch, z_batch])

        #######################################################
        # (2) Update D network
        ######################################################

        d_loss = 0
        with tf.GradientTape() as d_tape:
            for i, d_net in enumerate(self.d_nets):
                pass
                # if i == 0:
                #     fake_images = fake_images_64
                # elif i == 1:
                #     fake_images = fake_images_128
                # elif i == 2:
                #     fake_images = fake_images_256

                # real_preds, real_preds_uncond = d_net(encoded_labels_batch, axis=0), tf.stack([
                #     image_batch_scales[i], fake_images_64], axis=0)])
                #fake_preds, fake_preds_uncond = d_net([encoded_labels_batch, fake_images])
                #fake_preds, fake_preds_uncond=d_net([encoded_labels_batch, image_batch_scales[i]], training = True)

                #     d_loss_real_uncond = self.cfg.TRAIN.COEFF.UNCOND_LOSS * self.bce(
                #         tf.ones(it_batch_size,), real_preds_uncond)
                #     d_loss_real = self.bce(tf.ones(it_batch_size,), real_preds) + d_loss_real_uncond

                #     d_loss_fake_uncond = self.cfg.TRAIN.COEFF.UNCOND_LOSS *
                #     self.bce(tf.zeros(it_batch_size,), fake_preds_uncond)
                #     d_loss_fake = self.bce(tf.zeros(it_batch_size,), fake_preds) + d_loss_fake_uncond

                #     d_loss += d_loss_real + d_loss_fake
                # d_loss /= len(self.d_nets)

        return d_loss, None, None

    def train(self,):

        self.dataset = self.dataset.shuffle(self.n_samples).batch(self.batch_size)
        for epoch in range(self.cfg.TRAIN.MAX_EPOCH):

            with tqdm(total=self.n_samples, postfix={"d_loss": 0, "g_loss": 0}) as pbar:
                for image_batch, labels_batch, encoded_labels_batch in self.dataset:

                    image_batch_scales = []
                    if self.cfg.TREE.BRANCH_NUM > 0:
                        image_batch_scales.append(tf.image.resize(image_batch, [64, 64]))
                    if self.cfg.TREE.BRANCH_NUM > 1:
                        image_batch_scales.append(tf.image.resize(image_batch, [128, 128]))
                    if self.cfg.TREE.BRANCH_NUM > 2:
                        image_batch_scales.append(tf.image.resize(image_batch, [256, 256]))

                    it_batch_size = tf.shape(encoded_labels_batch)[0]
                    z_batch = tf.random.normal(shape=(it_batch_size, self.cfg.GAN.Z_DIM))

                    fake_images_64 = self.g_net([encoded_labels_batch, z_batch])
                    #d_loss, _, _ = self.train_step(image_batch_scales, encoded_labels_batch, image_batch_scales[0])
                    # pbar.set_postfix({"d_loss": d_loss.numpy(), "g_loss": 0})

                    pbar.update(self.batch_size)
