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
import matplotlib.pyplot as plt


class GLU(tf.keras.layers.Layer):
    def __init__(self,):
        super(GLU, self).__init__()

    def build(self, input_shape):
        self.n_channels = input_shape[-1] // 2

    def call(self, input):
        return input[..., :self.n_channels] * K.sigmoid(input[..., self.n_channels:])


class Generators:
    def __init__(self, branch_num=1, Ng=16, c_dim=10, z_dim=100):
        self.branch_num = branch_num
        self.Ng = Ng
        self.c_dim = c_dim
        self.z_dim = z_dim

    # def GLU(self, x):
    #     # Gated Linear Unit activation; see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.glu
    #     # Halfway split to form a and b
    #     n_channels = int(x.get_shape().as_list()[-1]/2)
    #     # a * sigmoid(b)
    #     return x[..., :n_channels] * K.sigmoid(x[..., n_channels:])

    def upsampling_block(self, x, n_filters):
        x = UpSampling2D(size=(2, 2), interpolation="nearest")(x)
        x = self.conv3x3_block(x, n_filters*2)
        return x

    def conv3x3_block(self, x, n_filters):
        x = Conv2D(n_filters, (3, 3), strides=1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = GLU()(x)
        return x

    def to_image_block(self, x):
        x = Conv2D(3, (3, 3), strides=1, padding="same")(x)
        x = Activation('tanh')(x)
        return x

    def joining_block(self, c_code, h_code, n_filters):
        s_size = h_code.get_shape().as_list()[2]

        c_code = Reshape((1, 1, self.c_dim))(c_code)
        c_code = Lambda(lambda x: K.tile(x, (1, s_size, s_size, 1,)))(c_code)

        joint = Concatenate()([c_code, h_code])

        joint = self.conv3x3_block(joint, n_filters*2)
        return joint

    def residual_block(self, x, n_filters):
        Fx = Conv2D(n_filters*2, kernel_size=(3, 3), strides=1, padding="same", use_bias=False)(x)
        Fx = BatchNormalization()(Fx)
        Fx = GLU()(x)

        Fx = Conv2D(n_filters, kernel_size=(3, 3), strides=1, padding="same", use_bias=False)(Fx)
        Fx = BatchNormalization()(Fx)

        x = Add()([Fx, x])
        return Fx + x

    def G_NET64(self, c, z):
        input_concat = Concatenate()([c, z])

        x = Dense(4*4*64*self.Ng * 2, use_bias=False)(input_concat)
        x = BatchNormalization()(x)
        x = GLU()(x)

        x = Reshape((4, 4, 64*self.Ng))(x)

        x = self.upsampling_block(x, 32 * self.Ng)
        x = self.upsampling_block(x, 16 * self.Ng)
        x = self.upsampling_block(x, 8 * self.Ng)
        out_hidden = self.upsampling_block(x, 4 * self.Ng)

        out_im_64 = self.to_image_block(out_hidden)
        return out_hidden, out_im_64

    def G_NET128(self, c, h_code):
        joint = self.joining_block(c, h_code, self.Ng * 2)

        x = self.residual_block(joint, self.Ng * 2)
        x = self.residual_block(x, self.Ng * 2)

        out_hidden = self.upsampling_block(x, self.Ng * 2)

        out_img_128 = self.to_image_block(out_hidden)
        return out_hidden, out_img_128

    def G_NET256(self, c, h_code):
        joint = self.joining_block(c, h_code, self.Ng)

        x = self.residual_block(joint, self.Ng)
        x = self.residual_block(x, self.Ng)

        out_hidden = self.upsampling_block(x, self.Ng)

        out_img_256 = self.to_image_block(out_hidden)
        return out_hidden, out_img_256

    def model(self,):
        input_c = Input(shape=(self.c_dim,))
        input_z = Input(shape=(self.z_dim,))

        outputs = []
        if self.branch_num > 0:
            hidden_0, out_im_64 = self.G_NET64(input_c, input_z)
            outputs.append(out_im_64)
        if self.branch_num > 1:
            hidden_1, out_im_128 = self.G_NET128(input_c, hidden_0)
            outputs.append(out_im_128)
        if self.branch_num > 2:
            hidden_2, out_im_256 = self.G_NET256(input_c, hidden_1)
            outputs.append(out_im_256)

        return Model(inputs=[input_c, input_z], outputs=outputs)


class Discriminators:
    def __init__(self, Nd=64, c_dim=50,):
        self.Nd = Nd
        self.c_dim = c_dim

    def conv3x3_block(self, x, n_filters):
        x = Conv2D(n_filters, (3, 3), strides=1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    def downsampling_block(self, x, n_filters, batch_norm=True):
        x = Conv2D(n_filters, (4, 4), strides=2, padding='same', use_bias=False)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    def encode_x16(self, x):
        x = self.downsampling_block(x, self.Nd, batch_norm=False)
        x = self.downsampling_block(x, self.Nd * 2)
        x = self.downsampling_block(x, self.Nd*4)
        x = self.downsampling_block(x, self.Nd * 8)
        return x

    def joining_block(self, c_code, h_code, n_filters):
        s_size = h_code.get_shape().as_list()[2]

        c_code = Reshape((1, 1, self.c_dim))(c_code)
        c_code = Lambda(lambda x: K.tile(x, (1, s_size, s_size, 1,)))(c_code)

        joint = Concatenate()([c_code, h_code])

        joint = self.conv3x3_block(joint, n_filters)
        return joint

    def logits(self, x):
        x = Conv2D(1, kernel_size=(4, 4), strides=4, padding="same")(x)
        x = Activation('sigmoid')(x)
        x = Reshape((1,))(x)
        return x

    def D_NET64(self,):
        input_c = Input(shape=(self.c_dim,))
        input_im = Input(shape=(64, 64, 3))

        x_code = self.encode_x16(input_im)

        joint = self.joining_block(input_c, x_code, self.Nd*8)

        out_cond = self.logits(joint)

        out_uncond = self.logits(x_code)

        return Model(inputs=[input_c, input_im], outputs=[out_cond, out_uncond])

    def D_NET128(self,):
        input_c = Input(shape=(self.c_dim,))
        input_im = Input(shape=(128, 128, 3))

        x_code = self.encode_x16(input_im)
        x_code = self.downsampling_block(x_code, self.Nd * 16)
        x_code = self.conv3x3_block(x_code, self.Nd * 8)

        joint = self.joining_block(input_c, x_code, self.Nd*8)

        out_cond = self.logits(joint)

        out_uncond = self.logits(x_code)

        return Model(inputs=[input_c, input_im], outputs=[out_cond, out_uncond])

    def D_NET256(self,):
        input_c = Input(shape=(self.c_dim,))
        input_im = Input(shape=(256, 256, 3))

        x_code = self.encode_x16(input_im)
        x_code = self.downsampling_block(x_code, self.Nd * 16)
        x_code = self.downsampling_block(x_code, self.Nd * 32)
        x_code = self.conv3x3_block(x_code, self.Nd * 16)

        x_code = self.conv3x3_block(x_code, self.Nd * 8)

        joint = self.joining_block(input_c, x_code, self.Nd*8)

        out_cond = self.logits(joint)

        out_uncond = self.logits(x_code)

        return Model(inputs=[input_c, input_im], outputs=[out_cond, out_uncond])


# generators = Generators()
# gen = generators.model()
# gen.summary()

# im_0, im_1, im_2 = gen([tf.random.normal(shape=(1000, 50)), tf.random.normal(shape=(1000, 100))])

# discriminators = Discriminators()
# D_64 = discriminators.D_NET64()
# D_64.summary()

# D_128 = discriminators.D_NET128()
# D_128.summary()

# D_256 = discriminators.D_NET256()
# D_256.summary()

# cond, uncond = D_64([tf.random.normal(shape=(1, 50)), im_0])
# print(cond, uncond)

# cond, uncond = D_128([tf.random.normal(shape=(1, 50)), im_1])
# print(cond, uncond)

# cond, uncond = D_256([tf.random.normal(shape=(1, 50)), im_2])
# print(cond, uncond)
# plt.imshow(im_0[0])
# plt.show()

# plt.imshow(im_1[0])
# plt.show()

# plt.imshow(im_2[0])
# plt.show()
