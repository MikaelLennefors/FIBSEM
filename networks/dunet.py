from __future__ import print_function
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.activations import relu, elu
import math
import h5py
from tensorflow.keras import backend as keras
import tensorflow as tf
import math
from matplotlib import pyplot as plt
import time
import random
import copy
import os
import losses

# DIMENSION-FUSION U-NET
# ======================
# link to github of authours: https://github.com/SZUHvern/D-UNet/blob/master/model.py
# link to article: https://arxiv.org/pdf/1908.05104.pdf

def expand(x):
    x = keras.expand_dims(x, axis=-1)
    return x

def squeeze(x):
    x = keras.squeeze(x, axis=-1)
    return x

def squeeze_2(x):
    x = keras.squeeze(x, axis=-2)
    return x

def BN_block(filter_num, input, activation):
    '''
    Batch normilization block. The block consists of a conv-layers, activations
    function and batch normilization in serial, two times. NOTE! batch normilization
    was not used in the master thesis, and therefore is omitted. It's a reminant
    from the original authurs.

    Input:
        filter_num (int): number of filters in use.
        input: input data.
        activation (fun): activation funtion in use.

    '''
    x = Conv2D(filter_num, 3, padding='same', kernel_initializer='he_normal')(input)
    x = activation(x)
    #x = BatchNormalization()(x)


    x = Conv2D(filter_num, 3, padding='same', kernel_initializer='he_normal')(x)
    x = activation(x)
    #x = BatchNormalization()(x)

    return x


def BN_block3d(filter_num, input, activation):
    '''
    Batch normilization block in 3D. The block consists of a conv-layers, activations
    function and batch normilization in serial, two times. NOTE! batch normilization
    was not used in the master thesis, and therefore is omitted. It's a reminant
    from the original authurs.

    Input:
        filter_num (int): number of filters in use.
        input: input data.
        activation (fun): activation funtion in use.

    '''
    x = Conv3D(filter_num, 3, padding='same', kernel_initializer='he_normal')(input)
    x = activation(x)
    #x = BatchNormalization()(x)

    x = Conv3D(filter_num, 3, padding='same', kernel_initializer='he_normal')(x)
    x = activation(x)
    #x = BatchNormalization()(x)

    return x


def D_Add(filter_num, input3d, input2d, activation):
    x = Conv3D(1, 1, padding='same', kernel_initializer='he_normal')(input3d)
    x = Lambda(squeeze)(x)
    x = Conv2D(filter_num, 3, padding='same', kernel_initializer='he_normal')(x)
    x = activation(x)
    x = Add()([x, input2d])
    return x


def D_concat(filter_num, input3d, input2d, activation):
    x = Conv3D(1, 1, padding='same', kernel_initializer='he_normal')(input3d)
    x = Lambda(squeeze)(x)
    x = Conv2D(filter_num, 3, padding='same', kernel_initializer='he_normal')(x)
    x = activation(x)
    x = Concatenate()([x, input2d])
    x = Conv2D(filter_num, 1, padding='same', kernel_initializer='he_normal')(x)
    x = activation(x)
    return x


def D_SE_concat(filter_num, input3d, input2d, activation):
    x = Conv3D(1, 1, padding='same', kernel_initializer='he_normal')(input3d)
    x = Lambda(squeeze)(x)
    x = Conv2D(filter_num, 3, padding='same', kernel_initializer='he_normal')(x)
    x = activation(x)
    x = squeeze_excite_block(x, activation)
    input2d = squeeze_excite_block(input2d, activation)
    x = Concatenate()([x, input2d])
    x = Conv2D(filter_num, 1, padding='same', kernel_initializer='he_normal')(x)
    x = activation(x)
    return x


def D_Add_SE(filter_num, input3d, input2d, activation):
    x = Conv3D(1, 1, padding='same', kernel_initializer='he_normal')(input3d)
    x = Lambda(squeeze)(x)
    x = Conv2D(filter_num, 3, padding='same', kernel_initializer='he_normal')(x)
    x = activation(x)
    x = Add()([x, input2d])
    x = squeeze_excite_block(x, activation)
    return x


def D_SE_Add(filter_num, input3d, input2d, activation):
    x = Conv3D(1, 1, padding='same', kernel_initializer='he_normal')(input3d)
    x = Lambda(squeeze)(x)
    x = Conv2D(filter_num, 3, padding='same', kernel_initializer='he_normal')(x)
    x = activation(x)
    x = squeeze_excite_block(x, activation)
    input2d = squeeze_excite_block(input2d, activation)
    x = Add()([x, input2d])

    return x


def D_concat_SE(filter_num, input3d, input2d, activation):
    x = Conv3D(1, 1, padding='same', kernel_initializer='he_normal')(input3d)
    x = Lambda(squeeze)(x)
    x = Conv2D(filter_num, 3, padding='same', kernel_initializer='he_normal')(x)
    x = activation(x)
    x = Concatenate()([x, input2d])
    x = squeeze_excite_block(x, activation)
    x = Conv2D(filter_num, 1, padding='same', kernel_initializer='he_normal')(x)
    x = activation(x)
    return x


def squeeze_excite_block(input, activation, ratio=16):
    '''
    Create a squeeze-excite block.

    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if keras.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, kernel_initializer='he_normal', use_bias=False)(se)
    se = activation(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if keras.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

def main():

    model = D_Unet(input_size = (258,258,3,1))
    print(model.summary())

def D_Unet(pretrained_weights = None, input_size = (258,258,3,1), activation = 0, multiple = 32, dout = 0.5):
    '''
    Dimension fusion U-Net structure.

    Input:
        pretrained_weights: The pre-trained weights if any.
        input_size (tuple of ints): size of input data (height, widht, channels, 1).
        activation (fun): activation funtion, 0 = elu, 1 = RReLU.
        multiple (int): number of filters in the first level of the network.
        dout (float): Dropout level.

    Output:
        model: returns the model

    '''

    keras.clear_session()

    if activation == 0:
       activation_fun = elu
    else:
       activation_fun = losses.RReLU()

    inputs = Input(shape=input_size)
    input2d = Lambda(squeeze)(inputs)
    conv3d1 = Conv3D(multiple, (3,3,1), padding='valid', kernel_initializer='he_normal')(inputs)
    conv3d1 = activation_fun(conv3d1)

    conv3d1 = Conv3D(multiple, 3, padding='same', kernel_initializer='he_normal')(conv3d1)
    conv3d1 = activation_fun(conv3d1)

    pool3d1 = MaxPooling3D(pool_size=(2,2,3))(conv3d1)

    conv3d2 = BN_block3d(2*multiple, pool3d1, activation_fun)

    conv1 = Conv2D(multiple, 3, padding='valid', kernel_initializer='he_normal')(input2d)
    conv1 = activation_fun(conv1)


    conv1 = Conv2D(multiple, 3, padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = activation_fun(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = BN_block(2*multiple, pool1, activation_fun)
    conv2 = D_SE_Add(2*multiple, conv3d2, conv2, activation_fun)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = BN_block(4*multiple, pool2, activation_fun)
    # conv3 = D_SE_Add(4*multiple, conv3d3, conv3, activation_fun)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = BN_block(8*multiple, pool3, activation_fun)
    conv4 = Dropout(dout)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = BN_block(16*multiple, pool4, activation_fun)
    conv5 = Dropout(dout)(conv5)

    up6 = Conv2D(8*multiple, 2, padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    up6 = activation_fun(up6)
    merge6 = Concatenate()([conv4, up6])
    conv6 = BN_block(8*multiple, merge6, activation_fun)

    up7 = Conv2D(4*multiple, 2, padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    up7 = activation_fun(up7)
    merge7 = Concatenate()([conv3, up7])
    conv7 = BN_block(4*multiple, merge7, activation_fun)

    up8 = Conv2D(2*multiple, 2, padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    up8 = activation_fun(up8)
    merge8 = Concatenate()([conv2, up8])
    conv8 = BN_block(2*multiple, merge8, activation_fun)

    up9 = Conv2D(multiple, 2, padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    up9 = activation_fun(up9)
    merge9 = Concatenate()([conv1, up9])
    conv9 = BN_block(multiple, merge9, activation_fun)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    model = Model(inputs=inputs, outputs=conv10)
    return model

if __name__ == '__main__':
    main()
