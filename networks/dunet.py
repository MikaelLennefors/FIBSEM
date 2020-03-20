from __future__ import print_function
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import multi_gpu_model
import math
import h5py
from tensorflow.keras import backend as K
import tensorflow as tf
import math
from matplotlib import pyplot as plt
import time
import random
import copy
import os
import losses
def expand(x):
    x = K.expand_dims(x, axis=-1)
    return x


def squeeze(x):
    x = K.squeeze(x, axis=-1)
    return x


def BN_block(filter_num, input, activation):
    x = Conv2D(filter_num, 3, padding='same', kernel_initializer='he_normal')(input)
    x = BatchNormalization()(x)
    x1 = activation(x)
    x = Conv2D(filter_num, 3, padding='same', kernel_initializer='he_normal')(x1)
    x = BatchNormalization()(x)
    x = activation(x)
    return x


def BN_block3d(filter_num, input, activation):
    x = Conv3D(filter_num, 3, padding='same', kernel_initializer='he_normal')(input)
    x = BatchNormalization()(x)
    x1 = activation(x)
    x = Conv3D(filter_num, 3, padding='same', kernel_initializer='he_normal')(x1)
    x = BatchNormalization()(x)
    x = activation(x)
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
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, kernel_initializer='he_normal', use_bias=False)(se)
    se = activation(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def D_Unet(pretrained_weights = None, input_size = (256, 7), activation = 1, multiple = 64, learning_rate = 1e-4, dout = 0.5, reg_coeff = 0.001):
    K.clear_session()

    if activation == 0:
       activation_fun = relu
    else:
       activation_fun = losses.RReLU()

    inputs = Input(shape=(input_size[0],input_size[0],input_size[1],1))
    input2d = Lambda(squeeze)(inputs)
    conv3d1 = BN_block3d(multiple, inputs, activation_fun)

    pool3d1 = AveragePooling3D(pool_size=2)(conv3d1)

    conv3d2 = BN_block3d(2*multiple, pool3d1, activation_fun)
    if input_size[1] != 3:
        pool3d2 = MaxPooling3D(pool_size=2)(conv3d2)

        conv3d3 = BN_block3d(4*multiple, pool3d2, activation_fun)


    conv1 = BN_block(multiple, input2d, activation_fun)

    #conv1 = D_Add(32, conv3d1, conv1, activation_fun)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = BN_block(2*multiple, pool1, activation_fun)
    if input_size[1] > 3:
        conv2 = D_SE_Add(2*multiple, conv3d2, conv2, activation_fun)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = BN_block(4*multiple, pool2, activation_fun)
    if input_size[1] > 3:
        conv3 = D_SE_Add(4*multiple, conv3d3, conv3, activation_fun)
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
    model.compile(optimizer = Adam(lr = learning_rate), loss = losses.iou_loss, metrics = [losses.iou_coef, 'accuracy', losses.TP, losses.TN, losses.FP, losses.FN])
    return model


def Unet():
    inputs = Input(shape=(192, 192, 1))
    conv1 = BN_block(32, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = BN_block(64, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = BN_block(128, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = BN_block(256, pool3)
    drop4 = Dropout(0.3)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = BN_block(512, pool4)
    drop5 = Dropout(0.3)(conv5)

    up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = Concatenate()([drop4, up6])
    conv6 = BN_block(256, merge6)

    up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = Concatenate()([conv3, up7])
    conv7 = BN_block(128, merge7)

    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = Concatenate()([conv2, up8])
    conv8 = BN_block(64, merge8)

    up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = Concatenate()([conv1, up9])
    conv9 = BN_block(32, merge9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)  # conv10作为输出

    model = Model(input=inputs, output=conv10)

    return model


def origin_block(filter_num, input):
    x = Conv2D(filter_num, 3, padding='same', kernel_initializer='he_normal')(input)
    x1 = Activation('relu')(x)
    x = Conv2D(filter_num, 3, padding='same', kernel_initializer='he_normal')(x1)
    x = Activation('relu')(x)
    return x


def Unet_origin():
    inputs = Input(shape=(192, 192, 1))
    conv1 = origin_block(64, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = origin_block(128, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = origin_block(256, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = origin_block(512, pool3)
    drop4 = Dropout(0.3)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = origin_block(1024, pool4)
    drop5 = Dropout(0.3)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = Concatenate()([drop4, up6])
    conv6 = origin_block(512, merge6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = Concatenate()([conv3, up7])
    conv7 = origin_block(256, merge7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = Concatenate()([conv2, up8])
    conv8 = origin_block(128, merge8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = Concatenate()([conv1, up9])
    conv9 = origin_block(64, merge9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)  # conv10作为输出

    model = Model(input=inputs, output=conv10)

    return model

def Unet3d():
    inputs = Input(shape=(192, 192, 4))
    input3d = Lambda(expand)(inputs)
    conv1 = BN_block3d(32, input3d)
    pool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1)

    conv2 = BN_block3d(64, pool1)
    pool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2)

    conv3 = BN_block3d(128, pool2)
    pool3 = MaxPooling3D(pool_size=(2, 2, 1))(conv3)

    conv4 = BN_block3d(256, pool3)
    drop4 = Dropout(0.3)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 1))(drop4)

    conv5 = BN_block3d(512, pool4)
    drop5 = Dropout(0.3)(conv5)

    up6 = Conv3D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='6')(
        UpSampling3D(size=(2, 2, 1))(drop5))
    merge6 = Concatenate()([drop4, up6])
    conv6 = BN_block3d(256, merge6)

    up7 = Conv3D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='8')(
        UpSampling3D(size=(2, 2, 1))(conv6))
    merge7 = Concatenate()([conv3, up7])
    conv7 = BN_block3d(128, merge7)

    up8 = Conv3D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='10')(
        UpSampling3D(size=(2, 2, 1))(conv7))
    merge8 = Concatenate()([conv2, up8])
    conv8 = BN_block3d(64, merge8)

    up9 = Conv3D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='12')(
        UpSampling3D(size=(2, 2, 1))(conv8))
    merge9 = Concatenate()([conv1, up9])
    conv9 = BN_block3d(32, merge9)
    conv10 = Conv3D(1, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Lambda(squeeze)(conv10)
    # '''
    # conv11 = Lambda(squeeze)(conv10)
    conv11 = BN_block(32, conv10)
    conv11 = Conv2D(1, 1, activation='sigmoid')(conv11)
    # '''
    model = Model(input=inputs, output=conv11)

    return model

def conv_bn_block(x, filter):
    x = Conv3D(filter, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x1 = Activation('relu')(x)
    x = Conv3D(filter, 3, padding='same', kernel_initializer='he_normal')(x1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Concatenate()([x, x1])
    return x

def main():

    model = D_Unet(input_size = (256,3))
    print(model.summary())

if __name__ == '__main__':
    main()
