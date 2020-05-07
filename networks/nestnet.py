import tensorflow.keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as keras
from tensorflow.keras.layers import Input, Conv2D, ZeroPadding2D, UpSampling2D, Dense, concatenate, Conv2DTranspose,  MaxPooling2D, GlobalAveragePooling2D, MaxPooling2D, Dense, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import GaussianDropout
from tensorflow.keras.activations import relu, elu
import numpy as np
import losses

########################################
# 2D Standard
########################################

def squeeze(x):
    x = keras.squeeze(x, axis=-1)
    return x

def standard_unit(input_tensor, stage, nb_filter, dropout = 0.5, kernel_size=3, activation = 0):

    x = Conv2D(nb_filter, (kernel_size, kernel_size), name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    x = activation(x)
    x = Dropout(dropout, name='dp'+stage+'_1')(x)
    x = Conv2D(nb_filter, (kernel_size, kernel_size), name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = activation(x)
    x = Dropout(dropout, name='dp'+stage+'_2')(x)

    return x

"""
Standard UNet++ [Zhou et.al, 2018]
Total params: 9,041,601
"""
def Nest_Net(input_size = (258,258,3,1), activation = 0, multiple = 32, dout = 0.5):

    nb_filter = [multiple,2*multiple,4*multiple,8*multiple,16*multiple]

    keras.clear_session()

    if activation == 0:
       activation_fun = elu
    else:
       activation_fun = losses.RReLU()

    # Handle Dimension Ordering for different backends
    global bn_axis
    bn_axis = 3
    img_input = Input(shape=input_size)
    conv1 = Lambda(squeeze)(img_input)
    if input_size[0] > 256:
        conv = Conv2D(nb_filter[0], (input_size[0] - 256 + 2) // 2, padding = 'valid', strides = 1, kernel_initializer = 'he_normal')(conv1)
        # prev_layer = BatchNormalization()(conv)
        prev_layer = activation_fun(conv)

        conv = Conv2D(nb_filter[0], (input_size[0] - 256 + 2) // 2, padding = 'valid', strides = 1, kernel_initializer = 'he_normal')(prev_layer)
        # prev_layer = BatchNormalization()(conv)
        conv1 = activation_fun(conv)
    conv1_1 = standard_unit(conv1, stage='11', nb_filter=nb_filter[0], activation = activation_fun)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1], activation = activation_fun)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0], activation = activation_fun)

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2], activation = activation_fun)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1], activation = activation_fun)

    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
    conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0], activation = activation_fun)

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3], activation = activation_fun)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2], activation = activation_fun)

    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
    conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1], activation = activation_fun)

    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
    conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0], activation = activation_fun)

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4], activation = activation_fun)

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3], activation = activation_fun)

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2], activation = activation_fun)

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1], activation = activation_fun)

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0], activation = activation_fun)

    nestnet_output_1 = Conv2D(1, (1, 1), activation='sigmoid', name='output_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
    nestnet_output_2 = Conv2D(1, (1, 1), activation='sigmoid', name='output_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
    nestnet_output_3 = Conv2D(1, (1, 1), activation='sigmoid', name='output_3', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
    nestnet_output_4 = Conv2D(1, (1, 1), activation='sigmoid', name='output_4', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    model = Model(inputs=img_input, outputs=[nestnet_output_4])

    return model


if __name__ == '__main__':

    # model = U_Net(96,96,1)
    # model.summary()
    #
    # model = wU_Net(96,96,1)
    # model.summary()

    model = Nest_Net(input_size = (258,258,3,1))
    model.summary()
