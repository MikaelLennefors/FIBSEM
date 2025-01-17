import math
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.activations import relu, elu
from tensorflow.keras import backend as keras
from tensorflow.keras.constraints import unit_norm, max_norm
import losses

# U-Net
# ======================
# link to article: https://arxiv.org/pdf/1505.04597.pdf
# Slightly modified to accomodate for 3D-2D data

def squeeze(x):
    x = keras.squeeze(x, axis=-1)
    return x

def conv_block_down(prev_layer, n_filters, activation):
    for i in range(2):
        conv = Conv2D(n_filters, 3, padding = 'same', kernel_initializer = 'he_normal')(prev_layer)
        prev_layer = activation(conv)

    return prev_layer

def conv_block_up(prev_layer, concat_layer, n_filters, activation):
    up_test = Conv2D(n_filters, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D()(prev_layer))
    up_test = activation(up_test)
    # up_test = Conv2DTranspose(n_filters, 2, strides=(2, 2), padding='same')(prev_layer)
    # up_test = activation(up_test)
    merge = concatenate([concat_layer,up_test], axis = 3)
    output = conv_block_down(merge, n_filters, activation)

    return output

def unet(pretrained_weights = None, input_size = (258, 258, 1, 1), activation = 0, multiple = 32, dout = 0.5):
    keras.clear_session()

    if activation == 0:
       activation_fun = elu
    else:
       activation_fun = losses.RReLU()

    inputs = Input(shape=input_size)
    conv1 = Lambda(squeeze)(inputs)

    if input_size[0] > 256:
        conv = Conv2D(multiple, (input_size[0] - 256 + 2) // 2, padding = 'valid', strides = 1, kernel_initializer = 'he_normal')(conv1)
        prev_layer = activation_fun(conv)

        conv = Conv2D(multiple, (input_size[0] - 256 + 2) // 2, padding = 'valid', strides = 1, kernel_initializer = 'he_normal')(prev_layer)
        conv1 = activation_fun(conv)
    elif input_size[0] == 256:
        conv = Conv2D(multiple, 3, padding = 'same', strides = 1, kernel_initializer = 'he_normal')(conv1)
        prev_layer = activation_fun(conv)

        conv = Conv2D(multiple, 3, padding = 'same', strides = 1, kernel_initializer = 'he_normal')(prev_layer)
        conv1 = activation_fun(conv)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block_down(pool1, 2*multiple, activation_fun)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block_down(pool2, 4*multiple, activation_fun)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block_down(pool3, 8*multiple, activation_fun)
    drop4 = Dropout(dout)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = conv_block_down(pool4, 16*multiple, activation_fun)
    drop5 = Dropout(dout)(conv5)

    up6 = conv_block_up(drop5, drop4, 8*multiple, activation_fun)
    up7 = conv_block_up(up6, conv3, 4*multiple, activation_fun)
    up8 = conv_block_up(up7, conv2, 2*multiple, activation_fun)
    up9 = conv_block_up(up8, conv1, multiple, activation_fun)
    conv9 = Conv2D(2, 3, padding = 'same', kernel_initializer = 'he_normal')(up9)
    act9 = activation_fun(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(act9)

    model = Model(inputs = inputs, outputs = conv10)

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def main():

    model = unet(input_size = (258,258,1,1))
    print(model.summary())

if __name__ == '__main__':
    main()
