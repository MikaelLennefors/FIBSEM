from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import ELU, LeakyReLU
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as keras
import losses


def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), name=None):
    '''
    2D Convolutional layers

    Arguments:
        x {keras layer} -- input layer
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters

    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})
        name {str} -- name of the layer (default: {None})

    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    return x



def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
    '''
    2D Transposed Convolutional layers

    Arguments:
        x {keras layer} -- input layer
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters

    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(2, 2)})
        name {str} -- name of the layer (default: {None})

    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    return x


def MultiResBlock(U, inp, activation, alpha = 1.67):
    '''
    MultiRes Block

    Arguments:
        U {int} -- Number of filters in a corrsponding UNet stage
        inp {keras layer} -- input layer

    Returns:
        [keras layer] -- [output layer]
    '''

    W = alpha * U

    shortcut = inp

    shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
                         int(W*0.5), 1, 1, padding='same')

    conv3x3 = conv2d_bn(inp, int(W*0.167), 3, 3, padding='same')
    conv3x3 = activation(conv3x3)

    conv5x5 = conv2d_bn(conv3x3, int(W*0.333), 3, 3, padding='same')
    conv5x5 = activation(conv5x5)

    conv7x7 = conv2d_bn(conv5x5, int(W*0.5), 3, 3, padding='same')
    conv7x7 = activation(conv7x7)

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    out = BatchNormalization(axis=3)(out)

    out = add([shortcut, out])
    out = activation(out)
    out = BatchNormalization(axis=3)(out)

    return out


def ResPath(filters, length, inp, activation):
    '''
    ResPath

    Arguments:
        filters {int} -- [description]
        length {int} -- length of ResPath
        inp {keras layer} -- input layer

    Returns:
        [keras layer] -- [output layer]
    '''


    shortcut = inp
    shortcut = conv2d_bn(shortcut, filters, 1, 1, padding='same')

    out = conv2d_bn(inp, filters, 3, 3, padding='same')
    out = activation(out)
    out = add([shortcut, out])
    out = activation(out)
    out = BatchNormalization(axis=3)(out)

    for i in range(length-1):

        shortcut = out
        shortcut = conv2d_bn(shortcut, filters, 1, 1, padding='same')

        out = conv2d_bn(out, filters, 3, 3, padding='same')
        out = activation(out)
        out = add([shortcut, out])
        out = activation(out)
        out = BatchNormalization(axis=3)(out)

    return out


def MultiResUnet(input_size = (256,256,1), activation = 1, multiple = 32, dout = 0.5):
    '''
    MultiResUNet

    Arguments:
        height {int} -- height of image
        width {int} -- width of image
        n_channels {int} -- number of channels in image

    Returns:
        [keras model] -- MultiResUNet model
    '''

    keras.clear_session()

    if activation == 0:
       activation_fun = elu
    else:
       activation_fun = losses.RReLU()

    inputs = Input(input_size)
    conv1 = inputs
    if input_size[0] > 256:
        conv = Conv2D(multiple, (input_size[0] - 256 + 2) // 2, padding = 'valid', strides = 1, kernel_initializer = 'he_normal')(inputs)
        # prev_layer = BatchNormalization()(conv)
        prev_layer = activation_fun(conv)

        conv = Conv2D(multiple, (input_size[0] - 256 + 2) // 2, padding = 'valid', strides = 1, kernel_initializer = 'he_normal')(prev_layer)
        # prev_layer = BatchNormalization()(conv)
        conv1 = activation_fun(conv)
    mresblock1 = MultiResBlock(multiple, conv1, activation = activation_fun)
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = ResPath(multiple, 4, mresblock1, activation = activation_fun)

    mresblock2 = MultiResBlock(multiple*2, pool1, activation = activation_fun)
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = ResPath(multiple*2, 3, mresblock2, activation = activation_fun)

    mresblock3 = MultiResBlock(multiple*4, pool2, activation = activation_fun)
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = ResPath(multiple*4, 2, mresblock3, activation = activation_fun)

    mresblock4 = MultiResBlock(multiple*8, pool3, activation = activation_fun)
    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = ResPath(multiple*8, 1, mresblock4, activation = activation_fun)

    mresblock5 = MultiResBlock(multiple*16, pool4, activation = activation_fun)

    up6 = concatenate([Conv2DTranspose(
        multiple*8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)
    mresblock6 = MultiResBlock(multiple*8, up6, activation = activation_fun)

    up7 = concatenate([Conv2DTranspose(
        multiple*4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)
    mresblock7 = MultiResBlock(multiple*4, up7, activation = activation_fun)

    up8 = concatenate([Conv2DTranspose(
        multiple*2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)
    mresblock8 = MultiResBlock(multiple*2, up8, activation = activation_fun)

    up9 = concatenate([Conv2DTranspose(multiple, (2, 2), strides=(
        2, 2), padding='same')(mresblock8), mresblock1], axis=3)
    mresblock9 = MultiResBlock(multiple, up9, activation = activation_fun)

    conv10 = conv2d_bn(mresblock9, 1, 1, 1)
    conv10 = sigmoid(conv10)

    model = Model(inputs=inputs, outputs=conv10)

    return model



def main():

    # Define the model

    model = MultiResUnet(input_size = (258, 258,1))
    print(model.summary())



if __name__ == '__main__':
    main()
