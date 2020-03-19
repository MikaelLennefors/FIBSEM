from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add, Dropout
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import sigmoid, relu
from tensorflow.keras import backend as keras
import losses


def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation=None, name=None, dout = 0.5):
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
    #print("before")
    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    #x = Dropout(dout)(x)
    # x = BatchNormalization(axis=3, scale=False, renorm = True)(x)
    #print("after")
    if(activation == None):
        return x
    elif(activation == 'sigmoid'):
        return sigmoid(x)
    elif activation == 0:
        #x = Activation('relu', name=name)(x)
        return relu(x)
    else:
        return losses.RReLU()(x)
    #x = Activation(activation, name=name)(x)




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
    #x = BatchNormalization(axis=3, scale=False)(x)
   
    return x


def MultiResBlock(U, inp, alpha = 1.67, act = 0,dout = 0.5):

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
                         int(W*0.5), 1, 1, activation=None, padding='same')

    conv3x3 = conv2d_bn(inp, int(W*0.167), 3, 3, activation = act, padding='same')
    #print(conv3x3)
    conv5x5 = conv2d_bn(conv3x3, int(W*0.333), 3, 3, activation = act, padding='same')

    conv7x7 = conv2d_bn(conv5x5, int(W*0.5), 3, 3, activation = act, padding='same')

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    #out = BatchNormalization(axis=3)(out)

    out = add([shortcut, out])
    if act == 0:
        out = relu(out)
    else:
        out = losses.RReLU()(out)
    #out = Dropout(dout)(out)
    # out = BatchNormalization(axis=3, renorm = True)(out)

    return out


def ResPath(filters, length, inp, act, dout = 0.5):
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
    shortcut = conv2d_bn(shortcut, filters, 1, 1,
                         activation=None, padding='same')

    out = conv2d_bn(inp, filters, 3, 3, activation=act, padding='same')

    out = add([shortcut, out])
    if act == 0:
        out = relu(out)
    else:
        out = losses.RReLU()(out)
    # out = BatchNormalization(axis=3, renorm = True)(out)
    #out = Dropout(dout)(out)

    for i in range(length-1):

        shortcut = out
        shortcut = conv2d_bn(shortcut, filters, 1, 1,
                             activation=None, padding='same')

        out = conv2d_bn(out, filters, 3, 3, activation=act, padding='same')

        out = add([shortcut, out])
        if act == 0:
            out = relu(out)
        else:
            out = losses.RReLU()(out)
        # out = BatchNormalization(axis=3, renorm = True)(out)

    return out


def MultiResUnet(input_size = 256, activation = 0, multiple = 32, learning_rate = 1e-4, bin_weight = 0.3, dout = 0.5):
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
    inputs = Input((input_size, input_size, 1))

    mresblock1 = MultiResBlock(multiple, inputs, act = activation)

    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = ResPath(multiple, 4, mresblock1, act = activation)

    mresblock2 = MultiResBlock(multiple*2, pool1, act = activation)
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = ResPath(multiple*2, 3, mresblock2, act = activation)

    mresblock3 = MultiResBlock(multiple*4, pool2, act = activation)
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = ResPath(multiple*4, 2, mresblock3, act = activation)

    mresblock4 = MultiResBlock(multiple*8, pool3, act = activation)
    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = ResPath(multiple*8, 1, mresblock4, act = activation)
    drop4 = Dropout(dout)(mresblock4)

    mresblock5 = MultiResBlock(multiple*16, pool4, act = activation)
    drop5 = Dropout(dout)(mresblock5)

    up6 = concatenate([Conv2DTranspose(
        multiple*8, (2, 2), strides=(2, 2), padding='same')(drop5), drop4], axis=3)
    mresblock6 = MultiResBlock(multiple*8, up6, act = activation)

    up7 = concatenate([Conv2DTranspose(
        multiple*4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)
    mresblock7 = MultiResBlock(multiple*4, up7, act = activation)

    up8 = concatenate([Conv2DTranspose(
        multiple*2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)
    mresblock8 = MultiResBlock(multiple*2, up8, act = activation)

    up9 = concatenate([Conv2DTranspose(multiple, (2, 2), strides=(
        2, 2), padding='same')(mresblock8), mresblock1], axis=3)
    mresblock9 = MultiResBlock(multiple, up9, act = activation)

    conv10 = conv2d_bn(mresblock9, 1, 1, 1, activation='sigmoid')
   
    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer = Adam(lr = learning_rate), loss = losses.iou_loss, metrics = [losses.iou_coef, 'accuracy', losses.TP, losses.TN, losses.FP, losses.FN])
    return model
   


def main():

    # Define the model

    model = MultiResUnet(128, 128,3)
    print(model.summary())



if __name__ == '__main__':
    main()