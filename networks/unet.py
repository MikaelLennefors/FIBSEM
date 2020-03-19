from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.activations import relu
from tensorflow.keras import backend as keras
from tensorflow.keras.constraints import unit_norm, max_norm
import losses

def conv_block_down(prev_layer, n_filters, activation):
    for i in range(2):
        conv = Conv2D(n_filters, 3, kernel_constraint = max_norm(3), padding = 'same', kernel_initializer = 'he_normal')(prev_layer)

        prev_layer = activation(conv)

    return prev_layer

def conv_block_up(prev_layer, concat_layer, n_filters, activation):

    up_test = Conv2DTranspose(n_filters, 2, strides=(2, 2), padding='same')(prev_layer)
    merge = concatenate([concat_layer,up_test], axis = 3)
    output = conv_block_down(merge, n_filters, activation)

    return output

def unet(pretrained_weights = None, input_size = 256, activation = 1, multiple = 32, learning_rate = 1e-4, dout = 0.5):
    keras.clear_session()

    if activation == 0:
       activation_fun = relu
    else:
       activation_fun = losses.RReLU()

    inputs = Input((input_size, input_size, 1))
    conv1 = conv_block_down(inputs, multiple, activation_fun)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = conv_block_down(pool1, 2*multiple, activation_fun)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = conv_block_down(pool2, 4*multiple, activation_fun)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = conv_block_down(pool3, 8*multiple, activation_fun)
    drop4 = Dropout(dout)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv_block_down(pool4, 16*multiple, activation_fun)
    drop5 = Dropout(dout)(conv5)

    up6 = conv_block_up(drop5, conv4, 8*multiple, activation_fun)
    up7 = conv_block_up(up6, conv3, 4*multiple, activation_fun)
    up8 = conv_block_up(up7, conv2, 2*multiple, activation_fun)
    up9 = conv_block_up(up8, conv1, multiple, activation_fun)
    
    conv9 = conv_block_down(up9, multiple, activation_fun)

    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(lr = learning_rate), loss = losses.iou_loss, metrics = [losses.iou_coef, 'accuracy', losses.TP, losses.TN, losses.FP, losses.FN])

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def main():
    
    model = unet()
    print(model.summary())

if __name__ == '__main__':
    main()