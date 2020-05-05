from tensorflow.keras import backend as keras
from tensorflow.keras.layers import *
import numpy as np

def weighted_binary_crossentropy(y_true, y_pred):
    zero_weight = 0.31
    one_weight = 1 - zero_weight
    b_ce = keras.binary_crossentropy(y_true, y_pred)

    weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
    weighted_b_ce = weight_vector * b_ce

    return weighted_b_ce

def iou_loss(y_true, y_pred, smooth=1.):
    y_true_c = keras.flatten(y_true)
    y_pred_c = keras.flatten(y_pred)
    intersection = keras.sum(y_true_c * y_pred_c)
    sum_ = keras.sum(y_true) + keras.sum(y_pred)
    jac = (intersection+smooth) / (sum_ - intersection+smooth)

    # intersection_c = keras.sum(keras.abs(y_true_c * y_pred_c), axis=-1)
    # sum_c = keras.sum(keras.abs(y_true_c) + keras.abs(y_pred_c), axis=-1)
    # jac_c = (intersection + smooth) / (sum_c - intersection_c + smooth)

    return jac
# Custom loss function
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * weighted_binary_crossentropy(y_true, y_pred) - iou_loss(y_true, y_pred)

def iou_coef(y_true, y_pred):
    # y_true_f = 1 - keras.flatten(y_true)
    # y_pred_f = 1 - keras.flatten(y_pred)
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    union = keras.sum (y_true_f + y_pred_f - y_true_f * y_pred_f)

    return intersection / union

class RReLU(Layer):
    '''Randomized Leaky Rectified Linear Unit
    that uses a random alpha in training while using the average of alphas
    in testing phase:
    During training
    f(x) = alpha * x for x < 0, where alpha ~ U(l, u), l < u,
    f(x) = x for x >= 0.
    During testing:
    f(x) = (l + u) / 2 * x for x < 0,
    f(x) = x for x >= 0.
    # Input shape
        Arbitrary. Use the keyword argument input_shape
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        l: lower bound of the uniform distribution, default is 1/8
        u: upper bound of the uniform distribution, default is 1/3
    # References
        - [Empirical Evaluation of Rectified Activations in Convolution Network](https://arxiv.org/pdf/1505.00853v2.pdf)
    '''
    def __init__(self, l=1/8., u=1/3., **kwargs):
        self.supports_masking = True
        self.l = l
        self.u = u
        self.average = (l + u) / 2
        self.uses_learning_phase = True
        super(RReLU, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return keras.in_train_phase(keras.relu(x, np.random.uniform(self.l, self.u)),
                                keras.relu(x, self.average))

    def get_config(self):
        config = {'l': self.l, 'u': self.u}
        base_config = super(RReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def TP(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    y_pred_f01 = keras.round(keras.clip(y_pred_f, 0, 1))
    true_positives = keras.sum(y_pred_f01*y_true_f)
    return true_positives/65536

def FP(y_true, y_pred):
    y_true = 1 - y_true
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    y_pred_f01 = keras.round(keras.clip(y_pred_f, 0, 1))
    false_positives = keras.sum(y_pred_f01*y_true_f)
    return false_positives/65536

def TN(y_true, y_pred):
    y_true = 1 - y_true
    y_pred = 1 - y_pred
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    y_pred_f01 = keras.round(keras.clip(y_pred_f, 0, 1))
    true_negatives = keras.sum(y_pred_f01*y_true_f)
    return true_negatives/65536


def FN(y_true, y_pred):
    y_pred = 1 - y_pred
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    tp_f01 = keras.round(keras.clip(y_pred_f, 0, 1))
    false_negatives = keras.sum(y_true_f*tp_f01)
    return false_negatives/65536
