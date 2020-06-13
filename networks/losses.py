from tensorflow.keras import backend as keras
from tensorflow.keras.layers import *
import numpy as np

def weighted_binary_crossentropy(y_true, y_pred, zero_weight = 0.31):
    '''
    Weighted Binary Cross-Entropy. Calculates the Weighted Binary Cross-Entropy
    loss given a true label, a prediction and a weight for one of the classes.
    Since it's a binary problem, the weight for the other class will be the
    complement to the first.

    Input:
        y_true: true label matrix
        y_pred: prediction matrix
        zero_weight: weight for pores, default 0.31 since approx porosity
    Output:
        weighted_b_ce: weighted binary cross-entropy score tensor
    '''
    one_weight = 1 - zero_weight
    b_ce = keras.binary_crossentropy(y_true, y_pred)

    weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
    weighted_b_ce = weight_vector * b_ce

    return weighted_b_ce

def iou_loss(y_true, y_pred, smooth=1.):
    '''
    Intersection over union loss (IoU). Calculates the Intersection over union
    loss given a true label, a prediction and a smooth-parameter.
    Since it's a binary problem, the weight for the other class will be the
    complement to the first.

    Input:
        y_true: true label matrix
        y_pred: prediction matrix
        smooth: in need to produce smooth gradients
    Output:
        jac: IoU score tensor
    '''
    y_true_c = keras.flatten(y_true)
    y_pred_c = keras.flatten(y_pred)
    intersection = keras.sum(y_true_c * y_pred_c)
    sum_ = keras.sum(y_true) + keras.sum(y_pred)
    jac = (intersection+smooth) / (sum_ - intersection+smooth)

    # intersection_c = keras.sum(keras.abs(y_true_c * y_pred_c), axis=-1)
    # sum_c = keras.sum(keras.abs(y_true_c) + keras.abs(y_pred_c), axis=-1)
    # jac_c = (intersection + smooth) / (sum_c - intersection_c + smooth)

    return jac

def bce_iou_loss(zero_weight):
    '''
    Weighted Binary Cross-Entropy mixed with IoU loss. Calculates the a mixed
    Weighted Binary Cross-Entropy and IoU loss given a weight for one class.
    Since it's a binary problem, the weight for the other class will be the
    complement to the first.

    Input:
        y_true: true label matrix
        y_pred: prediction matrix
    Output:
        bce_dice_loss: mixed score
    '''
    def bce_dice_loss(y_true, y_pred):
        return 0.5 * weighted_binary_crossentropy(y_true, y_pred, zero_weight) - iou_loss(y_true, y_pred)
    return bce_dice_loss

def iou_coef(y_true, y_pred):
    '''
    Intersection over union score (IoU). Calculates the a IoU loss given given a
    true label and a prediction.

    Input:
        zero_weight: weight for pores.
    Output:
        IoU score tensor
    '''
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
