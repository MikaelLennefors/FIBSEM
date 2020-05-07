import numpy as np
import tensorflow as tf
from tensorflow import keras


class EarlyStoppingBaseline(tf.keras.callbacks.Callback):
  """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

  def __init__(self, patience):
    super(EarlyStoppingBaseline, self).__init__()

    self.patience = patience

    # best_weights to store the weights at which the minimum loss occurs.
    self.best_weights = None

  def on_train_begin(self, logs=None):
    # The number of epoch it has waited when loss is no longer minimum.
    self.wait = 0
    # The epoch the training stops at.
    self.stopped_epoch = 0
    # Initialize the best as infinity.
    # self.best = np.Inf
    self.best = 0.72

  def on_epoch_end(self, epoch, logs=None):
    current = logs.get('val_accuracy')
    if np.greater(current, self.best):
      # self.best = 0.5
      # self.patience = 1
      self.wait = 0
      # Record the best weights if current results is better (greater).
      # self.best_weights = self.model.get_weights()
      # self.best = 0.6
    else:
      self.wait += 1
      if self.wait >= self.patience:
        self.stopped_epoch = epoch
        self.model.stop_training = True
        # print('Restoring model weights from the end of the best epoch.')
        # self.model.set_weights(self.best_weights)

  def on_train_end(self, logs=None):
    if self.stopped_epoch > 0:
      print('Epoch %05d: early stopping baseline'  % (self.stopped_epoch + 1))

class EarlyStoppingDelta(tf.keras.callbacks.Callback):
  """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

  def __init__(self, patience):
    super(EarlyStoppingDelta, self).__init__()

    self.patience = patience

  def on_train_begin(self, logs=None):
    # The number of epoch it has waited when loss is no longer minimum.
    self.wait = 0
    # The epoch the training stops at.
    self.stopped_epoch = 0
    # Initialize the best as 0.
    self.best = 0

  def on_epoch_end(self, epoch, logs=None):
    current = logs.get('val_iou_coef')
    if current - self.best > 0.001:
      self.best = current
      self.wait = 0
    else:
      self.wait += 1
      if self.wait >= self.patience:
        self.stopped_epoch = epoch
        self.model.stop_training = True

  def on_train_end(self, logs=None):
    if self.stopped_epoch > 0:
      print('Epoch %05d: early stopping min delta' % (self.stopped_epoch + 1))
