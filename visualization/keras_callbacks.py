import cv2
import numpy as np
import tensorflow as tf
import datetime

class PredictionCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_img, test_mask, callback_path, network, channels):
        self.test_img = test_img
        self.test_mask = test_mask
        self.callback_path = callback_path
        self.predicted_prop = np.array([])
        self.network = network
        self.channels = channels

    def on_epoch_end(self, epoch, logs={}):
        predic_masks = self.model.predict(np.expand_dims(self.test_img, axis = -1))
        testy_masks = np.mean(np.around(predic_masks))
        self.predicted_prop = np.append(self.predicted_prop, testy_masks)
