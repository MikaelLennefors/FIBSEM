import cv2
import numpy as np
import tensorflow as tf
import datetime

class PredictionCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_img, test_mask, callback_path, network):
        self.test_img = test_img
        self.test_mask = test_mask
        self.callback_path = callback_path
        self.predicted_prop = np.array([])
        self.network = network

    def on_epoch_end(self, epoch, logs={}):
        predic_masks = self.model.predict(np.expand_dims(self.test_img, axis = -1))
        testy_masks = np.mean(np.around(predic_masks))
        print('')
        print("True proportion:", np.mean(self.test_mask), "Predicted proportion:", testy_masks)
        self.predicted_prop = np.append(self.predicted_prop, testy_masks)
        print('')
        np.savetxt(self.callback_path + self.network + '_test_proportions.txt', self.predicted_prop)
