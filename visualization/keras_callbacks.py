import cv2
import numpy as np
import tensorflow as tf

class PredictionCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_img, test_mask, callback_path):
        self.test_img = test_img
        self.test_mask = test_mask
        self.callback_path = callback_path

    def on_epoch_end(self, epoch, logs={}):
        predic_mask = self.model.predict(np.expand_dims(self.test_img[0], axis = 0))
        testy_mask = 255. * np.around(predic_mask).reshape(np.shape(predic_mask)[1],np.shape(predic_mask)[1]).astype(np.uint8)
        print('')
        print("True proportion:", np.mean(self.test_mask[0]), "Predicted proportion:", np.mean(testy_mask) / 255.)
        print('')
        cv2.imwrite(self.callback_path + 'pred_mask_' + str(epoch).zfill(3) + '.png', testy_mask)
