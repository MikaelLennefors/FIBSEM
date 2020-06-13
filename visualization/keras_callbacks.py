import numpy as np
import tensorflow as tf

class PredictionCallback(tf.keras.callbacks.Callback):
    '''
    Creates prediction masks on validation data.

    Output:
        prediction masks
        predicted porosity
    '''
    def __init__(self, test_img):
        self.test_img = test_img
        self.predicted_prop = np.array([])

    def on_epoch_end(self, epoch, logs={}):
        predic_masks = self.model.predict(np.expand_dims(self.test_img, axis = -1))
        testy_masks = np.mean(np.around(predic_masks))
        self.predicted_prop = np.append(self.predicted_prop, testy_masks)
