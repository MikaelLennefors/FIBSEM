import numpy as np
import os
import random
import sys

# import statistics
import time
import math
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.optimizers import Adam

from bayes_opt import BayesianOptimization
from PIL import Image

sys.path.insert(1, './networks')
import losses
from unet import unet
from dunet import D_Unet
from multiresunet import MultiResUnet
from nestnet import Nest_Net, bce_dice_loss

sys.path.insert(1, './processing')
from elastic_deformation import elastic_transform
from split_data import gen_data_split
from datetime import datetime
from extract_data import extract_data
from whitening import zca_whitening
from split_grid import split_grid
from keras_augmentation import gen_aug



def evaluate_network(net_drop, net_filters, net_lr, prop_elastic):
    mean_benchmark = []
    net_lr = math.pow(10,-net_lr)
    net_filters = int(math.pow(2,math.floor(net_filters)+4))
    # print(net_lr)
    # print(net_filters)
    # raise
    for i in range(3):
        train_gen = t_gen[i]
        val_img = v_img[i]
        val_mask = v_mask[i]

        if channels == 1:
            input_size = np.shape(val_img)[1]//(2**grid_split)
            m = unet(input_size = input_size, multiple = net_filters, activation = net_activ_fun, learning_rate = net_lr, dout = net_drop)
        else:
            input_size = (np.shape(val_img)[1]//(2**grid_split),channels)
            m = D_Unet(input_size = input_size, multiple = net_filters, activation = net_activ_fun, learning_rate = net_lr, dout = net_drop)
        m.compile(optimizer = Adam(lr = net_lr), loss = losses.iou_loss, metrics = [losses.iou_coef, 'accuracy'])



        earlystopping1 = EarlyStopping(monitor = 'val_iou_coef', min_delta = 0.01, patience = 100, mode = 'max')
        earlystopping2 = EarlyStopping(monitor = 'val_iou_coef', baseline = 0.6, patience = 50)
        # class PredictionCallback(tf.keras.callbacks.Callback):
        #     def on_epoch_end(self, epoch, logs={}):
        #         predic_mask = self.model.predict(np.expand_dims(test_img[0], axis = 0))
        #         testy_mask = np.around(predic_mask).reshape(256,256).astype(np.uint8)*255

        #         im = Image.fromarray(testy_mask)
        #         im.save(callback_path + 'pred_mask_' + str(epoch).zfill(3) + '.png')


        callbacks_list = [earlystopping1, earlystopping2]

        count = 0
        for c in train_gen:
            y = np.array(c[-1])
            x = []
            for j in range(len(c) - 1):
                x.append(c[j])
            x = np.array(x)

            x = np.swapaxes(x,0,1)
            x = np.swapaxes(x,1,2)
            x = np.swapaxes(x,2,3)
            if np.shape(x)[3] == 1:
                x = np.squeeze(x, axis = 3)
            if elast_deform == True:
                for j in range(np.shape(x)[0]):
                    if random.random() < prop_elastic:
                        randoint = random.randint(0, 1e3)
                        for k in range(channels):
                            seed = np.random.RandomState(randoint)
                            img = x[j,:,:,k]
                            im_merge_t = elastic_transform(img, img.shape[1] * elast_alpha, img.shape[1] * elast_sigma, img.shape[1] * elast_affine_alpha, random_state = seed)
                            if channels > 1:
                                x[j,:,:,k] = im_merge_t
                            else:
                                x[j,:,:,k] = im_merge_t.reshape(img.shape[0],img.shape[1])
                        mask = y[j].copy().reshape(256,256)
                        seed = np.random.RandomState(randoint)
                        im_mask_t = elastic_transform(mask, mask.shape[1] * elast_alpha, mask.shape[1] * elast_sigma, mask.shape[1] * elast_affine_alpha, random_state = seed)
                        #print(np.shape(im_mask_t))
                        # im_mask_t = im_merge_t[...,0]
                        y[j] = im_mask_t#.reshape(256,256,1)
            y = np.around(y / 255.)

            results = m.fit(x, y, verbose = 0, batch_size = b_size, epochs=NO_OF_EPOCHS, validation_data=(val_img, val_mask), callbacks = callbacks_list)
            # i['val_iou'] = max(i['val_iou'], max(results.history['val_iou_coef']))
            # i['val_accuracy'] = max(i['val_accuracy'], max(results.history['val_accuracy']))
            # i['val_TP'] = max(i['val_TP'], max(results.history['val_TP']))
            # i['val_TN'] = max(i['val_TN'], max(results.history['val_TN']))
            # i['val_FP'] = max(i['val_FP'], max(results.history['val_FP']))
            # i['val_FN'] = max(i['val_FN'], max(results.history['val_FN']))

            count += 1
            #print(max_count - count)
            if count >= max_count:
                break
        pred = m.evaluate(test_img, test_mask, verbose = 0)
        score = pred[1]

        mean_benchmark.append(score)
    m1 = np.mean(mean_benchmark)
    # mdev = np.std(mean_benchmark)
    return m1
# test = evaluate_network(4,2)
