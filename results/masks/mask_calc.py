import numpy as np
import os
import cv2
import itertools
import sys

def iou_loss(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    sum_ = np.sum(y_true + y_pred)
    jac = intersection / (sum_ - intersection)

    # intersection_c = keras.sum(keras.abs(y_true_c * y_pred_c), axis=-1)
    # sum_c = keras.sum(keras.abs(y_true_c) + keras.abs(y_pred_c), axis=-1)
    # jac_c = (intersection + smooth) / (sum_c - intersection_c + smooth)

    return jac

def extract_data(path):

    og_dir = os.listdir(path)
    og_dir.sort()
    masks = {}
    for file_name in og_dir:
        name = file_name.split('_')
        if name[0] != 'mask':
            masks[name[0]] = {}

        # print('-')
        # print(name)
        # print(name[1].isdigit())
        #
    for file_name in og_dir:
        name = file_name.split('_')
        if name[0] != 'mask':
            masks[name[0]][name[1]] = cv2.imread(path + file_name, 0)
    for key, value in masks.items():
         for inner_key, inner_value in masks[key].items():
             print(key, inner_key, np.mean(inner_value)/ 255. )


    true_mask = masks['truth']['1'] / 255.
    test_mask = masks['nestnet']['1'] / 255.

    print(test_mask)
    print(np.min(true_mask))
    print(np.min(test_mask))
    print(np.max(true_mask))
    print(np.max(test_mask))

    print(iou_loss(test_mask, test_mask))

if __name__ == '__main__':
    extract_data('./')
