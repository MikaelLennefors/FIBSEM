import math
import numpy as np
import os

from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from whitening import zca_whitening
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import array_to_img

def extract_data(path, channels):
    og_dir = os.listdir(path)
    og_dir.sort()

    imgs = []
    masks = []

    for file_name in og_dir:
        img = Image.open(path + file_name)
        if 'image' in file_name:
            if channels != 1:
                imgs.append(img.getdata())
            else:
                if (int(file_name[-7:-4]) % 7) == 3:
                    imgs.append(img.getdata())

        elif 'mask' in file_name:
            masks.append(img.getdata())

    if np.shape(imgs)[1] // 256 == 256:
        imgs_shape = 256
    elif np.shape(imgs)[1] // 258 == 258:
        imgs_shape = 258
    else:
        raise
    imgs = np.array(imgs).reshape((-1, imgs_shape, imgs_shape, 1)).astype('float32')
    masks = np.array(masks).reshape((-1, 256, 256, 1))

    if channels > 1:
        imgs = np.array(np.split(imgs, np.shape(imgs)[0]/7))
        imgs = np.swapaxes(imgs,1,2)
        imgs = np.swapaxes(imgs,2,3)
    if channels == 3:
        imgs = np.delete(imgs,[0,1,5,6], axis = 3)
    elif channels == 5:
        imgs = np.delete(imgs,[0,6], axis = 3)

    return imgs, masks
