import math
import numpy as np
import os

from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from whitening import zca_whitening
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import array_to_img

def split_grid(images, masks, grid_split):
    if channels > 1:
        if grid_split > 0:
            imgs = imgs.reshape(-1,(2**grid_split),imgs_shape//(2**grid_split),(2**grid_split),imgs_shape//(2**grid_split),channels,1)
            imgs = np.swapaxes(imgs, 2, 3)
            imgs = imgs.reshape(-1, imgs_shape//(2**grid_split), imgs_shape//(2**grid_split), channels, 1)
            masks = masks.reshape((-1,(2**grid_split),256//(2**grid_split), (2**grid_split), 256//(2**grid_split),1))
            masks = np.swapaxes(masks, 2, 3)
            masks = masks.reshape(-1, 256//(2**grid_split), 256//(2**grid_split), 1)
    else:
        if grid_split > 0:
            imgs = imgs.reshape(-1,(2**grid_split),imgs_shape//(2**grid_split),(2**grid_split),imgs_shape//(2**grid_split),1)
            imgs = np.swapaxes(imgs, 2, 3)
            imgs = imgs.reshape(-1, imgs_shape//(2**grid_split), imgs_shape//(2**grid_split), 1)
            masks = masks.reshape((-1,(2**grid_split),256//(2**grid_split), (2**grid_split), 256//(2**grid_split),1))
            masks = np.swapaxes(masks, 2, 3)
            masks = masks.reshape(-1, 256//(2**grid_split), 256//(2**grid_split), 1)

    return imgs, masks
