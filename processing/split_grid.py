import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

def split_grid(images, masks, grid_split):
    img_shape = int(np.shape(masks)[1]/grid_split)

    img_split = []
    mask_split = []

    for i in range(grid_split):
        for j in range(grid_split):
            img_split.append(images[:,img_shape*i:img_shape+2+img_shape*i,img_shape*j:img_shape+2+img_shape*j])
            mask_split.append(masks[:,img_shape*i:img_shape+img_shape*i,img_shape*j:img_shape+img_shape*j])

    if np.shape(images)[3] > 1:
        img_split = np.array(img_split).reshape(-1,img_shape+2,img_shape+2,np.shape(images)[3],1)
    else:
        img_split = np.array(img_split).reshape(-1,img_shape+2,img_shape+2,1)
    mask_split = np.array(mask_split).reshape(-1,img_shape,img_shape,1)

    # print('img shape: ', np.shape(img_split))
    # print('mask shape: ', np.shape(mask_split))
    # raise

    return img_split, mask_split
