import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

def split_grid(images, masks, grid_split):
    grid_split = 4

    img_split = []
    mask_split = []

    for i in range(grid_split):
        for j in range(grid_split):
            img_split.append(images[:,64*i:66+64*i,64*j:66+64*j])
            mask_split.append(masks[:,64*i:64+64*i,64*j:64+64*j])


    if np.shape(images)[3] > 1:
        img_split = np.array(img_split).reshape(-1,66,66,np.shape(images)[3],1)
    else:
        img_split = np.array(img_split).reshape(-1,66,66,1)
    mask_split = np.array(mask_split).reshape(-1,64,64,1)

    return img_split, mask_split
