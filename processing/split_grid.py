import numpy as np
import random

def split_grid(images, masks, grid_split, test_set = False):
    img_shape = int(np.shape(masks)[1]/grid_split)

    img_split = []
    mask_split = []

    if test_set:
        n_patches = grid_split
        rows = np.linspace(0,256 - img_shape, grid_split)
        rows = rows.astype(int)
        cols = rows
    else:
        n_patches = 30
        rows = np.sort(random.sample(range(0, 256 - img_shape), n_patches))
        cols = np.sort(random.sample(range(0, 256 - img_shape), n_patches))

    for i,j in zip(rows, cols):
        img_split.append(images[:, i:img_shape+2+i, j:img_shape+2+j])
        mask_split.append(masks[:, i:img_shape+i, j:img_shape+j])

    if np.shape(images)[3] > 1:
        img_split = np.array(img_split).reshape(-1,img_shape+2,img_shape+2,np.shape(images)[3],1)
    else:
        img_split = np.array(img_split).reshape(-1,img_shape+2,img_shape+2,1)
    mask_split = np.array(mask_split).reshape(-1,img_shape,img_shape,1)

    return img_split, mask_split
