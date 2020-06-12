import numpy as np
import random

def split_grid(images, masks, grid_split, test_set = False, rand_seed = 1):
    '''
    Splits images of size 256x256 in (256/2^grid_split) x (256/2^grid_split) patches. The
    number of produces patches n_patches can be set by the user. The program
    samples n_patches randomly in the image. Also, the user
    can set if the the set to be splitted is the test set. If that is the case,
    the whole test image is used instead of taking a random sample of the image.

    Input:
        images (png): images
        masks (png): mask
        grid_split (int): 256/2^grid_split will be witdh and hight of patch.
        test_set (boolean): True if test set is to be split. Default False.
        rand_seed (int): seed for random sampling.
    '''
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
        random.seed(rand_seed)
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
