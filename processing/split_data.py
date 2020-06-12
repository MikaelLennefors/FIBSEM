import math
import numpy as np

def gen_data_split(images, masks, random_seed):
    '''
    Splits image data into training (75%) and validation set (25%).

    Input:
        images (png): the image data set.
        masks (png): the maske corresponding to the images.
        random_seed (int): random seed for split.
    Output:
        train_images (png): the training images
        train_mask (png): the training masks
        val_images (png): the validation images
        val_mask (png): the validation masks
    '''
    n_images = np.shape(images)[0]

    image_indices = np.random.RandomState(seed=random_seed).permutation(n_images)
    n_training_image = math.floor(0.75*n_images)

    im_split = np.vsplit(images[image_indices],[n_training_image])
    mask_split = np.split(masks[image_indices],[n_training_image])

    train_images = im_split[0]
    val_images = im_split[1]

    train_mask = mask_split[0].astype(np.uint8)
    val_mask = mask_split[1]
    return train_images, train_mask, val_images, val_mask
