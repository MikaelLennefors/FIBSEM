import math
import numpy as np

def gen_data_split(images, masks):
    n_images = np.shape(images)[0]

    image_indices = np.random.permutation(n_images)

    n_training_image = math.floor(0.75*n_images)

    im_split = np.vsplit(images[image_indices],[n_training_image])
    mask_split = np.split(masks[image_indices],[n_training_image])

    train_images = im_split[0]
    val_images = im_split[1]

    train_mask = mask_split[0].astype(np.uint8)
    val_mask = mask_split[1]
    return train_images, train_mask, val_images, val_mask
