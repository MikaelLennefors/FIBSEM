import math
import numpy as np
import time

from whitening import zca_whitening

def gen_data_split(images, masks, whitening_coeff = 5e-3):
    n_images = np.shape(images)[0]

    image_indices = np.random.permutation(n_images)

    n_training_image = math.floor(0.75*n_images)

    im_split = np.vsplit(images[image_indices],[n_training_image])
    mask_split = np.split(masks[image_indices],[n_training_image])

    train_images = im_split[0]
    val_images = im_split[1]

    train_mask = mask_split[0].astype(np.uint8)
    start_time = time.time()
    train_images = zca_whitening(train_images, whitening_coeff)
    end_time = time.time()-start_time
    #print(end_time)
    #print(time.time())
    #time.sleep(10)
    #print(time.time())
    #start_time = time.time()
    val_images = zca_whitening(val_images, whitening_coeff)
    end_time = time.time()-start_time
    print("hejhej", end_time)
    val_mask = mask_split[1] / 255.
    return train_images, train_mask, val_images, val_mask
