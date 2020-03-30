import math
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from whitening import zca_whitening

def gen_data_split(images, masks, channels = 1, b_size = 180, whitening_coeff = 5e-3, maskgen_args = dict(vertical_flip = True)):

    n_images = np.shape(images)[0]

    image_indices = np.random.permutation(n_images)

    n_training_image = math.floor(0.75*n_images)

    im_split = np.vsplit(images[image_indices],[n_training_image])
    mask_split = np.split(masks[image_indices],[n_training_image])

    train_images = im_split[0]
    val_images = im_split[1]

    train_mask = mask_split[0].astype(np.uint8)


    train_images = zca_whitening(train_images, whitening_coeff)
    val_images = zca_whitening(val_images, whitening_coeff)
    val_mask = mask_split[1] / 255.

    img_datagen = ImageDataGenerator(**maskgen_args)
    seed = 1
    mask_generator = img_datagen.flow(train_mask, batch_size=b_size, seed = seed)
    if channels == 1:
        img_generator = img_datagen.flow(train_images, batch_size=b_size, seed = seed)
        train_generator = zip(img_generator, mask_generator)
    else:
        img_generator = []
        for i in range(channels):
            img_generator.append(img_datagen.flow(train_images[:,:,:,i,:], batch_size=b_size, seed = seed))
        if channels == 7:
            train_generator = zip(img_generator[0], img_generator[1], img_generator[2], img_generator[3],
                img_generator[4], img_generator[5], img_generator[6], mask_generator)
        elif channels == 5:
            train_generator = zip(img_generator[0], img_generator[1], img_generator[2], img_generator[3],
                img_generator[4], mask_generator)
        elif channels == 3:
            train_generator = zip(img_generator[0], img_generator[1], img_generator[2], mask_generator)
    return train_generator, val_images, val_mask
