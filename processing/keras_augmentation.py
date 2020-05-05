import numpy as np
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def gen_aug(train_images, train_mask, maskgen_args, b_size):
    channels = np.shape(train_images)[3]
    img_datagen = ImageDataGenerator(**maskgen_args)
    seed = 139
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

    return train_generator
