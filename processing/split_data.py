import math
import numpy as np
import os

from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from whitening import zca_whitening
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import array_to_img

def extract_data(path, channels, grid_split = 0):
    og_dir = os.listdir(path)
    og_dir.sort()

    imgs = []
    masks = []

    for file_name in og_dir:
        img = Image.open(path + file_name)
        if 'image' in file_name:
            imgs.append(img.getdata())
        elif 'mask' in file_name:
            masks.append(img.getdata())

    imgs = np.array(imgs).reshape((-1, 256, 256, 1)).astype('float32')
    masks = np.array(masks).reshape((-1, 256, 256, 1))

    if channels > 1:
        imgs = np.array(np.split(imgs, np.shape(imgs)[0]/7))
        imgs = np.swapaxes(imgs,1,2)
        imgs = np.swapaxes(imgs,2,3)

        if grid_split > 0:
            imgs = imgs.reshape(-1,(2**grid_split),256//(2**grid_split),(2**grid_split),256//(2**grid_split),channels,1)
            imgs = np.swapaxes(imgs, 2, 3)
            imgs = imgs.reshape(-1, 256//(2**grid_split), 256//(2**grid_split), channels, 1)
            masks = masks.reshape((-1,(2**grid_split),256//(2**grid_split), (2**grid_split), 256//(2**grid_split),1))
            masks = np.swapaxes(masks, 2, 3)
            masks = masks.reshape(-1, 256//(2**grid_split), 256//(2**grid_split), 1)
    else:
        if grid_split > 0:
            imgs = imgs.reshape(-1,(2**grid_split),256//(2**grid_split),(2**grid_split),256//(2**grid_split),1)
            imgs = np.swapaxes(imgs, 2, 3)
            imgs = imgs.reshape(-1, 256//(2**grid_split), 256//(2**grid_split), 1)
            masks = masks.reshape((-1,(2**grid_split),256//(2**grid_split), (2**grid_split), 256//(2**grid_split),1))
            masks = np.swapaxes(masks, 2, 3)
            masks = masks.reshape(-1, 256//(2**grid_split), 256//(2**grid_split), 1)
    if channels == 3:
        imgs = np.delete(imgs,[0,1,5,6], axis = 3)
    elif channels == 5:
        imgs = np.delete(imgs,[0,6], axis = 3)


    return imgs, masks

def gen_data_split(path_to_data = '../data/train_val_data/', path_to_test = '../data/test_data/', channels = 1, b_size = 180, whitening_coeff = 5e-3, maskgen_args = dict(vertical_flip = True), grid_split = 0):

    images, masks = extract_data(path_to_data, channels, grid_split)
    test_images, test_masks = extract_data(path_to_test, channels, grid_split)

    if grid_split > 0:
        indices = []
        for i in range(0, np.shape(masks)[0], 2):
            if np.count_nonzero(masks[i]) < 400:
                indices.append(i)
        masks = np.delete(masks, indices, axis = 0)
        images = np.delete(images, indices, axis = 0)

    n_images = np.shape(images)[0]

    image_indices = np.random.permutation(n_images)

    n_training_image = math.floor(0.75*n_images)

    im_split = np.vsplit(images[image_indices],[n_training_image])
    mask_split = np.split(masks[image_indices],[n_training_image])

    train_images = im_split[0]
    val_images = im_split[1]

    train_mask = mask_split[0].astype(np.uint8)
    val_mask = mask_split[1] / 255.
    test_mask = test_masks / 255.

    train_images, val_images, test_images = zca_whitening([train_images, val_images, test_images], whitening_coeff)


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
    return train_generator, val_images, val_mask, test_images, test_mask




if __name__ == '__main__':
    train_gen, val_img, val_mask, test_img, test_mask = gen_data_split(path_to_data = '../data/train_val_data_raw/', path_to_test = '../data/test_data_raw/',
        channels = 7)
    print(np.shape(val_img))
    print(np.shape(val_mask))

    test = val_mask.reshape((60,4,64, 4, 64,1))
    test = np.swapaxes(test, 2, 3)
    test = test.reshape(960, 64, 64, 1)
    val_img = val_img.reshape((60,4,64,4,64,7,1))
    val_img = np.swapaxes(val_img,2,3)
    #test = test.reshape((60,16,64,64,7,1))
    val_img = val_img.reshape((960,64,64,7,1))

    # for i in range(np.shape(val_img)[0]):
    #     for j in range(np.shape(val_img)[3]):
    #         plt.imshow(array_to_img(val_img[i,:,:,j]), cmap='gray', vmin=0,vmax =255)
    #         plt.show()
    #         test = val_img[i,:,:,j,0].reshape(4,64,256,1)
    #         test = test.reshape(4,64,4,64,1)
    #         test = np.swapaxes(test, 1, 2)
    #         test = test.reshape(16,64,64,1)
    #         print(np.shape(test))
    #         raise



