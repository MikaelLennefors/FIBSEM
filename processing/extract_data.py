import numpy as np
import os
import cv2
import itertools

def extract_data(path, channels, standardize = False):

    ###
    # Input: path to data: string of absolute or relative path, number of channels: int in {1, 3, 5, 7} , standardize: Boolean to return standardized or normalized data
    # Output: numpy array of extracted data dimensions (#files, shape_x, shape_y, channels, 1)
    # Reads a directory containing data of porosities 22, 30 and 45%, can standardize data but most are reshapes to fit further input
    ###
    og_dir = os.listdir(path)
    og_dir.sort()

    imgs = {'22': [], '30': [], '45': []}
    masks = {'22': [], '30': [], '45': []}

    for file_name in og_dir:
        img = cv2.imread(path + file_name, 0)
        if 'image' in file_name:
            porosity = file_name[5:7]
            if channels != 1:
                imgs[porosity].append(img)
            else:
                if (int(file_name[-7:-4]) % 7) == 3:
                    imgs[porosity].append(img)

        elif 'mask' in file_name:
            porosity = file_name[4:6]
            masks[porosity].append(img)

    if standardize == True:
        mu_22 = np.mean(imgs['22'])
        mu_30 = np.mean(imgs['30'])
        mu_45 = np.mean(imgs['45'])

        sig_22 = np.std(imgs['22'])
        sig_30 = np.std(imgs['30'])
        sig_45 = np.std(imgs['45'])

        imgs['22'] = (imgs['22'] - mu_22) / sig_22
        imgs['30'] = (imgs['30'] - mu_30) / sig_30
        imgs['45'] = (imgs['45'] - mu_45) / sig_45

    imgs = np.array(list(itertools.chain.from_iterable(imgs.values())))
    masks = np.array(list(itertools.chain.from_iterable(masks.values())))

    imgs = np.array(imgs).reshape((-1, 258, 258, 1)).astype('float32')
    masks = np.array(masks).reshape((-1, 256, 256, 1)) / 255.

    if channels == 1:
        imgs = np.expand_dims(imgs, axis = -1)
    if channels > 1:
        imgs = imgs.reshape(-1, 7, 258, 258, 1)
        imgs = np.swapaxes(imgs,1,2)
        imgs = np.swapaxes(imgs,2,3)
    if channels == 3:
        imgs = np.delete(imgs,[0,1,5,6], axis = 3)
    elif channels == 5:
        imgs = np.delete(imgs,[0,6], axis = 3)
    return imgs, masks

if __name__ == '__main__':
    imgs, masks = extract_data('../data/train_val_data_border_clean/', channels = 1, standardize = True)

    print(np.shape(imgs))
    print(np.shape(masks))
