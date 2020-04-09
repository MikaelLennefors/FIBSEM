import numpy as np
import os
import cv2

def extract_data(path, channels):
    og_dir = os.listdir(path)
    og_dir.sort()

    imgs = []
    masks = []

    for file_name in og_dir:
        img = cv2.imread(path + file_name, 0)
        if 'image' in file_name:
            if channels != 1:
                imgs.append(img)
            else:
                if (int(file_name[-7:-4]) % 7) == 3:
                    imgs.append(img)

        elif 'mask' in file_name:
            masks.append(img)
    imgs = np.array(imgs).reshape((-1, 258, 258, 1)).astype('float32')
    masks = np.array(masks).reshape((-1, 256, 256, 1))
    if channels > 1:
        imgs = imgs.reshape(-1, 7, 258, 258, 1)
        imgs = np.swapaxes(imgs,1,2)
        imgs = np.swapaxes(imgs,2,3)
    if channels == 3:
        imgs = np.delete(imgs,[0,1,5,6], axis = 3)
    elif channels == 5:
        imgs = np.delete(imgs,[0,6], axis = 3)
    return imgs, masks
