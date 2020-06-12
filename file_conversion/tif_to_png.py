import numpy as np
import cv2
from libtiff import TIFF
# TODO: Change data folder?
path = '../data/'
save_path = '../data/all_data_384/'


def read_tiff(path):
    '''
    Converts a tiff-file to png. The function calls the user to input the
    porosity of the file to be read.

    Input:
        path (str): path to tiff-file as string.
    Output:
        save_name (png): png image from tiff-file.
    '''
    img1 = TIFF.open(path)
    images = []
    for image in img1.iter_images():
        tmp = np.around(image*(255/65535))
        images.append(tmp)

    return images

if __name__ == '__main__':
    ratio = input("Porositet: ")
    filename = 'regions_HPC{}.tif'.format(ratio)
    file = path + filename
    img = read_tiff(file)
    save_name = 'image{}_'.format(ratio)

    for image in range(np.shape(img)[0]):
        curr_img = img[image]
        curr_img.astype(int)
        cv2.imwrite(save_path + save_name + str(image).zfill(3) + '.png', curr_img)
