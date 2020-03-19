from PIL import Image
import numpy as np

path = '/home/sms/github/unet_data/'

save_path = '/home/sms/github/unet_data/test_data/'


def read_tiff(path):
    img = Image.open(path)
    images = []
    n_images = img.n_frames
    for i in range(n_images):
        try:
            img.seek(i)
            images.append(np.array(img))

        except EOFError:
            break

    return np.array(images)

if __name__ == '__main__':
    #ratio = input("Porositet: ")
    filename = 'test-volume.tif'
    file = path + filename
    img = read_tiff(file)
    #print(np.shape(img))
    #img = img[:,:,:,0]
    #img = img[3::7]
    #img = img[:,64:320:,64:320]
    for i in range(np.shape(img)[0]):
        im = Image.fromarray(img[i,:,:])
        im.save(save_path + 'image_' + str(i).zfill(2) + '.png')
