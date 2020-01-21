from PIL import Image
import numpy as np

path = '/home/sms/magnusro/fib_sem/manual_segmentation/'
filename = 'segmentation_regions_HPC30.tif'

# im = Image.open(path+filname)
#
# imageArray = np.array(im)
# print(np.shape(imageArray))
# # print(imageArray)
# img = im.seek(0)
# img1 = np.array(img)
# print(np.shape(img1))
# # im.show()


def read_tiff(path):
    """
    path - Path to the multipage-tiff file
    n_images - Number of pages in the tiff file
    """
    img = Image.open(path)
    images = []
    n_images = img.n_frames
    for i in range(n_images):
        try:
            img.seek(i)
            images.append(np.array(img))

        except EOFError:
            # Not enough frames in img
            break

    return np.array(images)

if __name__ == '__main__':
    file = path+filename
    img = read_tiff(file)
    print(np.shape(img))
    newImg = img[:,:,:,0]
    print(np.shape(newImg))
    newImg = newImg[3::7]
    print(np.shape(newImg))
    newImg = newImg[:,64:320:,64:320]
    print(np.shape(newImg))
    im = Image.fromarray(newImg[0,:,:])
    im.save('testPNG.png')
