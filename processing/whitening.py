import numpy as onp
import jax.numpy as np
from jax import jit

def zca_whitening(img, epsilon = 1e-3):
    '''
    Performs ZCA on input image. Function need jax-library to be installed. Jax
    is for the moment (april 2020) only avilable on Linux.

    Input:
        img (png): image to be processed
        epsilon (float): whitening-coefficient

    Output:
        jax array of processed image
    '''
    @jit
    def jax_whitening(img, epsilon):
        x = np.shape(img)[1]
        y = np.shape(img)[2]

        channels = np.shape(img)[3]

        if channels > 1:
            img = np.swapaxes(img,2,3)
            img = np.swapaxes(img,1,2)
            img = img.reshape(-1,np.shape(img)[2],np.shape(img)[3],1)

        og_shape = np.shape(img)

        img = img.reshape(og_shape[0], -1)


        img = img / 255.
        img = img - np.mean(img,axis=0)
        U, S, V = np.linalg.svd(np.cov(img, rowvar=True))

        img_ZCA = np.dot(np.dot(np.dot(U,np.diag(1.0/np.sqrt(S + epsilon))),U.T),img)

        min_ZCA = np.min(img_ZCA)
        max_ZCA = np.max(img_ZCA)

        img_ZCA = (img_ZCA - min_ZCA) / (max_ZCA - min_ZCA)

        img_ZCA = img_ZCA.reshape(og_shape)

        if channels > 1:
            img_ZCA = img_ZCA.reshape(-1, channels, x, y, 1)
            img_ZCA = np.swapaxes(img_ZCA,1,2)
            img_ZCA = np.swapaxes(img_ZCA,2,3)
        return img_ZCA
    return onp.array(jax_whitening(img, epsilon))
