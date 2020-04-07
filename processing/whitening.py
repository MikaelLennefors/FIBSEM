import numpy as np
import time
import scipy.linalg
# import jax.numpy as jnp
# import jax.scipy as jsp
def zca_whitening(img, epsilon = 1e-3):

    channels = np.shape(img)[3]
    if channels > 1:
        img = np.swapaxes(img,2,3)
        img = np.swapaxes(img,1,2)
        img = img.reshape(-1,np.shape(img)[2],np.shape(img)[3],1)
    og_shape = np.shape(img)

    img = img.reshape(og_shape[0], -1)

    img = img / 255.

    img = img - img.mean(axis=0)
    start_time = time.time()
    co = np.cov(img, rowvar=True)
    cov_time = time.time() - start_time
    start_time2 = time.time()
    U, S, V = scipy.linalg.svd(co)
    svd_time = time.time() - start_time2
    print("svd time:",svd_time)
    print("svd over cov:", svd_time/cov_time)
    print("--"*20)
    img_ZCA = U.dot(np.diag(1.0/np.sqrt(S + epsilon))).dot(U.T).dot(img)

    min_ZCA = img_ZCA.min()
    max_ZCA = img_ZCA.max()

    img_ZCA = (img_ZCA - min_ZCA) / (max_ZCA - min_ZCA)

    img_ZCA = img_ZCA.reshape(og_shape)

    if channels > 1:
        img_ZCA = np.array(np.split(img_ZCA, np.shape(img_ZCA)[0]/channels))
        img_ZCA = np.swapaxes(img_ZCA,1,2)
        img_ZCA = np.swapaxes(img_ZCA,2,3)


    return img_ZCA
