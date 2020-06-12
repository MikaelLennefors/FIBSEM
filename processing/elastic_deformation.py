import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
	'''
	Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5 and
	 https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation

	Input:
		image (png): image to be transformed
		alpha (float): paramter impacting the gaussian transformations.
		sigma (float): standard deviation of the gaussian transformations.
		alpha_affine (float): paramter impacting the gaussian transformations.
	Output:
		deformed image (png)
	'''

	if random_state is None:
		random_state = np.random.RandomState(None)

	shape = image.shape
	shape_size = shape[:2]

	center_square = np.float32(shape_size) // 2
	square_size = min(shape_size) // 3

	pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
	pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)

	M = cv2.getAffineTransform(pts1, pts2)
	image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
	dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
	dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

	x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
	indices = np.reshape(y+dy.reshape(shape_size), (-1, 1)), np.reshape(x+dx.reshape(shape_size), (-1, 1))

	return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape[0], shape[1], 1)
