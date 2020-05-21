import h5py
import numpy as np
import imageio
import scipy.io
import pandas as pd
import sys
import shutil, os

save_path = '../data/magnus_bilder/'
path_to_data = '/home/ubuntu/Downloads/'


def extract_mat_data(file):
	str1 = []
	data = {'train': [],
			'val':[],
			'test': []}
	f = scipy.io.loadmat(path_to_data + file)
	# print(f)
	# sys.exit()
	for key, value in f.items():
		if 'ind' in key:
			value = value[0].astype(int)
			test_array = 7*(np.sort(value) - 1)
			image_array = []

			for i in test_array:
				image_array.extend(np.linspace(i, i+6, 7, dtype = int))

			image_array = np.array(image_array, dtype = str)
			image_array = np.char.zfill(image_array, 3)
			image_array = ['image' + key[-2:] + '_' + element + '.png' for element in image_array]

			mask_array = np.array(test_array // 7, dtype = str)
			mask_array = np.char.zfill(mask_array, 2)
			mask_array = ['mask' + key[-2:] + '_' + element + '.png' for element in mask_array]
			data[key[12:-6]].extend(image_array)
			data[key[12:-6]].extend(mask_array)

	return(data)

if __name__ == '__main__':
	file = 'data_split.mat'
	extracted_data = extract_mat_data(file)
	for set, files in extracted_data.items():
		for f in files:
			try:
				shutil.move('../data/all_data_384/{}'.format(f), '../data/magnus_data/{}/'.format(set))
			except:
				print(f + 'does not exist')
	sys.exit()
	# for i in range(np.shape(extracted_data)[0]):
	# 	imageio.imwrite(save_path + 'mask' + str(ratio)+ '_' + str(i).zfill(2) + '.png', extracted_data[i].T)
	#np.save(save_path + 'mask' + ratio + '.npy', extracted_data)
