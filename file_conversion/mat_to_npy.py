import h5py
import numpy as np
import imageio


save_path = '../data/'
path_to_data = '/home/sms/magnusro/fib_sem/manual_segmentation/'


def extract_mat_data(file):
	str1 = []

	f = h5py.File(path_to_data + file, 'r')

	data = f['M']
	for i in range(np.shape(data)[1]):
		st = data[0][i]
		obj = f[st]
		str1.append([x for x in obj])
	return(np.array(str1))

if __name__ == '__main__':
	ratio = input("Porositet: ")
	file = 'manual_segmentation_HPC' + ratio + '.mat'
	extracted_data = (1 - extract_mat_data(file))*255
	for i in range(np.shape(extracted_data)[0]):
		imageio.imwrite(save_path + 'mask' + str(ratio)+ '_' + str(i).zfill(2) + '.png', extracted_data[i].T)
	#np.save(save_path + 'mask' + ratio + '.npy', extracted_data)
