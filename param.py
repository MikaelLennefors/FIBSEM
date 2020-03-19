import numpy as np
import random

class HyperParam:
	def __init__(self):
		#PREPROCESSING
		#self.zca_coeff = np.logspace(-4, -1, 8)
		
		#NETWORK
		self.net_filters = np.logspace(4, 6, 3, base=2)
		self.net_lr = np.logspace(-6, -3.5, 10)
		#self.net_bin_split = np.linspace(0.3, 0.4, 3)
		self.net_drop = np.linspace(0.3, 0.6, 4)
		#self.net_activ_fun = np.linspace(0, 1, 2)
		
		self.prop_elastic = np.linspace(0, 0.2, 3)
		#self.vertical_flip = np.linspace(0, 1, 2)
		#self.shear_range = np.linspace(0, 1, 11)
		#self.rotation_range = np.linspace(0, 90, 6)
		#self.zoom_range = np.linspace(0, 1, 11)
		#self.fill_mode = ['constant', 'nearest', 'reflect', 'wrap']


	def generate_hyperparameters(self):
		params = {}
		for i in vars(self).keys():
			exec("params[i] = np.random.choice(self.{}, 1).item()".format(i))
		return params


if __name__ == '__main__':
	test = HyperParam()
	test.generate_hyperparameters()




# self.net_filters = np.logspace(4, 6, 3, base=2)
# self.net_lr = np.logspace(-6, -3, 20)
# self.net_bin_split = np.linspace(0.1, 0.6, 20)
# self.net_beta1 = np.linspace(0, 0.9, 2)
# self.net_beta2 = 1-np.logspace(-5, -1, 5)
# self.net_drop = np.linspace(0, 0.6, 20)
# self.net_activ_fun = np.linspace(0, 1, 2)

# self.prop_elastic = np.linspace(0, 0.6, 7)