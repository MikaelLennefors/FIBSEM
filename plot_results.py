import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd

gpu = int(input("Choose GPU, 0 for Xp, 1 for V: "))
if gpu == 0:
    path = '../testXp.txt'
else:
	path = '../testV.txt'
# headers = ['net_filters','net_lr','net_bin_split','net_beta1','net_beta2','net_drop','net_activ_fun','prop_elastic', 'val_iou','val_accuracy']
# result = pd.DataFrame(np.random.randint(0,100,size=(100, 10)), columns=headers)
result = pd.read_csv(path, sep="\t")# , header=None
# print(result.columns)
result.columns = ['net_filters','net_lr','net_bin_split','net_beta1','net_beta2','net_drop','net_activ_fun','prop_elastic', 'val_iou','val_accuracy']
headers = result.columns
# result = pd.read_csv(path, sep="\t",header=None)
# print(result)
# print(result.columns)


# # Read header with parameters
# f = open(path)
# headers = f.readline()
# headers = headers.split('\t')
# headers[-1]= headers[-1].strip('\n')

# # Load file
# result = np.genfromtxt(path, delimiter = '\t', skip_header = True, usecols=(0,1,2,3,4,5,6,7,8,9))

# Sort after bes IoU
# result = result[result[:,8].argsort()]
result.sort_values(by=['val_iou'])

# Print 10 best reults
np.set_printoptions(suppress=True)
# iou = result[:,8]
print('The top 10 best results are:\n')
# print(result[-10:])
result = result.sort_values(by=['val_iou'], ascending=False)
print(result.iloc[:10,:])

# # Draw 2D histogram of results
# n_plots = len(headers)-2
# fig, axs = plt.subplots(1,n_plots)
# fig.suptitle('2D histogram of IoU vs hyperparameter space')


# my_cmap = plt.cm.jet
# # my_cmap.set_under('w',1)

# for i in range(0,n_plots):
#     im = axs[i].hist2d(result[:,i], iou,cmap = my_cmap)
#     axs[i].set_title('IoU vs {}'.format(headers[i]))
#     if i ==0:
#         axs[i].set(xlabel='{}'.format(headers[i]), ylabel='IoU')
#     else:
#         axs[i].set(xlabel='{}'.format(headers[i]))
# fig.colorbar(im[3], ax=axs.ravel().tolist())

# plt.show()
# sns.jointplot(x="net_filters", y="val_iou",data = result, kind="kde",n_levels=20)
# sns.pairplot(result, diag_kind="kde", y_vars = ['val_iou'], x_vars = ['net_filters','net_lr','net_bin_split','net_beta1','net_beta2','net_drop','net_activ_fun','prop_elastic'])

g = sns.PairGrid(result)
g.map_diag(sns.kdeplot)
g = g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot,n_levels=10)
plt.show()
