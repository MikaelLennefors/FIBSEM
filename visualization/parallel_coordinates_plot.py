import numpy as np
import pandas as pd
import plotly.express as px

gpu = int(input("Choose GPU, 0 for Xp, 1 for V: "))

if gpu == 0:
    path = '../results/Xp/'
else:
 	path = '../results/V/'

network = input('Choose network, unet, dunet, multi, nesnet: ')

if network == 'unet':
    path = path + 'unet_'
elif network == 'dunet':
 	path = path + 'dunet_'
elif network == 'multi':
 	path = path + 'multi_'
elif network == 'nesnet':
 	path = path + 'nesnet_'

channels = int(input('Number of channels: '))

if channels == 1:
    path = path +  str(channels) + '_channels.csv'
elif channels == 3:
 	path = path +  str(channels) + '_channels.csv'
elif channels == 5:
 	path = path +  str(channels) + '_channels.csv'
elif channels == 7:
 	path = path +  str(channels) + '_channels.csv'

result = pd.read_csv(path, sep=",", usecols=(2, 3, 4, 5, 6, 7, 8))# , header=None
# result = result.applymap(str)
#print(result['val_iou'].to_numpy())
# for i in range(len(result['val_iou'].to_numpy())):
# 	splittat = result['val_iou'].to_numpy()[i].split('0.')
# 	if len(splittat) > 1:
# 		result['val_iou'].iloc[i] = '0.' + splittat[1]
# 	else:
# 		result['val_iou'].iloc[i] = splittat[0]

# result['val_iou'] = pd.to_numeric(result['val_iou'])
# for i in result.columns:
# 	print(i, result[i].unique())
# #print(result['prop_elastic'].unique())
# raise

# result.columns = ['net_filters','net_lr','net_bin_split','net_beta1','net_beta2','net_drop','net_activ_fun','prop_elastic', 'val_iou','val_accuracy']
#result.columns = ['net_filters','net_lr','net_bin_split', 'val_iou','val_accuracy']



# Sort after bes IoU
# result.sort_values(by=['val_iou'])

# Convert som paramters to right log scale
# result['net_filters'] = np.log2(result['net_filters'])
# result['net_lr'] = np.log10(result['net_lr'])
# result['net_beta2'] = -1*np.log10(1 - result['net_beta2'])

# # Print 10 best reults
# np.set_printoptions(suppress=True)
# print('The top 10 best results are:\n')
# result = result.sort_values(by=['val_iou'], ascending=False)
# print(result.head(10))
#
# # Remove accuracy from the fram to plot in parallelogram
# parallel_frame = result.iloc[:,0:result.shape[1]]

# Plot with plotly
fig = px.parallel_coordinates(result, color = 'Mean IoU',
                             color_continuous_scale=px.colors.sequential.Inferno,
                             color_continuous_midpoint=0.5)
fig.layout.title = network
fig.write_html('first_figure.html', auto_open=True)
