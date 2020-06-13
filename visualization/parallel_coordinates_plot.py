import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

'''
    Produces parallel coordinates plot with Plotly. The user is asked for which
    network shouuld be plotted and how many channels that are used in that case.

    Output:
        interactive html file 
'''

path = '../results/hyper_opt/'

network = input('Choose network, unet, dunet, multi, nesnet: ')
path = path  + network + '_'
channels = int(input('Number of channels: '))
path = path + str(channels)+ '_channels.csv'
result = pd.read_csv(path, sep=",", usecols=(2, 3, 4, 5, 6, 7, 8, 9))# , header=None

count = 0
for val in result['Pre processing']:
    if val == 'Standardize':
        result.at[count, 'Pre processing'] = 0
    if val == 'Normalize':
        result.at[count, 'Pre processing'] = 1
    if val == 'ZCA':
        result.at[count, 'Pre processing'] = 2
    count = count + 1
count = 0

for val in result['Whitening']:
    if val !=0:
        result.at[count, 'Whitening'] = -1*np.log10(val)
    count = count + 1

result['Pre processing'] = result['Pre processing'].astype(int)
result['Filters'] = np.log2(result['Filters'])
result['Learning rate'] = -1*np.log10(result['Learning rate'])


maxIoU = np.max(result['Mean IoU'])
minIoU = np.min(result['Mean IoU'])
fig = go.Figure(data=
    go.Parcoords(
        line = dict(color = result['Mean IoU'],
                   colorscale = 'Viridis',
                   showscale = True,
                   cmin = minIoU,
                   cmax = maxIoU),
        dimensions = list([
            dict(range = [3,6],
                 label = '-log10(Learning rate)', values = result['Learning rate']),
            dict(tickvals = [4,5,6],
                 ticktext = ['16','32','64'],
                 label = 'Filters', values = result['Filters']),
            dict(tickvals = [0,1,2],
                 ticktext = ['Standardized','Normalized','ZCA'],
                 label = 'Pre processing', values = result['Pre processing']),
            dict(range = [0,4],
                 label = 'Whitening', values = result['Whitening']),
            dict(tickvals = [1,2,3,4,5,6],
                 ticktext = ['1','2','3','4','5','6'],
                 label = 'Batch size', values = result['Batch size']),
            dict(range = [0.3,0.5],
                 label = 'Dropout', values = result['Dropout']),
            dict(range = [0,0.2],
                 label = 'Elastic proportion', values = result['Elastic proportion']),
            dict(range = [0,0.8],
                label = 'Mean IoU', values = result['Mean IoU'])])
    )
)
fig.update_layout(
    title={
        'text': "Hyperparameter space",
        'y':0.98,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.show()
