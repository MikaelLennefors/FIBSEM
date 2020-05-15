import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

gpu = int(input("Choose GPU, 0 for Xp, 1 for V: "))

if gpu == 0:
    path = '../results/Xp/'
else:
 	path = '../results/V/'

network = input('Choose network, unet, dunet, multi, nesnet: ')
path = path  + network + '_'
channels = int(input('Number of channels: '))
path = path + str(channels)+ '_channels.csv'
result = pd.read_csv(path, sep=",", usecols=(2, 3, 4, 5, 6, 7, 8))# , header=None

count = 0
for val in result['Pre processing']:
    if val == 'Standardized':
        result.at[count, 'Pre processing'] = 0
    if val == 'Normalized':
        result.at[count, 'Pre processing'] = 1
    if val == 'ZCA: 1e-1':
        result.at[count, 'Pre processing'] = 2
    if val == 'ZCA: 1e-2':
        result.at[count, 'Pre processing'] = 3
    if val == 'ZCA: 1e-3':
        result.at[count, 'Pre processing'] = 4
    if val == 'ZCA: 1e-4':
        result.at[count, 'Pre processing'] = 5
    if val == 'ZCA: 1e-5':
        result.at[count, 'Pre processing'] = 6
    if val == 'ZCA: 1e-6':
        result.at[count, 'Pre processing'] = 7
    count = count + 1

result['Pre processing'] = result['Pre processing'].astype(int)
result['Filters'] = np.log2(result['Filters'])
result['Learning rate'] = -1*np.log10(result['Learning rate'])

test = 'prut ' + result['Learning rate'].astype(str)
test = list(test)
# print(test.type)
# raise

fig = go.Figure(data=
    go.Parcoords(
        line = dict(color = result['Mean IoU'],
                   colorscale = 'Viridis',
                   showscale = True,
                   cmin = 0,
                   cmax = 0.8),
        dimensions = list([
            dict(range = [3,6],
                 label = '-log10(Learning rate)', values = result['Learning rate']),
            dict(tickvals = [4,5,6],
                 ticktext = ['16','32','64'],
                 label = 'Filters', values = result['Filters']),
            dict(tickvals = [0,1,2,3,4,5,6,7],
                 ticktext = ['Standardized','Normalized','ZCA: 1e-1', 'ZCA: 1e-2', 'ZCA: 1e-3', 'ZCA: 1e-4', 'ZCA: 1e-5', 'ZCA: 1e-6'],
                 label = 'Pre processing', values = result['Pre processing']),
            dict(tickvals = [0,1,2,3,4,5,6],
                 ticktext = ['0','1','2','3','4','5','6'],
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
        'text': "Hyperparameter space for " + network,
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.show()
