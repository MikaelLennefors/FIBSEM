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
    count = count + 1

result['Pre processing'] = result['Pre processing'].astype(int)

# Plot with plotly
# fig = px.parallel_coordinates(result, color = 'Mean IoU',
#                              color_continuous_scale=px.colors.diverging.Tealrose,
#                              color_continuous_midpoint=0.5.
#                              dimensions = list([
#                             dict(range = [0,5],
#                                  tickvals = [0,1,2,3,4,5],
#                                  label = 'Pre processing',values = result['Pre processing'],
#                                  ticktext = ['Standardized', 'Normalized', 'ZCA: 1e-1', 'ZCA: 1e-2', 'ZCA: 1e-3', 'ZCA: 1e-4']),
#                         ])
# fig.layout.title = network
# fig.write_html('first_figure.html', auto_open=True)

df = pd.read_csv("https://raw.githubusercontent.com/bcdunbar/datasets/master/parcoords_data.csv")

fig = go.Figure(data=
    go.Parcoords(
        line = dict(color = result['Mean IoU'],
                   colorscale = 'Electric',
                   showscale = True,
                   cmin = 0,
                   cmax = 1),
        dimensions = list([
            dict(range = [0,1],
                 constraintrange = [100000,150000],
                 label = "Block Height", values = df['blockHeight']),
            dict(range = [0,700000],
                 label = 'Block Width', values = df['blockWidth']),
            dict(tickvals = [0,0.5,1,2,3],
                 ticktext = ['A','AB','B','Y','Z'],
                 label = 'Cyclinder Material', values = df['cycMaterial']),
            dict(range = [-1,4],
                 tickvals = [0,1,2,3],
                 label = 'Block Material', values = df['blockMaterial']),
            dict(range = [134,3154],
                 visible = True,
                 label = 'Total Weight', values = df['totalWeight']),
            dict(range = [9,19984],
                 label = 'Assembly Penalty Wt', values = df['assemblyPW']),
            dict(range = [49000,568000],
                 label = 'Height st Width', values = df['HstW'])])
    )
)
fig.show()
