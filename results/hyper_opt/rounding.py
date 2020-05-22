import pandas as pd
# path ='unet_1_channels_old.csv'
path ='multi_1_channels.csv'
# path ='multiresunet_5_channels.csv'
# path ='multiresunet_3_channels.csv'
# path ='dunet_3_good.csv'
# path ='nestnet_1_channels.csv'
# path ='nestnet_3_channels.csv'
# result = pd.read_csv(path, sep=",", usecols=(0,1,2, 3, 4, 5, 7, 8,9))# , header=None
result = pd.read_csv(path, sep=",", usecols=(0,1,2, 3, 4, 5, 6, 7, 8))# , header=None
# result = pd.read_csv(path, sep=",", usecols=(0,1,2, 3, 4, 5, 6, 7))# , header=None
result = result.round({'Mean IoU': 3, 'Dropout': 3, 'Elastic proportion': 3})
result['Learning rate'] = result['Learning rate'].map('{:.3e}'.format)
# result['Whitening'] = result['Whitening'].map('{:.3e}'.format)

if __name__ == '__main__':
    print(result.to_latex(index=False))
