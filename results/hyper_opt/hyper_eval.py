import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


if __name__ == '__main__':
    path ='unet_1_channels_old.csv'
    result1 = pd.read_csv(path, sep=",", usecols=(1,2))
    path ='multi_1_channels.csv'
    result2 = pd.read_csv(path, sep=",", usecols=(1,2))
    path ='multiresunet_5_channels.csv'
    result3 = pd.read_csv(path, sep=",", usecols=(1,2))
    path ='multiresunet_3_channels.csv'
    result4 = pd.read_csv(path, sep=",", usecols=(1,2))
    path ='nestnet_1_channels.csv'
    result5 = pd.read_csv(path, sep=",", usecols=(1,2))
    path ='nestnet_3_channels.csv'
    result6 = pd.read_csv(path, sep=",", usecols=(1,2))

    sns.lineplot(x = 'Iteration', y = 'Mean IoU', data = result1, label = 'U1')
    sns.lineplot(x = 'Iteration', y = 'Mean IoU', data = result2, label = 'M1')
    sns.lineplot(x = 'Iteration', y = 'Mean IoU', data = result3, label = 'M3')
    # sns.lineplot(x = 'Iteration', y = 'Mean IoU', data = result4, label = 'M5')
    # sns.lineplot(x = 'Iteration', y = 'Mean IoU', data = result5, label = 'N1')
    # sns.lineplot(x = 'Iteration', y = 'Mean IoU', data = result6, label = 'N5')
    plt.show()
