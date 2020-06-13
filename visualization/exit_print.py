import pandas as pd
import tabulate
import sys


def exit_print(list_of_dicts, gpu, network, channels):
    '''
    Both prints in terminal, and saves results to csv-file. Programs asks user to
    of numbers of results to print in terminal. If list_of_dicts is empty, this
    is reported.

    Input:
        list_of_dicts (dict): dictionary of results.
        gpu (int: 0 or 1): which gpu, XP = 0 ,V = 1.
        network (str): network name.
        channels (int): number of channels in current network.
    Output:
        print of result.
        res (csv): file of results.
    '''
    print('\n-')
    try:
        max_lines = int(input("Input the number of top results to print: "))
    except:
        max_lines = len(list_of_dicts)
    print('-')
    max_lines = min(len(list_of_dicts), max_lines)
    if max_lines < 1:
        print('Dict of results is empty')
        sys.exit()
    header = list_of_dicts[0].keys()
    rows =  [list(x.values()) for x in sorted(list_of_dicts, key = lambda m: m['Mean IoU'],reverse=True)[:max_lines]]

    res = pd.DataFrame(data = rows, columns = header)
    res.index += 1
    res.to_csv('./results/{}/metrics/{}_{}_channels.csv'.format(gpu, network, channels), index_label = 'Index')

    print('\n\n')
    print(tabulate.tabulate(res, headers='keys', tablefmt="fancy_grid", stralign="right", floatfmt=("d", "d", ".3f", "d", ".3e", "s", ".3e", "d", ".3f", ".3f")))
    print('\n\n')
    sys.exit()
