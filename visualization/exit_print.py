import pandas as pd
import tabulate
import sys


def exit_print(list_of_dicts, gpu, network, channels):
    print('\n-')
    max_lines = int(input("Input the number of top results to print: "))
    print('-')
    max_lines = min(len(list_of_dicts), max_lines)
    header = list_of_dicts[0].keys()
    rows =  [list(x.values()) for x in sorted(list_of_dicts, key = lambda m: m['Mean IoU'],reverse=True)[:max_lines]]


    #print(rows)

    test = pd.DataFrame(data = rows, columns = header)
    test.to_csv('./results/{}/{}_{}_channels.csv'.format(gpu, network, channels), index_label = 'Index')

    print('\n\n')
    print(tabulate.tabulate(rows, header, tablefmt="fancy_grid", stralign="right", floatfmt=(".3f", "2d", ".2e", "d", "d", ".3f", ".3f")))
    print('\n\n')
    sys.exit()
