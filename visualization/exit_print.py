import pandas as pd
import tabulate
import sys


def exit_print(list_of_dicts, gpu, network, channels):
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
    #print(rows)

    test = pd.DataFrame(data = rows, columns = header)
    test.index += 1
    test.to_csv('./results/{}/metrics/{}_{}_channels.csv'.format(gpu, network, channels), index_label = 'Index')

    print('\n\n')
    print(tabulate.tabulate(test, headers='keys', tablefmt="fancy_grid", stralign="right", floatfmt=("d", "d", ".3f", "d", ".3e", "s", ".3e", "d", ".3f", ".3f")))
    print('\n\n')
    sys.exit()
