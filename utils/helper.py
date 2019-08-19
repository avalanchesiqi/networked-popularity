import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

DATEFORMAT = {'tweet': '%a %b %d %H:%M:%S %z %Y',
              'youtube': '%Y-%m-%d'}


class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        print('>>> Elapsed time: {0}\n'.format(str(timedelta(seconds=time.time() - self.start_time))[:-3]))


def str2obj(str, fmt='youtube'):
    if fmt == 'tweet' or fmt == 'youtube':
        return datetime.strptime(str, DATEFORMAT[fmt])
    else:
        return datetime.strptime(str, fmt)


def obj2str(obj, fmt='youtube'):
    if fmt == 'tweet' or fmt == 'youtube':
        return obj.strftime(DATEFORMAT[fmt])
    else:
        return obj.strftime(fmt)


def strify(arr, delimiter=',', decimal=0):
    if decimal == 0:
        return delimiter.join(map(lambda x: '{0:.0f}'.format(x), arr))
    else:
        return delimiter.join(map(lambda x: '{0:.4f}'.format(x), arr))


def intify(arr):
    return list(map(int, arr))


def is_persistent_link(lst):
    n = len(lst)
    if sum(lst[:4]) < 2:
        return False
    elif sum(lst[:5]) < 3:
        return False
    elif sum(lst[-4:]) < 2:
        return False
    elif sum(lst[-5:]) < 3:
        return False
    else:
        for i in range(3, n-4):
            if sum(lst[i-3: i+4]) < 4:
                return False
    return True


def is_same_genre(lst1, lst2):
    if len(lst1) == 0 or len(lst2) == 0:
        return False
    for i in lst1:
        if i in lst2:
            return True
    return False


def gini(x, w=None):
    """ Compute the Gini coefficient given a list x.
    """
    # array indexing requires reset indexes
    x = pd.Series(x).reset_index(drop=True)
    if w is None:
        w = np.ones_like(x)
    w = pd.Series(w).reset_index(drop=True)
    n = x.size
    wxsum = sum(w * x)
    wsum = sum(w)
    sxw = np.argsort(x)
    sx = x[sxw] * w[sxw]
    sw = w[sxw]
    pxi = np.cumsum(sx) / wxsum
    pci = np.cumsum(sw) / wsum
    g = 0.0
    for i in np.arange(1, n):
        g = g + pxi.iloc[i] * pci.iloc[i - 1] - pci.iloc[i] * pxi.iloc[i - 1]
    return g
