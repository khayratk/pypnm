import errno
import os

import numpy as np


def all_unique(x):
    seen = set()
    return not any(i in seen or seen.add(i) for i in x)


def sum_by_group(data, groups):
    """
    Sum array by grouping

    Taken from http://stackoverflow.com/a/8732260/2587757


    Parameters
    ----------
    data : ndarray of size N
        Data to be summed up
    groups : ndarray of size N
        Array identifying the corresponding group of the values in 'data'
    """
    order = np.argsort(groups)
    groups = groups[order]
    data = data[order]
    data.cumsum(out=data)
    index = np.ones(len(groups), 'bool')
    index[:-1] = groups[1:] != groups[:-1]
    data = data[index]
    groups = groups[index]
    data[1:] = data[1:] - data[:-1]
    return data, groups


def require_path(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise