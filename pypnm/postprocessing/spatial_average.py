import numpy as np


def moving_window_average_1d(pos, field, window_size, num_points, weights=None):
    """
    Parameters
    ----------
    pos: ndarray
        Coordinate field of pores in a given direction (x, y or z) on which the averaging will be made
    field: ndarray
        Corresponding quantity that will be averaged.
    window_size: float
        Width of the averaging window
    num_points: int
        Number of points around which the averaging window will be used to average.
    weights: ndarray
        Weights for averaging. Default is uniform

    Returns
    -------
    Tuple (ndarray, ndarray)
        numpy arrays of averaged quantity and positions of averaging windows
    """

    assert len(pos) == len(field)
    assert window_size > 0.0

    if weights is not None:
        assert len(weights) == len(field)

    eps = 1e-8

    window_centers = np.linspace(np.min(pos)+0.5*window_size - eps, np.max(pos)-0.5*window_size+eps, num=num_points)

    avg = []
    avg_pos = []

    for i in xrange(num_points):
        mask = (pos > (window_centers[i]-0.5*window_size)) & (pos < (window_centers[i]+0.5*window_size))
        truncated_field = field[mask]
        if weights is not None:
            truncated_weights = weights[mask]
        else:
            truncated_weights = None

        avg.append(np.average(truncated_field, weights=truncated_weights))
        avg_pos.append(window_centers[i])

    return np.asarray(avg), np.asarray(avg_pos)
