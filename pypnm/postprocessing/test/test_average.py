import numpy as np
from pypnm.postprocessing import spatial_average


def test_spatial_average():
    x = np.random.rand(1000)
    field = np.random.rand(1000)
    size = x.max() - x.min()
    window_size = size / 10.
    num_points = 20
    #Without weights
    avg_data, avg_pos = spatial_average.moving_window_average_1d(x, field, window_size, num_points, weights=None)
    assert len(avg_data) == num_points

    #With weights
    avg_data, avg_pos = spatial_average.moving_window_average_1d(x, field, window_size, num_points, weights=field)
    assert len(avg_data) == num_points

if __name__ == "__main__":
    test_spatial_average()
