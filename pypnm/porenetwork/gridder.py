import itertools
from itertools import product

import numpy as np

from pypnm.porenetwork import component
from pypnm.util.bounding_box import BoundingBox


def grid3d_of_bounding_boxes(network, nx, ny, nz):
    min_x, max_x = np.min(network.pores.x), np.max(network.pores.x)
    min_y, max_y = np.min(network.pores.y), np.max(network.pores.y)
    min_z, max_z = np.min(network.pores.z), np.max(network.pores.z)

    len_x = max_x - min_x
    len_y = max_y - min_y
    len_z = max_z - min_z

    # Coordinates of all nodes in the set of bounding boxes
    x_coords = min_x + np.linspace(start=0.0, stop=len_x, num=nx+1, endpoint=True)
    y_coords = min_y + np.linspace(start=0.0, stop=len_y, num=ny+1, endpoint=True)
    z_coords = min_z + np.linspace(start=0.0, stop=len_z, num=nz+1, endpoint=True)

    # If a bounding box coincides with any pore body center, then shift appropriate face by eps
    pores = network.pores
    eps = 1e-5 * max(len_x, len_y, len_z) / network.nr_p

    for i in xrange(nx+1):
        if np.any(np.isclose(x_coords[i], pores.x, atol=len_x/1e-5)):
            if i == 0:
                x_coords[i] -= eps
            else:
                x_coords[i] += eps

    for j in xrange(ny+1):
        if np.any(np.isclose(y_coords[j], pores.y, atol=len_y/1e-5)):
            if j == 0:
                y_coords[j] -= eps
            else:
                y_coords[j] += eps

    for k in xrange(nz+1):
        if np.any(np.isclose(z_coords[k], pores.z, atol=len_z/1e-5)):
            if k == 0:
                z_coords[k] -= eps
            else:
                z_coords[k] += eps

    # Create bounding_boxes
    bounding_boxes = {}
    for i, j, k in product(xrange(nx), xrange(ny), xrange(nz)):
        bounding_boxes[i, j, k] = BoundingBox(x_coords[i], x_coords[i+1],
                                              y_coords[j], y_coords[j+1],
                                              z_coords[k], z_coords[k+1])

    return bounding_boxes


def grid3d_of_pore_lists(network, nx, ny, nz):
    """
    Parameters
    __________
    network: Pore Network
            Pore network to be partitioned
    nx, ny, nz: int
        number of coarse cells in x, y and z directions

    Returns
    -------
    partition : dictionary
        dictionary containing list of pore indices for each subnetwork indexed by the subnetwork's i, j, k coordinate.
    """
    pi_lists = {}

    bounding_boxes = grid3d_of_bounding_boxes(network, nx, ny, nz)
    for i, j, k in product(xrange(nx), xrange(ny), xrange(nz)):
        pi_lists[i, j, k] = component.pore_list_from_bbox(network, bounding_boxes[i, j, k])

    all_pores = np.zeros(0, dtype=np.int)
    for i, j, k in itertools.product(xrange(nx), xrange(ny), xrange(nz)):
        all_pores = np.union1d(all_pores, pi_lists[i, j, k])
    assert len(all_pores) == network.nr_p
    return pi_lists
