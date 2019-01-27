from collections import defaultdict
from itertools import product

import numpy as np
from scipy import spatial
from scipy.interpolate import RegularGridInterpolator


def _neighboring_indices(num_partitions, index):
    i, j, k = index
    nx, ny, nz = num_partitions
    neighbors = list()
    for di, dj, dk in product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]):
        i_ngh, j_ngh, k_ngh = i + di, j + dj, k + dk
        if (i, j, k) == (i_ngh, j_ngh, k_ngh):
            continue
        if i_ngh >= nx or j_ngh >= ny or k_ngh >= nz:
            continue
        if i_ngh < 0 or j_ngh < 0 or k_ngh < 0:
            continue
        neighbors.append((i_ngh, j_ngh, k_ngh))
    return neighbors


def sphere_packing(rad, domain_size, is_2d=False):
    """
    Computes a non-overlapping random sphere packing

    Parameters
    ----------
    rad: ndarray
        Array of sphere radii

    domain_size: 3-tuple
        domain size in x, y and z directions

    is_2d: bool (optional)
        if True, then z-coordinate of all spheres will be 0

    Returns
    -------
    x: ndarray
        Array of sphere x-coords

    y: ndarray
        Array of sphere y-coords

    z: ndarray
        Array of sphere z-coords

    r: ndarray
        Array of sphere radii
    """
    rad = np.sort(rad)
    rad[:] = rad[::-1]

    cell_to_sphere_ids = defaultdict(list)
    ngh_indices_map = dict()

    len_x = domain_size[0]
    len_y = domain_size[1]
    len_z = domain_size[2]

    dimensions = np.asarray([len_x, len_y, len_z])

    nr_pores = len(rad)
    r_max = np.max(rad)

    if is_2d:
        len_z = r_max

    coords = np.zeros([nr_pores, 3])

    max_r = np.max(rad)
    delta_h = 2.0 * max_r
    nx, ny, nz = int(len_x / delta_h), int(len_y / delta_h), max(int(len_z / delta_h), 1)

    for index in product(xrange(nx), xrange(ny), xrange(nz)):
        ngh_indices_map[index] = _neighboring_indices(num_partitions=(nx, ny, nz), index=index)

    for i in xrange(nr_pores):
        counter = 0
        while True:
            new_coord = np.random.rand(3) * dimensions
            if is_2d:
                new_coord[2] = 0.0
            index = int(new_coord[0] * nx / len_x), int(new_coord[1] * ny / len_y), int(new_coord[2] * nz / len_z)
            nghs = ngh_indices_map[index]
            overlap = False

            for ngh in [index] + nghs:
                if overlap:
                    break
                for sphere_id in cell_to_sphere_ids[ngh]:
                    overlap = (sum((coords[sphere_id] - new_coord) ** 2) - (rad[sphere_id] + rad[i]) ** 2) < 0
                    if overlap:
                        break

            if not overlap:
                cell_to_sphere_ids[index].append(i)
                coords[i][:] = new_coord
                break

            counter += 1
            if counter == 1000:
                raise RuntimeError("Sphere packing stuck")

    points = zip(coords[:, 0], coords[:, 1], coords[:, 2])
    tree = spatial.cKDTree(points)

    for i in xrange(nr_pores):
        dist, i_nghs = tree.query((coords[i, 0], coords[i, 1], coords[i, 2]), k=20)
        for local_index, i_ngh in enumerate(i_nghs):
            if i_ngh == i:
                continue
            assert dist[local_index] > (rad[i] + rad[i_ngh])

    return coords[:, 0], coords[:, 1], coords[:, 2], rad


def sphere_packing_from_field(nr_pores, field, domain_size):
    """
    Computes a non-overlapping random sphere packing given a field representing the size of a sphere at each location

    Parameters
    ----------
    nr_pores: int
        Number of spheres

    field: ndarray
        3 - Dimensional array representing a field. The location of the values in field are inferred from domain_size.
        It is assumed that the points in the field are arranged in a regular lattice.

    domain_size: 3-tuple
        domain size in x, y and z directions


    Returns
    -------
    x: ndarray
        Array of sphere x-coords

    y: ndarray
        Array of sphere y-coords

    z: ndarray
        Array of sphere z-coords

    r: ndarray
        Array of sphere radii
    """

    cell_to_sphere_ids = defaultdict(list)
    ngh_indices_map = dict()

    len_x = domain_size[0]
    len_y = domain_size[1]
    len_z = domain_size[2]

    x = np.linspace(0, len_x, field.shape[0])
    y = np.linspace(0, len_y, field.shape[1])
    z = np.linspace(0, len_z, field.shape[2])
    my_interpolating_function = RegularGridInterpolator((x, y, z), field, method="nearest")

    dimensions = np.asarray([len_x, len_y, len_z])
    coords = np.zeros([nr_pores, 3])

    max_r = np.max(field)
    delta_h = 2.0 * max_r
    nx, ny, nz = int(len_x / delta_h), int(len_y / delta_h), max(int(len_z / delta_h), 1)

    for index in product(xrange(nx), xrange(ny), xrange(nz)):
        ngh_indices_map[index] = _neighboring_indices(num_partitions=(nx, ny, nz), index=index)

    rad = np.zeros(nr_pores)

    for i in xrange(nr_pores):
        counter = 0
        if i % 10000 == 0:
            print i, np.sum(4 / 3. * rad ** 3 * np.pi) / (len_x * len_y * len_z)
        while True:
            new_coord = np.random.rand(3) * dimensions
            index = int(new_coord[0] * nx / len_x), int(new_coord[1] * ny / len_y), int(new_coord[2] * nz / len_z)
            nghs = ngh_indices_map[index]
            overlap = False

            rad_current = my_interpolating_function(new_coord)

            for ngh in [index] + nghs:
                for sphere_id in cell_to_sphere_ids[ngh]:
                    overlap = (sum((coords[sphere_id] - new_coord) ** 2) - (rad[sphere_id] + rad_current) ** 2) < 0
                    if overlap:
                        break
                if overlap:
                    break

            if not overlap:
                cell_to_sphere_ids[index].append(i)
                coords[i][:] = new_coord
                rad[i] = rad_current
                break

            counter += 1
            if counter == 1000:
                raise RuntimeError("Sphere packing stuck")

    rad = np.asarray(rad)
    points = zip(coords[:, 0], coords[:, 1], coords[:, 2])
    tree = spatial.cKDTree(points)

    for i in xrange(nr_pores):
        dist, i_nghs = tree.query((coords[i, 0], coords[i, 1], coords[i, 2]), k=20)
        for local_index, i_ngh in enumerate(i_nghs):
            if i_ngh == i:
                continue
            assert dist[local_index] > (rad[i] + rad[i_ngh])

    return coords[:, 0], coords[:, 1], coords[:, 2], rad
