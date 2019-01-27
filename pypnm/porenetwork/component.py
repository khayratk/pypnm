import itertools
import operator

import numpy as np


def element_mask_bbox(elements, bounding_box):
    element_mask = elements.x[:] > bounding_box.xmin
    element_mask = np.logical_and(elements.x < bounding_box.xmax, element_mask)
    element_mask = np.logical_and(elements.y > bounding_box.ymin, element_mask)
    element_mask = np.logical_and(elements.y < bounding_box.ymax, element_mask)
    element_mask = np.logical_and(elements.z > bounding_box.zmin, element_mask)
    element_mask = np.logical_and(elements.z < bounding_box.zmax, element_mask)
    return element_mask


def pore_mask_bbox(network, bounding_box):
    return element_mask_bbox(network.pores, bounding_box)


def pore_mask_outside_bbox(network, bounding_box):
    return np.logical_not(pore_mask_bbox(network, bounding_box))


def tube_mask_bbox(network, bounding_box):
    return element_mask_bbox(network.tubes, bounding_box)


def tube_mask_outside_bbox(network, bounding_box):
    return np.logical_not(tube_mask_bbox(network, bounding_box))


def tube_list_outside_bbox(network, bounding_box):
    return tube_mask_outside_bbox(network, bounding_box).nonzero()[0]


def pore_list_from_bbox(network, bounding_box):
    pore_mask = pore_mask_bbox(network, bounding_box)
    return pore_mask.nonzero()[0]


def pore_list_ngh_to_pore_list(network, pi_list):
    """
    Returns a list of pores which are adjacent to pores in pi_list. This list may include repetitions when
    pores in pi_list have common neighboring pores.

    Parameters
    ----------
    network: PoreNetwork
    pi_list: list
        list of pores to fine

    Returns
    -------
    ngh_pores_list: ndarray
        Index array of pores adjacent to list of pores.

    Notes
    -------
    The order of the pore indices in the returned list corresponds to pi_list.
    """
    if len(pi_list) == 0:
        return []

    temp_marker = np.zeros(network.nr_p, dtype=np.int32)
    temp_marker[pi_list] = 1

    ngh_pores_list = np.ones([len(pi_list)], dtype=np.object)

    for s, pi in enumerate(pi_list):
        ngh_pores_list[s] = network.ngh_pores[pi][temp_marker[network.ngh_pores[pi]] == 0]

    ngh_pores_list = np.hstack(ngh_pores_list)

    return ngh_pores_list


def tube_list_ngh_to_pore_list(network, pi_list):
    """
    Returns a list of tubes which are adjacent to pores in pi_list. This list may include repetitions when
    pores in pi_list have common neighboring tubes.

    Parameters
    ----------
    network: PoreNetwork
    pi_list: list
        list of pores to fine


    Returns
    -------
    ngh_tubes_list: ndarray
        Index array of tubes adjacent to list of pores.

    """
    temp_marker = np.zeros(network.nr_p, dtype=np.int32)
    temp_marker[pi_list] = 1

    ngh_tubes_list = np.ones([len(pi_list)], dtype=np.object)

    for s, pi in enumerate(pi_list):
        ngh_tubes_list[s] = network.ngh_tubes[pi][temp_marker[network.ngh_pores[pi]] == 0]

    ngh_tubes_list = np.hstack(ngh_tubes_list)

    return ngh_tubes_list


def neighboring_edges_to_vertices(network, vlist):
    if len(vlist) == 0:
        return []

    ngh_tubes = [network.ngh_tubes[v] for v in vlist]
    ngh_tubes = np.asarray(list(itertools.chain(*ngh_tubes)), dtype=np.int32)
    ngh_tubes = np.unique(ngh_tubes)
    return ngh_tubes


def __tube_list_plane(network, pore_coordinates, coordinate):
    edgelist = network.edgelist
    pi_list1 = edgelist[:, 0]
    pi_list2 = edgelist[:, 1]
    mask = (pore_coordinates[pi_list1] < coordinate) & (pore_coordinates[pi_list2] > coordinate) | \
           (pore_coordinates[pi_list1] > coordinate) & (pore_coordinates[pi_list2] < coordinate)

    return mask.nonzero()[0]


def tube_list_x_plane(network, x_pos):
    return __tube_list_plane(network, network.pores.x, x_pos)


def tube_list_y_plane(network, y_pos):
    return __tube_list_plane(network, network.pores.y, y_pos)


def tube_list_z_plane(network, z_pos):
    return __tube_list_plane(network, network.pores.z, z_pos)


def __filtered_pi_list(list_1, list_2, array, comparison):
    first_pore_mask = comparison(array[list_1], array[list_2])
    second_pore_mask = comparison(array[list_2], array[list_1])
    return list_1*first_pore_mask + list_2*second_pore_mask


def pore_list_ngh_of_tubes(network, ti_list=None):
    if ti_list is not None:
        pi_list_1 = network.edgelist[ti_list, 0]
        pi_list_2 = network.edgelist[ti_list, 1]
    else:
        pi_list_1 = network.edgelist[:, 0]
        pi_list_2 = network.edgelist[:, 1]

    return pi_list_1, pi_list_2


def __pore_list_plane(network, pore_coordinates, plane_coordinate, relation_to_coordinate):
    ti_list = __tube_list_plane(network, pore_coordinates, plane_coordinate)
    pi_list_1, pi_list_2 = pore_list_ngh_of_tubes(network, ti_list)
    return __filtered_pi_list(pi_list_1, pi_list_2, pore_coordinates, relation_to_coordinate)


def pore_list_x_plane_plus(network, x_pos):
    return __pore_list_plane(network, network.pores.x, x_pos, operator.gt)


def pore_list_y_plane_plus(network, y_pos):
    return __pore_list_plane(network, network.pores.y, y_pos, operator.gt)


def pore_list_z_plane_plus(network, z_pos):
    return __pore_list_plane(network, network.pores.z, z_pos, operator.gt)


def complement_pore_set(network, pi_list):
    return np.setdiff1d(np.arange(network.nr_p,dtype=int), pi_list)


def complement_tube_set(network, ti_list):
    return np.setdiff1d(np.arange(network.nr_t, dtype=np.int), ti_list)


def tubes_from_to_pore_sets(network, pi_list1, pi_list2):
    """
    Parameters
    ----------
    network: PoreNetwork
    pi_list1: list
        source pore indices
    pi_list2: list
        destination pore indices

    Returns
    -------
    list of tube indices where each tube which have a pore in pi_list1 as its first neighbor
    and in pi_list2 as its second neigbhor.

    Notes
    ______
    Even though there a tube is undirected, this is needed to compute the flux from a list of pores to other pores.

    """

    mask_pores_1 = np.zeros(network.nr_p, dtype=np.bool)
    mask_pores_2 = np.zeros(network.nr_p, dtype=np.bool)

    mask_pores_1[pi_list1] = 1
    mask_pores_2[pi_list2] = 1

    ngh_pores_1, ngh_pores_2 = pore_list_ngh_of_tubes(network)

    mask_tubes = mask_pores_1[ngh_pores_1] & mask_pores_2[ngh_pores_2]

    ti_list = mask_tubes.nonzero()[0]

    return ti_list


def tubes_within_pore_set(network, pi_list):
    """
    Index array of tubes having both neighboring pores in pi_list

    Parameters
    ----------
    network: PoreNetwork
    pi_list: list
        list of pore indices

    Returns
    -------
    ti_list: ndarray
        Index array of tubes having both neighboring pores in pi_list

    """
    mask_pores = np.zeros(network.nr_p, dtype=np.bool)

    mask_pores[pi_list] = 1
    ngh_pores_1 = network.edgelist[:, 0]
    ngh_pores_2 = network.edgelist[:, 1]

    mask_tubes = np.logical_and(mask_pores[ngh_pores_1], mask_pores[ngh_pores_2])

    ti_list = mask_tubes.nonzero()[0]

    return ti_list
