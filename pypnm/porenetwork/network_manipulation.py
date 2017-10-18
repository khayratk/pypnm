import numpy as np

from pypnm.percolation import graph_algs
from pypnm.porenetwork.component import tubes_within_pore_set
from pypnm.porenetwork.subnetwork import SubNetwork
from pypnm.util.bounding_box import BoundingBox
from pypnm.porenetwork.constants import FACES
from pypnm.linalg.laplacianmatrix import laplacian_from_network
from pypnm.linalg.petsc_interface import scipy_to_petsc_matrix

import petsc4py
import sys

petsc4py.init(sys.argv)
from petsc4py import PETSc


def reorder_network(network):
    A = laplacian_from_network(network)
    A = scipy_to_petsc_matrix(A)
    a_is, b_is = A.getOrdering(PETSc.Mat.OrderingType.RCM)
    a = np.asarray(a_is.getIndices())
    b = np.asarray(b_is.getIndices())
    assert np.all(a == b)
    permuted_network = SubNetwork(network, a)
    return permuted_network


def prune_network(network, bounding_percent=(-1, 2, -1, 2, -1, 2)):
    """
    :param network: Network to be pruned
    :param bounding_percent: list of 6 fractions defining the bounding box
    :return: Subnetwork of the network defined by the bounding box, with its dead-ends removed
    """
    pores = network.pores

    bounding_box = BoundingBox(min(pores.x) + (max(pores.x) - min(pores.x)) * bounding_percent[0],
                               min(pores.x) + (max(pores.x) - min(pores.x)) * bounding_percent[1],
                               min(pores.y) + (max(pores.y) - min(pores.y)) * bounding_percent[2],
                               min(pores.y) + (max(pores.y) - min(pores.y)) * bounding_percent[3],
                               min(pores.z) + (max(pores.z) - min(pores.z)) * bounding_percent[4],
                               min(pores.z) + (max(pores.z) - min(pores.z)) * bounding_percent[5])
    network = SubNetwork.from_bounding_box(network, bounding_box)
    pi_biconnected = graph_algs.get_pi_list_biconnected(network)
    network = SubNetwork(network, pi_biconnected)
    assert np.all(network.nr_nghs > 0)

    print "Number of pores", network.nr_p
    print "Number of tubes", network.nr_t
    return network


def remove_tubes_between_face_pores(network, FACE):
    """
    Removes pore throats between pores at a given Face. If, as a result, pores at the face are left isolated in islands,
    they are also removed.

    Parameters
    ----------
    network: PoreNetwork
    FACE: int
        Integer for face defined in pypnm.porenetwork.constants

    Returns
    -------
    out: PoreNetwork
        Porenetwork with tubes between face pores removed

    """

    for _ in xrange(2):
        tubes_at_face = tubes_within_pore_set(network, network.pi_list_face[FACE])
        network.remove_throats(tubes_at_face)
        pores_to_remove_face = np.flatnonzero(network.nr_nghs[network.pi_list_face[FACE]] == 0)
        pores_to_remove = network.pi_list_face[FACE][pores_to_remove_face]

        pores_to_keep = np.setdiff1d(np.arange(network.nr_p), pores_to_remove)
        network = SubNetwork(network, pores_to_keep)
        network = prune_network(network)

    for face in FACES:
        assert len(np.flatnonzero(network.nr_nghs[network.pi_list_face[face]] == 0)) == 0

    return network
