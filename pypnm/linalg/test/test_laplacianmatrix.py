import igraph as ig

import numpy as np

from pypnm.linalg.laplacianmatrix import *
from pypnm.linalg.laplacianmatrix import laplacian_from_igraph
from pypnm.porenetwork.network_factory import cube_network


def test_laplacianmatrix():
    network = cube_network(N=20)
    laplacian = laplacian_from_network(network, weights=network.tubes.r)


def test_laplacian_from_igraph():
    """
    Test creation of _laplacian
    """
    graph = ig.Graph.GRG(1000, 0.1)

    # Test wrong length as argument
    try:
        random_weights = np.random.rand(graph.vcount())
        laplacian_from_igraph(graph, random_weights)
    except ValueError:
        pass

    # Test without dirichlet boundaries
    random_weights = np.random.rand(graph.ecount())
    A = laplacian_from_igraph(graph, random_weights)

    sum_cols_of_A = A*np.ones(graph.vcount())

    assert np.allclose(sum_cols_of_A,  0.0, rtol=1e-10)

    # Test with random number of dirichlet boundaries
    num_of_dirichlet = np.random.randint(0, graph.vcount())
    ind_dirichlet = np.random.choice(graph.vcount(), num_of_dirichlet, replace=False)
    random_weights = np.random.rand(graph.ecount())
    A = laplacian_from_igraph(graph, weights=random_weights, ind_dirichlet=ind_dirichlet)

    sum_cols_of_A = A*np.ones(graph.vcount())

    assert np.isclose(np.sum(sum_cols_of_A),  num_of_dirichlet, rtol=1e-10)