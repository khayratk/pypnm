import igraph as ig

from pypnm.linalg.laplacianmatrix import *
from pypnm.linalg.laplacianmatrix import laplacian_from_igraph, flow_matrix_from_graph
from pypnm.porenetwork.network_factory import cube_network


def test_flow_matrix():
    g = ig.Graph()
    g.add_vertices(5)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(4, 2)

    g.es["cond"] = [1, 1, 1, 1]
    g.vs["p"] = [10, 5, 4, 1, 10]

    A = flow_matrix_from_graph(g, g.vs["p"], g.es["cond"], ind_dirichlet=[3], val_dirichlet=0.0)

    return A


def test_laplacianmatrix():
    network = cube_network(N=20)
    A = laplacian_from_network(network, weights=network.tubes.r)
    sum_cols = A * np.ones(A.shape[0])

    assert np.allclose(sum_cols, 0.0, rtol=1e-10)


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

    sum_cols = A * np.ones(graph.vcount())

    assert np.allclose(sum_cols, 0.0, rtol=1e-10)

    # Test with random number of dirichlet boundaries
    num_of_dirichlet = np.random.randint(0, graph.vcount())
    ind_dirichlet = np.random.choice(graph.vcount(), num_of_dirichlet, replace=False)
    random_weights = np.random.rand(graph.ecount())
    A = laplacian_from_igraph(graph, weights=random_weights, ind_dirichlet=ind_dirichlet)

    sum_cols = A * np.ones(graph.vcount())

    assert np.isclose(np.sum(sum_cols), num_of_dirichlet, rtol=1e-10)
