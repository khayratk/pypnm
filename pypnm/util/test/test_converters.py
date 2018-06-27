import cProfile
import pstats

from pypnm.linalg.laplacianmatrix import laplacian_from_network
from pypnm.porenetwork.network_factory import *
from pypnm.util.igraph_utils import scipy_matrix_to_igraph


def test_scipy_to_igraph():
    network = structured_network(Nx=30, Ny=30, Nz=30)
    A = laplacian_from_network(network)
    scipy_matrix_to_igraph(A)


def print_profiling_info(filename):
    p = pstats.Stats(filename)
    p.sort_stats('cumulative').print_stats(20)
    p.sort_stats('time').print_stats(20)


if __name__ == "__main__":
    exec_string = 'test_scipy_to_igraph()'

    cProfile.run(exec_string, 'restats')
    print_profiling_info('restats')

