import cProfile
import pstats

from pypnm.attribute_calculators.conductance_calc import ConductanceCalc
from pypnm.linalg.laplacianmatrix import LaplacianMatrix
from pypnm.porenetwork.network_factory import *
from pypnm.util.igraph_utils import scipy_matrix_to_igraph


def test_scipy_to_igraph():
    nx = 30
    ny = 30
    nz = 30
    network = structured_network(Nx=nx, Ny=ny, Nz=nz)
    k_computer = ConductanceCalc(network)
    k_computer.compute()

    A = LaplacianMatrix(network)
    A.set_edge_weights(network.tubes.k_w)

    scipy_matrix_to_igraph(A.get_csr_matrix())


def print_profiling_info(filename):
    p = pstats.Stats(filename)
    p.sort_stats('cumulative').print_stats(20)
    p.sort_stats('time').print_stats(20)

if __name__ == "__main__":
    exec_string = 'test_scipy_to_igraph()'

    cProfile.run(exec_string, 'restats')
    print_profiling_info('restats')

