import cProfile
import pstats

import numpy as np

from pypnm.ams.msrb_funcs import solve_with_msrsb
from pypnm.attribute_calculators.conductance_calc import ConductanceCalc
from pypnm.linalg.laplacianmatrix import LaplacianMatrix
from pypnm.linalg.linear_system_solver import solve_pyamg
from pypnm.porenetwork.network_factory import structured_network
from pypnm.porenetwork.network_factory import unstructured_network


def test_msrsb_unstructured():
    network = unstructured_network(nr_pores=10000)
    k_computer = ConductanceCalc(network)
    k_computer.compute()

    A = LaplacianMatrix(network)
    A.set_edge_weights(network.tubes.k_w + network.tubes.k_n)
    A = A.get_csr_matrix()
    rhs = np.zeros(network.nr_p)
    rhs[0] = 1e-10
    rhs[100] = -1e-10
    sol = solve_with_msrsb(A.indices, A.indptr, A.data, rhs)

    A[1, :] = 0.0
    A[1, 1] = 1.0
    sol_exact = solve_pyamg(A, rhs, tol=1e-10)

    sol = (sol - np.min(sol))
    sol_exact = (sol_exact-np.min(sol_exact))
    error = sol - sol_exact

    Linf = np.linalg.norm(error, ord=np.inf) / np.linalg.norm(sol_exact, ord=np.inf)
    L2_error = np.linalg.norm(error, ord=2) / np.linalg.norm(sol_exact, ord=2)
    assert ((Linf < 10e-2) & (L2_error < 10e-5)), "L1 error: %s  L2 error: %s" % (Linf, L2_error)


def test_msrsb_structured():
    nx, ny, nz = 3, 7, 11
    n_fine_per_cell = 5
    network = structured_network(Nx=n_fine_per_cell * nx, Ny=n_fine_per_cell * ny, Nz=n_fine_per_cell * nz)
    k_computer = ConductanceCalc(network)
    k_computer.compute()

    A = LaplacianMatrix(network)
    A.set_edge_weights(network.tubes.k_w + network.tubes.k_n)
    A = A.get_csr_matrix()
    rhs = np.zeros(network.nr_p)
    rhs[0] = 1e-10
    rhs[100] = -1e-10
    sol = solve_with_msrsb(A.indices, A.indptr, A.data, rhs)

    A[1, :] = 0.0
    A[1, 1] = 1.0
    sol_exact = solve_pyamg(A, rhs, tol=1e-10)

    sol = (sol - np.min(sol))
    sol_exact = (sol_exact-np.min(sol_exact))
    error = sol - sol_exact

    Linf = np.linalg.norm(error, ord=np.inf) / np.linalg.norm(sol_exact, ord=np.inf)
    L2_error = np.linalg.norm(error, ord=2) / np.linalg.norm(sol_exact, ord=2)
    assert ((Linf < 10e-2) & (L2_error < 10e-5)), "L1 error: %s  L2 error: %s" % (Linf, L2_error)


def print_profiling_info(filename):
    p = pstats.Stats(filename)
    p.sort_stats('cumulative').print_stats(20)
    p.sort_stats('time').print_stats(20)


if __name__ == "__main__":
    cProfile.run('test_msrsb()', 'restats')
    print_profiling_info('restats')
