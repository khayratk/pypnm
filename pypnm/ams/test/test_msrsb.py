import cProfile
import pstats

import numpy as np

from pypnm.ams.msrb_funcs import solve_with_msrsb
from pypnm.attribute_calculators.conductance_calc import ConductanceCalc
from pypnm.linalg.laplacianmatrix import LaplacianMatrix
from pypnm.porenetwork.network_factory import structured_network
from pypnm.porenetwork.network_factory import unstructured_network_delaunay
from pypnm.util.utils_testing import run_ip_algorithm
from scipy.sparse.linalg import spsolve

sim_settings = dict()
sim_settings["fluid_properties"] = dict()
sim_settings["fluid_properties"]['mu_n'] = 0.001
sim_settings["fluid_properties"]['mu_w'] = 1.0
sim_settings["fluid_properties"]['gamma'] = 1.0


def test_msrsb_two_phase():
    nx, ny, nz = 3, 7, 11
    n_fine_per_cell = 5
    network = structured_network(Nx=n_fine_per_cell * nx, Ny=n_fine_per_cell * ny, Nz=n_fine_per_cell * nz)
    run_ip_algorithm(network, 0.7)
    p_c = network.pores.p_c
    assert np.sum(p_c) > 0.0

    k_computer = ConductanceCalc(network, sim_settings["fluid_properties"], pores_have_conductance=True)
    k_computer.compute()

    assert np.sum(network.tubes.k_n) > 0.0

    A = LaplacianMatrix(network)
    A.set_edge_weights(network.tubes.k_w + network.tubes.k_n)
    A = A.get_csr_matrix()

    An = LaplacianMatrix(network)
    An.set_edge_weights(network.tubes.k_n)
    An = An.get_csr_matrix()

    k_tot = network.tubes.k_n+network.tubes.k_w
    print "max/min total conductance ratio is ", max(k_tot)/min(k_tot)

    rhs = np.zeros(network.nr_p)
    rhs[0] = 1e-10
    rhs[100] = -1e-10
    rhs += -An * p_c

    sol, _, _ = solve_with_msrsb(A.indices, A.indptr, A.data, rhs, tol=1e-12)

    A[1, :] = 0.0
    A[1, 1] = 1.0

    sol_exact = spsolve(A=A, b=rhs)

    sol -= np.min(sol)
    sol_exact -= np.min(sol_exact)
    error = sol - sol_exact

    Linf = np.linalg.norm(error, ord=np.inf) / np.linalg.norm(sol_exact, ord=np.inf)
    L2_error = np.linalg.norm(error, ord=2) / np.linalg.norm(sol_exact, ord=2)
    assert ((Linf < 10e-2) & (L2_error < 10e-5)), "L1 error: %s  L2 error: %s" % (Linf, L2_error)


def test_msrsb_unstructured():
    network = unstructured_network_delaunay(nr_pores=10000)
    k_computer = ConductanceCalc(network, sim_settings["fluid_properties"], pores_have_conductance=True)
    k_computer.compute()

    A = LaplacianMatrix(network)
    A.set_edge_weights(network.tubes.k_w + network.tubes.k_n)
    A = A.get_csr_matrix()
    rhs = np.zeros(network.nr_p)
    rhs[0] = 1e-10
    rhs[100] = -1e-10
    sol, _, _ = solve_with_msrsb(A.indices, A.indptr, A.data, rhs, tol=1e-12)

    A[1, :] = 0.0
    A[1, 1] = 1.0
    sol_exact = spsolve(A=A, b=rhs)

    sol -= np.min(sol)
    sol_exact -= np.min(sol_exact)
    error = sol - sol_exact

    Linf = np.linalg.norm(error, ord=np.inf) / np.linalg.norm(sol_exact, ord=np.inf)
    L2_error = np.linalg.norm(error, ord=2) / np.linalg.norm(sol_exact, ord=2)
    assert ((Linf < 10e-2) & (L2_error < 10e-5)), "L1 error: %s  L2 error: %s" % (Linf, L2_error)


def test_msrsb_structured():
    nx, ny, nz = 3, 7, 11
    n_fine_per_cell = 5
    network = structured_network(Nx=n_fine_per_cell * nx, Ny=n_fine_per_cell * ny, Nz=n_fine_per_cell * nz)
    k_computer = ConductanceCalc(network, sim_settings["fluid_properties"], pores_have_conductance=True)
    k_computer.compute()

    A = LaplacianMatrix(network)
    A.set_edge_weights(network.tubes.k_w + network.tubes.k_n)
    A = A.get_csr_matrix()
    rhs = np.zeros(network.nr_p)
    rhs[0] = 1e-10
    rhs[100] = -1e-10
    sol, _, _ = solve_with_msrsb(A.indices, A.indptr, A.data, rhs, tol=1e-12)

    A[1, :] = 0.0
    A[1, 1] = 1.0
    sol_exact = spsolve(A=A, b=rhs)

    sol -= np.min(sol)
    sol_exact -= np.min(sol_exact)
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
