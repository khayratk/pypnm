import pstats

import numpy as np
import pyamg

from pypnm.ams.msfv import MSFV
from pypnm.ams.wire_basket import create_wire_basket
from pypnm.attribute_calculators.conductance_calc import ConductanceCalc
from pypnm.linalg.laplacianmatrix import LaplacianMatrix
from pypnm.porenetwork import gridder
from pypnm.porenetwork.constants import *
from pypnm.porenetwork.network_factory import structured_network
from pypnm.util.indexing import GridIndexer3D
from pypnm.util.utils_testing import run_ip_algorithm

nx, ny, nz = 5, 3, 3
n_fine_per_cell = 6


def cartesion_subgraph_labels(network, nx, ny, nz):
    gi_subnet_primal = GridIndexer3D(nx, ny, nz)
    subnetwork_indices = gridder.grid3d_of_pore_lists(network, nx, ny, nz)
    subgraph_ids = np.zeros(network.nr_p, dtype=np.int)
    for index_tuple in subnetwork_indices:
        subgraph_ids[subnetwork_indices[index_tuple]] = gi_subnet_primal.get_index(*index_tuple)
    return subgraph_ids


def network_to_mat(network):
    k_computer = ConductanceCalc(network)
    k_computer.compute()

    A = LaplacianMatrix(network)
    A.set_edge_weights(network.tubes.k_w + network.tubes.k_n)
    A = A.get_csr_matrix()
    return A


def test_msfv_single_phase_neumann():
    network = structured_network(Nx=n_fine_per_cell * nx, Ny=n_fine_per_cell * ny, Nz=n_fine_per_cell * nz)

    subgraph_ids = cartesion_subgraph_labels(network, nx, ny, nz)

    A = network_to_mat(network)

    q_tilde = np.zeros(network.nr_p)
    q_tilde[0] = 1e-10
    q_tilde[network.nr_p - 1] = -1e-10
    wirebasket = create_wire_basket(network, nx, ny, nz)
    amsfv = MSFV(A=A.copy(), source_term=q_tilde, wirebasket=wirebasket, primal_cell_labels=subgraph_ids)

    ml = pyamg.smoothed_aggregation_solver(A, max_coarse=10)
    p_w_exact = ml.solve(b=q_tilde, tol=1.0e-16, accel='gmres')

    for restriction_operator in ["msfv", "msfe"]:
        p_w = amsfv.solve(iterative=True, tol=1e-6, restriction_operator=restriction_operator)

        A[3, :] = 0.0
        A[3, 3] = 1.e-10
        q_tilde[3] = 0

        # Enforce minimum value of zero for comparison
        p_w_exact -= np.min(p_w_exact)
        p_w -= np.min(p_w)
        error = p_w - p_w_exact

        Linf = np.linalg.norm(error, ord=np.inf) / np.linalg.norm(p_w_exact, ord=np.inf)
        L2_error = np.linalg.norm(error, ord=2) / np.linalg.norm(p_w_exact, ord=2)
        print "Linf and L2 errors are: %e, %e" % (Linf, L2_error)
        assert ((Linf < 10e-2) & (L2_error < 10e-5)), "L1 error: %s  L2 error: %s" % (Linf, L2_error)

        print "Restarting"
        p_w = amsfv.solve(iterative=True, tol=1e-8, restriction_operator="msfv", x0=p_w)


def test_msfv_single_phase_dirichlet_large_source():
    network = structured_network(Nx=n_fine_per_cell * nx, Ny=n_fine_per_cell * ny, Nz=n_fine_per_cell * nz)

    subgraph_ids = cartesion_subgraph_labels(network, nx, ny, nz)

    A = network_to_mat(network)

    q_tilde = np.zeros(network.nr_p)

    large_val = np.max(np.abs(A.data)) * 1.e6

    pi_1 = np.argmin(network.pores.x)
    pi_2 = np.argmax(network.pores.x)

    A[pi_1, pi_1] += large_val
    q_tilde[pi_1] = large_val * 2.0e6

    A[pi_2, pi_2] += large_val
    q_tilde[pi_2] = large_val * -4.7e6
    wirebasket = create_wire_basket(network, nx, ny, nz)
    amsfv = MSFV(A=A.copy(), source_term=q_tilde, wirebasket=wirebasket, primal_cell_labels=subgraph_ids)

    ml = pyamg.smoothed_aggregation_solver(A, max_coarse=10)
    p_w_exact = ml.solve(b=q_tilde, tol=1.0e-16, accel='gmres')

    for restriction_operator in ["msfe", "msfv"]:
        p_w = amsfv.solve(iterative=True, tol=1e-6, restriction_operator=restriction_operator)
        error = p_w - p_w_exact

        Linf = np.linalg.norm(error, ord=np.inf) / np.linalg.norm(p_w_exact, ord=np.inf)
        L2_error = np.linalg.norm(error, ord=2) / np.linalg.norm(p_w_exact, ord=2)
        print "MSFV pressure at boundaries are %e and %e" % (p_w[pi_1], p_w[pi_1])
        print "Exact pressure at boundaries are %e and %e" % (p_w_exact[pi_2], p_w_exact[pi_2])
        print "Linf and L2 errors are: %e, %e" % (Linf, L2_error)
        assert ((Linf < 10e-2) & (L2_error < 10e-5)), "L1 error: %s  L2 error: %s" % (Linf, L2_error)

        print "Restarting"
        p_w = amsfv.solve(iterative=True, tol=1e-8, restriction_operator="msfv", x0=p_w)


def test_msfv_single_phase_dirichlet_direct():
    network = structured_network(Nx=n_fine_per_cell * nx, Ny=n_fine_per_cell * ny, Nz=n_fine_per_cell * nz)
    subgraph_ids = cartesion_subgraph_labels(network, nx, ny, nz)

    A = network_to_mat(network)

    q_tilde = np.zeros(network.nr_p)

    pi_1 = np.argmin(network.pores.x)
    pi_2 = np.argmax(network.pores.x)

    A[pi_1, :] = 0.0
    A[pi_1, pi_1] = 1.0
    q_tilde[pi_1] = 2.0e6

    A[pi_2, :] = 0.0

    A[pi_2, pi_2] = 1.0
    q_tilde[pi_2] = 4.7e6
    wirebasket = create_wire_basket(network, nx, ny, nz)

    amsfv = MSFV(A=A.copy(), source_term=q_tilde, wirebasket=wirebasket, primal_cell_labels=subgraph_ids)
    ml = pyamg.smoothed_aggregation_solver(A, max_coarse=10)
    p_w_exact = ml.solve(b=q_tilde, tol=1.0e-16, accel='gmres')

    for restriction_operator in ["msfe", "msfv"]:
        p_w = amsfv.solve(iterative=True, tol=1e-6, restriction_operator=restriction_operator)
        error = p_w - p_w_exact

        Linf = np.linalg.norm(error, ord=np.inf) / np.linalg.norm(p_w_exact, ord=np.inf)
        L2_error = np.linalg.norm(error, ord=2) / np.linalg.norm(p_w_exact, ord=2)
        print "MSFV pressure at boundaries are %e and %e" % (p_w[pi_1], p_w[pi_1])
        print "MSFV pressure at boundaries are %e and %e" % (p_w_exact[pi_2], p_w_exact[pi_2])
        print "Linf and L2 errors are: %e, %e" % (Linf, L2_error)
        assert ((Linf < 10e-2) & (L2_error < 10e-5)), "L1 error: %s  L2 error: %s" % (Linf, L2_error)

        print "Restarting"
        p_w = amsfv.solve(iterative=True, tol=1e-8, restriction_operator="msfv", x0=p_w)


def test_msfv_two_phase():
    network = structured_network(Nx=n_fine_per_cell * nx, Ny=n_fine_per_cell * ny, Nz=n_fine_per_cell * nz)
    subgraph_ids = cartesion_subgraph_labels(network, nx, ny, nz)

    print "running invasion percolation algorithm"
    run_ip_algorithm(network, 0.7)
    p_c = network.pores.p_c

    k_computer = ConductanceCalc(network)
    k_computer.compute()

    assert np.sum(network.tubes.k_n) > 0.0
    assert np.sum(p_c) > 0.0

    A = network_to_mat(network)

    An = LaplacianMatrix(network)
    An.set_edge_weights(network.tubes.k_n)
    An = An.get_csr_matrix()

    q_tilde = np.zeros(network.nr_p)
    q_tilde[network.pi_list_face[WEST]] = 1e-10
    q_tilde[network.pi_list_face[EAST]] = -1e-10

    #for pi in np.union1d(network.pi_list_face[WEST], network.pi_list_face[WEST]):
    #    An.data[An.indptr[pi]:An.indptr[pi + 1]] = 0.0  # Set row of A_n matrix to zero

    wirebasket = create_wire_basket(network, nx, ny, nz)

    amsfv = MSFV(A=A.copy(), source_term=q_tilde.copy(), wirebasket=wirebasket, primal_cell_labels=subgraph_ids,
                 div_source_term=(-An.copy(), p_c.copy()))

    p_w = amsfv.solve(iterative=True, tol=1e-5, restriction_operator="msfe")

    q_div_source = -An * p_c
    print np.sum(np.abs(q_div_source))
    print np.sum(np.abs(q_tilde))

    ml = pyamg.smoothed_aggregation_solver(A, max_coarse=10)
    p_w_exact = ml.solve(b=q_tilde + q_div_source, tol=1.0e-21, accel='gmres')

    # Enforce minimum value of zero for comparison
    p_w_exact -= np.min(p_w_exact)
    p_w -= np.min(p_w)
    error = p_w - p_w_exact

    Linf = np.linalg.norm(error, ord=np.inf) / np.linalg.norm(p_w_exact, ord=np.inf)
    L2_error = np.linalg.norm(error, ord=2) / np.linalg.norm(p_w_exact, ord=2)
    print "Linf and L2 errors are: %e, %e" % (Linf, L2_error)
    assert ((Linf < 10e-2) & (L2_error < 10e-5)), "L1 error: %s  L2 error: %s" % (Linf, L2_error)

    print "Restarting"
    p_w = amsfv.solve(iterative=True, tol=1e-5, restriction_operator="msfv", x0=p_w)


def print_profiling_info(filename):
    p = pstats.Stats(filename)
    p.sort_stats('cumulative').print_stats(20)
    p.sort_stats('time').print_stats(20)

if __name__ == "__main__":
    test_msfv_two_phase()
    # test_msfv_single_phase_dirichlet_direct()
    # test_msfv_single_phase_dirichlet_large_source()
    # test_msfv_single_phase_neumann()

    print_profiling_info('restats')

