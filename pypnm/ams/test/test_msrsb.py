import cProfile
import pstats

import numpy as np

from pypnm.ams.msrb_funcs import solve_with_msrsb
from pypnm.attribute_calculators.conductance_calc import ConductanceCalc
from pypnm.linalg.laplacianmatrix import laplacian_from_network
from pypnm.porenetwork.network_factory import structured_network
from pypnm.porenetwork.network_factory import unstructured_network_delaunay
from pypnm.util.utils_testing import run_ip_algorithm
from scipy.sparse.linalg import spsolve
from pypnm.porenetwork.porenetwork import PoreNetwork

fluid_properties = dict()
fluid_properties['mu_n'] = 0.01
fluid_properties['mu_w'] = 1.0
fluid_properties['gamma'] = 1.0


def _solve_system_multigrid(A, rhs, nsmooth_ilu, nsmooth_gmres):
    for smoother, n_smooth in [('ilu', nsmooth_ilu),  ('gmres', nsmooth_gmres) ]:

        sol, hist_noms = solve_with_msrsb(A, rhs, tol=1e-7, smoother=smoother, n_smooth=n_smooth,
                               with_multiscale=False, tol_basis=1e-2, adapt_smoothing=False, conv_history=True)

        sol, hist_ms = solve_with_msrsb(A, rhs, tol=1e-7, smoother=smoother, n_smooth=n_smooth,
                               with_multiscale=True, tol_basis=1e-2, adapt_smoothing=False, conv_history=True)

        assert len(hist_ms["residual"]) < len(hist_noms["residual"]),\
            "iMSFV too slow. Smoother: %s, Niter ms: %d, Niter noms: %d" % (smoother, len(hist_ms["residual"]),
                                                                                     len(hist_noms["residual"]))
    return sol


def _compare_solutions(sol_ms, sol_exact):
    sol_ms -= np.min(sol_ms)
    sol_exact -= np.min(sol_exact)
    error = sol_ms - sol_exact

    Linf = np.linalg.norm(error, ord=np.inf) / np.linalg.norm(sol_exact, ord=np.inf)
    L2_error = np.linalg.norm(error, ord=2) / np.linalg.norm(sol_exact, ord=2)
    assert ((Linf < 10e-2) & (L2_error < 10e-5)), "L1 error: %s  L2 error: %s" % (Linf, L2_error)


def _create_system(network, qin=1e-9):
    # Create laplacian matrix
    k_computer = ConductanceCalc(network, fluid_properties, pores_have_conductance=True)
    k_computer.compute()
    A = laplacian_from_network(network, weights=network.tubes.k_w + network.tubes.k_n)

    print np.max(np.abs(A.data))/np.min(np.abs(A.data))
    # Create right hand-side
    rhs = np.zeros(network.nr_p)
    rhs[network.pi_list_face["WEST"]] = qin / len(network.pi_list_face["WEST"])
    rhs[network.pi_list_face["EAST"]] = -qin / len(network.pi_list_face["EAST"])

    return A, rhs


def test_msrsb_two_phase():
    try:
        network = PoreNetwork.load("network_test_2_phase")
    except IOError:
        network = unstructured_network_delaunay(nr_pores=80000)
        run_ip_algorithm(network, 0.7)
        network.save("network_test_2_phase")

    p_c = network.pores.p_c
    assert np.sum(p_c) > 0.0

    k_computer = ConductanceCalc(network, fluid_properties, pores_have_conductance=False)
    k_computer.compute()
    assert np.sum(network.tubes.k_n) > 0.0

    A = laplacian_from_network(network, weights=network.tubes.k_w + network.tubes.k_n)
    An = laplacian_from_network(network, weights=network.tubes.k_n)

    qin = 1e-9
    rhs = np.zeros(network.nr_p)
    rhs[network.pi_list_face["WEST"]] = qin / len(network.pi_list_face["WEST"])
    rhs[network.pi_list_face["EAST"]] = -qin / len(network.pi_list_face["EAST"])

    rhs += -An * p_c

    sol_ms = solve_with_msrsb(A, rhs, tol=1e-9, smoother="gmres", n_smooth=5, verbose=True,
                           with_multiscale=True, tol_basis=1e-2, adapt_smoothing=False)

    A[1, :] = 0.0
    A[1, 1] = 1.0

    sol_exact = spsolve(A=A, b=rhs)
    _compare_solutions(sol_ms, sol_exact)

def test_msrsb_unstructured():
    # Create or load network
    try:
        network = PoreNetwork.load("network_test_unstructured")
    except IOError:
        network = unstructured_network_delaunay(nr_pores=80000)
        network.save("network_test_unstructured")

    A, rhs = _create_system(network)

    sol_ms = _solve_system_multigrid(A, rhs, 40, 6)

    A[1, :] = 0.0
    A[1, 1] = 1.0
    sol_exact = spsolve(A=A, b=rhs)

    _compare_solutions(sol_ms, sol_exact)


def test_msrsb_structured():
    try:
        network = PoreNetwork.load("network_test_structured")
    except IOError:
        network = structured_network(Nx=40, Ny=40, Nz=40)
        network.save("network_test_structured")

    A, rhs = _create_system(network)

    sol_ms = _solve_system_multigrid(A, rhs, 40, 6)

    A[1, :] = 0.0
    A[1, 1] = 1.0
    sol_exact = spsolve(A=A, b=rhs)

    _compare_solutions(sol_ms, sol_exact)


def print_profiling_info(filename):
    p = pstats.Stats(filename)
    p.sort_stats('cumulative').print_stats(20)
    p.sort_stats('time').print_stats(20)


if __name__ == "__main__":
    cProfile.run('test_msrsb_unstructured()', 'restats')
    print_profiling_info('restats')
