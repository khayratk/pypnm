import numpy as np

from pypnm.ams.msrb_funcs import solve_with_msrsb
from pypnm.attribute_calculators.conductance_calc import ConductanceCalc
from pypnm.linalg.laplacianmatrix import LaplacianMatrix
from pypnm.porenetwork.network_factory import structured_network, unstructured_network_delaunay
from pypnm.util.utils_testing import run_ip_algorithm
from scipy.sparse.linalg import spsolve
from pypnm.porenetwork.porenetwork import PoreNetwork

import matplotlib.pyplot as plt
sim_settings = dict()
sim_settings["fluid_properties"] = dict()
sim_settings["fluid_properties"]['mu_n'] = 0.001
sim_settings["fluid_properties"]['mu_w'] = 1.0
sim_settings["fluid_properties"]['gamma'] = 1.0


def msrsb_two_phase():
    try:
        network = PoreNetwork.load("benchmark_network.pkl")

    except IOError:
        network = unstructured_network_delaunay(40000)
        run_ip_algorithm(network, 0.7)

        network.save("benchmark_network.pkl")

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
    print "max/min total conductance ratio is %g " % (max(k_tot)/min(k_tot))

    rhs = np.zeros(network.nr_p)
    rhs[np.argmin(network.pores.x)] = 1e-10
    rhs[np.argmax(network.pores.x)] = -1e-10
    rhs += -An * p_c

    sol, history = solve_with_msrsb(A, rhs, tol=1e-7, smoother="ilu", v_per_subdomain=100, conv_history=True,
                                    with_multiscale=True, max_iter=10000, tol_basis=1e-3, n_smooth=80,
                                    adapt_smoothing=False, verbose=True)
    print "converged in", len(history["residual"])

    A[1, :] = 0.0
    A[1, 1] = 1.0

    sol_exact = spsolve(A=A*1e10, b=rhs*1e10)

    sol -= np.min(sol)
    sol_exact -= np.min(sol_exact)
    error = sol - sol_exact

    Linf = np.linalg.norm(error, ord=np.inf) / np.linalg.norm(sol_exact, ord=np.inf)
    L2_error = np.linalg.norm(error, ord=2) / np.linalg.norm(sol_exact, ord=2)

    plt.semilogy(history["residual"])
    plt.show()

    assert ((Linf < 10e-2) & (L2_error < 10e-5)), "L1 error: %s  L2 error: %s" % (Linf, L2_error)

if __name__ == "__main__":
    msrsb_two_phase()
