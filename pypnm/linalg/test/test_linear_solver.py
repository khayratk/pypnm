import numpy as np
from scipy.sparse import csc_matrix

from pypnm.attribute_calculators.conductance_calc import ConductanceCalc
from pypnm.linalg.linear_system_solver import LinearSystemStandard, LinearSystemSimple
from pypnm.linalg.linear_system_solver import solve_sparse_mat_mat_lu
from pypnm.porenetwork.network_factory import cube_network


def test_mat_mat_sparse_solve():
    """
    Solve `AX=B` given that `B=A`. Should return `X=I`
    """
    row = np.array([0, 2, 2, 0, 1, 2])
    col = np.array([0, 0, 1, 2, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    A = csc_matrix((data, (row, col)), shape=(3, 3))

    row = np.array([0, 2, 2, 0, 1, 2])
    col = np.array([0, 0, 1, 2, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    B = csc_matrix((data, (row, col)), shape=(3, 3))

    X = solve_sparse_mat_mat_lu(A, B)
    assert np.isclose(np.sum(X.data), 3, rtol=1e-10)


def test_symmetric_dirichlet_bc():
    """
    Test setting dirichlet boundary condition in a manner which leaves the matrix symmetric
    """
    network = cube_network(N=40)

    k_computer = ConductanceCalc(network)
    k_computer.compute()

    p_dirichlet = np.array([1., 23., 43.])
    pi_dirichlet = [2, 7200, 2300]

    # Set dirichlet boundary conditions using symmetric dirichlet boundary conditions
    LS_sym = LinearSystemStandard(network)
    LS_sym.fill_matrix(network.tubes.k_w)
    LS_sym.set_dirichlet_pores_symmetric(pi_list=pi_dirichlet, value=p_dirichlet)

    sol_sym = LS_sym.solve(solver="PETSC", tol=1e-10)

    # Set dirichlet boundary conditions using non-symmetric dirichlet boundary conditions
    LS = LinearSystemStandard(network)
    LS.fill_matrix(network.tubes.k_w)
    LS.set_dirichlet_pores(pi_list=pi_dirichlet, value=p_dirichlet)

    sol = LS.solve(solver="AMG", tol=1e-10)

    assert np.allclose(sol_sym, sol), np.linalg.norm(sol_sym-sol)


def test_general_dirichlet_bc():
    """
    Test setting arbitrary dirichlet boundary condition
    """
    network = cube_network(N=10)

    k_computer = ConductanceCalc(network)
    k_computer.compute()

    press_in = 121.1
    press_out = 0.901

    # Set dirichlet boundary conditions in one command
    LS = LinearSystemStandard(network)
    LS.fill_matrix(network.tubes.k_w)
    LS.set_dirichlet_pores([2, 92, 29], np.array([press_in, press_in, press_out]))
    sol1 = LS.solve()

    # Set dirichlet boundary condition separately
    LS2 = LinearSystemStandard(network)
    LS2.fill_matrix(network.tubes.k_w)
    LS2.set_dirichlet_pores(2, press_in)
    LS2.set_dirichlet_pores(92, press_in)
    LS2.set_dirichlet_pores(29, press_out)
    sol2 = LS2.solve()

    assert np.allclose(sol1, sol2)


def test_general_dirichlet_bc_simple():
    """
    Test setting arbitrary dirichlet boundary condition
    """
    network = cube_network(N=10)

    k_computer = ConductanceCalc(network)
    k_computer.compute()

    press_in = 121.1
    press_out = 0.901

    LS1 = LinearSystemSimple(network, network.tubes.k_w, [2, 92, 29], np.array([press_in, press_in, press_out]))
    sol1 = LS1.solve()

    LS2 = LinearSystemStandard(network)
    LS2.fill_matrix(network.tubes.k_w)
    LS2.set_dirichlet_pores([2, 92, 29], np.array([press_in, press_in, press_out]))
    sol2 = LS2.solve()

    assert np.allclose(sol1, sol2)
