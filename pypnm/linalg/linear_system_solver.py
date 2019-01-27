import logging
import sys

import numpy as np
import pyamg
from numpy.linalg import norm
from pypnm.linalg.laplacianmatrix import LaplacianMatrix
from pypnm.linalg.petsc_interface import get_petsc_ksp, petsc_solve_from_ksp, petsc_solve
from scipy.sparse import csc_matrix

try:
    from pypnm.linalg.trilinos_interface import matrix_scipy_to_epetra, vector_numpy_to_epetra, trilinos_ml_prec
    from pypnm.linalg.trilinos_interface import trilinos_solve, solve_aztec
    WITH_TRILINOS = True
except ImportError:
    WITH_TRILINOS = False


try:
    import petsc4py
    from mpi4py import MPI

    petsc4py.init(sys.argv)
    from petsc4py import PETSc

    WITH_PETSC = True
except ImportError:
    WITH_PETSC = False

try:
    from scikits.umfpack import splu, spsolve
    WITH_UMFPACK=True
except ImportError:
    pass


import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()},
                  reload_support=True)

logger = logging.getLogger('pypnm.linear_system_solver')

__author__ = """\n""".join(['Karim Khayrat (kkhayrat@gmail.com)'])


def _ref_residual_inf(A, rhs):
    x_1 = rhs / A.diagonal()
    B = csc_matrix(A)
    B.setdiag(0.0)
    x_2 = (rhs-B*x_1)/A.diagonal()
    ref_residual = norm(rhs - A * x_2, ord=np.inf)
    assert ref_residual > 1e-16
    return ref_residual


def solve_sparse_mat_mat_from_lu(lu, B):
    B = B.tocsc()  # Convert to csc to extract columns efficiently

    # Create a sparse output matrix by repeatedly applying
    # the sparse factorization to solve columns of b.
    # Adapted from scipy.sparse.linalg.dsolve.linsolve

    ind_of_nonzero_cols = np.unique(B.nonzero()[1])

    data_segs = []
    row_segs = []
    col_segs = []
    for j in ind_of_nonzero_cols:
        Bj = B[:, j].A.ravel()

        xj = lu.solve(Bj)

        w = np.flatnonzero(xj)
        segment_length = w.shape[0]

        row_segs.append(w)
        col_segs.append(np.ones(segment_length, dtype=int) * j)
        data_segs.append(np.asarray(xj[w], dtype=B.dtype))

    sparse_data = np.concatenate(data_segs)
    sparse_row = np.concatenate(row_segs)
    sparse_col = np.concatenate(col_segs)
    x = csc_matrix((sparse_data, (sparse_row, sparse_col)), shape=B.shape, dtype=B.dtype)

    return x


def solve_sparse_mat_mat_lu(A, B, solver="petsc"):
    """
    Solves AX=B for X

    Parameters
    ----------

    A: scipy sparse matrix
        NxN Matrix
    B: scipy sparse matrix
        NxP Matrix
    solver: string
        Choice of direct solver. "petsc" or "scipy"

    Returns
    _______

    out: scipy sparse matrix
        solution X

    Notes
    _____

    Ignores zero columns in B, and hence is faster than the existing scipy implementation
    """

    sf = 1.e15  # scaling factor for petsc

    assert solver in ["petsc", "scipy"]

    if solver == "petsc":
        lu_A = get_petsc_ksp(A * sf, pctype="lu", ksptype="preonly", tol=1e-25, max_it=100)
    elif solver == "scipy":
        lu_A = splu(A)

    B = B.tocsc()  # Convert to csc to extract columns efficiently

    # Create a sparse output matrix by repeatedly applying
    # the sparse factorization to solve columns of b.
    # Adapted from scipy.sparse.linalg.dsolve.linsolve

    ind_of_nonzero_cols = np.unique(B.nonzero()[1])

    data_segs = []
    row_segs = []
    col_segs = []
    for j in ind_of_nonzero_cols:
        Bj = B[:, j].A.ravel()

        if solver == "scipy":
            xj = lu_A.solve(Bj)
        elif solver == "petsc":
            xj = petsc_solve_from_ksp(lu_A, Bj * sf, x0=None, tol=1e-5)

        w = np.flatnonzero(xj)
        segment_length = w.shape[0]

        row_segs.append(w)
        col_segs.append(np.ones(segment_length, dtype=int) * j)
        data_segs.append(np.asarray(xj[w], dtype=A.dtype))

    sparse_data = np.concatenate(data_segs)
    sparse_row = np.concatenate(row_segs)
    sparse_col = np.concatenate(col_segs)
    x = csc_matrix((sparse_data, (sparse_row, sparse_col)), shape=B.shape, dtype=B.dtype)

    return x


def solve_pyamg(A, b, x0=None, tol=1e-5):
    """
    Solves Ax=b using pyamg package

    Parameters
    ----------
    A: scipy matrix
    b: ndarray
    x0: ndarray, optional
        initial guess
    tol: int, optional
        convergence tolerance
    accel: string, optional
        type of Krylov accelerator

    Returns
    -------
    out: ndarray
        solution x to equation Ax=b

    """
    ml = pyamg.smoothed_aggregation_solver(A, max_coarse=10)
    return ml.solve(b=b, tol=tol, x0=x0, accel='gmres')


class RHSStandard(object):
    """Small class to hold rhs of a linear system equation defined on a network"""

    def __init__(self, network):
        self.val = np.zeros(network.nr_p)
        self.network = network

    def set_dirichlet_pores(self, pi_list, value):
        self.val[pi_list] = value


class LinearSystemStandard(object):
    """
    Wrapper for a linear system

    Parameters
    ___________
    network: PoreNetwork

    """
    def __init__(self, network):
        self.network = network
        self.matrix = LaplacianMatrix(network)
        self.rhs = RHSStandard(network)
        self.sol = np.zeros(network.nr_p)

    def fill_matrix(self, conductances):
        """
        Parameters
        ----------
        conductances: ndarray
            conductance of throats to be used to construct laplacian matrix

        """
        self.matrix.set_edge_weights(conductances)

    def solve(self, solver="LU", x0=None, tol=1e-5):
        """
        Solves linear system Ax=b

        Parameters
        ----------
        solver: string, optional
            Type of solver to use. "LU", "AMG" or "PETSC"
        x0: ndarray, optional
            initial guess
        tol: float
            convergence tolerance

        Returns
        -------
        out: ndarray
            Solution of Ax=b

        """
        if x0 is None:
            x0 = self.sol

        A = self.matrix.get_csr_matrix()

        if solver == "LU":
            self.sol = spsolve(A=A, b=self.rhs.val)

        elif solver == "AMG":
            self.sol = solve_pyamg(A=A, b=self.rhs.val, tol=tol, x0=x0)

        elif solver == "PETSC":
            self.sol = petsc_solve(A=A*1e16, b=self.rhs.val*1e16, tol=tol, x0=x0, ksptype="bcgs")

        return self.sol

    def set_dirichlet_pores(self, pi_list, value):
        """
        Sets boundary conditions for selected pores

        Parameters
        ----------
        pi_list: ndarray
            indices of pores to set boundary condition for

        value: ndarray
            values to set at selected pores
        """
        self.matrix.set_selected_rows_to_dirichlet(pi_list)
        self.rhs.set_dirichlet_pores(pi_list, value)

    def set_dirichlet_pores_symmetric(self, pi_list, value):
        """
        Sets boundary conditions for selected pores while maintaining symmetry of underlying laplacian matrix

        Parameters
        ----------
        pi_list: ndarray
            indices of pores to set boundary condition for

        value: ndarray
            values to set at selected pores
        """
        A_init = self.matrix.get_coo_matrix()

        self.matrix.set_offdiag_col_entries_to_zero(pi_list)

        sol_dummy = np.zeros(self.network.nr_p)
        sol_dummy[pi_list] = value

        A_diff = A_init - self.matrix.get_coo_matrix()

        rhs_new = A_diff * sol_dummy

        self.matrix.set_selected_rows_to_dirichlet(pi_list)
        self.rhs.val[:] = - rhs_new
        self.rhs.val[pi_list] = value
