import pyamg

import logging
import sys

import numpy as np
from numpy.linalg import norm
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

from pypnm.linalg.laplacianmatrix import LaplacianMatrix
from pypnm.linalg.petsc_interface import get_petsc_ksp, petsc_solve_from_ksp, petsc_solve

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
except ImportError:
    pass

try:
    from scikits.umfpack import splu, spsolve
except ImportError:
    pass

import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()},
                  reload_support=True)

logger = logging.getLogger('pypnm.linear_system_solver')

__author__ = """\n""".join(['Karim Khayrat (kkhayrat@gmail.com)'])


def _ref_residual_inf(A, rhs):
    return norm(rhs - A * (rhs / A.diagonal()), ord=np.inf)


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

    sf = 1.e30  # scaling factor for petsc

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
            xj = petsc_solve_from_ksp(lu_A, Bj * sf, x=None, tol=1e-5)

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


class PressureSolverDynamicDirichlet(object):
    def __init__(self, network):
        self.network = network
        self.rhs_matrix = LaplacianMatrix(self.network)
        self.rhs_matrix_csr = self.rhs_matrix.get_csr_matrix()
        self.solver_matrix = LaplacianMatrix(self.network)
        self.csr_solver_matrix = self.solver_matrix.get_csr_matrix()
        self.rhs = np.zeros(network.nr_p)
        self.sol = np.zeros(network.nr_p)

    def create_flux_matrix(self, cond):
        matrix = self.rhs_matrix
        matrix.fill_csr_matrix_with_edge_weights(self.rhs_matrix_csr, cond)
        return self.rhs_matrix_csr

    def compute_nonwetting_flux(self):
        network = self.network
        pores = network.pores

        A = self.create_flux_matrix(network.tubes.k_n)
        flux_n = A * pores.p_n
        return flux_n

    def compute_wetting_flux(self):
        network = self.network
        pores = network.pores

        A = self.create_flux_matrix(network.tubes.k_w)
        flux_w = A * pores.p_w
        return flux_w

    @staticmethod
    def compute_mass_residual(A, rhs, sol):
        residual = rhs - A * sol
        return norm(residual, ord=np.inf) / _ref_residual_inf(A, rhs)

    def add_source_rhs(self, source):
        assert len(source) == self.network.nr_p
        self.rhs[0:self.network.nr_p] += source

    def __set_matrix(self, k_n, k_w):
        self.solver_matrix.fill_csr_matrix_with_edge_weights(self.csr_solver_matrix, k_n + k_w)

    def set_dirichlet_pores(self, pi_list, value):
        if len(pi_list) > 0:
            self.solver_matrix.set_csr_matrix_rows_to_dirichlet(self.csr_solver_matrix, pi_list)
            self.rhs[pi_list] = value

    def set_rhs(self, k_n, p_c):
        self.rhs_matrix.fill_csr_matrix_with_edge_weights(self.rhs_matrix_csr, k_n)
        A = self.rhs_matrix_csr
        self.rhs[:] = -(A * p_c)

    def setup_linear_system(self, k_n, k_w, p_c):
        self.__set_matrix(k_n, k_w)
        self.set_rhs(k_n, p_c)

    def solve(self, solver="lu", x0=None, tol=1.e-9):
        solver = solver.lower()
        sf = 1e30
        pores = self.network.pores
        A = self.csr_solver_matrix
        self.solver_matrix.set_csr_singular_rows_to_dirichlet(A)
        self.A = A

        mass_residual = 1.0

        if x0 is not None:
            self.sol[:] = x0

        if solver == "lu":
            lu_A = splu(A)

        elif solver == "amg":
            ml = pyamg.rootnode_solver(A, max_coarse=10)

        elif solver == "petsc":
            comm = MPI.COMM_SELF
            ksp = get_petsc_ksp(A=A * sf, ksptype="minres", tol=tol, max_it=1000)
            petsc_rhs = PETSc.Vec().createWithArray(self.rhs * sf, comm=comm)

        elif "trilinos" in solver:
            epetra_mat = matrix_scipy_to_epetra(A * sf)
            epetra_rhs = vector_numpy_to_epetra(self.rhs * sf)

            if "ml" in solver:
                epetra_prec = trilinos_ml_prec(epetra_mat)

        def inner_loop_solve(tol):
            if solver == "lu":
                self.sol[:] = lu_A.solve(self.rhs)

            elif solver == "amg":
                self.sol[:] = ml.solve(b=self.rhs, x0=self.sol, tol=tol, accel='gmres')

            elif solver == "petsc":
                ksp.setTolerances(rtol=tol)
                ksp.setFromOptions()
                petsc_sol = PETSc.Vec().createWithArray(self.sol, comm=comm)
                ksp.setInitialGuessNonzero(True)
                ksp.solve(petsc_rhs, petsc_sol)

                self.sol[:] = petsc_sol.getArray()

            elif "trilinos" in solver:
                epetra_sol = vector_numpy_to_epetra(self.sol)

                if "ml" in solver:
                    x = trilinos_solve(epetra_mat, epetra_rhs, epetra_prec, x=epetra_sol, tol=tol)
                else:
                    x = solve_aztec(epetra_mat, epetra_rhs, x=epetra_sol, tol=tol)
                self.sol[:] = x

            pores.p_w[:] = self.sol[:]  # This side-effect is required for the compute_mass_residual function
            pores.p_n[:] = pores.p_w + pores.p_c

        count = 0
        while (mass_residual > 1e-5) and (count < 100):
            count += 1
            inner_loop_solve(tol)
            mass_residual = self.compute_mass_residual(A, self.rhs, self.sol)
            logger.debug("Mass flux residual %e", mass_residual)
            if count == 99:
                logger.warn("Failed to converge. Residual %e. Falling back to mltrilinos solver", mass_residual)
                return self.solve(solver="mltrilinos") # Fall back to reliable solver

            tol /= 10.0

        if "ml" in solver:
            epetra_prec.DestroyPreconditioner();

        return np.copy(self.sol)
