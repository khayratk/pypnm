import pyamg
import numpy as np
from mpi4py import MPI
from numpy.linalg import norm
from petsc4py import PETSc
from pypnm.linalg.laplacianmatrix import LaplacianMatrix, laplacian_from_network
from pypnm.linalg.linear_system_solver import _ref_residual_inf, logger
from pypnm.linalg.petsc_interface import get_petsc_ksp, petsc_solve_lu
from pypnm.linalg.trilinos_interface import matrix_scipy_to_epetra, vector_numpy_to_epetra, trilinos_ml_prec, \
    trilinos_solve, solve_aztec
try:
    from scikits.umfpack import splu
except ImportError:
    pass

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
    def compute_mass_residual(A, rhs, sol, ref_residual):
        residual = rhs - A * sol
        return norm(residual, ord=np.inf) / ref_residual

    def add_source_rhs(self, source):
        assert len(source) == self.network.nr_p
        self.rhs[0:self.network.nr_p] += source

    def __set_matrix(self, k_n, k_w):
        self.solver_matrix.fill_csr_matrix_with_edge_weights(self.csr_solver_matrix, k_n + k_w)

    def set_dirichlet_pores(self, pi_list, value):
        if len(pi_list) > 0:
            self.solver_matrix.set_csr_matrix_rows_to_dirichlet(self.csr_solver_matrix, pi_list, val=1.e-16)
            self.rhs[pi_list] = value*1.e-16

    def set_rhs(self, k_n, p_c):
        self.rhs_matrix.fill_csr_matrix_with_edge_weights(self.rhs_matrix_csr, k_n)
        A = self.rhs_matrix_csr
        self.rhs[:] = -(A * p_c)

    def setup_linear_system(self, k_n, k_w, p_c):
        self.__set_matrix(k_n, k_w)
        self.set_rhs(k_n, p_c)

    def solve(self, solver="lu", x0=None, tol=1.e-6):
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

        elif solver == "petsclu":
            comm = MPI.COMM_SELF
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

            elif solver == "petsclu":
                self.sol[:] = petsc_solve_lu(A=A*sf, b=petsc_rhs)

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
        ref_residual = _ref_residual_inf(A, self.rhs)
        while (mass_residual > 1e-5) and (count < 20):
            count += 1
            inner_loop_solve(tol)
            mass_residual = self.compute_mass_residual(A, self.rhs, self.sol, ref_residual)
            logger.debug("Mass flux residual %e", mass_residual)
            if count == 99:
                logger.warn("Failed to converge. Residual %e. Falling back to mltrilinos solver", mass_residual)
                logger.warn("Solver which failed was %s", solver)
                return self.solve(solver="mltrilinos") # Fall back to reliable solver

            tol /= 10.0

        if "ml" in solver:
            epetra_prec.DestroyPreconditioner();

        return np.copy(self.sol)