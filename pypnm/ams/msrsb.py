import numpy as np
from PyTrilinos import Epetra, EpetraExt, IFPACK

from pypnm.linalg.trilinos_interface import DinvA, sum_of_columns, mat_multiply, solve_direct
from pypnm.linalg.trilinos_interface import epetra_set_matrow_to_zero, epetra_set_vecrow_to_zero
import logging
logger = logging.getLogger('pypnm.msrsb')


class MSRSB(object):
    """
    Solver for a linear system using the multi-scale restricted smoothed basis method
    (similar to energy optimized smoothed aggregation).

    Parameters
    ----------
    A: Epetra CrsMatrix
        Matrix representing linear set of equations

    my_restriction_supports: dict
        Dictionary mapping the id of each restriction support to its constituent set of vertices.

    my_basis_supports: dict
        Dictionary mapping the id of each basis function support to its constituent set of vertices


    Notes
    -----
    For runs with MPI, my_restriction_supports and my_basis_supports should contain only the vertices belonging
    to each processor.
    Notation follows Mandel et al 1999 - Energy Optimization of Algebraic Multigrid Bases

    """
    def __init__(self, A, my_restriction_supports, my_basis_supports):

        if len(my_basis_supports) < len(my_restriction_supports):
            raise ValueError()

        self.A = A
        map = self.A.RangeMap()
        self.P = self._prolongation_operator_init(map, my_restriction_supports, my_basis_supports)
        self.P_initial = Epetra.CrsMatrix(self.P)
        self.R = self._restriction_operator_msfv(map, my_restriction_supports)

        self.N = self._create_structure_matrix(map, my_basis_supports)  # Allowed nonzeros structure of P
        self.delta_P = self._create_structure_matrix(map, my_basis_supports)
        self.delta_P.PutScalar(0.0)

        self.num_overlaps = sum_of_columns(self.N)

        sum_cols_P = sum_of_columns(self.P)
        assert np.allclose(sum_cols_P[:], 1.0)

    @staticmethod
    def _create_structure_matrix(row_map, my_supports, val=1.0):
        range_map = row_map
        comm = range_map.Comm()
        domain_map = Epetra.Map(-1, my_supports.keys(), 0, comm)
        A = Epetra.CrsMatrix(Epetra.Copy, range_map, 30)

        for basis_id in my_supports:
            my_fine_cell_ids = my_supports[basis_id]
            size = len(my_fine_cell_ids)
            assert size > 0
            ierr = A.InsertGlobalValues(my_fine_cell_ids, [int(basis_id)] * size, [val] * size)
            assert ierr == 0

        A.FillComplete(domain_map, range_map)

        a = sum_of_columns(A)
        assert np.all(a[:] >= 1.0)

        return A

    @staticmethod
    def _restriction_operator_msfv(row_map, my_restriction_supports):
        R = MSRSB._create_structure_matrix(row_map, my_restriction_supports)
        a = sum_of_columns(R)
        assert np.all(a[:] == 1.0)
        return R

    @staticmethod
    def _prolongation_operator_init(row_map, my_restriction_supports, my_basis_supports):
        """
        Creates N x M matrix where N is the number of vertices in the graph and M is the number of basis functions
        """
        range_map = row_map
        comm = range_map.Comm()
        domain_map = Epetra.Map(-1, my_basis_supports.keys(), 0, comm)
        P = Epetra.CrsMatrix(Epetra.Copy, range_map, 30)

        for basis_id in my_basis_supports:
            my_fine_cell_ids = my_basis_supports[basis_id]
            size = len(my_fine_cell_ids)
            assert size > 0
            P.InsertGlobalValues(my_fine_cell_ids, [int(basis_id)] * size, [0.] * size)

        for basis_id in my_restriction_supports:
            my_fine_cell_ids = my_restriction_supports[basis_id]
            size = len(my_fine_cell_ids)
            assert size > 0
            ierr = P.InsertGlobalValues(my_fine_cell_ids, [int(basis_id)] * size, [1.] * size)
            assert ierr == 0

        P.FillComplete(domain_map, range_map)

        a = sum_of_columns(P)
        assert np.all(a[:] == 1.0)
        return P

    def smooth_prolongation_operator(self, niter=100):
        """
        Parameters
        ----------
        niter:
            Number of smoothing steps

        Notes
        -----
        See  Algorithm 2 in Mandel et al 1999. However Jacobi iteration is used for smoothing as opposed to
        gradient descent.
        """

        support_matrix_copy = Epetra.CrsMatrix(self.N)
        tau = Epetra.Vector(self.A.RangeMap())
        J = DinvA(self.A)
        ierr = 0
        delta_P_temp = Epetra.CrsMatrix(Epetra.Copy, self.A.RangeMap(), 40)

        for _ in xrange(niter):
            sum_cols_P = sum_of_columns(self.P)
            assert np.allclose(sum_cols_P[:], 1.0)

            ierr += EpetraExt.Multiply(J, False, self.P, False, delta_P_temp)

            self.delta_P.PutScalar(0.0)  # Drop entries of delta_P_temp not matching the structure of delta_P
            ierr += EpetraExt.Add(delta_P_temp, False, 1.0, self.delta_P, 1.0)  # delta_P = N*(D^-1 AP)

            sum_cols_delta_P = sum_of_columns(self.delta_P)

            assert np.all(self.num_overlaps[:] >= 1)

            tau[:] = sum_cols_delta_P[:] / self.num_overlaps[:]
            support_matrix_copy.PutScalar(1.0)
            support_matrix_copy.LeftScale(tau)   # last term in Equation (19)

            ierr += EpetraExt.Add(support_matrix_copy, False, -1.0, self.delta_P, 1.0)  # delta_P = Z(N*D^-1AP)

            sum_cols_delta_P = sum_of_columns(self.delta_P)
            assert np.allclose(sum_cols_delta_P[:], 0.0)

            ierr += EpetraExt.Add(self.delta_P, False, -0.5, self.P, 1.0)

        sum_cols_P = sum_of_columns(self.P)

        assert np.allclose(sum_cols_P[:], 1.0)

        smoothness = mat_multiply(self.A, self.P).NormFrobenius() / mat_multiply(self.A, self.P_initial).NormFrobenius()
        logger.debug("smoothness of prolongation operator is %g", smoothness)

        assert ierr == 0

    def solve_one_step_msfv(self, rhs):
        return self.__solve_one_step(rhs, "msfv")

    def solve_one_step_msfe(self, rhs):
        return self.__solve_one_step(rhs, "msfe")

    def __solve_one_step(self, rhs, method):
        # Solves P*(RAP)^-1* R*rhs
        A, P = self.A, self.P
        assert method in ["msfv", "msfe"]
        if method == "msfv":
            R = self.R
        if method == "msfe":
            R = self.P

        AP = mat_multiply(A, P)

        RAP = mat_multiply(R, AP, transpose_1=True)
        assert np.max(np.abs(sum_of_columns(RAP))) < RAP.NormInf() * 1e-10

        rhs_coarse = Epetra.Vector(R.DomainMap())

        R.Multiply(True, rhs, rhs_coarse)

        # Set dirichlet boundary condition at a point
        row = 0
        RAP = epetra_set_matrow_to_zero(RAP, row=0)
        rhs_coarse = epetra_set_vecrow_to_zero(rhs_coarse, row=0)
        if row in RAP.Map().MyGlobalElements():
            ierr = RAP.ReplaceGlobalValues([row], [row], 1.0)
            assert ierr == 0

        sol_coarse = solve_direct(RAP, rhs_coarse)

        sol_fine = Epetra.Vector(P.RangeMap())
        P.Multiply(False, sol_coarse, sol_fine)
        return sol_fine

    def __compute_residual(self, rhs, x0, residual):
        residual[:] = 0.0
        self.A.Multiply(False, x0, residual)
        residual[:] = rhs[:] - residual[:]
        return residual

    def iterative_solve(self, rhs, x0, tol=1.e-5, max_iter=200):
        A = self.A

        rhs_sum = rhs.Comm().SumAll(np.sum(rhs[:]))
        assert rhs_sum < rhs.NormInf() * 1e-8, rhs_sum

        residual = Epetra.Vector(A.RangeMap())

        ref_pressure = Epetra.Vector(A.RangeMap())
        ref_residual = Epetra.Vector(A.RangeMap())

        temp = Epetra.Vector(A.RangeMap())
        A_diagonal = Epetra.Vector(A.RangeMap())
        A.ExtractDiagonalCopy(A_diagonal)

        ref_pressure[:] = rhs[:] / A_diagonal[:]
        A.Multiply(False, ref_pressure, temp)
        ref_residual[:] = rhs[:] - temp[:]
        ref_residual_norm = ref_residual.NormInf()

        residual = self.__compute_residual(rhs, x0, residual)
        error = self.solve_one_step_msfv(residual)
        x0[:] += error[:]

        if residual.NormInf() / ref_residual_norm < tol:
            logger.debug("Solution already converged before iteration")
            return x0

        ilu = IFPACK.ILU(self.A)
        ilu.Initialize()
        ilu.Compute()

        residual_prev_norm = 1.e50
        n_ilu_iter = 5

        for iteration in xrange(max_iter):
            for __ in xrange(max(n_ilu_iter, 100)):
                residual = self.__compute_residual(rhs, x0, residual)
                ilu.ApplyInverse(residual, error)
                x0[:] += error[:]

            residual = self.__compute_residual(rhs, x0, residual)
            error = self.solve_one_step_msfe(residual)

            if residual.NormInf() / ref_residual_norm < tol:
                logger.debug("Number of iterations for convergence: %d", iteration)
                break

            if residual.NormInf() >= residual_prev_norm:
                n_ilu_iter *= 2
                logger.warn("Solver stagnated, increasing number of ilu iterations to %d", n_ilu_iter)

            residual_prev_norm = residual.NormInf()

            x0[:] += error[:]

        residual = self.__compute_residual(rhs, x0, residual)
        error = self.solve_one_step_msfv(residual)
        x0[:] += error[:]

        return x0
