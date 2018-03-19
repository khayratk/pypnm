import logging

import numpy as np
from PyTrilinos import Epetra, EpetraExt, IFPACK, AztecOO

from pypnm.linalg.trilinos_interface import DinvA, sum_of_columns, mat_multiply, solve_direct
from pypnm.linalg.trilinos_interface import epetra_set_matrow_to_zero, epetra_set_vecrow_to_zero

logger = logging.getLogger('pypnm.msrsb')
import warnings


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

        self.R = self._create_structure_matrix(map, my_restriction_supports)
        self.N = self._create_structure_matrix(map, my_basis_supports)  # Allowed nonzeros structure of P
        self.delta_P = Epetra.CrsMatrix(self.N)
        self.delta_P.PutScalar(0.0)

        self.num_overlaps = sum_of_columns(self.N)

        sum_cols_P = sum_of_columns(self.P)
        assert np.allclose(sum_cols_P[:], 1.0)

        a = sum_of_columns(self.R)
        assert np.all(a[:] == 1.0)

    @staticmethod
    def _create_structure_matrix(row_map, my_supports, val=1.0):
        """
        Creates N x M matrix where N is the number of vertices in the graph and M is the number of basis functions.
        This matrix encodes which basis function supports (which is identified by their column ids)
        a given vertex (which is identified by row id) belongs to.
        Note: one vertex may belong to multiple supports.
        """
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

    def smooth_prolongation_operator(self, A, max_iter=1000, tol=1.e-2, verbose=False):
        """
        Parameters
        ----------
        A: Epetra matrix

        max_iter: integer
            Number of smoothing steps

        verbose: bool
            Flag to output convergence information

        Notes
        -----
        See  Algorithm 2 in Mandel et al 1999. However Jacobi iteration is used for smoothing as opposed to
        gradient descent.
        """

        support_matrix_copy = Epetra.CrsMatrix(self.N)
        tau = Epetra.Vector(A.RangeMap())
        J = DinvA(A)
        ierr = 0
        delta_P_temp = Epetra.CrsMatrix(Epetra.Copy, A.RangeMap(), 40)

        for iter_n in xrange(max_iter):
            sum_cols_P = sum_of_columns(self.P)
            assert np.allclose(sum_cols_P[:], 1.0)

            ierr += EpetraExt.Multiply(J, False, self.P, False, delta_P_temp)

            # Drop entries of delta_P_temp not matching the structure of delta_P
            self.delta_P.PutScalar(0.0)
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

            error = self.delta_P.NormInf()
            if error < tol:
                break

        logger.debug("Basis function error: %g. Number of iterations required: %d", error, iter_n)

        if verbose:
            print "Basis function error: %g. Number of iterations required: %d"%(error, iter_n)

        sum_cols_P = sum_of_columns(self.P)

        assert np.allclose(sum_cols_P[:], 1.0, atol=1.e-1000, rtol=1e-12)

        # Important numerical  step to ensure that mass is exactly conserved to machine zero
        """
        tau[:] = 1./sum_cols_P[:]
        self.P.LeftScale(tau)
        sum_cols_P = sum_of_columns(self.P)

        assert np.allclose(sum_cols_P[:], 1.0, atol=1.e-1000, rtol=1.e-15)
        """
        assert ierr == 0

    def __solve_one_step(self, rhs, RAP, R):
        # returns P*(RAP)^-1* R*rhs

        rhs_coarse = Epetra.Vector(R.DomainMap())

        R.Multiply(True, rhs, rhs_coarse)

        if not np.max(np.abs(sum_of_columns(RAP))) < RAP.NormInf() * 1e-10:
            warnings.warn("sum of matrix columns does not equal to zero")

        # Set dirichlet boundary condition at a point
        if np.max(np.abs(sum_of_columns(RAP))) < RAP.NormInf() * 1e-10:
            row = 0
            RAP = Epetra.CrsMatrix(RAP)
            RAP = epetra_set_matrow_to_zero(RAP, row=0)
            rhs_coarse = epetra_set_vecrow_to_zero(rhs_coarse, row=0)
            if row in RAP.Map().MyGlobalElements():
                ierr = RAP.ReplaceGlobalValues([row], [row], 1.0)
                assert ierr == 0

        sol_coarse = solve_direct(RAP, rhs_coarse)

        sol_fine = Epetra.Vector(self.P.RangeMap())
        self.P.Multiply(False, sol_coarse, sol_fine)
        return sol_fine

    def __compute_residual(self, A, rhs, x0, residual):
        residual[:] = 0.0
        A.Multiply(False, x0, residual)
        residual[:] = rhs[:] - residual[:]
        return residual

    def iterative_solve(self, A, rhs, x0, tol=1.e-5, max_iter=200, n_smooth=10, smoother="gmres",
                        conv_history=False, with_multiscale=True, adapt_smoothing=True, verbose=False):

        history = dict()
        history["n_smooth"] = []
        history["residual"] = []

        assert smoother in ["gmres", "ilu", "jacobi"]

        rhs_sum = rhs.Comm().SumAll(np.sum(rhs[:]))
        if not abs(rhs_sum) < abs(rhs.NormInf() * 1e-8):
            logger.warn("sum of rhs does not equal to zero")

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

        AP = mat_multiply(A, self.P)

        RAP_msfv = mat_multiply(self.R, AP, transpose_1=True)
        RAP_msfe = mat_multiply(self.P, AP, transpose_1=True)

        residual = self.__compute_residual(A, rhs, x0, residual)

        error = Epetra.Vector(self.P.RangeMap())
        if with_multiscale:
            error = self.__solve_one_step(residual, RAP_msfv, self.R)
            x0[:] += error[:]

        if smoother == "ilu":
            ilu = IFPACK.ILU(self.A)
            ilu.Initialize()
            ilu.Compute()

        residual_prev_norm = 1.e50

        solver = AztecOO.AztecOO()
        if smoother == "ilu":
            solver.SetAztecOption(AztecOO.AZ_solver, AztecOO.AZ_fixed_pt)
            solver.SetAztecOption(AztecOO.AZ_precond, AztecOO.AZ_dom_decomp)
            solver.SetAztecOption(AztecOO.AZ_subdomain_solve, AztecOO.AZ_ilu)

        if smoother == "gmres":
            solver.SetAztecOption(AztecOO.AZ_solver, AztecOO.AZ_gmres)
            solver.SetAztecOption(AztecOO.AZ_precond, AztecOO.AZ_dom_decomp)
            solver.SetAztecOption(AztecOO.AZ_subdomain_solve, AztecOO.AZ_ilu)

        if smoother == "jacobi":
            solver.SetAztecOption(AztecOO.AZ_solver, AztecOO.AZ_fixed_pt)
            solver.SetAztecOption(AztecOO.AZ_precond, AztecOO.AZ_Jacobi)

        solver.SetAztecOption(AztecOO.AZ_conv, AztecOO.AZ_rhs)
        solver.SetAztecOption(AztecOO.AZ_output, 0)

        for iteration in xrange(max_iter):
            residual = self.__compute_residual(A, rhs, x0, residual)

            logger.debug("Residual at iteration %d: %g", iteration,  residual.NormInf()[0] / ref_residual_norm)

            error[:] = 0.0
            solver.Iterate(A, error, residual, n_smooth, 1e-20)
            x0[:] += error[:]

            residual = self.__compute_residual(A, rhs, x0, residual)

            if with_multiscale:
                error = self.__solve_one_step(residual, RAP_msfe, self.P)
                x0[:] += error[:]

            if residual.NormInf()[0] / ref_residual_norm < tol:
                logger.debug("Number of iterations for convergence: %d", iteration)
                break

            if residual.NormInf()[0] >= 1.01 * residual_prev_norm and adapt_smoothing:
                n_smooth = int(1.4 * n_smooth)
                logger.warn("Solver stagnated, increasing number of smoothing steps to %d", n_smooth)

            residual_prev_norm = residual.NormInf()

            history["n_smooth"].append(n_smooth)
            history["residual"].append(residual_prev_norm[0]/ref_residual_norm)

            if verbose:
                print iteration, history["residual"][-1]

        residual = self.__compute_residual(A, rhs, x0, residual)
        error[:] = 0.0

        solver.Iterate(A, error, residual, n_smooth, 1e-20)
        x0[:] += error[:]

        residual = self.__compute_residual(A, rhs, x0, residual)
        error = self.__solve_one_step(residual, RAP_msfv, self.R)
        x0[:] += error[:]

        history["n_smooth"].append(n_smooth)
        history["residual"].append(residual.NormInf()[0] / ref_residual_norm)

        # Check for convergence.:

        lhs_fine = Epetra.Vector(A.DomainMap())
        rhs_coarse = Epetra.Vector(self.R.DomainMap())
        lhs_coarse = Epetra.Vector(self.R.DomainMap())

        self.R.Multiply(True, rhs, rhs_coarse)
        A.Multiply(False, x0, lhs_fine)
        self.R.Multiply(True, lhs_fine, lhs_coarse)

        max_lhs = np.max(np.abs(lhs_coarse))
        max_rhs = np.max(np.abs(rhs_coarse))
        tol = max(max_lhs, max_rhs)*1e-10
        assert np.allclose(rhs_coarse[:], lhs_coarse[:], atol=tol), "max_lhs:%e max_rhs:%e " % (max_lhs, max_rhs)
        assert np.allclose(lhs_coarse[:], rhs_coarse[:], atol=tol)

        if conv_history:
            return x0, history
        else:
            return x0


