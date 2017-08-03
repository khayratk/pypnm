import sys

import numpy as np
import petsc4py
from numpy.linalg import norm
from scipy.sparse.linalg import splu

try:
    from scikits.umfpack import splu, spsolve
except ImportError:
    pass


from scipy import sparse
from scipy.sparse import coo_matrix

from pypnm.linalg.linear_system_solver import solve_sparse_mat_mat_lu
from pypnm.linalg.petsc_interface import get_petsc_ksp
from pypnm.util.igraph_utils import scipy_matrix_to_igraph
import logging
petsc4py.init(sys.argv)
from petsc4py import PETSc

color_interior = 0
color_face = 1
color_edge = 2
color_node = 3
logger = logging.getLogger('pypnm')


def create_permutation_matrix(old_indices, new_indices):
    """Creates permutation matrix given two index lists

    Parameters
    ----------

    old_indices : ndarray
        Initial index list.
    new_indices : ndarray
        Permuted index list

    """
    n = len(old_indices)
    assert len(old_indices) == len(new_indices)
    assert len(np.unique(old_indices)) == len(old_indices)
    assert np.all(np.unique(old_indices) == np.unique(new_indices))
    assert np.all(old_indices >= 0)
    assert np.all(new_indices >= 0)
    data = np.ones(n, dtype=np.int32)

    perm_mat = coo_matrix((data, (np.copy(old_indices), np.copy(new_indices))), shape=(n, n)).tocsc()

    assert np.all(perm_mat*old_indices == new_indices)
    assert np.all(perm_mat.transpose()*new_indices == old_indices)

    return perm_mat


def permute_matrix(A, P):
    return P*A*P.transpose()


class MSFV(object):
    """Multiscale Finite Volume Solver Class

    Parameters
    ----------

    A : sparse matrix
        matrix `A` in linear system equation `Ax=b`
    wirebasket : ndarray
        Integer labels of the rows in `A`. 0 for internal nodes, 1 for face nodes, 2 for edge nodes and  3 vertex nodes.
    primal_cell_labels : ndarray
        Array of integers assigning each row to a primal coarse cell
    source_term : ndarray
        Integer labels indicating which primal cell the rows in `A` belong to.
    div_source_term : (sparse matrix, ndarray)
        Tuple which describes the source terms constructed from a discrete divergence operator applied to a vector field.
        E.g. Capillary pressure source is (A_n, p_c) which is translated to  A_n*p_c.
    """

    def __init__(self, A, wirebasket, primal_cell_labels, source_term=None, div_source_term=None):

        assert A.shape[0] == A.shape[1], "Matrix is expected to be square"
        self._num_fine = A.shape[0]

        if source_term is not None:
            self._source_term_unordered = source_term
        else:
            self._source_term_unordered = np.zeros(self._num_fine)

        if div_source_term is not None:
            self._An_unordered = div_source_term[0]
            self._pc_unordered = div_source_term[1]
        else:
            self._An_unordered = None
            self._pc_unordered = None

        self._A_unordered = A

        self.wirebasket = wirebasket
        self.primal_cell_labels = primal_cell_labels

        self._num_coarse = np.max(primal_cell_labels) + 1
        assert np.min(primal_cell_labels) == 0

        self.id_internal_nodes = (wirebasket == 0).nonzero()[0]
        self.id_face_nodes = (wirebasket == 1).nonzero()[0]
        self.id_edge_nodes = (wirebasket == 2).nonzero()[0]
        self.id_corner_nodes = (wirebasket == 3).nonzero()[0]

        P0 = 0
        P1 = P0 + len(self.id_internal_nodes)
        P2 = P1 + len(self.id_face_nodes)
        P3 = P2 + len(self.id_edge_nodes)
        P4 = P3 + len(self.id_corner_nodes)

        self._ind_wirebasket_ptr = [P0, P1, P2, P3, P4]

        self.P, self.new_to_old_ind_map, self.old_to_new_ind_map, self.row_ptr_i, self.row_ptr_f, self.row_ptr_e = self.__get_permutation_arrays()

        self.restriction_operator_msfv = self._restriction_operator()

    def set_matrix(self, A):
        """sets matrix `A` in linear system equation `Ax=b`

        Parameters
        ----------

        A : sparse matrix
        """
        self._A_unordered = A
        assert A.shape[0] == A.shape[1], "Matrix is expected to be square"
        assert self._num_fine == A.shape[0]

    def set_div_source_term(self, div_source_term):
        """Sets source term constructed from a discrete divergence operator.
        Such source terms require different treatments than point source terms in the MSFV method.

        Parameters
        ----------

        div_source_term : (sparse matrix, ndarray)
            Tuple which describes the source terms constructed from a discrete divergence operator applied to a vector field.
        """
        self._An_unordered = div_source_term[0]
        self._pc_unordered = div_source_term[1]

    def set_source_term(self, source_term):
        self._source_term_unordered = source_term

    @staticmethod
    def get_tolerance(array, tol=1e-8):
        a = np.abs(array)
        return np.max(a)*tol

    def _get_reordered_source_term(self):
        P = self.P
        return P*self._source_term_unordered

    def _get_reordered_div_source_term(self):
        P = self.P
        if (self._An_unordered is not None) and (self._pc_unordered is not None):
            source = P*(self._An_unordered*self._pc_unordered)
        else:
            source = np.zeros(self._num_fine)
        return source

    def _get_reordered_div_source_term_reduced(self):
        P = self.P
        if (self._An_unordered is not None) and (self._pc_unordered is not None):
            An = permute_matrix(self._An_unordered, P)
            M = self._reduced_problem_matrix(An)
            source = M*P*self._pc_unordered
        else:
            source = np.zeros(self._num_fine)
        return source

    def _restriction_operator(self):
        data = np.zeros(self._num_fine)
        row = np.zeros(self._num_fine)
        col = np.zeros(self._num_fine)

        pointer = 0

        old_index = self.new_to_old_ind_map[self._ind_wirebasket_ptr[3]: self._ind_wirebasket_ptr[4]]

        assert self._num_coarse == (self._ind_wirebasket_ptr[4] - self._ind_wirebasket_ptr[3])

        for x in xrange(self._num_coarse):
            cell_label = self.primal_cell_labels[old_index[x]]
            indices = (self.primal_cell_labels == cell_label).nonzero()[0]
            row[pointer: (pointer + len(indices))] = x
            col[pointer: (pointer + len(indices))] = indices
            data[pointer: (pointer + len(indices))] = 1.0
            pointer += len(indices)

        operator = coo_matrix((data, (row, col)), shape=(self._num_coarse, self._num_fine)).tocsr()
        return operator*self.P.transpose()

    def __get_permutation_arrays(self):
        #First create permutation matrix for the different cell types,
        #with arbitrary ordering of the constituting indices

        #new_indices - Reordered indices to comply with wire-basket
        #new_new_indices - Reordered indices to comply with wire-basket and additionally by their grouping

        old_indices = np.arange(self._num_fine, dtype=np.int32)
        new_indices = np.hstack([(self.wirebasket == x).nonzero()[0] for x in xrange(4)])
        P = create_permutation_matrix(old_indices, new_indices)
        A = P*self._A_unordered*P.transpose()

        p0, p1, p2, p3, p4 = self._ind_wirebasket_ptr

        # Assert that the corner of the grid is an internal node.
        # This check is not general and should be removed for unstructured grids.
        assert self.wirebasket[self._num_fine-1] == 0
        old_to_new_indices = P.transpose()*old_indices
        assert old_to_new_indices[self._num_fine-1] == p1-1

        assert np.all(new_indices[p0:p1] == self.id_internal_nodes)
        assert np.all(new_indices[p1:p2] == self.id_face_nodes)
        assert np.all(new_indices[p2:p3] == self.id_edge_nodes)
        assert np.all(new_indices[p3:p4] == self.id_corner_nodes)

        A_submat = self._get_submatrices_A(A)

        new_new_indices = -np.ones_like(old_indices)

        graph_internal = scipy_matrix_to_igraph(A_submat['ii'])
        graph_face = scipy_matrix_to_igraph(A_submat['ff'])
        graph_edge = scipy_matrix_to_igraph(A_submat['ee'])

        graph_components = graph_internal.components()
        print "Length of internal  groups", len(graph_components)
        row_ptr_internal = np.append([0], np.cumsum([len(component) for component in graph_components]))
        reordered_indices = np.hstack(component for component in graph_components)
        new_new_indices[p0:p1] = new_indices[p0:p1][reordered_indices]

        graph_components = graph_face.components()
        print "Length of face groups", len(graph_components)
        row_ptr_face = np.append([0], np.cumsum([len(component) for component in graph_components]))
        reordered_indices = np.hstack(component for component in graph_components)
        new_new_indices[p1:p2] = new_indices[p1:p2][reordered_indices]

        graph_components = graph_edge.components()
        print "Length of edge groups", len(graph_components)
        row_ptr_edge = np.append([0], np.cumsum([len(component) for component in graph_components]))
        reordered_indices = np.hstack(component for component in graph_components)
        new_new_indices[p2:p3] = new_indices[p2:p3][reordered_indices]

        new_new_indices[p3:p4] = new_indices[p3:p4]

        P = create_permutation_matrix(old_indices, new_new_indices)

        old_to_new_ind_map = P.transpose()*old_indices

        return P, new_new_indices, old_to_new_ind_map, row_ptr_internal, row_ptr_face, row_ptr_edge

    def _get_submatrices_A(self, A):
        p0, p1, p2, p3, p4 = self._ind_wirebasket_ptr

        #assert (A-A.transpose()).nnz == 0

        A_submat = dict()

        A_submat['ii'] = A[p0:p1, p0:p1].tocsr()
        A_submat['ff'] = A[p1:p2, p1:p2].tocsr()
        A_submat['ee'] = A[p2:p3, p2:p3].tocsr()
        A_submat['nn'] = A[p3:p4, p3:p4].tocsr()

        A_submat['if'] = A[p0:p1, p1:p2].tocsr()
        A_submat['fe'] = A[p1:p2, p2:p3].tocsr()
        A_submat['en'] = A[p2:p3, p3:p4].tocsr()

        A_submat['fi'] = A[p1:p2, p0:p1].tocsr()
        A_submat['ef'] = A[p2:p3, p1:p2].tocsr()
        A_submat['ne'] = A[p3:p4, p2:p3].tocsr()

        A_submat['ie'] = A[p0:p1, p2:p3].tocsr()
        A_submat['in'] = A[p0:p1, p3:p4].tocsr()
        A_submat['fn'] = A[p1:p2, p3:p4].tocsr()

        assert A_submat['ie'].nnz == 0
        assert A_submat['in'].nnz == 0
        assert A_submat['fn'].nnz == 0

        assert ((A_submat['fi'] - A_submat['if'].transpose()).nnz == 0)
        assert ((A_submat['fe'] - A_submat['ef'].transpose()).nnz == 0)
        assert ((A_submat['en'] - A_submat['ne'].transpose()).nnz == 0)

        return A_submat

    def _reduced_problem_matrix(self, A):
        A_submat = self._get_submatrices_A(A)

        diagonal_entry = np.ravel(A_submat['if'].transpose().sum(axis=1))
        diag_matrix = sparse.spdiags(diagonal_entry, [0], len(diagonal_entry), len(diagonal_entry))
        M_ff = A_submat['ff'] + diag_matrix
        M_ff = M_ff

        diagonal_entry = np.ravel(A_submat['fe'].transpose().sum(axis=1))
        diag_matrix = sparse.spdiags(diagonal_entry, [0], len(diagonal_entry), len(diagonal_entry))
        M_ee = A_submat['ee'] + diag_matrix
        M_ee = M_ee

        zeros = sparse.identity(self._num_coarse) - sparse.identity(self._num_coarse)

        M = sparse.bmat([[A_submat['ii'], A_submat['if'], None, None], [None, M_ff, A_submat['fe'], None],
                         [None, None, M_ee, A_submat['en']], [None, None, None, zeros]]).tocsr()

        if len(A.data) > 0:
            eps_tol = self.get_tolerance(A.data)

            if len(A.data>0) and A.sum() < eps_tol:
                assert np.abs(M.sum()) < eps_tol

        assert M.shape == A.shape

        return M

    def __get_Cq(self, A_submat, lu_M_ee, lu_M_ff, lu_A_ii, q_source):
        P0, P1, P2, P3, P4 = self._ind_wirebasket_ptr

        q_i = q_source[P0:P1]
        q_f = q_source[P1:P2]
        q_e = q_source[P2:P3]

        C_33 = lu_M_ee.solve(q_e)
        C_23 = -lu_M_ff.solve(A_submat['fe']*C_33)
        C_13 = -lu_A_ii.solve(A_submat['if']*C_23)
        C_22 = lu_M_ff.solve(q_f)
        C_12 = -lu_A_ii.solve(A_submat['if']*C_22)
        C_11 = lu_A_ii.solve(q_i)

        Cq = np.zeros_like(q_source)

        Cq[P0:P1] = C_11 + C_12 + C_13
        Cq[P1:P2] = C_22 + C_23
        Cq[P2:P3] = C_33

        return Cq

    def __get_prolongation_operator(self, A_submat, M_ii, M_ff, M_ee):
        P0, P1, P2, P3, P4 = self._ind_wirebasket_ptr

        ptr_e = self.row_ptr_e
        ptr_f = self.row_ptr_f
        ptr_i = self.row_ptr_i
        solve = solve_sparse_mat_mat_lu

        mat =  A_submat['en']
        M_ee_inv_times_A_en_list = [[solve(M_ee[ptr_e[i]:ptr_e[i+1], ptr_e[i]:ptr_e[i+1]], mat[ptr_e[i]:ptr_e[i+1], :])] for i in xrange(len(ptr_e)-1)]
        M_ee_inv_times_A_en = sparse.bmat(M_ee_inv_times_A_en_list)

        mat = A_submat['fe']*M_ee_inv_times_A_en
        M_ff_inv_times_rest_list = [[solve(M_ff[ptr_f[i]:ptr_f[i+1], ptr_f[i]:ptr_f[i+1]], mat[ptr_f[i]:ptr_f[i+1], :])] for i in xrange(len(ptr_f)-1)]
        M_ff_inv_times_rest = sparse.bmat(M_ff_inv_times_rest_list)

        mat = A_submat['if']*M_ff_inv_times_rest
        M_ii_inv_times_rest = self.__M_ii_inv_times_rest(mat, M_ii)

        prolongation_operator = sparse.bmat([[-M_ii_inv_times_rest], [M_ff_inv_times_rest], [-M_ee_inv_times_A_en], [sparse.identity(P4-P3)]])

        return prolongation_operator

    def __M_ii_inv_times_rest(self, mat, M_ii):
        solve = solve_sparse_mat_mat_lu
        ptr_i = self.row_ptr_i
        M_ii_inv_times_rest_list = [[solve(M_ii[ptr_i[i]:ptr_i[i+1], ptr_i[i]:ptr_i[i+1]], mat[ptr_i[i]:ptr_i[i+1], :])] for i in xrange(len(ptr_i)-1)]
        M_ii_inv_times_rest = sparse.bmat(M_ii_inv_times_rest_list)
        return M_ii_inv_times_rest

    @staticmethod
    def __petsc_solve(lu, rhs):
        petsc_rhs = PETSc.Vec().createWithArray(rhs)
        petsc_sol = petsc_rhs.duplicate()
        lu.solve(petsc_rhs, petsc_sol)
        return petsc_sol.getArray()

    def solve(self, iterative=False, max_iter=200, restriction_operator="msfv", tol=1e-4, x0=None, recompute=True, return_stats=False):
        """
        Parameters
        ----------
        iterative: bool, optional
            Determines whether iterative procesdure (one-level V cycle) is to be used.

        max_iter: int, optional
            Maximum number of iterations. Solver will stop after maxit steps.

        restriction_operator: string, optional
            Determines the restriction operator to be used for the iterative solver.
            Acceptable values are "msfe" and "msfv"
            Note: Always uses the "msfv" operator in the last iteration

        tol: float, optional
            Tolerance at which to terminate the solver. Only valid if iterative is set to True.

        x0: array, optional
            Initial guess for the solution (a vector of zeros by default).

        recompute: bool, optional
            Determines if the prolongation operator is to be recomputed.

        return_stats: bool, optional
            Determines if the statistics of the solver are returned.

        Returns
        -------
        x: array
            approximate or converged solution

        """
        if restriction_operator not in ["msfe", "msfv"]:
            raise ValueError("restriction_operator argument error. Valid arguments are \"msfe\" or  \"msfv\"")

        P0, P1, P2, P3, P4 = self._ind_wirebasket_ptr
        sf = 1e30  # Factor used to scale the matrix and rhs so that petsc does not crash

        A = permute_matrix(self._A_unordered, self.P)

        A_submat = self._get_submatrices_A(A)

        M = self._reduced_problem_matrix(A)

        M_ii, M_ff, M_ee = M[P0:P1, P0:P1], M[P1:P2, P1:P2], M[P2:P3, P2:P3]

        lu_M_ii, lu_M_ff, lu_M_ee = (splu(M_xx) for M_xx in [M_ii, M_ff, M_ee])

        if recompute:
            self.prolongation_operator = self.__get_prolongation_operator(A_submat, M_ii, M_ff, M_ee)

        prolongation_operator = self.prolongation_operator

        if restriction_operator == "msfe":
            restriction_operator = prolongation_operator.transpose()
        elif restriction_operator == "msfv":
            restriction_operator = self.restriction_operator_msfv

        q_source = self._get_reordered_source_term()
        q_div_source = self._get_reordered_div_source_term()
        q_div_source_prime = self._get_reordered_div_source_term_reduced()

        q = q_source + q_div_source
        q_prime = q_source + q_div_source_prime

        # Solve for pressure field which is conservative at the course scale.
        M_nn_msfv = self.restriction_operator_msfv*A*prolongation_operator

        eps_tol = self.get_tolerance(M_nn_msfv.data)
        singular = M_nn_msfv.sum() < eps_tol

        if singular:
            logger.debug("Coarse System is singular")
            M_nn_msfv[0, :] = 0.0
            M_nn_msfv[0, 0] = 1.0

        lu_M_nn_msfv = get_petsc_ksp(M_nn_msfv.tocsr() * sf, pctype="lu", ksptype="preonly", tol=1e-30, max_it=1)

        Cq_prime = self.__get_Cq(A_submat, lu_M_ee, lu_M_ff, lu_M_ii, q_prime)
        q_v = self.restriction_operator_msfv*(q_source - q_div_source - A*Cq_prime)
        x_n = self.__petsc_solve(lu_M_nn_msfv, q_v * sf)
        p_f = prolongation_operator * x_n + Cq_prime

        if singular:
            coarse_residual = self.restriction_operator_msfv * (A * p_f - q)
            tol_rhs = np.max(np.abs(q))*1e-8
            assert abs(np.sum(coarse_residual)) < tol_rhs,  "tol:%e, residual:%e, Max_A:%e" % \
                                                        (tol_rhs, np.sum(coarse_residual), A.max())

        if not iterative:
            return self.P.transpose()*p_f
        else:
            if x0 is not None:
                p_f = self.P*x0
            M_nn = restriction_operator*A*prolongation_operator

            if singular:
                M_nn[0, :] = 0.0
                M_nn[0, 0] = 1.0

            lu_M_nn = get_petsc_ksp(M_nn * sf, pctype="lu", ksptype="preonly", tol=1e-25, max_it=1)

            ilu_A = get_petsc_ksp(A * sf, pctype="ilu", ksptype="richardson", tol=1e-25, max_it=10)

            n_ilu_iter = 1

            residual = q-A*p_f
            residual_prev = residual

            ref_residual_norm = norm(q - A * (q / A.diagonal()), ord=np.inf)

            for iteration in xrange(max_iter):
                # ILU preconditioning step
                for _ in xrange(n_ilu_iter):
                    error = self.__petsc_solve(ilu_A, residual * sf)
                    p_f = p_f + error
                    residual = q-A*p_f

                # Multi-scale step
                residual_coarse = restriction_operator * residual
                if singular:
                    residual_coarse[0] = 0.0

                error = self.__petsc_solve(lu_M_nn, residual_coarse*sf)
                p_f = p_f + prolongation_operator*error

                residual = q-A*p_f

                logger.debug("residual: %e", norm(residual, ord=np.inf)/ref_residual_norm)

                if norm(residual, ord=np.inf)/ref_residual_norm < tol:
                    print "Number of iterations is", iteration
                    break

                if norm(residual, ord=np.inf) >= norm(residual_prev, ord=np.inf):
                    n_ilu_iter += 1
                    logger.warning("Solver stagnated. Increasing number of iterations")

                residual_prev = residual

            residual = q - A * p_f
            print "residual after final msfe step: %e" % (norm(residual, ord=np.inf)/ref_residual_norm)
            print "tol", tol

            residual_coarse = self.restriction_operator_msfv * residual

            if singular:
                residual_coarse[0] = 0.0

            error = self.__petsc_solve(lu_M_nn_msfv, residual_coarse *sf)
            p_f = p_f + prolongation_operator*error
            residual = q-A*p_f

            logger.debug("residual after final msfv step: %e", norm(residual, ord=np.inf)/ref_residual_norm)

            print "residual after final msfv step: %e" % (norm(residual, ord=np.inf)/ref_residual_norm)

            mass_balance = self.restriction_operator_msfv*residual
            assert np.all(np.abs(mass_balance) < np.max(np.abs(q))*10e-9), "balance: %e, tol: %e" %(np.max(np.abs(mass_balance)), np.max(np.abs(q))*10e-10)

            if return_stats:
                return self.P.transpose()*p_f, iteration
            else:
                return self.P.transpose()*p_f
