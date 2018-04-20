import itertools

import numpy as np
from scipy.sparse import csr_matrix, eye, coo_matrix, csc_matrix


def get_adjlist(A):
    """
    Parameters
    ----------
    A: Scipy matrix

    Returns
    -------
    adjacency list of equivalent undirected graph
    """

    A = A.tocsr()
    indices = A.indices
    indptr = A.indptr
    adjlist = [filter(lambda x: x != i, indices[indptr[i]:indptr[i+1]].tolist()) for i in xrange(A.shape[0])]

    return adjlist


def _laplacian(edgelist, weights=None, ind_dirichlet=None):
    rows = edgelist[0]
    cols = edgelist[1]
    nv = max(np.max(rows), np.max(cols)) + 1

    if weights is None:
        weights = np.ones(len(rows))
    else:
        if len(weights) != len(rows):
            raise ValueError("The number of weights have to be equal to number of edges")

    # Make matrix symmetric
    rows, cols = np.hstack((rows, cols)), np.hstack((cols, rows))
    data = -np.hstack((weights, weights))

    diagonal_dirichlet = np.zeros(nv)

    if ind_dirichlet is not None:
        mask = np.zeros(nv, dtype=np.bool)
        mask[ind_dirichlet] = True
        mask_rows = mask[rows]  # Selects all the entries belonging to rows which have a dirichlet value
        data[mask_rows] = 0.0

        diagonal_dirichlet[ind_dirichlet] = 1.0

    off_diag = csr_matrix((data, (rows, cols)), shape=(nv, nv))

    diag = eye(nv, format="csr")
    diag.setdiag(-(off_diag*np.ones(nv)) + diagonal_dirichlet)

    return off_diag + diag


def laplacian_from_network(network, weights=None, ind_dirichlet=None):
    """
    Parameters
    ----------
    network: PoreNetwork
    weights: array_like, optional
        edge weights
    ind_dirichlet: array_like, optional
        indices of vertices where the corresponding rows in the laplacian
        will have single entries of 1.0 on the diagonal

    Returns
    -------
    out: Laplacian matrix in scipy format

    """
    row = network.edgelist[:, 0]
    col = network.edgelist[:, 1]
    return _laplacian((row, col), weights, ind_dirichlet)


def laplacian_from_igraph(graph, weights=None, ind_dirichlet=None):
    """
    Parameters
    ----------
    network: igraph
    weights: ndarray, optional
        edge weights
    ind_dirichlet, ndarray, optional
        indices of vertices where the corresponding rows in the laplacian
        will have single entries of 1.0 on the diagonal

    Returns
    -------
    out: Laplacian matrix in scipy format

    """
    edges = graph.get_edgelist()
    edges = zip(*edges)
    row = edges[0]
    col = edges[1]

    return _laplacian((row, col), weights, ind_dirichlet)


class LaplacianMatrix(object):
    """
    Class to store the laplacian matrix obtained from a network and efficiently update its entries.

    Parameters
    ----------
    network: PoreNetwork

    Notes
    ----------
    This class when several laplacian matrices need to be created with the same structure
    """

    def __init__(self, network):
        self.coo_len = 2 * network.nr_t + network.nr_p
        self.data = np.zeros(self.coo_len)
        self.N = network.nr_p

        self.row, self.col = self.__get_row_and_col_arrays(network)
        self._edge_to_data_ind = np.hstack(network.ngh_tubes.flat).astype(np.int)  # hstack does not preserve type

        assert len(self._edge_to_data_ind) == 2 * network.nr_t

        self.m_ones = -np.ones(self.N)

        self.__create_index_lists(self.row, self.col)

    def __get_row_and_col_arrays(self, network):
        """
        Storage format: COL: [sorted(N_1_1 N_1_2 N_1_3 N_1_4 P_1)  sorted(N_2_1 N_2_2 N_2_3 N_2_4 P_2) ...]
                        ROW: [P_1    P_1   P_1   P_1   P_1  P_2   P_2   P_2   P_2   P_2 ...]

        N_n_m:  Vertex id of mth neighbour of nth vertex
        P_1, P_2... always in increasing order.
        """

        row = [(network.nr_nghs[p_i] + 1) * [p_i] for p_i in xrange(self.N)]
        row = np.asarray(list(itertools.chain(*row)), dtype=np.int32)

        col = [sorted(network.ngh_pores[p_i].tolist() + [p_i]) for p_i in xrange(self.N)]
        col = np.asarray(list(itertools.chain(*col)), dtype=np.int32)

        assert (len(row[row < 0]) == 0)
        assert (len(col[col < 0]) == 0)
        assert (len(col) == self.coo_len)
        assert (len(row) == self.coo_len), "num rows %d coo len %d" % (len(row), self.coo_len)

        return row, col

    def __create_index_lists(self, row, col):
        data_nondiag_mask = (row != col)
        self._data_diag_mask = (row == col)
        self._data_diag_ind = self._data_diag_mask.nonzero()[0]
        self._data_nondiag_ind = data_nondiag_mask.nonzero()[0]
        assert len(self._data_diag_ind) == self.N
        assert np.sum(data_nondiag_mask) == len(self._edge_to_data_ind)

    def set_edge_weights(self, weights):
        """
        Sets the coefficients of the laplacian matrix with the provided edge weights.

        Parameters
        ----------

        weights : ndarray
           Edge weights

        """
        nr_p = self.N
        self.data[:] = 0.0

        # Set non-diagonal entries
        self.data[self._data_nondiag_ind] = -weights[self._edge_to_data_ind]

        # Set diagonal entries
        A = csr_matrix((self.data, (self.row, self.col)), shape=(nr_p, nr_p))
        self.data[self._data_diag_ind] = A * self.m_ones

        if np.sum(weights) > 0.0:
            tol = max(abs(self.data.min()), self.data.max()) / 1.0e10
            assert np.isclose(np.sum(self.data), 0.0, atol=tol)

    def fill_csr_matrix_with_edge_weights(self, csr_matrix, weights):
        """
        Resets the coefficient matrix of the csr_matrix with the provided weights of the network edges.

        Parameters
        ----------
        csr_matrix: scipy.sparse.csr_matrix
            A csr_matrix generated from LaplacianMatrix
        weights : ndarray
            Weights of the network edges

        Notes
        -----
        This is currently more efficient than using scipy's inbuilt functions.

        """
        csr_matrix.data.fill(0.0)
        csr_matrix.data[self._data_nondiag_ind] = -weights.take(self._edge_to_data_ind, axis=0)
        csr_matrix.data[self._data_diag_ind] = csr_matrix * self.m_ones

    def set_csr_matrix_rows_to_dirichlet(self, csr_matrix, row_indices, val=1.0):
        """
        Deletes selected rows in matrix and inserts 1.0 on their diagonal entries

        Parameters
        ----------
        csr_matrix: scipy csr matrix
        row_indices: array
            indices of rows for which the dirichlet boundary conditions are set

        """
        csr_matrix.data[self._data_diag_ind[row_indices]] = val

        mask = self.mask_from_indices(row_indices, self.N)
        # data_diag_mask_bnd = (mask[self.row]) & (mask[self.col]) & self._data_diag_mask
        # data_nondiag_mask_bnd = (mask[self.row]) & np.logical_not(data_diag_mask_bnd)
        data_nondiag_mask_bnd = np.logical_not(self._data_diag_mask) & (mask.take(self.row))
        csr_matrix.data[data_nondiag_mask_bnd] = 0.0

    def set_csr_singular_rows_to_dirichlet(self, csr_matrix):
        """
        Finds rows in matrix which have no entries (i.e. singular) and inserts 1.0 on their diagonal entries

        Parameters
        ----------
        csr_matrix: scipy csr matrix
        row_indices: array
            indices of rows for which the dirichlet boundary conditions are set

        """
        data_diag_mask = np.zeros(self.coo_len, dtype=np.bool)
        data_diag_mask[self._data_diag_ind] = True
        csr_matrix.data[data_diag_mask & (csr_matrix.data == 0.0)] = 1.0

    @staticmethod
    def mask_from_indices(indices, N):
        mask = np.zeros(N, dtype=np.bool)
        mask[indices] = True
        return mask

    def set_selected_rows_to_dirichlet(self, row_indices):
        """
        Deletes selected rows in matrix and inserts 1.0 on their diagonal entries

        Parameters
        ----------
        row_indices: array
            indices of rows for which the dirichlet boundary conditions are set

        """
        if type(row_indices).__module__ == np.__name__:
            assert np.issubdtype(row_indices.dtype, np.integer)

        self.data[self._data_diag_ind[row_indices]] = 1.0
        self.set_offdiag_row_entries_to_zero(row_indices)
        self.set_singular_rows_to_dirichlet()

    def set_offdiag_col_entries_to_zero(self, col_indices):
        mask = self.mask_from_indices(col_indices, self.N)
        data_diag_mask = (mask[self.row]) & (mask[self.col]) & self._data_diag_mask
        data_nondiag_mask = (mask[self.col]) & np.logical_not(data_diag_mask)
        self.data[data_nondiag_mask] = 0.0

    def set_offdiag_row_entries_to_zero(self, row_indices):
        mask = self.mask_from_indices(row_indices, self.N)
        data_diag_mask = (mask[self.row]) & (mask[self.col]) & self._data_diag_mask
        data_nondiag_mask = (mask[self.row]) & np.logical_not(data_diag_mask)
        self.data[data_nondiag_mask] = 0.0

    def set_singular_rows_to_dirichlet(self):
        data_diag_mask = np.zeros(self.coo_len, dtype=np.bool)
        data_diag_mask[self._data_diag_ind] = True
        self.data[data_diag_mask & (self.data == 0.0)] = 1.0

    def get_coo_matrix(self):
        nr_p = self.N
        return coo_matrix((np.copy(self.data), (np.copy(self.row), np.copy(self.col))), shape=(nr_p, nr_p))

    def get_csr_matrix(self):
        nr_p = self.N
        return csr_matrix((np.copy(self.data), (np.copy(self.row), np.copy(self.col))), shape=(nr_p, nr_p))

    def get_csc_matrix(self):
        nr_p = self.N
        return csc_matrix((np.copy(self.data), (np.copy(self.row), np.copy(self.col))), shape=(nr_p, nr_p))