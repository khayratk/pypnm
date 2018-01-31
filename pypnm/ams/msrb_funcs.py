import pymetis

from PyTrilinos import Epetra
from scipy.sparse.csgraph import connected_components
import numpy as np

from pypnm.ams.msrsb import MSRSB
from pypnm.linalg.laplacianmatrix import get_adjlist
from pypnm.linalg.trilinos_interface import matrix_scipy_to_epetra, vector_numpy_to_epetra
from pypnm.util.igraph_utils import scipy_matrix_to_igraph, coarse_graph_from_partition, support_of_basis_function

from pypnm.multiscale.multiscale_unstructured import MultiScaleSimUnstructured
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import norm

def partition_multicomponent_graph(A_scipy, v_per_subdomain=1000):
    final_labels = -np.ones(A_scipy.shape[0], dtype=np.int)
    partition_ind_shift = 0
    n_components, labels_components = connected_components(A_scipy, directed=False)

    for component_id in xrange(n_components):
        component_n = (labels_components == component_id).nonzero()[0]
        A_sub = A_scipy[component_n, :][:, component_n]
        num_subpartitions = max(len(component_n) / v_per_subdomain, 2)
        _, labels_subpartition = pymetis.part_graph(num_subpartitions, get_adjlist(A_sub))
        assert max(labels_subpartition) + 1 == num_subpartitions
        assert np.all(final_labels[component_n] == -1)
        final_labels[component_n] = np.asarray(labels_subpartition) + partition_ind_shift

        partition_ind_shift += num_subpartitions

    assert np.all(final_labels != -1), len((final_labels == -1).nonzero()[0])

    return partition_ind_shift, final_labels


def solve_with_msrsb(A_scipy, rhs, x0=None, tol=1e-5, v_per_subdomain=1000, n_smooth=10, smoother="ilu", with_multiscale=True,
                     conv_history=False, max_iter=200, tol_basis=1e-2, adapt_smoothing=True, verbose=False):

    A_epetra = matrix_scipy_to_epetra(A_scipy)

    try:
        max_diff = np.max((A_scipy - A_scipy.T).data)
    except ValueError:
        max_diff = 0

    if max_diff < norm(A_scipy)*1e-10:
        print "matrix is symmetric"
        A_b_scipy = A_scipy
        A_b_epetra = A_epetra
    else:
        print "matrix is non symmetric"
        diagonal = dia_matrix(((A_scipy + A_scipy.T) * np.ones(A_scipy.shape[0]), [0]), shape=A_scipy.shape)
        A_b_scipy = 0.5 * (A_scipy + A_scipy.T - diagonal)
        A_b_epetra = matrix_scipy_to_epetra(A_b_scipy)

    my_restriction_supports, my_basis_support = get_supports(A_b_scipy, v_per_subdomain)

    ms = MSRSB(A_b_epetra, my_restriction_supports, my_basis_support)

    ms.smooth_prolongation_operator(A_b_epetra, max_iter=10000, tol=tol_basis)

    sol = Epetra.Vector(A_epetra.RangeMap())

    if x0 is not None:
        sol[:] = x0[:]

    sol, history = ms.iterative_solve(A_epetra, vector_numpy_to_epetra(rhs), sol, tol=tol, max_iter=max_iter,
                                      n_smooth=n_smooth, smoother=smoother, with_multiscale=with_multiscale,
                                      conv_history=True, adapt_smoothing=adapt_smoothing, verbose=verbose)

    sol = np.asarray(sol)
    if conv_history:
        return sol, history
    else:
        return sol


def get_supports(A_scipy, v_per_subdomain=1000):
    comm = Epetra.PyComm()
    num_proc = comm.NumProc()

    graph = scipy_matrix_to_igraph(A_scipy)

    num_subdomains = A_scipy.get_shape()[0] / v_per_subdomain

    # create global_id attributes before creating subgraphs.
    graph.vs["global_id"] = np.arange(graph.vcount())
    graph.es["global_id"] = np.arange(graph.ecount())

    n_components, _ = connected_components(A_scipy, directed=False)
    print "number of components", n_components

    if n_components == 1:
        _, subgraph_ids_each_vertex = pymetis.part_graph(num_subdomains, graph.get_adjlist())
    else:
        _, subgraph_ids_each_vertex = partition_multicomponent_graph(A_scipy, v_per_subdomain)
        print "num of subdomains created", _

    subgraph_ids_each_vertex = np.asarray(subgraph_ids_each_vertex)
    graph.vs["subgraph_id"] = subgraph_ids_each_vertex

    # Assign a processor id to each subgraph
    coarse_graph = coarse_graph_from_partition(graph, subgraph_ids_each_vertex)

    _, proc_ids = pymetis.part_graph(num_proc, coarse_graph.get_adjlist())
    coarse_graph.vs['proc_id'] = proc_ids
    coarse_graph.vs["subgraph_id"] = np.arange(coarse_graph.vcount())

    # Assign a processor id to each pore
    subgraph_id_to_proc_id = {v["subgraph_id"]: v['proc_id'] for v in coarse_graph.vs}
    graph.vs["proc_id"] = [subgraph_id_to_proc_id[v["subgraph_id"]] for v in graph.vs]

    subgraph_ids = np.unique(subgraph_ids_each_vertex)

    subgraph_id_to_v_center_id = MultiScaleSimUnstructured.subgraph_central_vertices(graph, subgraph_ids)

    # Epetra maps to facilitate data transfer between processors
    unique_map, nonunique_map, subgraph_ids_vec = MultiScaleSimUnstructured.create_maps(graph, comm)

    epetra_importer = Epetra.Import(nonunique_map, unique_map)
    epetra_importer.NumPermuteIDs() == 0
    epetra_importer.NumSameIDs() == unique_map.NumMyElements()

    my_basis_support = dict()

    my_global_elements = unique_map.MyGlobalElements()

    graph["global_to_local"] = dict((v["global_id"], v.index) for v in graph.vs)
    graph["local_to_global"] = dict((v.index, v["global_id"]) for v in graph.vs)

    my_restriction_supports = dict()

    subgraph_id_vec = np.asarray(graph.vs['subgraph_id'])
    for i in subgraph_ids:
        vs_subgraph = (subgraph_id_vec == i).nonzero()[0]
        my_restriction_supports[i] = graph.vs.select(vs_subgraph)["global_id"]

    for i in subgraph_ids:
        support_vertices = support_of_basis_function(i, graph, coarse_graph, subgraph_id_to_v_center_id,
                                                     my_restriction_supports)

        my_basis_support[i] = np.intersect1d(support_vertices, my_global_elements).astype(np.int32)

    return my_restriction_supports, my_basis_support
