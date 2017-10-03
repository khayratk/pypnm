import pymetis
from PyTrilinos import Epetra
from scipy.sparse import csr_matrix
import numpy as np
from pypnm.ams.msrsb import MSRSB
from pypnm.linalg.trilinos_interface import matrix_scipy_to_epetra, vector_numpy_to_epetra
from pypnm.multiscale.multiscale_unstructured import MultiScaleSimUnstructured
from pypnm.util.igraph_utils import scipy_matrix_to_igraph, coarse_graph_from_partition, support_of_basis_function


def solve_multiscale(ms, A, b, tol=1e-5):
    ms.A = A
    ms.smooth_prolongation_operator(100)
    sol = Epetra.Vector(A.RangeMap())
    sol = ms.iterative_solve(b, sol, tol=tol, max_iter=5000)
    return sol


def solve_with_msrsb(ia, ja, a, rhs, tol=1e-3, v_per_subdomain=1000):
    comm = Epetra.PyComm()
    num_proc = comm.NumProc()

    A_scipy = csr_matrix((a, ia, ja))
    graph = scipy_matrix_to_igraph(A_scipy)

    A = matrix_scipy_to_epetra(A_scipy)

    num_subdomains = A_scipy.get_shape()[0] / v_per_subdomain

    print "num of subdomains created", num_subdomains

    # create global_id attributes before creating subgraphs.
    graph.vs["global_id"] = np.arange(graph.vcount())
    graph.es["global_id"] = np.arange(graph.ecount())

    _, subgraph_ids_each_vertex = pymetis.part_graph(num_subdomains, graph.get_adjlist())

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

    ms = MSRSB(A, my_restriction_supports, my_basis_support)
    return np.asarray(solve_multiscale(ms, A, vector_numpy_to_epetra(rhs), tol=tol)), ms