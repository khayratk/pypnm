from scipy.sparse import csr_matrix
from PyTrilinos import Epetra
from pypnm.linalg.trilinos_interface import matrix_scipy_to_epetra, vector_numpy_to_epetra
import pymetis
from pypnm.multiscale.multiscale_sim import MultiScaleSimUnstructured
from pypnm.attribute_calculators.conductance_calc import ConductanceCalc
from pypnm.linalg.laplacianmatrix import LaplacianMatrix
from pypnm.porenetwork.network_factory import structured_network
import numpy as np
from pypnm.util.igraph_utils import coarse_graph_from_partition, graph_central_vertex, support_of_basis_function, \
    scipy_matrix_to_igraph
from pypnm.ams.msrsb import MSRSB
from pypnm.linalg.linear_system_solver import solve_pyamg
import cProfile
import pstats


def solve_multiscale(ms, A, b, tol=1e-10):
    ms.A = A
    ms.smooth_prolongation_operator(100)
    sol = Epetra.Vector(A.RangeMap())
    sol = ms.iterative_solve(b, sol, tol=tol, max_iter=5000)
    return sol


def solve_with_msrsb(ia, ja, a, rhs):
    comm = Epetra.PyComm()
    num_proc = comm.NumProc()

    A_scipy = csr_matrix((a, ia, ja))
    graph = scipy_matrix_to_igraph(A_scipy)

    A = matrix_scipy_to_epetra(A_scipy)

    num_subnetworks = A_scipy.get_shape()[0] / 1000

    print "num of subgraphs created", num_subnetworks
    # create global_id attributes before creating subgraphs.
    graph.vs["global_id"] = np.arange(graph.vcount())
    graph.es["global_id"] = np.arange(graph.ecount())

    _, subgraph_ids_each_vertex = pymetis.part_graph(num_subnetworks, graph.get_adjlist())

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
        support_vertices = support_of_basis_function(i, graph, coarse_graph, subgraph_id_to_v_center_id, my_restriction_supports)

        my_basis_support[i] = np.intersect1d(support_vertices, my_global_elements).astype(np.int32)

    ms = MSRSB(A, my_restriction_supports, my_basis_support)
    return np.asarray(solve_multiscale(ms, A, vector_numpy_to_epetra(rhs), tol=1.0e-10))


def test_msrsb():
    nx, ny, nz = 10, 10, 10
    n_fine_per_cell = 10
    network = structured_network(Nx=n_fine_per_cell * nx, Ny=n_fine_per_cell * ny, Nz=n_fine_per_cell * nz)
    k_computer = ConductanceCalc(network)
    k_computer.compute()

    A = LaplacianMatrix(network)
    A.set_edge_weights(network.tubes.k_w + network.tubes.k_n)
    A = A.get_csr_matrix()
    rhs = np.zeros(network.nr_p)
    rhs[0] = 1e-10
    rhs[100] = -1e-10
    sol = solve_with_msrsb(A.indices, A.indptr, A.data, rhs)

    A[1, :] = 0.0
    A[1, 1] = 1.0
    sol_exact = solve_pyamg(A, rhs, tol=1e-10)

    sol = (sol - np.min(sol))
    sol_exact = (sol_exact-np.min(sol_exact))
    error = sol - sol_exact

    Linf = np.linalg.norm(error, ord=np.inf) / np.linalg.norm(sol_exact, ord=np.inf)
    L2_error = np.linalg.norm(error, ord=2) / np.linalg.norm(sol_exact, ord=2)
    assert ((Linf < 10e-2) & (L2_error < 10e-5)), "L1 error: %s  L2 error: %s" % (Linf, L2_error)


def print_profiling_info(filename):
    p = pstats.Stats(filename)
    p.sort_stats('cumulative').print_stats(20)
    p.sort_stats('time').print_stats(20)


if __name__ == "__main__":
    cProfile.run('test_msrsb()', 'restats')
    print_profiling_info('restats')
