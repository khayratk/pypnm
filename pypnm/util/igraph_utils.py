import igraph as ig

import numpy as np


def coarse_graph_from_partition(g, partition):
    g = g.copy()
    g.contract_vertices(partition)
    g.simplify()
    return g


def graph_central_vertex(g):
    closeness = np.array(g.closeness())
    v_central = np.argmax(closeness)
    return v_central


def support_of_basis_function(basis_id, graph, coarse_graph, basis_id_to_v_center_id, my_restriction_supports):
    """
    Computes the global indices of vertices belonging to the support of the basis function

    Parameters
    ----------
    basis_id : int
        Id of basis function
    graph : igraph Graph
        Graph representing e.g. a fine-scale grid or pore-network.
        Vertex attributes "subgraph_id", which identifies the subgraph that a vertex belongs to,
        and "global_id", which attributes the global id of a vertex, have to be defined.
    coarse_graph : igraph Graph
        Graph representing e.g. the coarse-scale grid or grid of subnetwork
    basis_id_to_v_center_id: dictionary
        Mapping from id of basis function to the global id of the basis function' central vertex

    my_restriction_supports: dict
        Mapping from id of basis function to corresponding support of the restriction operator
    Returns
    -------
    b : list
        list of global vertex indices belonging to the support of the basis function

    """
    assert "subgraph_id" in graph.vs.attributes()
    assert "global_id" in graph.vs.attributes()

    assert len(graph.vs['global_id']) == len(np.unique(graph.vs['global_id']))

    ngh_basis_ids = list(set(coarse_graph.neighbors(basis_id)).intersection(basis_id_to_v_center_id.keys()))

    v_center_global = basis_id_to_v_center_id[basis_id]
    v_center_local = graph["global_to_local"][v_center_global]

    ngh_vs_centers_global = [basis_id_to_v_center_id[i] for i in ngh_basis_ids]
    ngh_vs_center_local = [graph["global_to_local"][i] for i in ngh_vs_centers_global]
    distances = graph.shortest_paths(v_center_local, ngh_vs_center_local)[0]
    basis_id_to_max_distance = dict((i, j) for i, j in zip(ngh_basis_ids, distances))

    max_distance = max(basis_id_to_max_distance.values())

    basis_id_to_max_distance[basis_id] = 1000000

    pore_list = []
    allowed_coarse_ids = [basis_id] + ngh_basis_ids

    bfs_iter = graph.bfsiter(v_center_local, advanced=True)
    for v, dist, v_parent in bfs_iter:
        if dist == max_distance:
            break

        if v['subgraph_id'] in allowed_coarse_ids:
            if dist < basis_id_to_max_distance[v['subgraph_id']]:
                pore_list.append(v.index)

    assert v_center_local in pore_list

    basis_support = graph.vs[pore_list]['global_id']

    assert len(np.unique(basis_support)) == len(basis_support)

    for i in ngh_basis_ids:
        ngh_v_center_global = basis_id_to_v_center_id[i]
        assert ngh_v_center_global not in basis_support, max_distance

    # Ensure that the basis_support is a superset of the restriction support
    restriction_support = my_restriction_supports[basis_id]
    basis_support = list(set(basis_support))
    basis_support = list(set(basis_support).union(restriction_support))
    assert len(basis_support) > len(restriction_support)

    return basis_support


def scipy_matrix_to_igraph(matrix, directed=False):
    sources, targets = matrix.nonzero()
    edges = np.vstack((sources, targets)).T.tolist()
    return ig.Graph(edges, directed=directed).simplify()


def network_to_igraph(network, vertex_mask=None, edge_mask=None, edge_attributes=None, vertex_attributes=None):
    """
    Parameters
    ----------
    network: PoreNetwork
    vertex_mask: ndarray bool
    edge_mask: ndarray bool
    edge_attributes: string array
    vertex_attributes: string array

    Returns
    -------
    out: igraph

    """

    if vertex_mask is None:
        vertex_mask = np.ones_like(network.pores.invaded)

    if edge_mask is None:
        edge_mask = np.ones_like(network.tubes.invaded)

    A = network.edgelist
    G = ig.Graph()
    G.add_vertices(network.nr_p)

    pore_list1 = A[:, 0]  # First column of tube adjacency list
    pore_list2 = A[:, 1]  # Second column of tube adjacency list
    pore_status1 = vertex_mask[pore_list1]
    pore_status2 = vertex_mask[pore_list2]

    # Add only edges to graph which are marked and whose neighboring pores are also marked
    edge_indices = np.nonzero((pore_status1 + pore_status2 + edge_mask) == 3)[0]
    H = A[edge_indices, :]
    G.add_edges(H)

    if edge_attributes is not None:
        for attr in edge_attributes:
            G.es[attr] = getattr(network.tubes, attr)[edge_indices]

    if vertex_attributes is not None:
        for attr in vertex_attributes:
            G.vs[attr] = getattr(network.pores, attr)

    return G