from heapq import heappop, heappush

import numpy as np


def _add_cluster_to_frontier(c_id, clusters, frontier_vertices, frontier_edges,
                             frontier_vertex_marker, frontier_edge_marker,
                             cluster_id_vertices, cluster_id_edges,
                             edge_weights, vertex_weights, edge_list, ngh_edges):
    """
    Adds all vertices and edges of cluster with id c_id to frontier.
    """
    cluster_vertices, cluster_edges = clusters.pop(c_id)

    for e1 in cluster_edges:
        frontier_edge_marker[e1] = 1
        cluster_id_edges[e1] = -1
        for v_ngh in edge_list[e1]:
            if frontier_vertex_marker[v_ngh] == 0:
                heappush(frontier_vertices, (vertex_weights[v_ngh], v_ngh))
                frontier_vertex_marker[v_ngh] = 1

    for v1 in cluster_vertices:
        frontier_vertex_marker[v1] = 1
        cluster_id_vertices[v1] = -1
        for e_ngh in ngh_edges[v1]:
            if frontier_edge_marker[e_ngh] == 0:
                heappush(frontier_edges, (edge_weights[e_ngh], e_ngh))
                frontier_edge_marker[e_ngh] = 1

    return (clusters, frontier_vertices, frontier_edges,
            frontier_vertex_marker, frontier_edge_marker,
            cluster_id_vertices, cluster_id_edges)


def site_bond_invasion_percolation(graph, edge_weights, source_vertices,
                                   clusters=dict(), vertex_weights=None, cluster_id_vertices=None,
                                   cluster_id_edges=None):
    """
    Parameters
    ----------
    graph: igraph
    edge_weights: ndarray
    vertex_weights: ndarray
    source_vertices: list
        source vertices for invasion percolation
    clusters (optional): dictionary
        dictionary mapping cluster id to cluster given in the form of (V,E)
    cluster_id_vertices (optional): ndarray
        id of cluster that each vertex belongs to. -1 if not part of cluster
    cluster_id_edges (optional): ndarray
        id of cluster that each edge belongs to. -1 if not part of cluster
    Returns
    -------
    invasion_events: list
        list of tuples in the form (edge_id, vertex_id, weight) corresponding to invasion events

    Notes
    ------
    If cluster_id is positive, then vertex/edge is assumed to be connected
    """

    if not isinstance(edge_weights, (np.ndarray, np.generic)):
        raise TypeError("Edge weights should be given as an ndarray")

    if cluster_id_vertices is None:
        cluster_id_vertices = -np.ones(graph.vcount(), dtype=np.int)

    if cluster_id_edges is None:
        cluster_id_edges = -np.ones(graph.ecount(), dtype=np.int)

    if vertex_weights is None:
        vertex_weights = np.zeros(graph.vcount(), dtype=np.int)

    ngh_vertices = graph.get_adjlist()
    ngh_edges = graph.get_inclist()
    edge_list = graph.get_edgelist()

    # Markers arrays to mark if the vertex or edge has already been added to the frontier queue
    frontier_edge_marker = np.zeros(graph.ecount(), dtype=np.bool)
    frontier_vertex_marker = np.zeros(graph.vcount(), dtype=np.bool)

    edge_invaded = np.zeros(graph.ecount(), dtype=np.bool)
    vertex_invaded = np.zeros(graph.vcount(), dtype=np.bool)

    edge_invaded[cluster_id_edges >= 0] = 1
    vertex_invaded[cluster_id_vertices >= 0] = 1

    frontier_edges = []  # priority queue with list of tuples storing (entry pressure, edge index)
    frontier_vertices = []  # priority queue with list of tuples storing (entry pressure,  vertex index)

    for v in source_vertices:
        for e_ngh in ngh_edges[v]:
            if frontier_edge_marker[e_ngh] == 0:
                heappush(frontier_edges, (edge_weights[e_ngh], e_ngh))
                frontier_edge_marker[e_ngh] = 1

    while frontier_edges or frontier_vertices:
        # Invade edge if there are no frontier vertices or if the entry pressure of the frontier edge is smaller
        # than that of the frontier vertex
        if (not frontier_vertices) or (frontier_edges and (frontier_edges[0][0] < frontier_vertices[0][0])):
            weight, e = heappop(frontier_edges)
            if edge_invaded[e] == 0:
                edge_invaded[e] = 1
                yield (0, e, weight)

            # Check for reconnecting cluster and add its periphery to the frontier
            v1, v2 = edge_list[e]
            v = None
            if frontier_vertex_marker[v1] == 0:
                v = v1
            if frontier_vertex_marker[v2] == 0:
                v = v2

            if (v is not None) and (cluster_id_vertices[v] >= 0):
                c_id = cluster_id_vertices[v]

                (clusters, frontier_vertices, frontier_edges, frontier_vertex_marker, frontier_edge_marker,
                 cluster_id_vertices,
                 cluster_id_edges) = _add_cluster_to_frontier(c_id, clusters, frontier_vertices, frontier_edges,
                                                              frontier_vertex_marker, frontier_edge_marker,
                                                              cluster_id_vertices, cluster_id_edges,
                                                              edge_weights, vertex_weights, edge_list,
                                                              ngh_edges)

            # If vertex v does not belong to a cluster, add it to frontier vertices.
            elif (v is not None) and (cluster_id_vertices[v] == -1):
                heappush(frontier_vertices, (vertex_weights[v], v))
                frontier_vertex_marker[v] = 1

        elif frontier_vertices:
            weight, v = heappop(frontier_vertices)
            if vertex_invaded[v] == 0:
                vertex_invaded[v] = 1
                yield (1, v, weight)

            # Check for reconnecting clusters and add their periphery to the frontier
            for e_ngh in ngh_edges[v]:
                if frontier_edge_marker[e_ngh] == 0:

                    if cluster_id_edges[e_ngh] >= 0:
                        c_id = cluster_id_edges[e_ngh]

                        (clusters, frontier_vertices, frontier_edges, frontier_vertex_marker, frontier_edge_marker,
                         cluster_id_vertices,
                         cluster_id_edges) = _add_cluster_to_frontier(c_id, clusters, frontier_vertices, frontier_edges,
                                                                      frontier_vertex_marker, frontier_edge_marker,
                                                                      cluster_id_vertices, cluster_id_edges,
                                                                      edge_weights, vertex_weights, edge_list,
                                                                      ngh_edges)

                    elif cluster_id_edges[e_ngh] == -1:
                        heappush(frontier_edges, (edge_weights[e_ngh], e_ngh))
                        frontier_edge_marker[e_ngh] = 1
