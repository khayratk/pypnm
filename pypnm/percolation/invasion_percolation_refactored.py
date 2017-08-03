import numpy as np
from heapq import heappop, heappush

def site_bond_invasion_percolation(graph, edge_weights, vertex_weights, source_vertices,
                                   clusters = dict(), cluster_id_vertices=None, cluster_id_edges=None):
    """
    Parameters
    ----------
    graph: igraph
    edge_weights: ndarray
    vertex_weights: ndarray
    source_vertices: list
        source vertices
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
    """

    if not isinstance(edge_weights, (np.ndarray, np.generic)):
        raise TypeError("Edge weights should be given as an ndarray")

    if cluster_id_vertices is None:
        cluster_id_vertices = -np.ones(graph.vcount(), dtype=np.int)

    if cluster_id_edges is None:
        cluster_id_edges = -np.ones(graph.ecount(), dtype=np.int)

    ngh_vertices = graph.get_adjlist()
    ngh_edges = graph.get_inclist()
    edge_list = graph.get_edgelist()

    frontier_edge_marker = np.zeros(graph.ecount(), dtype=np.bool)
    frontier_vertex_marker = np.zeros(graph.vcount(), dtype=np.bool)

    edge_invaded = np.zeros(graph.ecount(), dtype=np.bool)
    vertex_invaded = np.zeros(graph.vcount(), dtype=np.bool)

    frontier_edges = []  # priority queue with list of tuples storing (entry pressure, edge index)
    frontier_vertices = []  # priority queue with list of tuples storing (entry pressure,  vertex index)

    for v in source_vertices:
        for e_ngh in ngh_edges[v]:
            if frontier_edge_marker[e_ngh] == 0:
                heappush(frontier_edges, (edge_weights[e_ngh], e_ngh))
                frontier_edge_marker[e_ngh] = 1

    while frontier_edges or frontier_vertices:
        # Invade edge
        if (not frontier_vertices) or frontier_edges[0][0] > frontier_vertices[0][0]:
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
                cluster = clusters.pop(c_id)
                for e1 in cluster[1]:
                    frontier_edge_marker[e1] = 1
                for v1 in cluster[0]:
                    frontier_vertex_marker[v1] = 1
                    cluster_id_vertices[v1] = -1
                    for e_ngh in ngh_edges[v]:
                        if frontier_edge_marker[e_ngh] == 0:
                            heappush(frontier_edges, (edge_weights[e_ngh], e_ngh))
                            frontier_edge_marker[e_ngh] = 1

        elif frontier_vertices:
            weight, v = heappop(frontier_vertices)
            if vertex_invaded[v] == 0:
                vertex_invaded[v] = 1
                yield (1, v, weight)

            # Check for reconnecting clusters and add their periphery to  the frontier
            for e_ngh in ngh_edges[v]:
                if cluster_id_edges[e_ngh] >= 0:
                    c_id = cluster_id_edges[e_ngh]
                    cluster = clusters.pop(c_id)
                    for e1 in cluster[1]:
                        frontier_edge_marker[e1] = 1
                    for v1 in cluster[0]:
                        cluster_id_vertices[v1] = -1
                        for v_ngh, e_ngh in zip(ngh_vertices[v], ngh_edges[v]):
                            if frontier_edge_marker[e_ngh] == 0:
                                heappush(frontier_edges, (edge_weights[e_ngh], e_ngh, v_ngh))
                                frontier_edge_marker[e_ngh] = 1

        for v_ngh, e_ngh in zip(ngh_vertices[v], ngh_edges[v]):
            if frontier_edge_marker[e_ngh] == 0:
                heappush(frontier_edges, (edge_weights[e_ngh], e_ngh, v_ngh))
                frontier_edge_marker[e_ngh] = 1
