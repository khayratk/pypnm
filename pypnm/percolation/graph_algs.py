import numpy as np
from pypnm.porenetwork.constants import *
from pypnm.util.igraph_utils import network_to_igraph
import pyximport
import pyximport
from pyximport import install

old_get_distutils_extension = pyximport.pyximport.get_distutils_extension

def new_get_distutils_extension(modname, pyxfilename, language_level=None):
    extension_mod, setup_args = old_get_distutils_extension(modname, pyxfilename, language_level)
    extension_mod.language='c++'
    return extension_mod,setup_args

pyximport.pyximport.get_distutils_extension = new_get_distutils_extension


pyximport.install(setup_args={"include_dirs": np.get_include()},
                  reload_support=True)

import graph_algs_cython

import itertools


def bfs_vertices(network, source, flag_pore, flag_tube, max_level=99999999, max_count=999999999):
    return graph_algs_cython.bfs_vertices(network, source, flag_pore, flag_tube, max_level, max_count)


def bfs_vertices_nw(network, source, max_level=99999999, max_count=999999999):
    return bfs_vertices(network, source, network.pores.connected, network.tubes.connected, max_level, max_count)


def bfs_vertices_nw_conn(network, source, max_level=99999999, max_count=999999999):
    flag_pore = np.minimum(network.pores.connected, network.pores.invaded)
    flag_tube = np.minimum(network.tubes.connected, network.tubes.invaded)

    return bfs_vertices(network, source, flag_pore, flag_tube, max_level, max_count)


def bfs_vertices_nw_disconn(network, source, max_level=99999999, max_count=999999999):
    flag_pore = network.pores.invaded - network.pores.connected
    flag_tube = network.tubes.invaded - network.tubes.connected
    return bfs_vertices(network, source, flag_pore, flag_tube, max_level, max_count)


def pore_connected_to_inlet(network, source):
    return graph_algs_cython.pore_connected_to_inlet(network, source)


def graph_igraph(network):
    return network_to_igraph(network)


def subgraph_nwett_igraph(network):
    return network_to_igraph(network, network.pores.invaded, network.tubes.invaded)


def subgraph_wett_igraph(network):
    return network_to_igraph(network, 1-network.pores.invaded, 1-network.tubes.invaded)


def subgraph_conn_igraph(network):
    return network_to_igraph(network, network.pores.connected, network.tubes.connected)


def get_pore_and_tube_connectivity_to_pore_list(network, pore_list):
    pores_connected = np.zeros(network.nr_p, dtype=np.int32)
    tubes_connected = np.zeros(network.nr_t, dtype=np.int32)

    G = graph_igraph(network)
    source = network.nr_p

    # Add dummy nodes connected to inlet and outlet nodes
    G.add_vertices(1)
    H = [(source, x) for x in pore_list]
    G.add_edges(H)
    components = G.subcomponent(source)
    components.remove(source)
    pores_connected[list(components)] = 1

    # Set tubes connectivity
    p1 = network.edgelist[:, 0]
    p2 = network.edgelist[:, 1]
    tubes_connected[:] = (pores_connected[p1] | pores_connected[p2])

    return pores_connected, tubes_connected


def get_pore_and_tube_nonwetting_connectivity_to_pore_list(network, pore_list):
    G = subgraph_nwett_igraph(network)

    # Add dummy node connected to pore_list
    G.add_vertices(1)

    # Index of dummy node
    pi_dummy = network.nr_p

    H = [(pi_dummy, x) for x in pore_list if network.pores.invaded[x] == NWETT]
    G.add_edges(H)
    components = G.subcomponent(pi_dummy)
    components.remove(pi_dummy)

    pores_connected = np.zeros(network.nr_p, dtype=np.int32)
    pores_connected[list(components)] = 1

    # Set tubes connectivity
    p1 = network.edgelist[:, 0]
    p2 = network.edgelist[:, 1]
    tubes_connected = (pores_connected[p1] | pores_connected[p2]) & network.tubes.invaded

    return pores_connected.astype(np.int8), tubes_connected.astype(np.int8)


def get_pore_and_tube_wetting_connectivity_to_pore_list(network, pore_list):
    G = subgraph_wett_igraph(network)

    # Add dummy node connected to pore_list
    G.add_vertices(1)

    # Index of dummy node
    pi_dummy = network.nr_p

    H = [(pi_dummy, x) for x in pore_list if network.pores.invaded[x] == WETT]
    G.add_edges(H)
    components = G.subcomponent(pi_dummy)
    components.remove(pi_dummy)

    pores_connected = np.zeros(network.nr_p, dtype=np.int32)
    pores_connected[list(components)] = 1

    # Set tubes connectivity
    p1 = network.edgelist[:, 0]
    p2 = network.edgelist[:, 1]
    tubes_connected = (pores_connected[p1] | pores_connected[p2]) & (1-network.tubes.invaded)

    return pores_connected.astype(np.int8), tubes_connected.astype(np.int8)


def get_pore_and_tube_connected_masks_to_pore_list(network, pore_list, pore_invasion_mask, tube_invasion_mask):
    G = subgraph_nwett_igraph(network)

    #Add dummy node connected to pore_list
    G.add_vertices(1)

    #Index of dummy node
    pi_dummy = network.nr_p

    H = [(pi_dummy, x) for x in pore_list if pore_invasion_mask[x]]
    G.add_edges(H)
    components = G.subcomponent(pi_dummy)
    components.remove(pi_dummy)

    pores_connected_mask = np.zeros(network.nr_p, dtype=np.bool)
    pores_connected_mask[list(components)] = 1

    #Set tubes connectivity
    p1 = network.edgelist[:, 0]
    p2 = network.edgelist[:, 1]
    tubes_connected_mask = (pores_connected_mask[p1] | pores_connected_mask[p2]) & tube_invasion_mask

    return pores_connected_mask, tubes_connected_mask


def shortest_pore_path_between_two_nodes(network, p1, p2, weights = None):
    G = network_to_igraph(network)
    return G.get_shortest_paths(p1, p2, weights)[0]


def update_pore_and_tube_nw_connectivity_to_inlet(network):
    pores = network.pores
    tubes = network.tubes
    pores.connected[:] = 0
    tubes.connected[:] = 0
    pores.connected[:], tubes.connected[:] = get_pore_and_tube_nonwetting_connectivity_to_pore_list(network, network.pi_in)


def update_tube_backbone_from_pores(network):
    pores = network.pores
    tubes = network.tubes
    A = network.edgelist
    tubes.bbone[:] = pores.bbone[A[:, 0]] & pores.bbone[A[:, 1]] & tubes.invaded[:]


def connect_inlet_to_outlet_pores(network, G):
    source = network.nr_p
    dest = network.nr_p + 1

    G.add_vertices(2)
    H = [(source, x) for x in network.pi_in]
    G.add_edges(H)
    H = [(dest, x) for x in network.pi_out]
    G.add_edges(H)
    G.add_edge(source, dest)


def get_pi_list_of_biconnected_components(network, G):
    """
    network: Pore network
    G: Igraph subgraph of network
    """
    connect_inlet_to_outlet_pores(network, G)
    components = G.biconnected_components()
    check = -1
    ind_c = check
    source = network.nr_p
    dest = network.nr_p + 1

    for ind in range(len(components)):
        if (source in components[ind]) and (dest in components[ind]):
            ind_c = ind

    assert (ind_c != check)

    s = components[ind_c]
    s.remove(dest)
    s.remove(source)
    return list(s)


def update_pore_backbone(network, G):
    pi_list = get_pi_list_of_biconnected_components(network, G)
    network.pores.bbone[pi_list] = 1


def update_pore_and_tube_backbone_general(network):
    network.pores.bbone[:] = 0
    network.tubes.bbone[:] = 0
    G = subgraph_conn_igraph(network)
    update_pore_backbone(network, G)
    update_tube_backbone_from_pores(network)


def get_pi_list_biconnected(network):
    G = graph_igraph(network)
    pi_list = get_pi_list_of_biconnected_components(network, G)
    return np.array(pi_list, dtype=np.int32)


def update_pore_and_tube_backbone(network):
    update_pore_and_tube_backbone_general(network)
