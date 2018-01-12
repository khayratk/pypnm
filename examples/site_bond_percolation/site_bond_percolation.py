import numpy as np

from pypnm.percolation.invasion_percolation_refactored import site_bond_invasion_percolation
from pypnm.porenetwork.network_factory import structured_network
from pypnm.util.igraph_utils import network_to_igraph

network = structured_network(4, 4, 4)
g = network_to_igraph(network, vertex_attributes=["x", "y"])

sat = 0
total_vol = np.sum(network.tubes.vol) + np.sum(network.pores.vol)
network.pores.invaded[0] = 1
for x, (element_type, element_id, weight) in enumerate(site_bond_invasion_percolation(g, 1. / network.tubes.r, [0])):
    if element_type == 0:
        sat += network.tubes.vol[element_id] / total_vol
        network.tubes.invaded[element_id] = 1
    else:
        sat += network.pores.vol[element_id] / total_vol
        network.pores.invaded[element_id] = 1

    if sat == 1.0:
        break
