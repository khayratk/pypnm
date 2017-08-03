import numpy as np
from scipy import spatial

from pypnm.porenetwork import component
from pypnm.porenetwork.coordination_number import choose_edges_for_target_coord_num
from pypnm.porenetwork.network_manipulation import prune_network
from pypnm.porenetwork.structured_porenetwork import StructuredPoreNetwork
from pypnm.util.sphere_packing import sphere_packing


def create_unstructured_network(nr_pores, pdf_pore_radius, pdf_tube_radius, pdf_coord_number, domain_size, is_2d=False):
    nr_new_pores = nr_pores - 1

    rad_generated = pdf_pore_radius.rvs(size=nr_new_pores)
    x_coord, y_coord, z_coord, rad = sphere_packing(rad=rad_generated, domain_size=domain_size, is_2d=is_2d)

    network = StructuredPoreNetwork([1, 1, 1], 1.0e-16)  # Pore-Network with only one pore as seed

    network.add_pores(x_coord, y_coord, z_coord, rad)

    vertex_degree = 100
    max_nr_new_tubes = vertex_degree * (nr_new_pores+1)/2

    points = zip(network.pores.x, network.pores.y, network.pores.z)
    tree = spatial.cKDTree(points)
    edgelist_1 = -np.ones(max_nr_new_tubes, dtype=np.int32)
    edgelist_2 = -np.ones(max_nr_new_tubes, dtype=np.int32)

    ptr = 0
    for pi in xrange(network.nr_p):
        d, pi_nghs = tree.query((network.pores.x[pi], network.pores.y[pi], network.pores.z[pi]), vertex_degree)
        for pi_ngh in pi_nghs:
            if pi_ngh > pi:
                edgelist_1[ptr] = pi
                edgelist_2[ptr] = pi_ngh
                ptr += 1

    edgelist_1 = edgelist_1[edgelist_1 > -1]
    edgelist_2 = edgelist_2[edgelist_2 > -1]
    edgelist = np.vstack([edgelist_1, edgelist_2]).T

    length = np.sqrt((network.pores.x[edgelist_1] - network.pores.x[edgelist_2]) ** 2 +
                     (network.pores.y[edgelist_1] - network.pores.y[edgelist_2]) ** 2 +
                     (network.pores.z[edgelist_1] - network.pores.z[edgelist_2]) ** 2)

    rad_tubes = pdf_tube_radius.rvs(size=len(edgelist_1))

    network.add_throats(edgelist, r=rad_tubes, l=length, G=np.ones(len(edgelist_1)) / 16.0)

    target_coord_number = pdf_coord_number.rvs(size=network.nr_p)

    ti_list = choose_edges_for_target_coord_num(network, target_coord_number)
    ti_list_remove = component.complement_tube_set(network, ti_list)
    network.remove_throats(ti_list_remove)

    network._fix_tubes_larger_than_ngh_pores()
    network = prune_network(network, [0.1, 0.9, 0.1, 0.9, 0.1, 0.9])
    network.network_type = "unstructured"

    return network
