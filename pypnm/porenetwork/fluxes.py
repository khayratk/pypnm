from component import tubes_from_to_pore_sets, complement_pore_set
import numpy as np


def flux_into_pores(network, pressure, conductance, pi_list_target):

    pi_list_complement = complement_pore_set(network, pi_list_target)

    return flux_from_into_pores(network, pressure, conductance, pi_list_complement, pi_list_target)


def flux_from_into_pores(network, pressure, conductance, pi_list_source, pi_list_target):

    ti_list1 = tubes_from_to_pore_sets(network, pi_list_source, pi_list_target)
    ti_list2 = tubes_from_to_pore_sets(network, pi_list_target, pi_list_source)

    flux = np.sum((pressure[network.edgelist[ti_list1, 0]] - pressure[network.edgelist[ti_list1, 1]]) *
                  conductance[ti_list1])
    flux -= np.sum((pressure[network.edgelist[ti_list2, 0]] - pressure[network.edgelist[ti_list2, 1]]) *
                   conductance[ti_list2])

    return flux
