import cProfile
import pstats

import numpy as np

from pypnm.linalg.laplacianmatrix import laplacian_from_network
from pypnm.porenetwork.constants import WEST, EAST
from pypnm.porenetwork.network_factory import unstructured_network_delaunay
from pypnm.porenetwork.porenetwork import PoreNetwork
from pypnm.linalg.petsc_interface import petsc_solve
from pypnm.percolation import graph_algs
from pypnm.porenetwork.component import tube_list_ngh_to_pore_list

def compute_conductance(r, l, mu):
    return np.pi * (r ** 4) / (8. * l * mu)


def compute_nonwetting_flux(network, conductance, pressure, source_n):
    g_n = np.zeros(network.nr_t)
    pi_nw = (network.tubes.invaded == 1).nonzero()[0]
    g_n[pi_nw] = conductance[pi_nw]
    A_n = laplacian_from_network(network, weights=g_n)
    flux_in_nw = -A_n * pressure + source_n
    return flux_in_nw


def compute_timestep(network, flux_in_nw):
    assert np.all(network.pores.sat >= -1e-10), np.min(network.pores.sat)

    vol = network.pores.vol
    sat = network.pores.sat

    dt_imb, dt_drain = np.inf, np.inf
    # timestep drainage

    pi_drain = (flux_in_nw > 1e-14).nonzero()[0]
    if len(pi_drain) > 0:
        dt_drain = np.min((1 - sat[pi_drain]) * vol[pi_drain] / flux_in_nw[pi_drain])

    pi_imb = (flux_in_nw < -1e-14).nonzero()[0]
    if len(pi_imb) > 0:
        dt_imb = np.min(sat[pi_imb] * vol[pi_imb] / -flux_in_nw[pi_imb])

    print dt_drain, dt_imb
    return min(dt_drain, dt_imb)


def update_sat(network, flux_in_nw, dt):
    network.pores.sat[:] = network.pores.sat[:] + flux_in_nw / network.pores.vol * dt
    assert np.all(network.pores.sat >= -1e-10), np.min(network.pores.sat)
    network.pores.sat[network.pores.sat < 0] = 0.0

    return network.pores.sat


def get_drained_tubes(network, pressure, entry_pressure):
    pore_list_1 = network.edgelist[:, 0]
    pore_list_2 = network.edgelist[:, 1]

    pores_invaded_1 = network.pores.invaded[pore_list_1]
    pores_invaded_2 = network.pores.invaded[pore_list_2]

    saturated_1 = network.pores.sat[pore_list_1] > 0.999
    saturated_2 = network.pores.sat[pore_list_2] > 0.999

    pressure_drop_1 = pressure[pore_list_1] - pressure[pore_list_2]
    pressure_drop_2 = - pressure_drop_1

    drain_criteria_1 = saturated_1 & (pressure_drop_1 > entry_pressure) & (pores_invaded_1 == 1)
    drain_criteria_2 = saturated_2 & (pressure_drop_2 > entry_pressure) & (pores_invaded_2 == 1)

    intfc_tubes_mask = (drain_criteria_1 | drain_criteria_2) & (network.tubes.invaded == 0)
    intfc_tubes = intfc_tubes_mask.nonzero()[0]
    return intfc_tubes


def get_blocked_tubes(network, pressure, entry_pressure):
    pore_list_1 = network.edgelist[:, 0]
    pore_list_2 = network.edgelist[:, 1]

    pores_invaded_1 = network.pores.invaded[pore_list_1]
    pores_invaded_2 = network.pores.invaded[pore_list_2]

    saturated_1 = network.pores.sat[pore_list_1] > 0.999
    saturated_2 = network.pores.sat[pore_list_2] > 0.999

    pressure_drop_1 = pressure[pore_list_1] - pressure[pore_list_2]
    pressure_drop_2 = - pressure_drop_1

    block_criteria_1 = saturated_1 & (pressure_drop_1 > 0) & (pressure_drop_1 < entry_pressure) & (pores_invaded_1 == 1)
    block_criteria_2 = saturated_2 & (pressure_drop_2 > 0) & (pressure_drop_2 < entry_pressure) & (pores_invaded_2 == 1)
    #block_criteria_3 = saturated_1 & saturated_2

    intfc_tubes_mask = (block_criteria_1 | block_criteria_2) & (network.tubes.invaded == 0)
    intfc_tubes = intfc_tubes_mask.nonzero()[0]
    return intfc_tubes


def get_imbibed_tubes(network, pressure):
    pore_list_1 = network.edgelist[:, 0]
    pore_list_2 = network.edgelist[:, 1]

    pores_invaded_1 = network.pores.invaded[pore_list_1]
    pores_invaded_2 = network.pores.invaded[pore_list_2]

    saturated_1 = network.pores.sat[pore_list_1] < 0.001
    saturated_2 = network.pores.sat[pore_list_2] < 0.001

    press_diff = pressure[pore_list_1] - pressure[pore_list_2]

    pressure_drop_1 = (pores_invaded_1 * press_diff)
    pressure_drop_2 = (pores_invaded_2 * (-press_diff))

    imb_criteria_1 = saturated_1 & (pressure_drop_1 > 0.0) & (pores_invaded_1 == 1)
    imb_criteria_2 = saturated_2 & (pressure_drop_2 > 0.0) & (pores_invaded_2 == 1)

    intfc_tubes_mask = (imb_criteria_1 | imb_criteria_2) & (network.tubes.invaded == 1)
    intfc_tubes = intfc_tubes_mask.nonzero()[0]
    return intfc_tubes


def run():
    try:
        network = PoreNetwork.load("benchmark_network.pkl")

    except IOError:
        network = unstructured_network_delaunay(50000, quasi_2d=True)
        #network = structured_network(50, 50, 5)
        network.save("benchmark_network.pkl")

    tubes = network.tubes
    pores = network.pores

    mu_w = 1.0
    mu_n = 0.1
    gamma = 1.0
    network.pores.invaded[network.pi_list_face[WEST]] = 1
    network.pores.sat[:] = 0.0
    entry_pressure = 2 * gamma / network.tubes.r

    ds_out = 0.01
    s_target = 0.01
    n_out = 0

    tube_conductances = compute_conductance(tubes.r, tubes.l, mu_w)

    A = laplacian_from_network(network, weights=tube_conductances, ind_dirichlet=network.pi_list_face[EAST])
    source_n = np.zeros(network.nr_p)
    source_n[network.pi_list_face[WEST]] = 1.e-7
    source = source_n
    outlet_pore_set = set(network.pi_list_face[EAST])

    pressure = np.zeros(network.nr_p)
    sf = 1e20
    try:
        for niter in xrange(100000):
            pores.invaded[network.pi_list_face[WEST]] = 1
            pores.sat[network.pi_list_face[EAST]] = 0.0

            # Initial calculation
            tube_conductances[tubes.invaded==1] = compute_conductance(tubes.r[tubes.invaded==1], tubes.l[tubes.invaded==1], mu_n)
            tube_conductances[tubes.invaded==0] = compute_conductance(tubes.r[tubes.invaded==0], tubes.l[tubes.invaded==0], mu_w)

            A = laplacian_from_network(network, weights=tube_conductances, ind_dirichlet=network.pi_list_face[EAST])
            pressure = petsc_solve(A * sf, source * sf, x0=pressure, tol=1e-10)
            network.pores.p_n[:] = pressure

            # Compute blocked throats
            ti_list_blocked = get_blocked_tubes(network, pressure, entry_pressure)
            print "blocked list", ti_list_blocked
            tube_conductances[ti_list_blocked] = 0.0

            A = laplacian_from_network(network, weights=tube_conductances, ind_dirichlet=network.pi_list_face[EAST])

            pressure = petsc_solve(A * sf, source * sf, x0=pressure, tol=1e-10)
            network.pores.p_n[:] = pressure


            # Compute drained throats
            ti_list_drained = get_drained_tubes(network, pressure, entry_pressure)

            print "drained_list", ti_list_drained

            for ti_displaced in ti_list_drained:
                network.tubes.invaded[ti_displaced] = 1
                network.pores.invaded[network.edgelist[ti_displaced]] = 1
                tube_conductances[ti_displaced] = compute_conductance(tubes.r[ti_displaced], tubes.l[ti_displaced], mu_n)

            A = laplacian_from_network(network, weights=tube_conductances, ind_dirichlet=network.pi_list_face[EAST])
            pressure = petsc_solve(A * sf, source * sf, x0=pressure, tol=1e-10)
            network.pores.p_n[:] = pressure

            # Compute imbibed throats
            ti_list_imbibed = get_imbibed_tubes(network, pressure)

            print "imbibed list", ti_list_imbibed

            for ti_displaced in ti_list_imbibed:

                network.tubes.invaded[ti_displaced] = 0
                pi_ngh_1, pi_ngh_2 = network.edgelist[ti_displaced]

                if pores.sat[pi_ngh_1] < 0.001 and np.all(tubes.invaded[network.ngh_tubes[pi_ngh_1]]==0):
                    network.pores.invaded[pi_ngh_1] = 0

                if pores.sat[pi_ngh_1] < 0.002 and np.all(tubes.invaded[network.ngh_tubes[pi_ngh_2]]==0):
                    network.pores.invaded[pi_ngh_2] = 0

                tube_conductances[ti_displaced] = compute_conductance(tubes.r[ti_displaced], tubes.l[ti_displaced],
                                                                      mu_w)

            A = laplacian_from_network(network, weights=tube_conductances, ind_dirichlet=network.pi_list_face[EAST])

            pressure = petsc_solve(A * sf, source * sf, x0=pressure, tol=1e-10)
            network.pores.p_n[:] = pressure

            # Compute blocked throats
            ti_list_blocked = get_blocked_tubes(network, pressure, entry_pressure)
            print "blocked list", ti_list_blocked
            tube_conductances[ti_list_blocked] = 0.0

            # Update saturation and all that
            flux_n = compute_nonwetting_flux(network, tube_conductances, pressure, source_n)
            dt = compute_timestep(network, flux_n)

            network.pores.sat = update_sat(network, flux_n, dt)

            sat_network = np.sum(network.pores.sat * network.pores.vol) / np.sum(network.pores.vol)

            if sat_network > s_target:
                network.pores.p_n[:] = pressure
                print "Number of throats invaded:", niter
                network.save("network_history/network"+str(n_out).zfill(5)+".pkl")
                s_target += ds_out
                n_out += 1
                network.export_to_vtk("test" + str(niter).zfill(3) + ".vtk")

            if niter % 10 == 0:
                print "saturation is:", sat_network


    except KeyboardInterrupt:
        pass


def print_profiling_info(filename):
    p = pstats.Stats(filename)
    p.sort_stats('cumulative').print_stats(50)
    p.sort_stats('time').print_stats(50)


if __name__ == "__main__":
    exec_string = 'run()'

    cProfile.run(exec_string, 'restats')
    print_profiling_info('restats')
    run()
