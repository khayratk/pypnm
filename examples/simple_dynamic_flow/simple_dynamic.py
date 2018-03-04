import cProfile
import pstats

import numpy as np

from pypnm.linalg.laplacianmatrix import laplacian_from_network
from pypnm.porenetwork.constants import WEST, EAST
from pypnm.porenetwork.network_factory import unstructured_network_delaunay
from pypnm.porenetwork.porenetwork import PoreNetwork
from pypnm.linalg.petsc_interface import petsc_solve
from sim_settings import sim_settings


def compute_conductance(r, l, mu):
    return np.pi * (r ** 4) / (8. * l * mu)


def compute_nonwetting_influx(network, conductance, pressure, q_n):
    g_n = np.zeros(network.nr_t)
    ti_nw = (network.tubes.invaded == 1).nonzero()[0]
    g_n[ti_nw] = conductance[ti_nw]
    A_n = laplacian_from_network(network, weights=g_n)
    flux_in_nw = -A_n * pressure + q_n
    return flux_in_nw

def print_pore_info(network, pi):
    print "pore with zero timestep", pi
    print "saturation", network.pores.sat[pi]
    print "invasion_state of pore", network.pores.invaded[pi]
    ngh_tubes = network.ngh_tubes[pi]
    ngh_pores = network.ngh_pores[pi]

    print "invasion status tubes", network.tubes.invaded[ngh_tubes]
    print "invasion status pores", network.pores.invaded[ngh_pores]
    print "saturation of ngh_pores", network.pores.sat[ngh_pores]
    print "pressure_difference", network.pores.p_n[pi] - network.pores.p_n[ngh_pores]

def compute_timestep(network, flux_in_nw):
    assert np.all(network.pores.sat >= -1e-10), np.min(network.pores.sat)

    vol = network.pores.vol
    sat = network.pores.sat

    dt_imb, dt_drain = np.inf, np.inf
    # timestep drainage

    tol_flux = np.max(np.abs(flux_in_nw))*1.e-6

    pi_drain = (flux_in_nw > tol_flux).nonzero()[0]
    if len(pi_drain) > 0:
        dt_list_drain = (1 - sat[pi_drain]) * vol[pi_drain] / flux_in_nw[pi_drain]
        dt_drain = np.min(dt_list_drain)

    if len(dt_list_drain) > 0 and any(dt_list_drain==0):
        pi_zero = pi_drain[np.argmin(dt_list_drain)]
        print_pore_info(network, pi_zero)

    pi_imb = (flux_in_nw < - tol_flux).nonzero()[0]
    if len(pi_imb) > 0:
        dt_list_imb = sat[pi_imb] * vol[pi_imb] / -flux_in_nw[pi_imb]
        dt_imb = np.min(dt_list_imb)

    if len(dt_list_imb) > 0 and any(dt_list_imb==0):
        pi_zero = pi_imb[np.argmin(dt_list_imb)]
        print_pore_info(network, pi_zero)


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

    potential = np.maximum(pressure_drop_1-entry_pressure, pressure_drop_2-entry_pressure)[intfc_tubes]

    sort_by_potential = np.argsort(-potential)
    intfc_tubes = intfc_tubes[sort_by_potential]

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

    intfc_tubes_mask = (block_criteria_1 | block_criteria_2) & (network.tubes.invaded == 0)
    intfc_tubes = intfc_tubes_mask.nonzero()[0]

    potential = np.maximum(pressure_drop_1-entry_pressure, pressure_drop_2-entry_pressure)[intfc_tubes]

    sort_by_potential = np.argsort(potential)
    intfc_tubes = intfc_tubes[sort_by_potential]

    return intfc_tubes


def get_imbibed_tubes(network, pressure):
    pore_list_1 = network.edgelist[:, 0]
    pore_list_2 = network.edgelist[:, 1]

    pores_invaded_1 = network.pores.invaded[pore_list_1]
    pores_invaded_2 = network.pores.invaded[pore_list_2]

    saturated_1 = network.pores.sat[pore_list_1] < 0.001
    saturated_2 = network.pores.sat[pore_list_2] < 0.001

    pressure_drop_1 = pressure[pore_list_1] - pressure[pore_list_2]
    pressure_drop_2 = - pressure_drop_1

    imb_criteria_1 = saturated_1 & (pressure_drop_1 > 0.0)  & (pores_invaded_1 == 1)
    imb_criteria_2 = saturated_2 & (pressure_drop_2 > 0.0)   & (pores_invaded_2 == 1)

    intfc_tubes_mask = (imb_criteria_1 | imb_criteria_2) & (network.tubes.invaded == 1)

    intfc_tubes = intfc_tubes_mask.nonzero()[0]
    potential = np.maximum(pressure_drop_1, pressure_drop_2)[intfc_tubes]

    sort_by_potential = np.argsort(-potential)
    intfc_tubes = intfc_tubes[sort_by_potential]

    return intfc_tubes


def run():

    # Create network
    try:
        network = PoreNetwork.load("benchmark_network.pkl")

    except IOError:
        network = unstructured_network_delaunay(5000, quasi_2d=True)
        network.save("benchmark_network.pkl")

    tubes = network.tubes
    pores = network.pores

    # load simulation settings
    mu_w = sim_settings['fluid_properties']["mu_w"]
    mu_n = sim_settings['fluid_properties']["mu_n"]
    gamma = sim_settings['fluid_properties']["gamma"]

    ds_out = 0.01
    s_target = 0.01
    n_out = 0

    # set boundary conditions
    pores.invaded[network.pi_list_face[WEST]] = 1
    pores.sat[:] = 0.0
    q_n = np.zeros(network.nr_p)
    q_n[network.pi_list_face[WEST]] = 1.e-8/len(network.pi_list_face[WEST])
    q = q_n

    # Compute tube entry pressure and conductances
    entry_pressure = 2 * gamma / network.tubes.r
    tube_conductances = compute_conductance(tubes.r, tubes.l, mu_w)

    pressure = np.zeros(network.nr_p)
    sf = 1.e20  # scaling factor

    try:
        for niter in xrange(100000):
            # Ensure pores at the inlet are invaded and outlet is always empty
            pores.invaded[network.pi_list_face[WEST]] = 1
            pores.sat[network.pi_list_face[WEST]] = 1.0
            pores.sat[network.pi_list_face[EAST]] = 0.0

            # Update tube conductance
            pi_drained = (tubes.invaded == 1).nonzero()[0]
            pi_imbibed = (tubes.invaded == 0).nonzero()[0]
            tube_conductances[pi_drained] = compute_conductance(tubes.r[pi_drained], tubes.l[pi_drained], mu_n)
            tube_conductances[pi_imbibed] = compute_conductance(tubes.r[pi_imbibed], tubes.l[pi_imbibed], mu_w)

            for _ in xrange(3):

                while True:
                    # Solve pressure
                    A = laplacian_from_network(network, weights=tube_conductances, ind_dirichlet=network.pi_list_face[EAST])
                    pressure = petsc_solve(A * sf, q * sf, x0=pressure, tol=1e-16)
                    pores.p_n[:] = pressure

                    # Compute blocked throats
                    ti_list_blocked = get_blocked_tubes(network, pressure, entry_pressure)
                    print "blocked list", ti_list_blocked

                    if len(ti_list_blocked) == 0:
                        break

                    ti_blocked = ti_list_blocked[0]
                    tube_conductances[ti_blocked] = 0.0
                    tubes.invaded[ti_blocked] = 2

                tubes.invaded[tubes.invaded == 2] = 1

                while True:
                    # Solve pressure
                    A = laplacian_from_network(network, weights=tube_conductances, ind_dirichlet=network.pi_list_face[EAST])
                    pressure = petsc_solve(A * sf, q * sf, x0=pressure, tol=1e-16)
                    network.pores.p_n[:] = pressure

                    # Compute drained throats
                    ti_list_drained = get_drained_tubes(network, pressure, entry_pressure)
                    print "drained_list", ti_list_drained

                    if len(ti_list_drained) == 0:
                        break

                    ti_drained = ti_list_drained[0]
                    tubes.invaded[ti_drained] = 1
                    pores.invaded[network.edgelist[ti_drained]] = 1
                    tube_conductances[ti_drained] = compute_conductance(tubes.r[ti_drained], tubes.l[ti_drained], mu_n)

                while True:
                    # Solve pressure
                    A = laplacian_from_network(network, weights=tube_conductances, ind_dirichlet=network.pi_list_face[EAST])
                    pressure = petsc_solve(A * sf, q * sf, x0=pressure, tol=1e-16)
                    network.pores.p_n[:] = pressure

                    # Compute imbibed throats
                    ti_list_imbibed = get_imbibed_tubes(network, pressure)

                    print "imbibed list", ti_list_imbibed

                    if len(ti_list_imbibed) == 0:
                        break

                    ti_imbibed = ti_list_imbibed[0]

                    network.tubes.invaded[ti_imbibed] = 0
                    pi_ngh_1, pi_ngh_2 = network.edgelist[ti_imbibed]

                    if (pores.sat[pi_ngh_1] < 0.001) and np.all(tubes.invaded[network.ngh_tubes[pi_ngh_1]]==0):
                        network.pores.invaded[pi_ngh_1] = 0

                    if (pores.sat[pi_ngh_2] < 0.001) and np.all(tubes.invaded[network.ngh_tubes[pi_ngh_2]]==0):
                        network.pores.invaded[pi_ngh_2] = 0

                    tube_conductances[ti_imbibed] = compute_conductance(tubes.r[ti_imbibed], tubes.l[ti_imbibed], mu_w)

            #  Update saturation
            flux_n = compute_nonwetting_influx(network, tube_conductances, pressure, q_n)
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
