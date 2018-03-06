import cProfile
import pstats

import numpy as np

from pypnm.linalg.laplacianmatrix import laplacian_from_network
from pypnm.porenetwork.constants import WEST, EAST
from pypnm.porenetwork.network_factory import unstructured_network_delaunay, structured_network
from pypnm.porenetwork.porenetwork import PoreNetwork
from pypnm.linalg.petsc_interface import petsc_solve, petsc_solve_lu
from sim_settings import sim_settings
from pypnm.porenetwork.component import tubes_within_pore_set


def compute_conductance(r, l, mu):
    """

    Returns
    -------
    out : array_like

    """
    return np.pi * (r ** 4) / (8. * l * mu)


def compute_nonwetting_influx(network, conductance, pressure, q_n):
    g_n = np.zeros(network.nr_t)
    ti_nw = (network.tubes.invaded == 1).nonzero()[0]
    g_n[ti_nw] = conductance[ti_nw]
    A_n = laplacian_from_network(network, weights=g_n)
    flux_in_nw = -A_n * pressure + q_n
    return flux_in_nw


def print_pore_info(network, pi):
    print "pore id", pi
    print "saturation", network.pores.sat[pi]
    print "invasion_state of pore", network.pores.invaded[pi]
    ngh_tubes = network.ngh_tubes[pi]
    ngh_pores = network.ngh_pores[pi]

    print "invasion status ngh tubes", network.tubes.invaded[ngh_tubes]
    print "indices of ngh tubes", ngh_tubes
    print "invasion status ngh pores", network.pores.invaded[ngh_pores]
    print "indices of ngh pores", ngh_pores
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

    if len(pi_drain) > 0 and len(dt_list_drain) > 0 and any(dt_list_drain == 0):
        pi_zero = pi_drain[np.argmin(dt_list_drain)]
        print_pore_info(network, pi_zero)

    pi_imb = (flux_in_nw < - tol_flux).nonzero()[0]
    if len(pi_imb) > 0:
        dt_list_imb = sat[pi_imb] * vol[pi_imb] / -flux_in_nw[pi_imb]
        dt_imb = np.min(dt_list_imb)

    if len(pi_imb) > 0 and len(dt_list_imb) > 0 and any(dt_list_imb==0):
        pi_zero = pi_imb[np.argmin(dt_list_imb)]
        print_pore_info(network, pi_zero)
        #raise ValueError("timestep zero")

    return dt_drain, dt_imb


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
    drain_criteria_3 = saturated_1 & saturated_2

    intfc_tubes_mask = (drain_criteria_1 | drain_criteria_2 | drain_criteria_3) & ((network.tubes.invaded == 0) | (network.tubes.invaded==2))
    intfc_tubes = intfc_tubes_mask.nonzero()[0]

    potential = np.maximum(pressure_drop_1-entry_pressure, pressure_drop_2-entry_pressure)[intfc_tubes]

    sort_by_potential = np.argsort(-potential)
    intfc_tubes = intfc_tubes[sort_by_potential]

    return intfc_tubes


def get_blocked_tubes(network, pressure, entry_pressure):
    """
    Notes
    -------
    Allows tube to be blocked between two drained pores

    """
    pore_list_1 = network.edgelist[:, 0]
    pore_list_2 = network.edgelist[:, 1]

    pores_invaded_1 = network.pores.invaded[pore_list_1]
    pores_invaded_2 = network.pores.invaded[pore_list_2]

    saturated_1 = network.pores.sat[pore_list_1] > 0.999
    saturated_2 = network.pores.sat[pore_list_2] > 0.999

    pressure_drop_1 = pressure[pore_list_1] - pressure[pore_list_2]
    pressure_drop_2 = -pressure_drop_1

    assert np.all((saturated_1 & (pores_invaded_1 == 1)) == saturated_1)
    assert np.all((saturated_2 & (pores_invaded_2 == 1)) == saturated_2)

    block_criteria_1 = saturated_1 & (pressure_drop_1 > 0) & (pressure_drop_1 < entry_pressure)
    block_criteria_2 = saturated_2 & (pressure_drop_2 > 0) & (pressure_drop_2 < entry_pressure)

    intfc_tubes_mask = (block_criteria_1 | block_criteria_2) & (network.tubes.invaded == 0)
    intfc_tubes = intfc_tubes_mask.nonzero()[0]

    potential = np.maximum(pressure_drop_1-entry_pressure, pressure_drop_2-entry_pressure)[intfc_tubes]

    sort_by_potential = np.argsort(potential)
    intfc_tubes = intfc_tubes[sort_by_potential]

    return intfc_tubes


def get_unblocked_tubes(network):
    """
    Notes
    -------
    Allows tube to be blocked between two drained pores

    """
    pore_list_1 = network.edgelist[:, 0]
    pore_list_2 = network.edgelist[:, 1]

    saturated_1 = network.pores.sat[pore_list_1] >= 0.999
    saturated_2 = network.pores.sat[pore_list_2] >= 0.999

    unblock_criteria = np.logical_not(saturated_1) & np.logical_not(saturated_2)
    mask_unblocked_tubes = unblock_criteria & (network.tubes.invaded == 2)
    ti_unblocked_tubes = mask_unblocked_tubes.nonzero()[0]

    return ti_unblocked_tubes


def get_imbibed_tubes(network, pressure):
    pore_list_1 = network.edgelist[:, 0]
    pore_list_2 = network.edgelist[:, 1]

    pores_invaded_1 = network.pores.invaded[pore_list_1]
    pores_invaded_2 = network.pores.invaded[pore_list_2]

    saturated_1 = network.pores.sat[pore_list_1] < 0.001
    saturated_2 = network.pores.sat[pore_list_2] < 0.001

    pressure_drop_1 = pressure[pore_list_1] - pressure[pore_list_2]
    pressure_drop_2 = - pressure_drop_1

    imb_criteria_1 = saturated_1 & (pressure_drop_1 > 0.0) & (pores_invaded_1 == 1)
    imb_criteria_2 = saturated_2 & (pressure_drop_2 > 0.0) & (pores_invaded_2 == 1)

    intfc_tubes_mask = (imb_criteria_1 | imb_criteria_2) & (network.tubes.invaded == 1)

    intfc_tubes = intfc_tubes_mask.nonzero()[0]
    potential = np.maximum(pressure_drop_1, pressure_drop_2)[intfc_tubes]

    sort_by_potential = np.argsort(-potential)
    intfc_tubes = intfc_tubes[sort_by_potential]

    return intfc_tubes


def drain_tube(network, ti, tube_conductances, mu_n):
    tubes = network.tubes
    pores = network.pores

    if tubes.invaded[ti]==2:
        print "unblocking tube and invading:", ti
    else:
        print "invading tube", ti

    tubes.invaded[ti] = 1
    pores.invaded[network.edgelist[ti]] = 1

    print "drained pores:", network.edgelist[ti]

    tube_conductances[ti] = compute_conductance(tubes.r[ti], tubes.l[ti], mu_n)


def fixed_blocked_pore(network, tube_conductances, mu_n, pi_blocked):
    print "pore completely blocked", pi_blocked
    print_pore_info(network, pi_blocked)
    for ti_untrap in network.ngh_tubes[pi_blocked]:
        drain_tube(network, ti_untrap, tube_conductances, mu_n)

def run():

    # Create network
    try:
        network = PoreNetwork.load("benchmark_network.pkl")

    except IOError:
        network = unstructured_network_delaunay(10000, quasi_2d=True)
        #network = structured_network(20, 50, 3)
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
    q_n[network.pi_list_face[WEST]] = sim_settings['dynamic']['q_n_inlet']/len(network.pi_list_face[WEST])
    q = q_n

    # Compute tube entry pressure and conductances
    entry_pressure = 2 * gamma / network.tubes.r
    tube_conductances = compute_conductance(tubes.r, tubes.l, mu_w)

    pressure = np.zeros(network.nr_p)
    sf = 1.e20  # scaling factor

    # TODO: unblock interface pore only if time-step is zero!

    try:
        pores.sat[network.pi_list_face[WEST]] = 0.01

        for niter in xrange(1000000):
            # Ensure pores at the inlet are invaded and outlet is always empty
            pores.invaded[network.pi_list_face[WEST]] = 1
            tubes.invaded[tubes_within_pore_set(network, network.pi_list_face[WEST])] =1

            pores.sat[network.pi_list_face[EAST]] = 0.0

            # Update tube conductance
            ti_drained = (tubes.invaded == 1).nonzero()[0]
            ti_imbibed = (tubes.invaded == 0).nonzero()[0]
            tube_conductances[:] = 0.0
            tube_conductances[ti_drained] = compute_conductance(tubes.r[ti_drained], tubes.l[ti_drained], mu_n)
            tube_conductances[ti_imbibed] = compute_conductance(tubes.r[ti_imbibed], tubes.l[ti_imbibed], mu_w)

            for _ in xrange(3):
                event = False

                # Unblock Tubes
                while True:
                    # Solve Pressure
                    A = laplacian_from_network(network, tube_conductances, ind_dirichlet=network.pi_list_face[EAST])
                    pressure = petsc_solve(A * sf, q * sf, x0=pressure, tol=1e-10)
                    pores.p_n[:] = pressure
                    assert not np.any(np.isnan(pressure))

                    # Compute unblocked tubes
                    ti_list_unblocked = get_unblocked_tubes(network)
                    print "num of unblocked tubes:", len(ti_list_unblocked)

                    if len(ti_list_unblocked) == 0:
                        break

                    event = True
                    network.tubes.invaded[ti_list_unblocked] = 0

                # Block tubes
                while True:
                    # Solve pressure
                    A = laplacian_from_network(network, tube_conductances, ind_dirichlet=network.pi_list_face[EAST])

                    pi_list_blocked = (A.diagonal() == 0).nonzero()[0]
                    for pi_blocked in pi_list_blocked:
                        fixed_blocked_pore(network, tube_conductances, mu_n, pi_blocked)
                        A = laplacian_from_network(network, tube_conductances, ind_dirichlet=network.pi_list_face[EAST])

                    pressure = petsc_solve(A * sf, q * sf, x0=pressure, tol=1e-10)
                    pores.p_n[:] = pressure
                    assert not np.any(np.isnan(pressure))

                    # Compute blocked throats
                    ti_list_blocked = get_blocked_tubes(network, pressure, entry_pressure)

                    for ti_blocked in ti_list_blocked[0:2*len(ti_list_blocked)/3+1]:
                        tube_conductances[ti_blocked] = 0.0
                        tubes.invaded[ti_blocked] = 2
                        print "blocked tube due to capillary:", ti_blocked
                        assert np.sum(network.pores.invaded[network.edgelist[ti_blocked]]) > 0

                    # If all interface pores are blocked, unlock one of them.
                    ti_blocked_all = (tubes.invaded == 2).nonzero()[0]
                    p_1, p_2 = network.edgelist[:, 0], network.edgelist[:, 1]
                    intfc_mask = ((pores.sat[p_1] >= 0.999) | (pores.sat[p_2] >= 0.999)) &( (tubes.invaded==0) | (tubes.invaded == 2))
                    ti_intfc = intfc_mask.nonzero()[0]

                    assert len(ti_intfc) >= len(ti_blocked_all), "%d, %d" %(len(ti_intfc), len(ti_blocked_all))
                    print "num tubes on interface, num of tubes blocked", len(ti_intfc), len(ti_blocked_all)

                    if len(ti_blocked_all)>0 and (len(ti_intfc) == len(ti_blocked_all)):
                        print "All interface tubes blocked"
                        flux_n = compute_nonwetting_influx(network, tube_conductances, pressure, q_n)
                        dt_drain, dt_imb = compute_timestep(network, flux_n)

                        if dt_drain == 0.0:
                            print "All interface tubes blocked, opening widest tube and timestep is zero"
                            ti_largest = ti_blocked_all[np.argmax(tubes.r[ti_blocked_all])]
                            drain_tube(network, ti_largest, tube_conductances, mu_n)

                    if len(ti_list_blocked) == 0:
                        break

                    event = True

                # Drain tubes
                while True:
                    # Solve pressure
                    A = laplacian_from_network(network, weights=tube_conductances, ind_dirichlet=network.pi_list_face[EAST])
                    pressure = petsc_solve(A * sf, q * sf, x0=pressure, tol=1e-10)
                    network.pores.p_n[:] = pressure
                    assert not np.any(np.isnan(pressure))

                    # Compute drained throats
                    ti_list_drained = get_drained_tubes(network, pressure, entry_pressure)
                    # print "length of drain list", len(ti_list_drained)

                    if len(ti_list_drained) == 0:
                        break

                    for ti_drained in ti_list_drained[0:2*len(ti_list_drained)/3+1]:
                        drain_tube(network, ti_drained, tube_conductances, mu_n)
                    event = True

                # Imbibe Tubes
                while True:
                    # Solve pressure
                    A = laplacian_from_network(network, weights=tube_conductances, ind_dirichlet=network.pi_list_face[EAST])

                    pressure = petsc_solve(A * sf, q * sf, x0=pressure, tol=1e-10)
                    network.pores.p_n[:] = pressure

                    assert not np.any(np.isnan(pressure))

                    # Compute imbibed throats
                    ti_list_imbibed = get_imbibed_tubes(network, pressure)
                    # print "len imbibed list", len(ti_list_imbibed)

                    if len(ti_list_imbibed) == 0:
                        break

                    ti_imbibed = ti_list_imbibed[0]

                    network.tubes.invaded[ti_imbibed] = 0
                    pi_ngh_1, pi_ngh_2 = network.edgelist[ti_imbibed]

                    print "imbibing tube:", ti_imbibed
                    if (pores.sat[pi_ngh_1] < 0.001) and np.all(tubes.invaded[network.ngh_tubes[pi_ngh_1]]==0):
                        network.pores.invaded[pi_ngh_1] = 0
                        print "imbibed pore:", pi_ngh_1

                    if (pores.sat[pi_ngh_2] < 0.001) and np.all(tubes.invaded[network.ngh_tubes[pi_ngh_2]]==0):
                        network.pores.invaded[pi_ngh_2] = 0
                        print "imbibed pore:", pi_ngh_2

                    tube_conductances[ti_imbibed] = compute_conductance(tubes.r[ti_imbibed], tubes.l[ti_imbibed], mu_w)
                    event = True

                if not event:
                    break

            A = laplacian_from_network(network, weights=tube_conductances, ind_dirichlet=network.pi_list_face[EAST])
            pressure = petsc_solve(A * sf, q * sf, x0=pressure, tol=1e-12)
            network.pores.p_n[:] = pressure

            #  Update saturation
            flux_n = compute_nonwetting_influx(network, tube_conductances, pressure, q_n)
            dt_drain, dt_imb = compute_timestep(network, flux_n)
            dt = min(dt_drain, dt_imb)

            network.pores.sat = update_sat(network, flux_n, dt)

            sat_network = np.sum(network.pores.sat * network.pores.vol) / np.sum(network.pores.vol)

            if sat_network > s_target:
                network.pores.p_n[:] = pressure
                print "Number of throats invaded:", niter
                network.save("network_history/network"+str(n_out).zfill(5)+".pkl")
                s_target += ds_out
                n_out += 1
                network.export_to_vtk("test" + str(n_out).zfill(3))

            if niter % 1 == 0:
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
