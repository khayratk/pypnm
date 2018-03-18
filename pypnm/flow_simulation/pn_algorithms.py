import logging

import numpy as np

logger = logging.getLogger('pypnm.pn_algorithms')

eps_sat = 1e-3


def get_piston_disp_tubes_nonwett(network, entry_pressure, flux_n, source_nonwett=None):
    p_c = network.pores.p_c
    sat = network.pores.sat

    pores_1 = network.edgelist[:, 0]
    pores_2 = network.edgelist[:, 1]

    sat_crit = 0.5

    if source_nonwett is None:
        dvn_dt = - flux_n
    else:
        dvn_dt = source_nonwett - flux_n

    sat_above_crit_1 = (sat[pores_1] > sat_crit)
    sat_above_crit_2 = (sat[pores_2] > sat_crit)

    tubes_wett = (network.tubes.invaded == 0)

    pc_above_pe_1 = p_c[pores_1] >= entry_pressure
    pc_above_pe_2 = p_c[pores_2] >= entry_pressure

    pc_diff_above_pe_1 = (network.pores.p_n[pores_1] - network.pores.p_w[pores_2]) > entry_pressure
    pc_diff_above_pe_2 = (network.pores.p_n[pores_2] - network.pores.p_w[pores_1]) > entry_pressure

    outward_wetting_flux_1 = (dvn_dt[pores_1] > 0.0)
    outward_wetting_flux_2 = (dvn_dt[pores_2] > 0.0)

    condition1 = pc_above_pe_1 & sat_above_crit_1 & outward_wetting_flux_1 & pc_diff_above_pe_1

    condition2 = pc_above_pe_2 & sat_above_crit_2 & outward_wetting_flux_2 & pc_diff_above_pe_2

    condition3 = pc_above_pe_1 & pc_above_pe_2 & sat_above_crit_1 & sat_above_crit_2

    condition = (condition1 | condition2 | condition3) & tubes_wett

    ti_displacement = condition.nonzero()[0]

    potential = entry_pressure

    sort_inds = np.argsort(potential[ti_displacement])

    return ti_displacement[sort_inds]


def get_piston_disp_tubes_wett(network, entry_pressure, flux_w, source_wett=None):
    if source_wett is None:
        dvw_dt = - flux_w
    else:
        dvw_dt = source_wett - flux_w

    p_n = network.pores.p_n
    sat = network.pores.sat
    tubes_nwett = (network.tubes.invaded == 1)
    pores_1 = network.edgelist[:, 0]
    pores_2 = network.edgelist[:, 1]

    sat_crit = eps_sat
    sat_below_crit_1 = (sat[pores_1] < sat_crit)
    sat_below_crit_2 = (sat[pores_2] < sat_crit)

    condition1 = sat_below_crit_1 & (p_n[pores_1] > p_n[pores_2]) & (dvw_dt[pores_1] > 0.0)
    condition2 = sat_below_crit_2 & (p_n[pores_2] > p_n[pores_1]) & (dvw_dt[pores_2] > 0.0)

    condition = (condition1 | condition2) & tubes_nwett

    ti_displacement = condition.nonzero()[0]

    potential = - entry_pressure

    sort_inds = np.argsort(potential[ti_displacement])

    return ti_displacement[sort_inds]


def invade_tube_nw(network, k):
    """
    Invades tube with nonwetting phase

    Parameters
    ----------
    network: PoreNetwork
    k: tube index
    """
    network.tubes.invaded[k] = 1
    p1, p2 = network.edgelist[k, :]
    network.pores.invaded[[p1, p2]] = 1
    logger.debug("TUBE %d NW INVADED. Radius is %f. Pore1 sat: %g Pore2 sat: %g",
             k, network.tubes.r[k], network.pores.sat[p1], network.pores.sat[p2])


def invade_tube_w(network, k):
    """
    Invades tube with wetting phase

    Parameters
    ----------
    network: PoreNetwork
    k: tube index
    """

    p1 = network.edgelist[k, 0]
    p2 = network.edgelist[k, 1]

    network.tubes.invaded[k] = 0
    logger.debug("TUBE %d W INVADED by piston mechanism. Radius is %f", k, network.tubes.r[k])

    if np.all(network.tubes.invaded[network.ngh_tubes[p1]] == 0) and (network.pores.sat[p1] < eps_sat):
        network.pores.invaded[p1] = 0
        logger.debug("Pore %d Invaded with wetting phasedomain type %d ", p1, network.pore_domain_type[p1] )

    if np.all(network.tubes.invaded[network.ngh_tubes[p2]] == 0) and (network.pores.sat[p2] < eps_sat):
        network.pores.invaded[p2] = 0
        logger.debug("Pore %d Invaded with wetting phase, domain type %d", p2, network.pore_domain_type[p2])


def update_tube_piston_w(network, entry_pressure, flux_w, source_wett):
    ti_disp = get_piston_disp_tubes_wett(network, entry_pressure, flux_w, source_wett)
    is_event = False

    if len(ti_disp) > 0:
        k = ti_disp[0]
        invade_tube_w(network, k)

    return is_event


def update_tube_snapoff(network, pe_comp):
    """
    Snaps off one tube at a time
    """
    p_c = network.tubes.p_c

    snapoff_pressure = pe_comp.snap_off_all_tubes()

    is_event = False

    t_inds = np.argsort(-snapoff_pressure)

    snap_off_mask = np.logical_and(p_c <= snapoff_pressure, network.tubes.invaded == 1)
    snap_off_mask = snap_off_mask[t_inds]

    ti_snap_off = np.nonzero(snap_off_mask)[0]

    if np.sum(snap_off_mask) > 0:
        k = ti_snap_off[0]
        k = t_inds[k]
        network.tubes.invaded[k] = 0
        is_event = True

        logger.debug("TUBE %d W Invaded after snap-off. Radius is %f", k,  network.tubes.r[k])
        logger.debug("Snap_off pressure of tubes: %f", snapoff_pressure[k])

    return is_event


def snapoff_all_tubes(network, pe_comp):
    """
    Snaps off all tubes which satisfy the criteria
    """
    p_c = network.tubes.p_c

    snapoff_pressure = pe_comp.snap_off_all_tubes(network)
    snap_off_mask = (p_c <= snapoff_pressure) & (network.tubes.invaded == 1)

    ti_snap_off = np.nonzero(snap_off_mask)[0]
    network.tubes.invaded[ti_snap_off] = 0

    logger.debug("Snapping off all tubes. Total number of tubes snapped off is %d", np.sum(snap_off_mask))
    return


def invade_pore_with_wett_phase(network, i):
    network.pores.invaded[i] = 0
    network.tubes.invaded[network.ngh_tubes[i]] = 0
    logger.debug("Tubes invaded after pore wetting invaded: %s", network.ngh_tubes[i])


def update_pore_status(network, flux_w, flux_n,  bool_accounted_pores=None, source_wett=None, source_nonwett=None):
    is_event = False
    pores = network.pores

    if source_wett is None:
        dvw_dt = -flux_w
    else:
        dvw_dt = source_wett - flux_w

    if source_nonwett is None:
        dvn_dt = -flux_n
    else:
        dvn_dt = source_nonwett - flux_n

    if bool_accounted_pores is None:
        bool_accounted_pores = np.ones(network.nr_p, dtype=np.bool)

    pores_to_be_imbibed_mask = (pores.sat < eps_sat) & (dvw_dt > 0.0) & (pores.invaded == 1) & bool_accounted_pores
    pores_to_be_imbibed = np.flatnonzero(pores_to_be_imbibed_mask)

    for i in pores_to_be_imbibed:
        invade_pore_with_wett_phase(network, i)
        logger.debug("Pore %d Invaded with wetting phase. Domain type %d", i, network.pore_domain_type[i])
        is_event = True

    pores_to_be_drained_mask = (source_nonwett > 0) & (dvn_dt>0.0) & (pores.invaded==0) & bool_accounted_pores
    pores_to_be_drained = np.flatnonzero(pores_to_be_drained_mask)

    for i in pores_to_be_drained:
        network.pores.invaded[i] = 1

        logger.debug("Pore %d Invaded with nonwetting phase", i)
        is_event = True

    return is_event
