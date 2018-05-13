import numpy as np
from pypnm.porenetwork.component import tube_list_x_plane


def time_avg_energy_dissip(phase):

    def _callback(simulation):
        network = simulation.network
        if _callback.phase == "NWETT":
            press = network.pores.p_n
            cond = network.tubes.k_n
        elif _callback.phase == "WETT":
            press = network.pores.p_w
            cond = network.tubes.k_w

        pi_1, pi_2 = network.edgelist[:, 0], network.edgelist[:, 1]
        energy_dissipated = np.sum((press[pi_2] - press[pi_1]) ** 2 * cond)

        _callback.total_time += simulation.dt
        _callback.work += energy_dissipated * simulation.dt
        _callback.energy_dissip = _callback.work/_callback.total_time

    _callback.phase = phase
    _callback.energy_dissip = 0.0
    _callback.total_time = 0.0
    _callback.work = 0.0
    return _callback


def time_avg_inlet_flux(phase, face):
    def _callback(simulation):
        network = simulation.network

        if _callback.phase == "NWETT":
            flux = simulation.flux_n
        elif _callback.phase == "WETT":
            flux = simulation.flux_w

        pi_inlet = network.pi_list_face[face]
        _callback.sum_flux += np.sum(flux[pi_inlet]) * simulation.dt
        _callback.total_time += simulation.dt
        _callback.avg_flux = _callback.sum_flux / _callback.total_time

    _callback.sum_flux = 0.0
    _callback.total_time = 0.0
    _callback.avg_flux = 0.0
    _callback.phase = phase

    return _callback


def time_avg_flux(phase):

    def _callback(simulation):
        network = simulation.network
        if _callback.phase == "NWETT":
            press = network.pores.p_n
            cond = network.tubes.k_n
        elif _callback.phase == "WETT":
            press = network.pores.p_w
            cond = network.tubes.k_w

        l_x = network.dim[0]

        ti_center = tube_list_x_plane(network, np.min(network.pores.x)+l_x/2)

        tubes = network.tubes
        orientation = network.edge_orientations.T[0]
        pi_1, pi_2 = network.edgelist[:, 0], network.edgelist[:, 1]
        flux_12 = -(press[pi_2] - press[pi_1])*cond
        dvdu = flux_12*orientation*tubes.l_tot
        avg_flux = np.sum(dvdu)/np.sum(tubes.A_tot*np.abs(orientation)*tubes.l_tot) * np.sum(tubes.A_tot[ti_center])

        _callback.total_time += simulation.dt
        _callback.avg_flux_dt_integral += avg_flux * simulation.dt
        _callback.avg_flux_time_avg = _callback.avg_flux_dt_integral / _callback.total_time

    _callback.phase = phase
    _callback.total_time = 0.0
    _callback.avg_flux_dt_integral = 0.0
    _callback.avg_flux_time_avg = 0.0
    return _callback

