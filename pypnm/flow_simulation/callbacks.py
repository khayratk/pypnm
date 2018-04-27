import numpy as np


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