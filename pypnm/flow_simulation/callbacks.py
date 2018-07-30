from pypnm.porenetwork.relperm import work_done_per_second, average_flux


def time_avg_energy_dissip(phase, ti_indices=Ellipsis):

    def _callback(simulation):
        network = simulation.network
        if _callback.phase == "NWETT":
            press = network.pores.p_n
            cond = network.tubes.k_n
        elif _callback.phase == "WETT":
            press = network.pores.p_w
            cond = network.tubes.k_w

        energy_dissipated = work_done_per_second(network, press, cond,ti_indices)

        _callback.total_time += simulation.dt
        _callback.work += energy_dissipated * simulation.dt
        _callback.energy_dissip = _callback.work/(_callback.total_time+1e-100)

    _callback.phase = phase
    _callback.energy_dissip = 0.0
    _callback.total_time = 0.0
    _callback.work = 0.0
    return _callback


def time_avg_flux(phase, ti_indices=Ellipsis):

    def _callback(simulation):
        network = simulation.network
        if _callback.phase == "NWETT":
            press = network.pores.p_n
            cond = network.tubes.k_n
        elif _callback.phase == "WETT":
            press = network.pores.p_w
            cond = network.tubes.k_w

        avg_flux = average_flux(network, press, cond, dir=0, ti_indices=ti_indices)

        _callback.total_time += simulation.dt
        _callback.avg_flux_dt_integral += avg_flux * simulation.dt
        _callback.avg_flux_time_avg = _callback.avg_flux_dt_integral / (_callback.total_time+1e-100)


    _callback.phase = phase
    _callback.total_time = 0.0
    _callback.avg_flux_dt_integral = 0.0
    _callback.avg_flux_time_avg = 0.0
    return _callback

