import cProfile
import pstats

from pypnm.multiscale.multiscale_structured import MultiScaleSimStructured
from pypnm.porenetwork.network_factory import structured_network
from pypnm.porenetwork.constants import EAST, WEST
from pypnm.porenetwork.network_manipulation import remove_tubes_between_face_pores
from sim_settings import sim_settings


def multiscale_fixed_pressure():
    nx, ny, nz = 6, 3, 3
    n_fine_per_cell = 5

    network = structured_network(Nx=n_fine_per_cell * nx, Ny=n_fine_per_cell * ny, Nz=n_fine_per_cell * nz)
    network.set_zero_volume_all_tubes()
    multiscale_sim = MultiScaleSimStructured(network, (nx, ny, nz))
    multiscale_sim.set_subnetwork_press_solver(sim_settings['multiscale']['solver'])
    multiscale_sim.set_delta_s_max(sim_settings['multiscale']['ds'])
    multiscale_sim.bc_const_press_xmin(wett_press=1.5e5, nwett_press=2*1.5e5)
    multiscale_sim.bc_const_press_xmax(wett_press=0.0, nwett_press=0.0)
    multiscale_sim.initialize()

    dt = network.total_vol/1e-8
    for i in xrange(3):
        multiscale_sim.advance_in_time(dt)
        multiscale_sim.output_vtk(i, "vtk_structured_pressure_bc")
        multiscale_sim.output_hd5(i, "hf5_structured_pressure_bc")


def multiscale_fixed_flux():
    nx, ny, nz = 6, 3, 3
    n_fine_per_cell = 5
    q_inlet = 1.e-7

    network = structured_network(Nx=n_fine_per_cell * nx, Ny=n_fine_per_cell * ny, Nz=n_fine_per_cell * nz)
    network.set_zero_volume_all_tubes()
    remove_tubes_between_face_pores(network, WEST)
    remove_tubes_between_face_pores(network, EAST)

    multiscale_sim = MultiScaleSimStructured(network, (nx, ny, nz))
    multiscale_sim.set_subnetwork_press_solver(sim_settings['multiscale']['solver'])
    multiscale_sim.set_delta_s_max(sim_settings['multiscale']['ds'])

    multiscale_sim.bc_const_source_xmin(wett_source=0.0, nwett_source=q_inlet)
    multiscale_sim.bc_const_source_xmax(wett_source=-q_inlet, nwett_source=0.0)
    multiscale_sim.initialize()

    dt = 0.01*network.total_vol / q_inlet
    for i in xrange(3):
        multiscale_sim.advance_in_time(dt)
        multiscale_sim.output_vtk(i, "vtk_structured_flux_bc")
        multiscale_sim.output_hd5(i, "hf5_structured_flux_bc")


def print_profiling_info(filename):
    p = pstats.Stats(filename)
    p.sort_stats('cumulative').print_stats(20)
    p.sort_stats('time').print_stats(20)


if __name__ == "__main__":
    import logging
    from pypnm.util.logging_pypnm import logger
    logger.setLevel(logging.DEBUG)

    cProfile.run("multiscale_fixed_flux()", 'restats')
    print_profiling_info('restats')



