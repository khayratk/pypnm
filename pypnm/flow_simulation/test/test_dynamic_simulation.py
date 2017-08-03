import cProfile
import os

import numpy as np

from pypnm.porenetwork.constants import WEST, EAST
from pypnm.porenetwork.porenetwork import PoreNetwork
from pypnm.flow_simulation.dynamic_simulation import DynamicSimulation
from pypnm.flow_simulation.simulation_bc import SimulationBoundaryCondition
from sim_settings import sim_settings


def setup_vtk_output(simulation, network):
    simulation.create_vtk_output_folder("paraview_dyn_run", delete_existing_files=True)

    simulation.add_vtk_output_pore_field(network.pores.invaded, "Pore_invaded")
    simulation.add_vtk_output_pore_field(network.pores.p_n, "Pn")
    simulation.add_vtk_output_pore_field(network.pores.p_w, "Pw")
    simulation.add_vtk_output_pore_field(network.pores.p_c, "Pc")
    simulation.add_vtk_output_pore_field(network.pores.sat, "Sat")
    simulation.add_vtk_output_tube_field(network.tubes.invaded, "Tube_invaded")
    simulation.write_vtk_output("initial_network")


def test_dynamic_simulation_sources():
    # print "creating unstructured network"
    # network = unstructured_network(nr_pores=5000)
    # network.set_zero_volume_all_tubes()
    # network.save("test_network_50000.pkl")

    network = PoreNetwork.load("test_network_5000.pkl")

    simulation = DynamicSimulation(network)
    simulation.solver_type = "trilinos"
    bc = SimulationBoundaryCondition()

    bc.set_nonwetting_source(network.pi_list_face[WEST][0:2], np.asarray([1e-6, 1e-6]))
    bc.set_wetting_sink(network.pi_list_face[EAST][0:2], -np.asarray([1e-6, 1e-6]))

    simulation.set_boundary_conditions(bc)
    setup_vtk_output(simulation, network)

    try_again_counter = 0
    delta_t = 0.02 * simulation.network.total_pore_vol/simulation.total_source_nonwett

    for n in xrange(20):
        print ("TimeStep: %g" % delta_t)
        dt_sim = simulation.advance_in_time(delta_t=delta_t)
        print "Nonwetting connected saturation:", simulation.get_nonwetting_connected_saturation()
        print "Nonwetting saturation:", simulation.nonwetting_saturation()

        if not np.isclose(dt_sim, delta_t):
            delta_t = dt_sim
            simulation.reset_status()
            print "Trying Again", try_again_counter
            try_again_counter += 1
            dt_sim = simulation.advance_in_time(delta_t=delta_t)
            assert np.isclose(dt_sim, delta_t), "%f, %f" % (dt_sim, delta_t)

        print "Nonwetting connected saturation:", simulation.get_nonwetting_connected_saturation()
        print "Nonwetting saturation:", simulation.nonwetting_saturation()

        output_filename = "dynamic" + str(n).zfill(8) + ".vtp"
        simulation.write_vtk_output(output_filename)

    os.system('rm -r plots')


def test_dynamic_simulation_sources_with_dirichlet():
    print "creating unstructured network"
    # network = unstructured_network(nr_pores=5000)
    # network.save("test_network_5000.pkl")

    network = PoreNetwork.load("test_network_5000.pkl")
    network.set_zero_volume_all_tubes()

    simulation = DynamicSimulation(network)
    simulation.solver_type = "trilinos"
    bc = SimulationBoundaryCondition()

    bc.set_pressure_inlet(pi_list=network.pi_list_face[WEST], p_wett=1.5e5, p_nwett=2*1.5e5)
    bc.set_wetting_sink(network.pi_list_face[EAST][0:2], -np.asarray([1e-6, 1e-6]))

    simulation.set_boundary_conditions(bc)

    setup_vtk_output(simulation, network)

    try_again_counter = 0
    delta_t = 0.02 * simulation.network.total_pore_vol/simulation.total_source_nonwett
    for n in xrange(40):
        print "TimeStep: %g" % delta_t
        dt_sim = simulation.advance_in_time(delta_t=delta_t)
        print "Nonwetting connected saturation:", simulation.get_nonwetting_connected_saturation()
        print "Nonwetting saturation:", simulation.nonwetting_saturation()

        if not np.isclose(dt_sim, delta_t):
            delta_t = dt_sim*0.98
            simulation.reset_status()
            print "Trying Again", try_again_counter
            try_again_counter += 1
            dt_sim = simulation.advance_in_time(delta_t=delta_t)
            assert np.isclose(dt_sim, delta_t), "%f, %f" % (dt_sim, delta_t)

        output_filename = "dynamic" + str(n).zfill(8)
        simulation.write_vtk_output(output_filename)
    os.system('rm -r plots')


def test_dynamic_simulation_dirichlet():
    print "creating unstructured network"
    # network = unstructured_network(nr_pores=5000)
    # network.save("test_network_5000.pkl")

    network = PoreNetwork.load("test_network_5000.pkl")
    network.set_zero_volume_all_tubes()

    simulation = DynamicSimulation(network)
    simulation.solver_type = "trilinos"
    bc = SimulationBoundaryCondition()

    bc.set_pressure_inlet(pi_list=network.pi_list_face[WEST], p_wett=1.5e5, p_nwett=2*1.5e5)
    bc.set_pressure_outlet(pi_list=network.pi_list_face[EAST], p_wett=0.0)

    simulation.set_boundary_conditions(bc)

    setup_vtk_output(simulation, network)

    try_again_counter = 0
    delta_t = 0.2 * simulation.network.total_pore_vol/1.e-8
    for n in xrange(40):
        print "TimeStep: %g" % delta_t
        dt_sim = simulation.advance_in_time(delta_t=delta_t)
        print "Nonwetting connected saturation:", simulation.get_nonwetting_connected_saturation()
        print "Nonwetting saturation:", simulation.nonwetting_saturation()

        if not np.isclose(dt_sim, delta_t):
            delta_t = dt_sim*0.98
            simulation.reset_status()
            print "Trying Again", try_again_counter
            try_again_counter += 1
            dt_sim = simulation.advance_in_time(delta_t=delta_t)
            assert np.isclose(dt_sim, delta_t), "%f, %f" % (dt_sim, delta_t)

        output_filename = "dynamic" + str(n).zfill(8)
        simulation.write_vtk_output(output_filename)
    os.system('rm -r plots')


def print_profiling_info(filename):
    import pstats
    p = pstats.Stats(filename)
    p.sort_stats('cumulative').print_stats(20)
    p.sort_stats('time').print_stats(20)


if __name__ == "__main__":
    import logging
    from pypnm.util.logging_pypnm import logger
    logger.setLevel(logging.DEBUG)

    enable_profiling = sim_settings['enable_profiling']

    if enable_profiling:
        cProfile.run("test_dynamic_simulation_sources_with_dirichlet()", 'restats')
        print_profiling_info('restats')
    else:
        test_dynamic_simulation_sources_with_dirichlet()