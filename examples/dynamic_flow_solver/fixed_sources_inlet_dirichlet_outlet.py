"""
This example sets up a dynamic two-phase flow simulation in a pore network.
The boundary conditions are set to be fixed nonwetting fluid source at the inlet
and fixed wetting pressure at the outlet.
The simulation stops when a domain saturation of 0.5 has been reached
"""

import numpy as np
from pypnm.porenetwork.constants import WEST, EAST
from pypnm.porenetwork.network_factory import unstructured_network_periodic_y, structured_network
from pypnm.flow_simulation.dynamic_simulation import DynamicSimulation
from pypnm.flow_simulation.simulation_bc import SimulationBoundaryCondition
from sim_settings import sim_settings
from pypnm.porenetwork.component import tube_list_ngh_to_pore_list, pore_list_ngh_to_pore_list
import logging
from pypnm.porenetwork.network_manipulation import remove_tubes_between_face_pores
from pypnm.porenetwork.porenetwork import  PoreNetwork
logger = logging.getLogger('pypnm')
logger.setLevel("INFO")
import pstats
import cProfile

def dynamic_simulation():
    try:
        network = PoreNetwork.load("network.pkl")
    except IOError:
        # Generate small unstructured network.
        network = structured_network(20, 10, 10, periodic=True)
        network = remove_tubes_between_face_pores(network, EAST)
        network = remove_tubes_between_face_pores(network, WEST)
        pi_inlet = network.pi_list_face[WEST]

        ti_list_inlet = np.unique(tube_list_ngh_to_pore_list(network, pi_inlet))
        network.set_radius_tubes(ti_list_inlet, r=np.mean(network.tubes.r))
        network.set_radius_pores(pi_inlet, r=np.mean(network.pores.r))
        pi_inlet_ngh = np.unique(pore_list_ngh_to_pore_list(network, pi_inlet))
        network.set_radius_pores(pi_inlet_ngh, r=np.mean(network.pores.r))
        network._fix_tubes_larger_than_ngh_pores()


    # The implemented dynamic flow solver can only work with zero volume pore throats
    network.set_zero_volume_all_tubes()

    # Initialize solver
    simulation = DynamicSimulation(network, sim_settings["fluid_properties"], explicit=True, delta_pc=0.01)

    # Set boundary conditions using list of pores and list of sources. Here a total inflow of q_total is used
    # distributed over the inlet and outlet pores
    bc = SimulationBoundaryCondition()
    q_total = sim_settings['dynamic']['q_n_inlet']   # All units are SI units

    pi_inlet = network.pi_list_face[WEST]
    pi_outlet = network.pi_list_face[EAST]

    bc.set_nonwetting_source(pi_inlet, q_total/len(pi_inlet)*np.ones(len(pi_inlet)))
    bc.set_pressure_outlet(pi_list=pi_outlet, p_wett=0.0)

    simulation.set_boundary_conditions(bc)

    # Set output-time interval for simulation
    delta_t_output = 0.01 * simulation.network.total_pore_vol/simulation.total_source_nonwett

    #  Before starting simulation set up the postprocessor

    # VTK post processing- To save space the user specifies which fields to output
    simulation.create_vtk_output_folder("paraview_dyn_run", delete_existing_files=True)

    simulation.add_vtk_output_pore_field(network.pores.invaded, "Pore_invaded")  # invaded = occupied by nonwett phase
    simulation.add_vtk_output_pore_field(network.pores.p_n, "Pn")
    simulation.add_vtk_output_pore_field(network.pores.p_w, "Pw")
    simulation.add_vtk_output_pore_field(network.pores.p_c, "Pc")
    simulation.add_vtk_output_pore_field(network.pores.sat, "Sat")
    simulation.add_vtk_output_tube_field(network.tubes.invaded, "Tube_invaded")
    simulation.write_vtk_output("initial_network")

    try:
        for n in xrange(100):
            print ("TimeStep: %g" % delta_t_output)
            simulation.advance_in_time(delta_t=delta_t_output)

            simulation.write_vtk_output(label=n)
            simulation.write_to_hdf(label=n, folder_name="paraview_dyn_run")

            network.save("network_history/network" + str(n).zfill(5) + ".pkl")

            print "Nonwetting saturation:", simulation.nonwetting_saturation()

    except KeyboardInterrupt:
        pass


def print_profiling_info(filename):
    p = pstats.Stats(filename)
    p.sort_stats('cumulative').print_stats(50)
    p.sort_stats('time').print_stats(50)


if __name__ == "__main__":
    exec_string = 'dynamic_simulation()'

    cProfile.run(exec_string, 'restats')
    print_profiling_info('restats')
