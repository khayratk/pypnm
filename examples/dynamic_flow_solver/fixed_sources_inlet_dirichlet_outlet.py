"""
This example sets up a dynamic two-phase flow simulation in a pore network.
The boundary conditions are set to be fixed nonwetting fluid source at the inlet
and fixed wetting pressure at the outlet.
The simulation stops when a domain saturation of 0.5 has been reached
"""

import numpy as np
from pypnm.porenetwork.constants import WEST, EAST
from pypnm.porenetwork.network_factory import unstructured_network
from pypnm.flow_simulation.dynamic_simulation import DynamicSimulation
from pypnm.flow_simulation.simulation_bc import SimulationBoundaryCondition


def dynamic_simulation():
    # Generate small unstructured network.
    network = unstructured_network(nr_pores=2000)

    # The implemented dynamic flow solver can only work with zero volume pore throats
    network.set_zero_volume_all_tubes()

    # Initialize solver
    simulation = DynamicSimulation(network)
    simulation.press_solver_type = "petsc"

    # Set boundary conditions using list of pores and list of sources. Here a total inflow of q_total is used
    # distributed over the inlet and outlet pores
    bc = SimulationBoundaryCondition()
    q_total = 1.e-8  # All units are SI units

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

    for n in xrange(50):
        print ("TimeStep: %g" % delta_t_output)
        dt_sim = simulation.advance_in_time(delta_t=delta_t_output)

        simulation.write_vtk_output(label=n)
        simulation.write_to_hdf(label=n, folder_name="paraview_dyn_run")
        print "Nonwetting saturation:", simulation.nonwetting_saturation()

if  __name__=="__main__":
    dynamic_simulation()