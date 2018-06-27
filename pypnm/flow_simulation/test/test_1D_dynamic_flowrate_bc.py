import numpy as np
from pypnm.porenetwork.constants import WEST, EAST
from pypnm.porenetwork.network_factory import structured_network
from pypnm.porenetwork.porenetwork import PoreNetwork
from pypnm.flow_simulation.dynamic_simulation import DynamicSimulation
from pypnm.flow_simulation.simulation_bc import SimulationBoundaryCondition
import logging

logger = logging.getLogger('pypnm')
logger.setLevel("INFO")

fluid_properties = {"gamma": 1.0, "mu_n": 0.1, "mu_w": 1.0}


def dynamic_simulation():
    # Generate small unstructured network.

    try:
        network = PoreNetwork.load("network_1D")
        print "loaded network"
    except IOError:
        network = structured_network(20, 1, 1)
        network.set_zero_volume_all_tubes()
        network.save("network_1D")

    # Initialize solver
    simulation = DynamicSimulation(network, fluid_properties, explicit=False, delta_pc=0.01)

    # Set boundary conditions using list of pores and list of sources. Here a total inflow of q_total is used
    # distributed over the inlet and outlet pores
    bc = SimulationBoundaryCondition()
    q_total = 1e-13  # All units are SI units

    pi_inlet = network.pi_list_face[WEST]
    pi_outlet = network.pi_list_face[EAST]

    bc.set_nonwetting_source(pi_inlet, q_total / len(pi_inlet) * np.ones(len(pi_inlet)))
    bc.set_pressure_outlet(pi_list=pi_outlet, p_wett=0.0)

    simulation.set_boundary_conditions(bc)

    # Set output-time interval for simulation
    delta_t_output = 0.01 * simulation.network.total_pore_vol / simulation.total_source_nonwett

    #  Before starting simulation set up the postprocessor

    # VTK post processing- To save space the user specifies which fields to output
    simulation.create_vtk_output_folder("implicit", delete_existing_files=True)

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

            print "Nonwetting saturation:", simulation.nonwetting_saturation()

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    dynamic_simulation()
