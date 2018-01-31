"""
This example sets up a dynamic two-phase flow simulation in a pore network using the approximate multiscale solver.
The boundary conditions are set to be fixed nonwetting fluid source and fixed wetting fluid sinks.
"""

from pypnm.porenetwork.constants import WEST, EAST
from pypnm.porenetwork.network_factory import unstructured_network_delaunay
from pypnm.multiscale.multiscale_unstructured import MultiScaleSimUnstructured
from pypnm.porenetwork.network_manipulation import remove_tubes_between_face_pores
from PyTrilinos import Epetra
from mpi4py import MPI
from pypnm.porenetwork.porenetwork import  PoreNetwork

from sim_settings import sim_settings
import logging

logger = logging.getLogger('pypnm')
logger.setLevel("WARN")


def multiscale_simulation(restart):
    comm = Epetra.PyComm()
    mpicomm = MPI.COMM_WORLD
    my_id = comm.MyPID()

    if restart:
        multiscale_sim = MultiScaleSimUnstructured.load()
        if my_id == 0:
            network_volume = multiscale_sim.network.total_vol
        else:
            network_volume = None

        mpicomm.Barrier()
        network_volume = mpicomm.bcast(network_volume)
        q_total = 1.e-8  # All units are SI units

    else:
        # Processor 0 loads the network
        if my_id == 0:
            # Generate  unstructured network.
            try:
                network = PoreNetwork.load("benchmark_network.pkl")

            except IOError:
                network = unstructured_network_delaunay(nr_pores=100000)
                # Remove pore throats between inlet and outlet pores. This is to avoid high pressure at the inlet
                network = remove_tubes_between_face_pores(network, EAST)
                network = remove_tubes_between_face_pores(network, WEST)
                network.save("benchmark_network.pkl")

            # The implemented multiscale flow solver only works with zero volume pore throats
            network.set_zero_volume_all_tubes()
            network_volume = network.total_vol
        else:
            network = None
            network_volume = None

        mpicomm.Barrier()

        network_volume = mpicomm.bcast(network_volume)

        # Set number of subnetworks
        num_subnetworks = 20

        # Initialize solver
        multiscale_sim = MultiScaleSimUnstructured(network, sim_settings["fluid_properties"], num_subnetworks)

        # Set boundary conditions using list of pores and list of sources. Here a total inflow of q_total is used
        # distributed over the inlet and outlet pores
        q_total = 1.e-8  # All units are SI units

        multiscale_sim.bc_const_source_xmin(wett_source=0.0, nwett_source=q_total)
        multiscale_sim.bc_const_source_xmax(wett_source=-q_total, nwett_source=0.0)

        print "Initializing solver"
        multiscale_sim.initialize()
        multiscale_sim.set_subnetwork_press_solver("petsc")

    print "starting simulation"

    dt = 0.01*network_volume / q_total
    print "time step is", dt

    for i in xrange(4):
        multiscale_sim.bc_const_source_xmin(wett_source=0.0, nwett_source=q_total)
        multiscale_sim.bc_const_source_xmax(wett_source=-q_total, nwett_source=0.0)
        multiscale_sim.initialize()

        multiscale_sim.advance_in_time(dt)
        # Output vtk and hf5 files for postprocessing
        multiscale_sim.output_vtk(i, "vtk_unstructured_flux_bc")
        multiscale_sim.output_hd5(i, "hf5_unstructured_flux_bc")
        multiscale_sim.save()


if __name__ == "__main__":
    multiscale_simulation(restart=False)
    print "Exiting simulation"
