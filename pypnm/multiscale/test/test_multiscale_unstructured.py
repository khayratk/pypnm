import cProfile
import pstats

from PyTrilinos import Epetra
from mpi4py import MPI

from pypnm.multiscale.multiscale_sim import MultiScaleSimUnstructured
from pypnm.porenetwork.network_factory import unstructured_network
import logging

def multiscale_fixed_flux():

    comm = Epetra.PyComm()
    my_id = comm.MyPID()
    mpicomm = MPI.COMM_WORLD

    num_subnetworks = 5
    if my_id == 0:
        print "creating unstructured"
        network = unstructured_network(2000)
        network.set_zero_volume_all_tubes()
        network_volume = network.total_vol
    else:
        network = None
        network_volume = None
    mpicomm.Barrier()

    network_volume = mpicomm.bcast(network_volume)

    print "setting up multiscale unstructured"
    multiscale_sim = MultiScaleSimUnstructured(network, num_subnetworks, comm=None, mpicomm=None, subgraph_ids=None)
    multiscale_sim.bc_const_source_xmin(wett_source=0.0, nwett_source=1e-8)
    multiscale_sim.bc_const_source_xmax(wett_source=-1e-8, nwett_source=0.0)

    print "Initializing solver"
    multiscale_sim.initialize()
    multiscale_sim.set_subnetwork_press_solver("mltrilinos")

    counter = 0
    logger = logging.getLogger('pypnm')
    logger.setLevel(logging.WARNING)

    print "starting simulation"

    dt = 0.01*network_volume / 1e-8
    print "time steps are", dt
    for i in xrange(3):
        multiscale_sim.advance_in_time(dt)
        multiscale_sim.output_vtk(i, "vtk_unstructured_flux_bc")
        multiscale_sim.output_hd5(i, "hf5_unstructured_flux_bc")

def print_profiling_info(filename):
    p = pstats.Stats(filename)
    p.sort_stats('cumulative').print_stats(20)
    p.sort_stats('time').print_stats(20)

if __name__ == "__main__":
    import logging
    from pypnm.util.logging_pypnm import logger
    logger.setLevel(logging.WARN)

    cProfile.run("test_multiscale_fixed_flux()", 'restats')
    print_profiling_info('restats')



