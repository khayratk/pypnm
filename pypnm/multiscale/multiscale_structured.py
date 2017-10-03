import numpy as np
from PyTrilinos import Epetra
from mpi4py import MPI

from pypnm.ams.msfv import MSFV
from pypnm.ams.wire_basket import create_wire_basket
from pypnm.attribute_calculators.conductance_calc import ConductanceCalc
from pypnm.attribute_calculators.pc_computer import DynamicCapillaryPressureComputer
from pypnm.flow_simulation.dynamic_simulation import DynamicSimulation
from pypnm.multiscale.multiscale_sim import MultiscaleSim, _create_subnetworks, _create_inter_subgraph_edgelist, logger, \
    _network_saturation, reduce_dictionary_values
from pypnm.multiscale.utils import update_inter_invasion_status_snap_off, create_matrix, create_rhs, \
    update_inter_invasion_status_piston, update_inter_invasion_status_piston_wetting, \
    create_subnetwork_boundary_conditions
from pypnm.porenetwork import gridder
from pypnm.util.igraph_utils import network_to_igraph, coarse_graph_from_partition
from pypnm.util.indexing import GridIndexer3D


class MultiScaleSimStructured(MultiscaleSim):
    def __init__(self, network, coarse_dimensions):
        self.network = network
        comm = Epetra.PyComm()
        mpicomm = MPI.COMM_WORLD
        self.comm, self.mpicomm = comm, mpicomm

        nx, ny, nz = coarse_dimensions
        subnetwork_indices = gridder.grid3d_of_pore_lists(network, nx, ny, nz)
        gi_subnet_primal = GridIndexer3D(nx, ny, nz)
        subgraph_ids = np.zeros(network.nr_p, dtype=np.int)

        for index in subnetwork_indices:
            subgraph_ids[subnetwork_indices[index]] = gi_subnet_primal.get_index(*index)

        self.graph = network_to_igraph(network, edge_attributes=["l", "A_tot", "r", "G"])

        self.unique_map = Epetra.Map(-1, network.nr_p, 0, comm)

        self.graph.vs["global_id"] = np.arange(self.graph.vcount())
        self.graph.es["global_id"] = np.arange(self.graph.ecount())
        self.graph.vs["proc_id"] = 0

        self.graph.vs["subgraph_id"] = subgraph_ids

        coarse_graph = coarse_graph_from_partition(self.graph, subgraph_ids)
        coarse_graph.vs["subgraph_id"] = np.arange(coarse_graph.vcount())
        coarse_graph.vs["proc_id"] = 0

        self.my_subnetworks = _create_subnetworks(network, subgraph_ids, coarse_graph, mpicomm)
        self.wire_basket = create_wire_basket(network, nx, ny, nz)

        self.my_subgraph_ids = np.unique([gi_subnet_primal.get_index(*index) for index in subnetwork_indices])
        self.inter_subgraph_edges = _create_inter_subgraph_edgelist(self.graph, self.my_subgraph_ids, mpicomm)

        self.p_c = Epetra.Vector(self.unique_map)
        self.p_w = Epetra.Vector(self.unique_map)
        self.sat = Epetra.Vector(self.unique_map)
        self.global_source_wett = Epetra.Vector(self.unique_map)
        self.global_source_nonwett = Epetra.Vector(self.unique_map)
        self.out_flux_w = Epetra.Vector(self.unique_map)
        self.out_flux_n = Epetra.Vector(self.unique_map)

        self.global_bound_press_wett = Epetra.Vector(self.unique_map)
        self.global_bound_press_nonwett = Epetra.Vector(self.unique_map)

        self.pi_list_press_inlet = []
        self.press_inlet_w = None
        self.press_inlet_nw = None

        self.pi_list_press_outlet = []
        self.press_outlet_w = None
        self.press_outlet_nw = None

        self.subgraph_ids = subgraph_ids
        self.simulations = dict()
        for i in self.my_subgraph_ids:
            self.simulations[i] = DynamicSimulation(self.my_subnetworks[i])
            self.simulations[i].solver_type = "lu"

            k_comp = ConductanceCalc(self.my_subnetworks[i])
            k_comp.compute()
            pc_comp = DynamicCapillaryPressureComputer(self.my_subnetworks[i])
            pc_comp.compute()

        self.time = 0.0
        self.stop_time = None

        self.delta_s_max = 0.02

    def initialize(self):
        self.p_c = self._update_pc_from_subnetworks(self.p_c, self.my_subnetworks)
        self.sat = self._update_sat_from_subnetworks(self.sat, self.my_subnetworks)

        self.inter_subgraph_edges = update_inter_invasion_status_snap_off(self.inter_subgraph_edges, self.p_c)
        self.inter_subgraph_edges = self._update_inter_conductances(self.inter_subgraph_edges,
                                                                                         self.p_c)

        A = create_matrix(self.unique_map, ["k_n", "k_w"], self.my_subnetworks, None, self.inter_subgraph_edges,
                          matformat="scipy")
        rhs = create_rhs(self.unique_map, self.my_subnetworks, None, self.inter_subgraph_edges, self.p_c * 0.0,
                         self.global_source_wett, self.global_source_nonwett, vecformat="numpy")
        A_n = create_matrix(self.unique_map, ["k_n"], self.my_subnetworks, None, self.inter_subgraph_edges,
                            matformat="scipy")

        self.msfv = MSFV(A=A, source_term=rhs, wirebasket=self.wire_basket, div_source_term=[A_n, np.copy(rhs)],
                         primal_cell_labels=self.subgraph_ids)

    def __solve_pressure(self, recompute=True, tol=1e-5):
        A = create_matrix(self.unique_map, ["k_n", "k_w"], self.my_subnetworks, None, self.inter_subgraph_edges,
                          matformat="scipy")
        A = A.tocsr()
        rhs = create_rhs(self.unique_map, self.my_subnetworks, None, self.inter_subgraph_edges, self.p_c * 0.0,
                         self.global_source_wett, self.global_source_nonwett, vecformat="numpy")
        A_n = create_matrix(self.unique_map, ["k_n"], self.my_subnetworks, None, self.inter_subgraph_edges,
                            matformat="scipy")

        for pi in np.union1d(self.pi_list_press_inlet, self.pi_list_press_outlet):
            A_n.data[A_n.indptr[pi]:A_n.indptr[pi + 1]] = 0.0  # Set row of A_n matrix to zero
            A.data[A.indptr[pi]:A.indptr[pi + 1]] = 0.0
            A[pi, pi] = 1.0

        rhs[self.pi_list_press_inlet] = self.press_inlet_w
        rhs[self.pi_list_press_outlet] = self.press_outlet_w

        logger.info("Solving using MSFV method")

        self.msfv.set_matrix(A)
        self.msfv.set_source_term(rhs)
        self.msfv.set_div_source_term([-A_n, self.p_c[:]])
        p_w_numpy = self.msfv.solve(iterative=True, tol=tol, restriction_operator="msfe", x0=self.p_w[:],
                                    recompute=recompute)
        self.p_w[:] = p_w_numpy[:]
        p_n = self.p_w + self.p_c
        print "Done solving using MSFV method"
        return p_n, self.p_w

    def advance_in_time(self, delta_t):
        self.stop_time = self.time + delta_t

        while True:
            sat_current = _network_saturation(self.my_subnetworks, self.mpicomm)
            print "Current Saturation of Complete Network", sat_current
            print "Current Time", self.time

            # Update global capillary pressure and saturation vectors from vectors contained in subnetworks
            self.p_c, self.sat = self._sync_pc_and_sat()

            # Snap-off boundary throats
            self.inter_subgraph_edges = update_inter_invasion_status_snap_off(self.inter_subgraph_edges, self.p_c)
            self.inter_subgraph_edges = self._update_inter_conductances(self.inter_subgraph_edges, self.p_c)

            # Invade boundary throats with nonwetting phase
            p_n, self.p_w = self.__solve_pressure(recompute=True)

            AminusD_n = create_matrix(self.unique_map, ["k_n"], None, None, self.inter_subgraph_edges)
            ierr = AminusD_n.Multiply(False, p_n, self.out_flux_n)
            assert ierr == 0

            self.inter_subgraph_edges = update_inter_invasion_status_piston(self.inter_subgraph_edges, self.p_w,
                                                                            self.p_c, self.sat, self.out_flux_n,
                                                                            self.global_source_nonwett)
            self.inter_subgraph_edges = self._update_inter_conductances(self.inter_subgraph_edges, self.p_c)

            # Snap-off tubes
            p_n, self.p_w = self.__solve_pressure(recompute=False)

            self.inter_subgraph_edges = update_inter_invasion_status_snap_off(self.inter_subgraph_edges, self.p_c)
            self.inter_subgraph_edges = self._update_inter_conductances(self.inter_subgraph_edges,
                                                                                             self.p_c)

            # Invade tubes with wetting phase
            p_n, self.p_w = self.__solve_pressure(recompute=False)

            inter_subgraph_edges = update_inter_invasion_status_piston_wetting(self.inter_subgraph_edges, self.p_w,
                                                                               self.p_c, self.sat)
            inter_subgraph_edges = self._update_inter_conductances(inter_subgraph_edges, self.p_c)

            # Setup boundary conditions for subnetwork simulations
            p_n, self.p_w = self.__solve_pressure(recompute=False)

            AminusD_w = create_matrix(self.unique_map, ["k_w"], None, None, inter_subgraph_edges)
            AminusD_n = create_matrix(self.unique_map, ["k_n"], None, None, inter_subgraph_edges)

            ierr = AminusD_w.Multiply(False, self.p_w, self.out_flux_w)
            assert ierr == 0
            ierr = AminusD_n.Multiply(False, p_n, self.out_flux_n)
            assert ierr == 0

            subgraph_ids_distributed = np.asarray(self.graph.vs['subgraph_id'])

            press_bc_inlet = {"pi_list": self.pi_list_press_inlet, "wett": self.press_inlet_w, "nwett": self.press_inlet_nw}
            press_bc_outlet = {"pi_list": self.pi_list_press_outlet, "wett": self.press_outlet_w,
                               "nwett": self.press_outlet_nw}

            bc = create_subnetwork_boundary_conditions(self.global_source_nonwett - self.out_flux_n,
                                                       self.global_source_wett - self.out_flux_w,
                                                       self.my_subnetworks, subgraph_ids_distributed,
                                                       press_bc_inlet, press_bc_outlet)

            for i in self.my_subgraph_ids:
                self.simulations[i].set_boundary_conditions(bc[i])
                if bc[i].no_dirichlet:
                    mass_balance = bc[i].mass_balance()
                    total_sources = self.simulations[i].total_source_nonwett + self.simulations[i].total_source_wett
                    assert mass_balance < total_sources * 1.e-10, "Mass balance: %e, Total sources: %g" % (
                    mass_balance, total_sources)

            subgraph_id_to_nw_influx = self.compute_nonwetting_influx(self.simulations)
            subgraph_id_to_volume = {i: self.my_subnetworks[i].total_vol for i in self.my_subgraph_ids}
            subgraph_id_to_saturation = {i: np.sum(self.my_subnetworks[i].pores.sat * self.my_subnetworks[i].pores.vol)
                                            / self.my_subnetworks[i].total_vol for i in self.my_subgraph_ids}

            for i in self.my_subnetworks:
                assert self.my_subnetworks[i].total_throat_vol == 0
                assert self.my_subnetworks[i].total_vol == np.sum(self.my_subnetworks[i].pores.vol)

            total_nw_influx = reduce_dictionary_values(subgraph_id_to_nw_influx, self.mpicomm)
            print "computed nonwetting influxes", total_nw_influx, np.sum(self.global_source_nonwett)

            dt = self._compute_timestep(subgraph_id_to_nw_influx, subgraph_id_to_saturation, subgraph_id_to_volume,
                                        self.delta_s_max)

            dt = min(dt, self.stop_time - self.time)

            dt_sim_min = 1.0e20

            print "advancing simulations with timestep", dt
            for i in self.simulations:
                dt_sim = self.simulations[i].advance_in_time(delta_t=dt)
                dt_sim_min = min(dt_sim, dt_sim_min)

            self.comm.Barrier()

            dt_sim_min = self.mpicomm.allreduce(dt_sim_min, op=MPI.MIN)

            backtracking = not np.isclose(dt_sim_min, dt)
            if backtracking:
                dt_sim_min *= 0.95
                print "Backtracking all simulations"

                for i in self.simulations:
                    self.simulations[i].reset_status()

                if dt_sim_min == 0.0:
                    continue

                print "Rerunning all simulations"
                for i in self.simulations:
                    dt_sim = self.simulations[i].advance_in_time(delta_t=dt_sim_min)
                    assert np.isclose(dt_sim, dt_sim_min), str(dt_sim) + "  " + str(dt_sim_min)

            self.comm.Barrier()
            self.time += dt_sim_min

            print "time is ", self.time
            print "stop time is", self.stop_time

            if np.isclose(self.time, self.stop_time):
                break