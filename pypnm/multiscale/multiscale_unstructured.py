import pymetis
import numpy as np
from PyTrilinos import Epetra
from mpi4py import MPI

from pypnm.ams.msrsb import MSRSB
from pypnm.attribute_calculators.conductance_calc import ConductanceCalc
from pypnm.attribute_calculators.pc_computer import DynamicCapillaryPressureComputer
from pypnm.flow_simulation.dynamic_simulation import DynamicSimulation


from pypnm.linalg.trilinos_interface import solve_aztec
from pypnm.multiscale.multiscale_sim import MultiscaleSim, _create_subnetworks, _create_inter_subgraph_edgelist, \
    _network_saturation, logger
from pypnm.multiscale.utils import update_inter_invasion_status_snap_off, create_matrix, create_rhs, solve_multiscale, \
    update_inter_invasion_status_piston, update_inter_invasion_status_piston_wetting, \
    create_subnetwork_boundary_conditions
from pypnm.util.igraph_utils import network_to_igraph, coarse_graph_from_partition, support_of_basis_function, \
    graph_central_vertex


class MultiScaleSimUnstructured(MultiscaleSim):
    """
    Parameters
    ----------
    network: PoreNetwork
    num_subnetworks: int
        Number of subnetworks to split up network
    comm: Epetra communicator, optional
    mpicomm: mpi4py communicator, optional
    subgraph_ids: numpy array, optional
        Integer array of length network.nr_p containing the partition of the network.
    """
    def __init__(self, network,  fluid_properties, num_subnetworks, comm=None, mpicomm=None, subgraph_ids=None,
                 delta_s_max=0.01):
        self.network = network

        if comm is None:
            comm = Epetra.PyComm()

        if mpicomm is None:
            mpicomm = MPI.COMM_WORLD

        self.comm, self.mpicomm = comm, mpicomm
        self.fluid_properties = fluid_properties
        self.my_id = comm.MyPID()
        my_id = self.my_id
        self.num_proc = comm.NumProc()

        self.num_subnetworks = num_subnetworks

        # Create graph corresponding to network and another coarser graph with its vertices corresponding to subnetworks
        if my_id == 0:
            self.graph = network_to_igraph(network, edge_attributes=["l", "A_tot", "r", "G"])

            # create global_id attributes before creating subgraphs.
            self.graph.vs["global_id"] = np.arange(self.graph.vcount())
            self.graph.es["global_id"] = np.arange(self.graph.ecount())

            if subgraph_ids is None:
                _, subgraph_ids = pymetis.part_graph(num_subnetworks, self.graph.get_adjlist())

            subgraph_ids = np.asarray(subgraph_ids)
            self.graph.vs["subgraph_id"] = subgraph_ids

            # Assign a processor id to each subgraph
            coarse_graph = coarse_graph_from_partition(self.graph, subgraph_ids)
            _, proc_ids = pymetis.part_graph(self.num_proc, coarse_graph.get_adjlist())
            coarse_graph.vs['proc_id'] = proc_ids
            coarse_graph.vs["subgraph_id"] = np.arange(coarse_graph.vcount())

            # Assign a processor id to each pore
            subgraph_id_to_proc_id = {v["subgraph_id"]: v['proc_id'] for v in coarse_graph.vs}
            self.graph.vs["proc_id"] = [subgraph_id_to_proc_id[v["subgraph_id"]] for v in self.graph.vs]

        if my_id != 0:
            network = None
            self.graph = None
            coarse_graph = None
            subgraph_ids = None

        self.coarse_graph = self.mpicomm.bcast(coarse_graph, root=0)
        self.graph = self.distribute_graph(self.graph, self.coarse_graph, self.mpicomm)

        self.my_subnetworks = _create_subnetworks(network, subgraph_ids, self.coarse_graph, self.mpicomm)

        self.my_subgraph_ids = self.my_subnetworks.keys()
        self.my_subgraph_ids_with_ghost = list(set().union(*self.coarse_graph.neighborhood(self.my_subgraph_ids)))

        self.inter_subgraph_edges = _create_inter_subgraph_edgelist(self.graph, self.my_subgraph_ids, self.mpicomm)
        self.inter_processor_edges = self.create_inter_processor_edgelist(self.graph, self.my_subgraph_ids, self.mpicomm)

        self.subgraph_id_to_v_center_id = self.subgraph_central_vertices(self.graph, self.my_subgraph_ids_with_ghost)

        # Epetra maps to facilitate data transfer between processors
        self.unique_map, self.nonunique_map, self.subgraph_ids_vec = self.create_maps(self.graph, self.comm)

        self.epetra_importer = Epetra.Import(self.nonunique_map, self.unique_map)
        assert self.epetra_importer.NumPermuteIDs() == 0
        assert self.epetra_importer.NumSameIDs() == self.unique_map.NumMyElements()

        # subgraph_id to support vertices map. Stores the support vertex ids (global ids) of both
        # subnetworks belonging to this processor as well as those belonging to ghost subnetworks.
        # Only support vertex ids belonging to this processor are stored.
        self.my_basis_support = dict()
        self.my_subgraph_support = dict()

        my_global_elements = self.unique_map.MyGlobalElements()


        self.graph["global_to_local"] = dict((v["global_id"], v.index) for v in self.graph.vs)
        self.graph["local_to_global"] = dict((v.index, v["global_id"]) for v in self.graph.vs)

        # support region for each subgraph
        for i in self.my_subgraph_ids:
            self.my_subgraph_support[i] = self.my_subnetworks[i].pi_local_to_global


        # support region for each subgraph
        self.my_subgraph_support_with_ghosts = dict()
        for i in self.my_subgraph_ids_with_ghost:
            self.my_subgraph_support_with_ghosts[i] = np.asarray(self.graph.vs.select(subgraph_id=i)["global_id"])

        for i in self.my_subgraph_ids:
            assert np.all(np.sort(self.my_subgraph_support[i]) == np.sort(self.my_subgraph_support_with_ghosts[i]))

        for i in self.my_subgraph_ids:
            if num_subnetworks == 1:
                support_vertices = self.my_subnetworks[0].pi_local_to_global
            else:
                support_vertices = support_of_basis_function(i, self.graph, self.coarse_graph,
                                                             self.subgraph_id_to_v_center_id, self.my_subgraph_support_with_ghosts)

            self.my_basis_support[i] = np.intersect1d(support_vertices, my_global_elements).astype(np.int32)


        # Create distributed arrays - Note: Memory wasted here by allocating extra arrays which include ghost cells.
        # This can be optimized but the python interface for PyTrilinos is not documented well enough.
        # Better would be to create only the arrays which include ghost cells.
        unique_map = self.unique_map
        nonunique_map = self.nonunique_map
        self.p_c = Epetra.Vector(unique_map)
        self.p_w = Epetra.Vector(unique_map)
        self.sat = Epetra.Vector(unique_map)
        self.global_source_wett = Epetra.Vector(unique_map)
        self.global_source_nonwett = Epetra.Vector(unique_map)
        self.out_flux_w = Epetra.Vector(unique_map)
        self.out_flux_n = Epetra.Vector(unique_map)

        self.p_c_with_ghost = Epetra.Vector(nonunique_map)
        self.p_w_with_ghost = Epetra.Vector(nonunique_map)
        self.sat_with_ghost = Epetra.Vector(nonunique_map)
        self.global_source_nonwett_with_ghost = Epetra.Vector(nonunique_map)
        self.out_flux_n_with_ghost = Epetra.Vector(nonunique_map)

        # Crate dynamic simulations
        self.simulations = dict()

        for i in self.my_subgraph_ids:
            self.simulations[i] = DynamicSimulation(self.my_subnetworks[i], self.fluid_properties)
            self.simulations[i].solver_type = "lu"

            k_comp = ConductanceCalc(self.my_subnetworks[i], self.fluid_properties)
            k_comp.compute()
            pc_comp = DynamicCapillaryPressureComputer(self.my_subnetworks[i])
            pc_comp.compute()

        self.delta_s_max = delta_s_max
        self.p_tol = 1.e-6
        self.time = 0.0
        self.stop_time = None

        self.pi_list_press_inlet = []
        self.press_inlet_w = None
        self.press_inlet_nw = None

        self.pi_list_press_outlet = []
        self.press_outlet_w = None
        self.press_outlet_nw = None



    @staticmethod
    def create_maps(graph, comm):
        """
        Parameters
        ----------
        graph: igraph
            igraph representing the network.
        comm: Epetra communicator

        Returns
        -------
        unique_map: Epetra Map
            Map representing the vertices belonging to each processor
        nonunique_map: Epetra Map
            Map representing the vertices belonging to each processor as well one ghost vertex neighborhood
        subgraph_ids_vec: Epetra Vector
            Epetra vector recording the subgraph id of each vertex
        """
        my_id = comm.MyPID()
        my_local_elements = graph.vs(proc_id_eq=my_id).indices
        my_global_elements = graph.vs[my_local_elements]['global_id']
        my_ghost_elements = set().union(*graph.neighborhood(my_local_elements))
        my_ghost_elements = set(graph.vs(my_ghost_elements)['global_id']) - set(my_global_elements)
        my_ghost_elements = sorted(list(my_ghost_elements))
        assert not set(my_global_elements).intersection(my_ghost_elements)

        unique_map = Epetra.Map(-1, my_global_elements, 0, comm)
        nonunique_map = Epetra.Map(-1, my_global_elements + my_ghost_elements, 0, comm)

        subgraph_ids_vec = Epetra.Vector(unique_map)
        subgraph_ids_vec[:] = np.asarray(graph.vs[my_local_elements]['subgraph_id'], dtype=np.int32)

        return unique_map, nonunique_map, subgraph_ids_vec

    @staticmethod
    def subgraph_central_vertices(graph, my_subgraph_ids_with_ghost):
        basis_id_to_v_center_id = {}
        subgraph_id_vec = np.asarray(graph.vs['subgraph_id'])
        for i in my_subgraph_ids_with_ghost:
            vs_subgraph = (subgraph_id_vec == i).nonzero()[0]
            assert len(vs_subgraph) > 0
            subgraph = graph.subgraph(vs_subgraph)
            v_central_local = graph_central_vertex(subgraph)
            basis_id_to_v_center_id[i] = subgraph.vs['global_id'][v_central_local]
        return basis_id_to_v_center_id

    @staticmethod
    def create_inter_processor_edgelist(graph, my_subgraph_ids, mpicomm):
        """
        Parameters
        ----------
        graph: igraph
            graph representing the pore network
        my_subgraph_ids: list
            indices indicating the indices of the subgraphs belonging to this processor
        mpicomm: mpi4py communicator

        Returns
        -------
        inter_processor_edges: dict
            data structure with properties of tubes as well as their incident vertices
        """

        my_id = mpicomm.rank
        proc_id = np.asarray(graph.vs["proc_id"], dtype=np.int32)
        subgraph_id = graph.vs["subgraph_id"]
        global_id = np.asarray(graph.vs["global_id"], dtype=np.int32)

        inter_processor_edgelist = [(u, v) if proc_id[u] == my_id else (v, u) for (u, v) in graph.get_edgelist()
                                    if (
                                        ((proc_id[u] == my_id and proc_id[v] != my_id) or
                                         (proc_id[u] != my_id and proc_id[v] == my_id)) and
                                        ((subgraph_id[u] in my_subgraph_ids) or (subgraph_id[v] in my_subgraph_ids))
                                    )
                                    ]

        eids = graph.get_eids(pairs=inter_processor_edgelist)

        inter_processor_edgelist = np.asarray(inter_processor_edgelist)

        inter_processor_edges = dict()

        for attr in ["G", "global_id", "l", "A_tot", "r"]:
            inter_processor_edges[attr] = np.asarray(graph.es[eids][attr])

        inter_processor_edges["invaded"] = np.zeros(len(inter_processor_edges["r"]), dtype=np.int32)

        if len(inter_processor_edgelist) == 0:
            inter_processor_edges["edgelist"] = [[], []]
        else:

            vertices_1 = global_id[inter_processor_edgelist.T[0]]
            vertices_2 = global_id[inter_processor_edgelist.T[1]]

            proc_1 = proc_id[inter_processor_edgelist.T[0]]
            proc_2 = proc_id[inter_processor_edgelist.T[1]]

            assert np.all(proc_1 != proc_2)

            inter_processor_edges["edgelist"] = [vertices_1, vertices_2]

        return inter_processor_edges

    @staticmethod
    def distribute_graph(graph, coarse_graph, mpicomm):
        """
        Distributes a fine-scale graph (graph) among the processors  using information stored in a coarse_graph.

        Parameters
        ----------
        graph: igraph
               graph with a "subgraph_id" vertex attribute which assigns each vertex to a subgraph
        coarse_graph: igraph
               coarse graph with the vertices representing the subgraph. Each vertex must have a proc_id vertex attribute
               which assigns each subgraph to the processor
        mpicomm: mpi4py communicator

        Returns
        -------
        graph: igraph
            graph consisting of the vertices of the subgraph which belong to the processor, the and vertices of the
             neighboring subgraphs
        """
        my_id = mpicomm.rank
        num_proc = mpicomm.size

        if my_id == 0:
            req = dict()
            for dest_id in xrange(1, num_proc):
                coarse_ids_for_proc = coarse_graph.vs.select(proc_id_eq=dest_id)
                coarse_ids_for_proc_with_ghosts = list(set().union(*coarse_graph.neighborhood(coarse_ids_for_proc)))
                graph_for_proc = graph.vs.select(subgraph_id_in=coarse_ids_for_proc_with_ghosts).subgraph()
                req[dest_id] = mpicomm.isend(graph_for_proc, dest=dest_id)

            for dest_id in req:
                req[dest_id].wait()

        if my_id != 0:
            graph = mpicomm.recv(source=0)

        return graph

    def bc_const_source_xmin(self, wett_source=0.0, nwett_source=0.0):
        MultiscaleSim.bc_const_source_xmin(self, wett_source, nwett_source)
        self.global_source_nonwett_with_ghost.Import(self.global_source_nonwett, self.epetra_importer, Epetra.Insert)

    def bc_const_source_xmax(self, wett_source=0.0, nwett_source=0.0):
        MultiscaleSim.bc_const_source_xmax(self, wett_source, nwett_source)
        self.global_source_nonwett_with_ghost.Import(self.global_source_nonwett, self.epetra_importer, Epetra.Insert)
        self.comm.Barrier()
        mpicomm = self.mpicomm
        assert np.isclose(mpicomm.allreduce(np.sum(self.global_source_wett[:] + self.global_source_nonwett[:])), 1e-15)

    def initialize(self):
        self.p_c = self._update_pc_from_subnetworks(self.p_c, self.my_subnetworks)
        ierr = self.p_c_with_ghost.Import(self.p_c, self.epetra_importer, Epetra.Insert)
        assert ierr == 0

        self.sat = self._update_sat_from_subnetworks(self.sat, self.my_subnetworks)
        ierr = self.sat_with_ghost.Import(self.sat, self.epetra_importer, Epetra.Insert)
        assert ierr == 0

        self.inter_subgraph_edges = update_inter_invasion_status_snap_off(self.inter_subgraph_edges,
                                                                          self.p_c_with_ghost)

        self.inter_processor_edges = update_inter_invasion_status_snap_off(self.inter_processor_edges,
                                                                           self.p_c_with_ghost)

        self.inter_subgraph_edges = self._update_inter_conductances(self.inter_subgraph_edges, self.p_c_with_ghost)
        self.inter_processor_edges = self._update_inter_conductances(self.inter_processor_edges, self.p_c_with_ghost)
        self.comm.Barrier()
        A = create_matrix(self.unique_map, ["k_n", "k_w"], self.my_subnetworks, self.inter_processor_edges,
                          self.inter_subgraph_edges)

        self.ms = MSRSB(A, self.my_subgraph_support, self.my_basis_support)
        self.ms.smooth_prolongation_operator(A, tol=1.e-3)

    def __solve_pressure(self, smooth_prolongator=True):
        A = create_matrix(self.unique_map, ["k_n", "k_w"], self.my_subnetworks, self.inter_processor_edges,
                          self.inter_subgraph_edges)
        rhs = create_rhs(self.unique_map, self.my_subnetworks, self.inter_processor_edges, self.inter_subgraph_edges,
                         self.p_c,
                         self.global_source_wett, self.global_source_nonwett)

        if self.num_subnetworks == 1:
            self.p_w = solve_aztec(A, rhs, self.p_w, tol=1e-8)
            self.p_n = self.p_w + self.p_c

        else:
            self.p_n, self.p_w = solve_multiscale(self.ms, A, rhs, self.p_c, p_w=self.p_w,
                                                  smooth_prolongator=smooth_prolongator, tol=self.p_tol)

        return self.p_n, self.p_w

    def advance_in_time(self, delta_t):
        comm = self.comm
        epetra_importer = self.epetra_importer
        simulations = self.simulations

        self.stop_time = self.time + delta_t
        while True:
            sat_current = _network_saturation(self.my_subnetworks, self.mpicomm)
            logger.info("Current Saturation of Complete Network: %g", sat_current)
            logger.info("Current Time: %g", self.time)

            # Update global capillary pressure and saturation vectors from vectors contained in subnetworks
            self.p_c, self.sat = self._sync_pc_and_sat()

            # Workaround for Python interface of Epetra -- An extra vector with overlapping mapping
            ierr = self.p_c_with_ghost.Import(self.p_c, self.epetra_importer, Epetra.Insert)
            assert ierr == 0
            ierr = self.sat_with_ghost.Import(self.sat, self.epetra_importer, Epetra.Insert)
            assert ierr == 0

            # Snap-off boundary throats
            self.inter_subgraph_edges = update_inter_invasion_status_snap_off(self.inter_subgraph_edges,
                                                                              self.p_c_with_ghost)

            self.inter_processor_edges = update_inter_invasion_status_snap_off(self.inter_processor_edges,
                                                                               self.p_c_with_ghost)

            self.inter_subgraph_edges = self._update_inter_conductances(self.inter_subgraph_edges, self.p_c_with_ghost)
            self.inter_processor_edges = self._update_inter_conductances(self.inter_processor_edges, self.p_c_with_ghost)
            self.comm.Barrier()

            # Invade boundary throats with nonwetting phase
            self.p_n, self.p_w = self.__solve_pressure(smooth_prolongator=True)

            self.p_w_with_ghost.Import(self.p_w, epetra_importer, Epetra.Insert)

            AminusD_n = create_matrix(self.unique_map, ["k_n"], None, self.inter_processor_edges,
                                      self.inter_subgraph_edges)
            ierr = AminusD_n.Multiply(False, self.p_n, self.out_flux_n)
            assert ierr == 0
            self.out_flux_n_with_ghost.Import(self.out_flux_n, epetra_importer, Epetra.Insert)

            comm.Barrier()

            self.inter_subgraph_edges = update_inter_invasion_status_piston(self.inter_subgraph_edges,
                                                                            self.p_w_with_ghost,
                                                                            self.p_c_with_ghost,
                                                                            self.sat_with_ghost,
                                                                            self.out_flux_n_with_ghost,
                                                                            self.global_source_nonwett_with_ghost)

            self.inter_processor_edges = update_inter_invasion_status_piston(self.inter_processor_edges,
                                                                             self.p_w_with_ghost,
                                                                             self.p_c_with_ghost,
                                                                             self.sat_with_ghost,
                                                                             self.out_flux_n_with_ghost,
                                                                             self.global_source_nonwett_with_ghost)

            self.inter_subgraph_edges = self._update_inter_conductances(self.inter_subgraph_edges,
                                                                        self.p_c_with_ghost)
            self.inter_processor_edges = self._update_inter_conductances(self.inter_processor_edges,
                                                                         self.p_c_with_ghost)
            comm.Barrier()

            # Invade tubes with wetting phase
            self.p_n, self.p_w = self.__solve_pressure(smooth_prolongator=True)

            self.p_w_with_ghost.Import(self.p_w, epetra_importer, Epetra.Insert)

            self.inter_subgraph_edges = update_inter_invasion_status_snap_off(self.inter_subgraph_edges,
                                                                              self.p_c_with_ghost)
            self.inter_processor_edges = update_inter_invasion_status_snap_off(self.inter_processor_edges,
                                                                               self.p_c_with_ghost)

            self.inter_subgraph_edges = self._update_inter_conductances(self.inter_subgraph_edges, self.p_c_with_ghost)
            self.inter_processor_edges = self._update_inter_conductances(self.inter_processor_edges, self.p_c_with_ghost)
            comm.Barrier()

            for _ in xrange(2):
                self.p_n, self.p_w = self.__solve_pressure(smooth_prolongator=True)

                self.p_w_with_ghost.Import(self.p_w, epetra_importer, Epetra.Insert)

                self.inter_subgraph_edges = update_inter_invasion_status_piston_wetting(self.inter_subgraph_edges,
                                                                                        self.p_w_with_ghost,
                                                                                        self.p_c_with_ghost,
                                                                                        self.sat_with_ghost)

                self.inter_processor_edges = update_inter_invasion_status_piston_wetting(self.inter_processor_edges,
                                                                                         self.p_w_with_ghost,
                                                                                         self.p_c_with_ghost,
                                                                                         self.sat_with_ghost)

                self.inter_subgraph_edges = self._update_inter_conductances(self.inter_subgraph_edges, self.p_c_with_ghost)
                self.inter_processor_edges = self._update_inter_conductances(self.inter_processor_edges, self.p_c_with_ghost)

                comm.Barrier()

            self.p_n, self.p_w = self.__solve_pressure(smooth_prolongator=True)

            self.p_w_with_ghost.Import(self.p_w, epetra_importer, Epetra.Insert)

            AminusD_w = create_matrix(self.unique_map, ["k_w"], None, self.inter_processor_edges,
                                      self.inter_subgraph_edges)
            AminusD_n = create_matrix(self.unique_map, ["k_n"], None, self.inter_processor_edges,
                                      self.inter_subgraph_edges)

            ierr = AminusD_w.Multiply(False, self.p_w, self.out_flux_w)
            assert ierr == 0
            ierr = AminusD_n.Multiply(False, self.p_n, self.out_flux_n)
            assert ierr == 0
            comm.Barrier()

            press_bc_inlet = {"pi_list": self.pi_list_press_inlet, "wett": self.press_inlet_w, "nwett": self.press_inlet_nw}
            press_bc_outlet = {"pi_list": self.pi_list_press_outlet, "wett": self.press_outlet_w,
                               "nwett": self.press_outlet_nw}

            bc = create_subnetwork_boundary_conditions(self.global_source_nonwett - self.out_flux_n,
                                                       self.global_source_wett - self.out_flux_w,
                                                       self.my_subnetworks, self.subgraph_ids_vec,
                                                       press_bc_inlet, press_bc_outlet)

            for i in self.my_subgraph_ids:
                self.simulations[i].set_boundary_conditions(bc[i])
                mass_balance = bc[i].mass_balance()
                total_sources = self.simulations[i].total_source_nonwett + self.simulations[i].total_source_wett
                assert mass_balance < total_sources * 1.e-8, "Mass balance: %e, Total sources: %g" % (mass_balance, total_sources)

            subgraph_id_to_nw_influx = self.compute_nonwetting_influx(simulations)
            subgraph_id_to_volume = {i: self.my_subnetworks[i].total_vol for i in self.my_subgraph_ids}
            subgraph_id_to_saturation = {i: np.sum(self.my_subnetworks[i].pores.sat * self.my_subnetworks[i].pores.vol)
                                            / self.my_subnetworks[i].total_vol for i in self.my_subgraph_ids}

            for i in self.my_subnetworks:
                assert self.my_subnetworks[i].total_throat_vol == 0
                assert self.my_subnetworks[i].total_vol == np.sum(self.my_subnetworks[i].pores.vol)

            dt = self._compute_timestep(subgraph_id_to_nw_influx, subgraph_id_to_saturation, subgraph_id_to_volume,
                                        self.delta_s_max)

            dt = min(dt, self.stop_time - self.time)

            dt = self.mpicomm.allreduce(dt, op=MPI.MIN)

            dt_sim_min = 1.0e20

            logger.info("advancing simulations with timestep %g", dt)

            for i in self.simulations:
                dt_sim = self.simulations[i].advance_in_time(delta_t=dt)
                dt_sim_min = min(dt_sim, dt_sim_min)
            comm.Barrier()

            dt_sim_min = self.mpicomm.allreduce(dt_sim_min, op=MPI.MIN)
            self.comm.Barrier()

            backtracking = not np.isclose(dt_sim_min, dt)
            if backtracking:
                logger.warn("Back Tracking because simulation was stuck with time-step %g", dt_sim_min)

                dt_sim_min *= 0.95
                logger.warn("advancing backtracked simulations with time-step %g", dt_sim_min)
                for i in simulations:
                    simulations[i].reset_status()

                if dt_sim_min == 0.0:
                    continue

                for i in simulations:
                    logger.debug("Simulation %d", i)
                    dt_sim = simulations[i].advance_in_time(delta_t=dt_sim_min)
                    logger.debug("Time step: %d, target time step: %d", dt_sim, dt_sim_min)
                    assert np.isclose(dt_sim, dt_sim_min), str(dt_sim) + "  " + str(dt_sim_min)

            self.comm.Barrier()
            self.time += dt_sim_min

            logger.info("time is %g", self.time)
            logger.info("stop time is %g", self.stop_time)

            if np.isclose(self.time, self.stop_time):
                break
