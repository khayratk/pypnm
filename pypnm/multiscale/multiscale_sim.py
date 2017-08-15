import errno
import os
from collections import defaultdict

import numpy as np
import pymetis
from PyTrilinos import Epetra
from mpi4py import MPI

from pypnm.ams.msfv import MSFV
from pypnm.ams.msrsb import MSRSB
from pypnm.ams.wire_basket import create_wire_basket
from pypnm.attribute_calculators.conductance_calc import ConductanceCalc
from pypnm.attribute_calculators.pc_computer import DynamicCapillaryPressureComputer
from pypnm.linalg.trilinos_interface import solve_aztec
from pypnm.multiscale.utils import create_matrix, create_rhs
from pypnm.multiscale.utils import solve_multiscale, create_subnetwork_boundary_conditions
from pypnm.multiscale.utils import update_inter_invasion_status_snap_off, update_inter_invasion_status_piston, \
    update_inter_invasion_status_piston_wetting
from pypnm.porenetwork import gridder
from pypnm.porenetwork.constants import EAST, WEST
from pypnm.porenetwork.subnetwork import SubNetwork
from pypnm.flow_simulation.dynamic_simulation import DynamicSimulation
from pypnm.util.hd5_output import add_field_to_hdf_file, add_attribute_to_hdf_file
from pypnm.util.igraph_utils import coarse_graph_from_partition, graph_central_vertex, support_of_basis_function, \
    network_to_igraph
from pypnm.util.indexing import GridIndexer3D
from sim_settings import sim_settings


def require_path(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def reduce_dictionary_values(key_to_scalar_distributed, mpicomm):
    """
    Parameters
    ----------
    key_to_scalar_distributed: dict
        Python dictionary which is divided among processors
    mpicomm: mpi4py communicator

    Returns
    -------
    global_sum: number
        The sum of all values of all dictionaries.

    """
    local_sum = np.sum(key_to_scalar_distributed.values())
    global_sum = mpicomm.allreduce(local_sum, op=MPI.SUM)
    return global_sum


def _create_subnetworks(network, subgraph_ids, coarse_graph, mpicomm):
    """
    Create subnetworks for each processor given a network and partition information.

    Parameters
    ----------
    network: Porenetwork
        Porenetwork from which subnetworks will be created from
    subgraph_ids: integer ndarray
        Vertex attribute assigning each vertex in Porenetwork to a subgraph index
    coarse_graph: igraph
        Graph consisting of subnetworks as vertices, with edges present between two subnetworks if they are adjacent.
        Course_graph must have a [proc_id] property
    mpicomm: mpi4py communicator

    Returns
    -------
    subnetworks: dict
        Dictionary with subnetwork ids as keys and references to subnetworks as values.
    """
    my_id = mpicomm.rank
    num_proc = mpicomm.size

    if my_id == 0:
        num_subnetworks = np.max(subgraph_ids) + 1
        proc_ids = coarse_graph.vs['proc_id']
        req = dict()

        indices_of_subgraph = defaultdict(list)
        for i, subgraph_id in enumerate(subgraph_ids):
            indices_of_subgraph[subgraph_id].append(i)
        indices_of_subgraph = dict(indices_of_subgraph)

        for dest_id in xrange(num_proc):
            send_dict = dict()
            for i in xrange(num_subnetworks):
                if proc_ids[i] == dest_id:
                    pi_list = indices_of_subgraph[i]
                    send_dict[i] = SubNetwork(network, pi_list)
                    assert len(network_to_igraph(send_dict[i]).components()) == 1

            if dest_id == 0:
                my_subnetworks = send_dict
            else:
                req[dest_id] = mpicomm.isend(send_dict, dest=dest_id)

        for dest_id in req:
            req[dest_id].wait()

    if my_id != 0:
        my_subnetworks = mpicomm.recv(source=0)

    return my_subnetworks


def _create_inter_subgraph_edgelist(graph, my_subgraph_ids, mpicomm):
    """
    Parameters
    ----------
    graph: igraph
        Distributed graph with "global_id" attribute to uniquely identify vertices
    my_subgraph_ids: numpy.ndarray
        List of subgraph ids  belonging to this processor.
    my_id:
        The id of this processor.
    Returns
    -------
    inter_subgraph_edges: dict
        Dictionary with several keys which completely specify the pore throats as well as adjacent pore bodies.

    """
    my_id = mpicomm.rank
    subgraph_id = graph.vs["subgraph_id"]
    global_id = np.asarray(graph.vs["global_id"], dtype=np.int32)
    proc_id = np.asarray(graph.vs["proc_id"], dtype=np.int32)

    inter_subgraph_edgelist = [(u, v) for (u, v) in graph.get_edgelist()
                               if (subgraph_id[u] in my_subgraph_ids and subgraph_id[v] in my_subgraph_ids) and
                               (subgraph_id[u] != subgraph_id[v])]

    eids = graph.get_eids(pairs=inter_subgraph_edgelist)

    inter_subgraph_edgelist = np.asarray(inter_subgraph_edgelist)

    inter_subgraph_edges = dict()

    for attr in ["G", "global_id", "l", "A_tot", "r"]:
        inter_subgraph_edges[attr] = np.asarray(graph.es[eids][attr])

    inter_subgraph_edges["invaded"] = np.zeros_like(inter_subgraph_edges["r"], dtype=np.int)

    if len(inter_subgraph_edgelist) == 0:
        inter_subgraph_edges["edgelist"] = [[], []]
    else:
        vertices_1 = global_id[inter_subgraph_edgelist.T[0]]
        vertices_2 = global_id[inter_subgraph_edgelist.T[1]]

        proc_1 = proc_id[inter_subgraph_edgelist.T[0]]
        proc_2 = proc_id[inter_subgraph_edgelist.T[1]]

        assert np.all(proc_1 == my_id)
        assert np.all(proc_2 == my_id)

        inter_subgraph_edges["edgelist"] = [vertices_1, vertices_2]

    return inter_subgraph_edges


def _network_saturation(subnetworks, mpicomm):
    """
    Computes the saturation of the complete network given a a dictionary of subnetworks distributed among the processors
    in a communicator.
    Parameters
    ----------
    subnetworks: dict
        Dictionary of subnetworks
    mpicomm: mpi4py communicator

    Returns
    -------
    saturation: float

    """
    subgraph_id_to_volume = {i: subnetworks[i].total_vol for i in subnetworks}
    subgraph_id_to_nw_volume = {i: np.sum(subnetworks[i].pores.sat * subnetworks[i].pores.vol) for i in subnetworks}
    total_network_volume = reduce_dictionary_values(subgraph_id_to_volume, mpicomm)
    total_nw_volume = reduce_dictionary_values(subgraph_id_to_nw_volume, mpicomm)
    saturation = total_nw_volume / total_network_volume
    return saturation


def _output_subnetworks_to_hdf(subnetworks, label, time, folder_name, mpicomm):
    require_path(folder_name)
    network_saturation = _network_saturation(subnetworks, mpicomm)
    for i in subnetworks:
        filename = folder_name + "/hdf_subnet" + str(i).zfill(4) + ".h5"
        add_attribute_to_hdf_file(filename, label, "network_saturation", network_saturation)
        add_attribute_to_hdf_file(filename, label, "time", time)
        add_field_to_hdf_file(filename, label, "saturation", subnetworks[i].pores.sat)
        add_field_to_hdf_file(filename, label, "p_c", subnetworks[i].pores.p_c)
        add_field_to_hdf_file(filename, label, "p_n", subnetworks[i].pores.p_n)
        add_field_to_hdf_file(filename, label, "p_w", subnetworks[i].pores.p_w)
        add_field_to_hdf_file(filename, label, "volume", subnetworks[i].pores.vol)
        add_field_to_hdf_file(filename, 0, "x", subnetworks[i].pores.x)
        add_field_to_hdf_file(filename, 0, "y", subnetworks[i].pores.y)
        add_field_to_hdf_file(filename, 0, "z", subnetworks[i].pores.z)


def _output_subnetworks_to_vtk(subnetworks, label, folder_name):
    require_path(folder_name)
    for i in subnetworks:
        subnetworks[i].export_to_vtk(filename="subnetwork_" + str(i) + "_" + str(label).zfill(4), folder_name=folder_name)


class MultiscaleSim(object):
    def _bc_const_source_face(self, wett_source, nwett_source, FACE):
        my_id = self.comm.MyPID()
        mpicomm = self.mpicomm
        if my_id == 0:
            pi_list_wetting_source_global = self.network.pi_list_face[FACE].astype(np.int32) # TODO: Separate
            pi_list_nonwetting_source_global = self.network.pi_list_face[FACE].astype(np.int32)

        if my_id != 0:
            pi_list_wetting_source_global = None
            pi_list_nonwetting_source_global = None

        pi_list_wetting_source_global = mpicomm.bcast(pi_list_wetting_source_global, root=0)
        pi_list_nonwetting_source_global = mpicomm.bcast(pi_list_nonwetting_source_global, root=0)

        q_wetting_source_global = np.ones_like(pi_list_wetting_source_global, dtype=np.float) * wett_source / len(
            pi_list_wetting_source_global)
        q_nonwetting_source_global = np.ones_like(pi_list_nonwetting_source_global,
                                                  dtype=np.float) * nwett_source / len(
            pi_list_nonwetting_source_global)

        self.global_source_wett.ReplaceGlobalValues(q_wetting_source_global, pi_list_wetting_source_global)
        self.global_source_nonwett.ReplaceGlobalValues(q_nonwetting_source_global, pi_list_nonwetting_source_global)

    def network_saturation(self, subnetworks, mpicomm):
        return _network_saturation(subnetworks, mpicomm)

    def bc_const_source_xmin(self, wett_source=0.0, nwett_source=0.0):
        """
        Sets total volumetric source term at the left face of the network
        This value is distributed among all pores at the left face

        Parameters
        ----------
        wett_source: float
                    Volumetric source term for the wetting phase.
        nwett_source: float
                    Volumetric source term for the nonwetting phase.
        """
        self._bc_const_source_face(wett_source, nwett_source, WEST)

    def bc_const_source_xmax(self, wett_source=0.0, nwett_source=0.0):
        """
        Sets total volumetric source term at the right face of the network
        This value is distributed among all pores at the right face

        Parameters
        ----------
        wett_source: float
                    Volumetric source term for the wetting phase.
        nwett_source: float
                    Volumetric source term for the nonwetting phase.
        """
        self._bc_const_source_face(wett_source, nwett_source, EAST)
        self.comm.Barrier()
        mpicomm = self.mpicomm
        assert np.isclose(mpicomm.allreduce(np.sum(self.global_source_wett[:] + self.global_source_nonwett[:])), 1e-15)

    def bc_const_press_xmin(self, wett_press, nwett_press):
        """
        Sets total pressures of the wetting and nonwetting phase at the left face.

        Parameters
        ----------
        wett_press: float
                    pressure of the wetting phase
        wett_press: float
                    pressure of the nonwetting phase
        """
        my_id = self.comm.MyPID()
        mpicomm = self.mpicomm
        if my_id == 0:
            pi_list_inlet = self.network.pi_list_face[WEST].astype(np.int32)

        if my_id != 0:
            pi_list_inlet = None

        self.pi_list_press_inlet = mpicomm.bcast(pi_list_inlet, root=0)

        self.press_inlet_w = wett_press
        self.press_inlet_nw = nwett_press

    def bc_const_press_xmax(self, wett_press, nwett_press):
        """
        Sets total pressures of the wetting and nonwetting phase at the right face.

        Parameters
        ----------
        wett_press: float
                    pressure of the wetting phase
        wett_press: float
                    pressure of the nonwetting phase
        """

        my_id = self.comm.MyPID()
        mpicomm = self.mpicomm
        if my_id == 0:
            pi_list_outlet = self.network.pi_list_face[EAST].astype(np.int32)

        if my_id != 0:
            pi_list_outlet = None

        self.pi_list_press_outlet = mpicomm.bcast(pi_list_outlet, root=0)

        self.press_outlet_w = wett_press
        self.press_outlet_nw = nwett_press

    def output_vtk(self, label, folder_name="vtk_multiscale_msfv"):
        _output_subnetworks_to_vtk(self.my_subnetworks, label, folder_name)

    def output_hd5(self, label, foldername="hf5_multiscale_msfv"):
        _output_subnetworks_to_hdf(self.my_subnetworks, label, self.time, foldername, self.mpicomm)

    @staticmethod
    def _compute_timestep(subgraph_id_to_nw_influx, subgraph_id_to_saturation, subgraph_id_to_volume, delta_s_max=0.02):
        dt_imbibition = 1.0
        dt_drainage = 1.0

        dt_drainage_fill = min(
            [subgraph_id_to_volume[i] * (1. - subgraph_id_to_saturation[i]) / subgraph_id_to_nw_influx[i]
             for i in subgraph_id_to_nw_influx if subgraph_id_to_nw_influx[i] > 0.0]
            or [dt_drainage])

        dt_drainage_delta_s = min([subgraph_id_to_volume[i] * delta_s_max / subgraph_id_to_nw_influx[i]
                                   for i in subgraph_id_to_nw_influx if subgraph_id_to_nw_influx[i] > 0.0]
                                  or [dt_drainage])

        dt_drainage = min(dt_drainage_fill, dt_drainage_delta_s)

        dt_imbibition_empty = min(
            [-subgraph_id_to_volume[i] * (subgraph_id_to_saturation[i]) / subgraph_id_to_nw_influx[i]
             for i in subgraph_id_to_nw_influx if subgraph_id_to_nw_influx[i] < 0.0]
            or [dt_imbibition])

        dt_imbibition_delta_s = min([-subgraph_id_to_volume[i] * delta_s_max / subgraph_id_to_nw_influx[i]
                                     for i in subgraph_id_to_nw_influx if subgraph_id_to_nw_influx[i] < 0.0]
                                    or [dt_imbibition])

        dt_imbibition = min(dt_imbibition_empty, dt_imbibition_delta_s)

        assert np.all(subgraph_id_to_saturation.values() >= 0.0)

        assert dt_drainage >= 0.0
        assert dt_imbibition >= 0.0
        return min(dt_drainage, dt_imbibition)

    def set_delta_s_max(self, delta_s):
        self.delta_s_max = delta_s

    def set_p_tol(self, p_tol):
        self.p_tol = p_tol

    @staticmethod
    def _update_inter_conductances(inter_edges, p_c):
        inter_edgelist_local_1 = np.asarray([p_c.Map().LID(i) for i in inter_edges['edgelist'][0]], dtype=np.int32)
        inter_edgelist_local_2 = np.asarray([p_c.Map().LID(i) for i in inter_edges['edgelist'][1]], dtype=np.int32)

        assert np.all(inter_edgelist_local_1 >= 0)
        assert np.all(inter_edgelist_local_2 >= 0)

        inter_pc_edge = np.maximum(p_c[inter_edgelist_local_1], p_c[inter_edgelist_local_2])

        el_rad = inter_edges["r"]
        el_len = inter_edges["l"]
        el_G = inter_edges["G"]
        el_A_tot = inter_edges["A_tot"]
        invasion_status = inter_edges["invaded"]
        k_n, k_w = ConductanceCalc.compute_conductances(sim_settings['fluid_properties'], el_rad, el_len, el_G,
                                                        el_A_tot, inter_pc_edge, invasion_status)

        inter_edges['k_n'] = k_n
        inter_edges['k_w'] = k_w

        return inter_edges

    @staticmethod
    def _update_pc_from_subnetworks(p_c, my_subnetworks):
        """
        Parameters
        ----------
        p_c: Epetra Vector
            Capillary pressure in pore bodies
        my_subnetworks: dictionary
            Dictionary of subnetworks belonging to current processor

        Returns
        -------
        p_c: Epetra Vector
            updated capillary pressure

        """
        for i in my_subnetworks:
            ierr = p_c.ReplaceGlobalValues(my_subnetworks[i].pores.p_c, my_subnetworks[i].pi_local_to_global)
            assert ierr == 0
        return p_c

    @staticmethod
    def _update_sat_from_subnetworks(sat, my_subnetworks):
        """
        Parameters
        ----------
        sat: Epetra Vector
            Saturation in pore bodies
        my_subnetworks: dictionary
            dictionary of subnetworks belonging to current processor

        Returns
        -------
        sat: Epetra Vector
             Saturation in pore bodies

        """
        for i in my_subnetworks:
            ierr = sat.ReplaceGlobalValues(my_subnetworks[i].pores.sat, my_subnetworks[i].pi_local_to_global)
            assert ierr == 0
        return sat

    def set_subnetwork_press_solver(self, solver_type):
        """
        Parameters
        ----------
        solver_type: {"lu", "petsc", "AMG", "trilinos", "mltrilinos"}
            Sets solver type for subnetwork.

        """
        assert solver_type in ["lu", "petsc", "AMG", "trilinos", "mltrilinos"]
        for i in self.my_subgraph_ids:
            self.simulations[i].press_solver_type = solver_type

    def _sync_pc_and_sat(self):
        self.p_c = self._update_pc_from_subnetworks(self.p_c, self.my_subnetworks)

        if len(self.pi_list_press_inlet) > 0:
            self.p_c[self.pi_list_press_inlet] = self.press_inlet_nw - self.press_inlet_w

        self.sat = self._update_sat_from_subnetworks(self.sat, self.my_subnetworks)
        return self.p_c, self.sat


    @staticmethod
    def compute_nonwetting_influx(simulations):
        return {i: simulations[i].total_source_nonwett + simulations[i].total_sink_nonwett
                if len(simulations[i].bc.pi_list_inlet) == 0
                else -simulations[i].total_source_wett - simulations[i].total_sink_wett
                for i in simulations}


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
    def __init__(self, network, num_subnetworks, comm=None, mpicomm=None, subgraph_ids=None):
        self.network = network

        if comm is None:
            comm = Epetra.PyComm()

        if mpicomm is None:
            mpicomm = MPI.COMM_WORLD

        self.comm, self.mpicomm = comm, mpicomm

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
            self.simulations[i] = DynamicSimulation(self.my_subnetworks[i])
            self.simulations[i].solver_type = "lu"

            k_comp = ConductanceCalc(self.my_subnetworks[i])
            k_comp.compute()
            pc_comp = DynamicCapillaryPressureComputer(self.my_subnetworks[i])
            pc_comp.compute()

        self.delta_s_max = 0.01
        self.p_tol = 1.e-5
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
        self.p_c = MultiScaleSimUnstructured._update_pc_from_subnetworks(self.p_c, self.my_subnetworks)
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
        self.ms.smooth_prolongation_operator(10)

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
            print "Current Saturation of Complete Network", sat_current
            print "Current Time", self.time

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
            self.p_n, self.p_w = self.__solve_pressure(smooth_prolongator=False)

            self.p_w_with_ghost.Import(self.p_w, epetra_importer, Epetra.Insert)

            self.inter_subgraph_edges = update_inter_invasion_status_snap_off(self.inter_subgraph_edges,
                                                                              self.p_c_with_ghost)
            self.inter_processor_edges = update_inter_invasion_status_snap_off(self.inter_processor_edges,
                                                                               self.p_c_with_ghost)

            self.inter_subgraph_edges = self._update_inter_conductances(self.inter_subgraph_edges, self.p_c_with_ghost)
            self.inter_processor_edges = self._update_inter_conductances(self.inter_processor_edges, self.p_c_with_ghost)
            comm.Barrier()

            for _ in xrange(2):
                self.p_n, self.p_w = self.__solve_pressure(smooth_prolongator=False)

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

            self.p_n, self.p_w = self.__solve_pressure(smooth_prolongator=False)

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

            print "advancing simulations with timestep", dt
            for i in self.simulations:
                dt_sim = self.simulations[i].advance_in_time(delta_t=dt)
                dt_sim_min = min(dt_sim, dt_sim_min)
            comm.Barrier()

            dt_sim_min = self.mpicomm.allreduce(dt_sim_min, op=MPI.MIN)
            self.comm.Barrier()

            backtracking = not np.isclose(dt_sim_min, dt)
            if backtracking:
                print "Back Tracking because simulation was stuck at time-step ", dt_sim_min

                dt_sim_min *= 0.95
                print "advancing backtracked simulations with time-step", dt_sim_min
                for i in simulations:
                    simulations[i].reset_status()

                if dt_sim_min == 0.0:
                    continue

                for i in simulations:
                    # print "SIMULATION", i
                    dt_sim = simulations[i].advance_in_time(delta_t=dt_sim_min)
                    # print dt_sim, dt_sim_min
                    assert np.isclose(dt_sim, dt_sim_min), str(dt_sim) + "  " + str(dt_sim_min)

            self.comm.Barrier()
            self.time += dt_sim_min

            print "time is ", self.time
            print "stop time is", self.stop_time

            if np.isclose(self.time, self.stop_time):
                break


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

        print "Solving using MSFV method"

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
