import cPickle as pickle
import errno
import logging
import os
from collections import defaultdict

import numpy as np
from PyTrilinos import Epetra
from mpi4py import MPI

from pypnm.ams.msrsb import MSRSB
from pypnm.attribute_calculators.conductance_calc import ConductanceCalc
from pypnm.multiscale.utils import create_matrix
from pypnm.porenetwork.constants import EAST, WEST
from pypnm.porenetwork.subnetwork import SubNetwork
from pypnm.util.hd5_output import add_field_to_hdf_file, add_attribute_to_hdf_file
from pypnm.util.igraph_utils import network_to_igraph
try:
    from sim_settings import sim_settings
except ImportError:
    sim_settings = dict()
    sim_settings["fluid_properties"] = dict()
    sim_settings["fluid_properties"]['mu_n'] = 1.0
    sim_settings["fluid_properties"]['mu_w'] = 1.0
    sim_settings["fluid_properties"]['gamma'] = 1.0


def require_path(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

logger = logging.getLogger('pypnm')


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
    Create subnetworks for each processor given a pore network and partition information.

    Parameters
    ----------
    network: Porenetwork
    subgraph_ids: ndarray
        mapping from vertex id to a subgraph id
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
                    n_components = len(network_to_igraph(send_dict[i]).components())
                    assert n_components == 1, n_components

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
    my_subgraph_ids: array_like
        List of subgraph ids  belonging to this processor.
    mpicomm: mpi4py communicator

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
        add_field_to_hdf_file(filename, label, "pore_sat", subnetworks[i].pores.sat)
        add_field_to_hdf_file(filename, label, "p_c", subnetworks[i].pores.p_c)
        add_field_to_hdf_file(filename, label, "p_n", subnetworks[i].pores.p_n)
        add_field_to_hdf_file(filename, label, "p_w", subnetworks[i].pores.p_w)
        add_field_to_hdf_file(filename, label, "tube_invaded", subnetworks[i].tubes.invaded)

        add_field_to_hdf_file(filename, 0, "tube_r", subnetworks[i].tubes.r)
        add_field_to_hdf_file(filename, 0, "tube_l", subnetworks[i].tubes.l)
        add_field_to_hdf_file(filename, 0, "tube_A_tot", subnetworks[i].tubes.A_tot)

        add_field_to_hdf_file(filename, 0, "pore_vol", subnetworks[i].pores.vol)
        add_field_to_hdf_file(filename, 0, "G", subnetworks[i].pores.G)
        add_field_to_hdf_file(filename, 0, "r", subnetworks[i].pores.r)
        add_field_to_hdf_file(filename, 0, "x", subnetworks[i].pores.x)
        add_field_to_hdf_file(filename, 0, "y", subnetworks[i].pores.y)
        add_field_to_hdf_file(filename, 0, "z", subnetworks[i].pores.z)


def _export_subnetwork_structure(subnetworks, folder_name):
    require_path(folder_name)
    for i in subnetworks:
        filename = folder_name + "/hdf_subnet_structure" + str(i).zfill(4) + ".h5"
        subnetworks[i].export_to_hdf(filename)


def _output_subnetworks_to_vtk(subnetworks, label, folder_name):
    require_path(folder_name)
    for i in subnetworks:
        subnetworks[i].export_to_vtk(filename="subnetwork_" + str(i) + "_" + str(label).zfill(4), folder_name=folder_name)


class MultiscaleSim(object):
    def save(self, filename="multiscale_sim"):
        del self.unique_map
        del self.nonunique_map
        del self.subgraph_ids_vec
        del self.epetra_importer

        del self.p_c
        del self.sat
        del self.p_w

        try:
            del self.p_n
        except:
            pass

        del self.out_flux_w
        del self.out_flux_n
        del self.p_c_with_ghost
        del self.sat_with_ghost
        del self.p_w_with_ghost
        del self.out_flux_n_with_ghost

        del self.ms

        del self.global_source_wett
        del self.global_source_nonwett
        del self.global_source_nonwett_with_ghost

        del self.comm

        comm = Epetra.PyComm()
        output_file = open(filename+"_proc"+str(comm.MyPID()), 'wb')
        pickle.dump(self, output_file, protocol=pickle.HIGHEST_PROTOCOL)
        output_file.close()
        self.re_init()

    def re_init(self):
        self.comm = Epetra.PyComm()
        self.mpicomm = MPI.COMM_WORLD

        self.unique_map, self.nonunique_map, self.subgraph_ids_vec = self.create_maps(self.graph, self.comm)
        self.epetra_importer = Epetra.Import(self.nonunique_map, self.unique_map)

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
        self.p_c = self._update_pc_from_subnetworks(self.p_c, self.my_subnetworks)
        ierr = self.p_c_with_ghost.Import(self.p_c, self.epetra_importer, Epetra.Insert)
        assert ierr == 0

        self.sat = self._update_sat_from_subnetworks(self.sat, self.my_subnetworks)
        ierr = self.sat_with_ghost.Import(self.sat, self.epetra_importer, Epetra.Insert)

        A = create_matrix(self.unique_map, ["k_n", "k_w"], self.my_subnetworks, self.inter_processor_edges,
                          self.inter_subgraph_edges)

        self.ms = MSRSB(A, self.my_subgraph_support, self.my_basis_support)
        self.ms.smooth_prolongation_operator(A, tol=1.e-3)

        assert ierr == 0

    @classmethod
    def load(cls, filename="multiscale_sim"):
        """
        loads simulation from a pkl file

        Parameters
        ----------
        filename: str

        """
        comm = Epetra.PyComm()
        input_file = open(filename+"_proc"+str(comm.MyPID()), 'rb')
        ms = pickle.load(input_file)
        ms.comm = Epetra.PyComm()
        ms.mpicomm = MPI.COMM_WORLD
        ms.re_init()

        return ms

    def _bc_const_source_face(self, wett_source, nwett_source, FACE):
        my_id = self.comm.MyPID()
        mpicomm = self.mpicomm
        if my_id == 0:
            pi_list_wetting_source_global = self.network.pi_list_face[FACE].astype(np.int32)  # TODO: Separate
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


