import logging

import dill as pickle
import numpy as np

from pypnm.percolation import graph_algs
from pypnm.porenetwork import component
from pypnm.porenetwork.backbone import BackBoneComputer
from pypnm.porenetwork.entry_press_computer import EntryPressureComputer
from pypnm.porenetwork.relperm import SimpleRelPermComputer
from pypnm.porenetwork.subnetwork import SubNetworkTightlyCoupled
from pypnm.postprocessing.vtk_output import VtkWriter
from pypnm.util.bounding_box import BoundingBox

logger = logging.getLogger('pypnm.simulation')

from pypnm.util.hd5_output import add_field_to_hdf_file, add_attribute_to_hdf_file
from pypnm.util.utils import require_path


class Simulation(object):
    def __init__(self, network):
        self.network = network

        self._relperm_comp = None
        self.pe_comp = EntryPressureComputer()

        self.vtk_writer = None
        self.network_pp = None  # pp denotes post processing

        self._relperm_comp_pp = None
        self._backbone_comp_pp = None

        self.sat_comp = None
        self._sat_comp_pp = None
        self.SatComputer = None

    def create_vtk_output_folder(self, folder_name, delete_existing_files=False):
        self.vtk_writer = VtkWriter(self.network, folder_name, delete_existing_files)
        self.vtk_writer.add_pore_field(self.network.pores.r, "PoreRadius")
        self.vtk_writer.add_tube_field(self.network.tubes.r, "TubeRadius")

    def add_vtk_output_pore_field(self, array, array_name):
        assert len(array) == self.network.nr_p
        self.vtk_writer.add_pore_field(array, array_name)

    def add_vtk_output_tube_field(self, array, array_name):
        assert len(array) == self.network.nr_t
        self.vtk_writer.add_tube_field(array, array_name)

    def write_vtk_output(self, label):
        output_filename = "paraview" + str(label).zfill(8)
        self.vtk_writer.write(output_filename)

    def set_post_processing_window_bounds(self, bounding_percent):
        eps = 1e-10
        pores = self.network.pores
        bounding_box = BoundingBox(np.max(pores.x)*bounding_percent[0]-eps,
                                   np.max(pores.x)*bounding_percent[1]+eps,
                                   np.max(pores.y)*bounding_percent[2]-eps,
                                   np.max(pores.y)*bounding_percent[3]+eps,
                                   np.max(pores.z)*bounding_percent[4]-eps,
                                   np.max(pores.z)*bounding_percent[5]+eps)
        self.network_pp = SubNetworkTightlyCoupled.from_bounding_box(self.network, bounding_box)
        self._sat_comp_pp = self.SatComputer(self.network_pp)

    def get_relative_permeability(self):
        self._relperm_comp = SimpleRelPermComputer(self.network)
        self._relperm_comp.compute()
        return self._relperm_comp.kr_n[0], self._relperm_comp.kr_w[0]

    def get_permeability(self):
        self._relperm_comp = SimpleRelPermComputer(self.network)
        assert self._relperm_comp.K > 0.0
        return self._relperm_comp.K

    def get_relative_permeability_pp_window(self):
        assert self.network_pp is not None, "Post processing window bounds have not been defined"

        if self._relperm_comp_pp is None:
            self._relperm_comp_pp = SimpleRelPermComputer(self.network_pp)

        self.__sync_postprocessing_network()

        self._relperm_comp_pp.compute()
        return self._relperm_comp_pp.kr_n[0], self._relperm_comp_pp.kr_w[0]

    @staticmethod
    def __update_network_connectivity(network):
        network.pores.connected[:] = 0.0
        network.tubes.connected[:] = 0.0
        graph_algs.update_pore_and_tube_nw_connectivity_to_inlet(network)

    @staticmethod
    def __update_network_backbone(network):
        network.pores.bbone[:] = 0.0
        network.tubes.bbone[:] = 0.0
        graph_algs.update_pore_and_tube_backbone(network)

    def get_nonwetting_backbone_saturation(self):
        self.update_network_connectivity_and_backbone()
        return self.sat_comp.sat_nw_bb()

    def get_nonwetting_backbone_saturation_pp_window(self):
        assert self.network_pp is not None

        if self._backbone_comp_pp is None:
            self._backbone_comp_pp = BackBoneComputer(self.network_pp)

        self.update_network_connectivity_and_backbone()
        self.__sync_postprocessing_network()

        return self._sat_comp_pp.sat_nw_bb()

    def update_network_connectivity_and_backbone(self):
        self.__update_network_connectivity(self.network)
        self.__update_network_backbone(self.network)

    def nonwetting_saturation(self):
        return self.sat_comp.sat_nw()

    def wetting_film_saturation(self):
        return self.sat_comp.sat_w_film()

    def wetting_connected_saturation(self):
        network = self.network
        prev_status = np.copy(network.pores.invaded[network.pi_in])

        network.pores.invaded[network.pi_in] = 0
        pore_connected_mask, tube_connected_mask = \
            graph_algs.get_pore_and_tube_wetting_connectivity_to_pore_list(network, network.pi_in)

        network.pores.invaded[network.pi_in] = prev_status

        vol = np.sum(network.pores.vol[pore_connected_mask == 1]) + np.sum(network.tubes.vol[tube_connected_mask == 1])
        total_vol = np.sum(network.pores.vol) + np.sum(network.tubes.vol)
        return vol/total_vol, pore_connected_mask, tube_connected_mask

    def wetting_trapped_saturation(self):
        network = self.network
        prev_status = np.copy(network.pores.invaded[network.pi_in])

        network.pores.invaded[network.pi_in] = 0
        pore_connected_mask, tube_connected_mask = \
            graph_algs.get_pore_and_tube_wetting_connectivity_to_pore_list(network, network.pi_in)

        network.pores.invaded[network.pi_in] = prev_status

        pore_mask = (pore_connected_mask == 0) & (network.pores.invaded == 0)
        tube_mask = (tube_connected_mask == 0) & (network.tubes.invaded == 0)

        vol = np.sum(network.pores.vol[pore_mask]) + np.sum(network.tubes.vol[tube_mask])
        total_vol = np.sum(network.pores.vol) + np.sum(network.tubes.vol)

        return vol/total_vol, pore_mask, tube_mask

    def get_wetting_backbone_saturation(self):
        network = self.network
        return self.__wetting_backbone_saturation(network)

    def get_wetting_backbone_saturation_pp_window(self):
        self.__sync_postprocessing_network()
        network = self.network_pp
        return self.__wetting_backbone_saturation(network)

    @staticmethod
    def __wetting_backbone_saturation(network):
        # Set inlet pores to be wetting only for the purpose of computing the backbone.
        prev_status = np.copy(network.pores.invaded[network.pi_in])
        network.pores.invaded[network.pi_in] = 0

        G = graph_algs.subgraph_wett_igraph(network)

        pi_w_backbone = graph_algs.get_pi_list_of_biconnected_components(network, G)

        ti_w_backbone = component.tubes_within_pore_set(network, pi_w_backbone)
        ti_w_backbone = ti_w_backbone[network.tubes.invaded[ti_w_backbone] == 0]

        # Reset invaded pores
        network.pores.invaded[network.pi_in] = prev_status

        vol = np.sum(network.pores.vol[pi_w_backbone]) + np.sum(network.tubes.vol[ti_w_backbone])
        total_vol = np.sum(network.pores.vol) + np.sum(network.tubes.vol)

        pi_w_backbone_mask = np.zeros(network.nr_p)
        pi_w_backbone_mask[pi_w_backbone] = 1

        ti_w_backbone_mask = np.zeros(network.nr_t)
        ti_w_backbone_mask[ti_w_backbone] = 1

        return vol/total_vol, pi_w_backbone_mask, ti_w_backbone_mask

    def get_nonwetting_connected_saturation(self):
        self.__update_network_connectivity(self.network)
        return self.sat_comp.sat_nw_conn()

    def __sync_postprocessing_network(self):
        self.network_pp.copy_status_fields_from_super_network()
        self.network_pp.copy_sat_from_super_network()
        self.network_pp.copy_pc_from_super_network()

    def get_nonwetting_saturation_pp_window(self):
        self.__sync_postprocessing_network()

        return self._sat_comp_pp.sat_nw()

    def get_wetting_film_saturation_pp_window(self):
        self.__sync_postprocessing_network()

        return self._sat_comp_pp.sat_w_film()

    def get_nonwetting_connected_euler_number_pp_window(self):
        self.__sync_postprocessing_network()

        euler_full = 1-(self.network_pp.nr_t - self.network_pp.nr_p+1)
        beta_1 = self.network_pp.tubes.connected.sum() - self.network_pp.pores.connected.sum() + 1
        beta_0 = 1
        return float(beta_0-beta_1)/euler_full

    def get_nonwetting_connected_euler_number(self):
        euler_full = 1-(self.network.nr_t - self.network.nr_p+1)
        beta_1 = self.network.tubes.connected.sum() - self.network.pores.connected.sum() + 1
        beta_0 = 1
        return float(beta_0-beta_1)/euler_full

    def get_nonwetting_connected_saturation_pp_window(self):
        self.__update_network_connectivity(self.network)
        self.__sync_postprocessing_network()
        return self._sat_comp_pp.sat_nw_conn()

    def save_to_file(self, filename):
        output_file = open(filename, 'wb')
        pickle.dump(self, output_file, protocol=pickle.HIGHEST_PROTOCOL)
        output_file.close()

    @classmethod
    def load_from_file(cls, filename):
        input_file = open(filename, 'rb')
        simulation = pickle.load(input_file)
        return simulation

    def write_to_hdf(self, label, folder_name):
        time = self.time
        network = self.network
        require_path(folder_name)
        network_saturation = self.nonwetting_saturation()
        filename = folder_name+"/hdf_net.h5"
        add_attribute_to_hdf_file(filename, label, "network_saturation", network_saturation)
        add_attribute_to_hdf_file(filename, label, "time", time)

        add_field_to_hdf_file(filename, label, "pore_sat", network.pores.sat)
        add_field_to_hdf_file(filename, label, "p_c", network.pores.p_c)
        add_field_to_hdf_file(filename, label, "p_w", network.pores.p_w)
        add_field_to_hdf_file(filename, label, "p_n", network.pores.p_n)

        add_field_to_hdf_file(filename, label, "pore_vol", network.pores.vol)
        add_field_to_hdf_file(filename, label, "pore_rad", network.pores.r)
        add_field_to_hdf_file(filename, label, "pore_status", network.pores.invaded)

        add_field_to_hdf_file(filename, label, "tube_r", network.tubes.r)
        add_field_to_hdf_file(filename, label, "tube_l", network.tubes.l)
        add_field_to_hdf_file(filename, label, "tube_A_tot", network.tubes.A_tot)


        add_field_to_hdf_file(filename, label, "tube_invaded", network.tubes.invaded)

        add_field_to_hdf_file(filename, 0, "G", network.pores.G)
        add_field_to_hdf_file(filename, 0, "r", network.pores.r)

        add_field_to_hdf_file(filename, 0, "x", network.pores.x)
        add_field_to_hdf_file(filename, 0, "y", network.pores.y)
        add_field_to_hdf_file(filename, 0, "z", network.pores.z)




