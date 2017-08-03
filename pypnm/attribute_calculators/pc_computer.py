from pypnm.porenetwork.constants import *
import numpy as np
from sim_settings import *
from pypnm.porenetwork.pore_element_models import BasePEModel


class CapillaryPressureComputer(object):
    def __init__(self, network):
        self.network = network
        self.mode = NEUTRAL
        self.p_c = 0.0
        self.initialize_pc_from_invasion_status()

    def initialize_pc_from_invasion_status(self):
        gamma = sim_settings['fluid_properties']['gamma']
        pores = self.network.pores

        self.set_constant_pc_from_pc_field()

        pores.p_c[pores.invaded == 1] = np.maximum(1.000001*BasePEModel.snap_off_pressure(gamma=gamma,
                                                                                     r=pores.r[pores.invaded == 1]),
                                                   pores.p_c[pores.invaded == 1])
        self.p_c = max(self.p_c, np.max(pores.p_c))
        self.compute()

    def set_constant_pc_from_pc_field(self):
        self.p_c = np.max(self.network.pores.p_c)

    def set_invasion_mode(self, mode):
        assert mode in INVASION_MODES
        self.mode = mode

    def compute(self):
        self.set_pc_zero_in_wetted_elements()
        self.set_pc_connected_pores()
        self.set_pc_invaded_tubes()

    def set_pc_zero_in_wetted_elements(self):
        elements_list = [self.network.tubes, self.network.pores]
        for elements in elements_list:
            mask_wetting = (elements.invaded == 0)
            elements.p_c[mask_wetting] = 0.0

    def set_pc_connected_pores(self):
        network = self.network
        mask_connected = (network.pores.connected == 1)
        network.pores.p_c[mask_connected] = self.p_c

    def set_pc_invaded_tubes(self):
        network = self.network
        pores_pc = network.pores.p_c
        tubes_pc = network.tubes.p_c
        edgelist = network.edgelist
        mask_invaded = (network.tubes.invaded == 1)
        tubes_pc[mask_invaded] = np.maximum(pores_pc[edgelist[mask_invaded, 0]], pores_pc[edgelist[mask_invaded, 1]])

    def receive_update(self):
        self.compute()

    def get_pc(self):
        network = self.network
        mask_connected = network.pores.connected == 1
        return np.max(network.pores.p_c[mask_connected])


class DynamicCapillaryPressureComputer(object):
    def __init__(self, network):
        self.network = network

    @staticmethod
    def sat_to_pc_func(sat, pore_rad):
        # Joekar-Niasar JFM (2010) Eq. 3.7
        gamma = sim_settings['fluid_properties']['gamma']
        assert np.all(sat < 1.0)
        assert np.all(sat >= 0.0)
        exp_part = np.exp(-6.83 * (1. - sat))
        assert (not np.isnan(np.sum(exp_part)))
        assert np.all(exp_part < 1.0)
        return 2 * gamma / (pore_rad * (1.0 - exp_part))

    @staticmethod
    def pc_to_sat_func(pore_rad, p_c):
        # Inverse of sat_to_pc_func
        gamma = sim_settings['fluid_properties']['gamma']
        assert (np.all(p_c > 0.0))
        exp_part = 1. - 2. * gamma / (p_c * pore_rad)
        assert np.all(exp_part > 0.0)
        sat = 1 + np.log(exp_part) / 6.83
        assert np.all(sat <= 1.0)
        assert np.all(sat >= 0.0)
        return sat

    def compute(self):
        network = self.network
        pores = network.pores
        network.pores.p_c[:] = 0.0
        network.tubes.p_c[:] = 0.0

        pores.p_c[:] = self.sat_to_pc_func(pores.sat, pores.r)

        assert np.all(pores.p_c > 0.0)
        assert (not np.isnan(np.sum(pores.p_c)))
        assert np.all(pores.p_c[pores.invaded == 1] > 0.0)

        self.set_pc_invaded_tubes()

    def set_pc_invaded_tubes(self):
        network = self.network
        pores_pc = self.network.pores.p_c
        edgelist = self.network.edgelist
        mask_invaded = (network.tubes.invaded == 1)
        network.tubes.p_c[mask_invaded] = np.maximum(pores_pc[edgelist[mask_invaded, 0]],
                                                     pores_pc[edgelist[mask_invaded, 1]])

        # assert (np.all(network.tubes.p_c[mask_invaded] > 0.0))

    def get_pc(self):
        network = self.network
        return np.sum(network.pores.p_c) / len(network.pores.p_c)
