import logging

import numpy as np

from pore_element_models import  ToraModel


try:
    from sim_settings import *
except ImportError:
    sim_settings = dict()
    sim_settings["fluid_properties"] = dict()
    sim_settings["fluid_properties"]['gamma'] = 1.0

logger = logging.getLogger('pypnm.saturation_computer')


class SaturationComputer(object):
    def __init__(self, network):
        self.network = network

    def sat_nw(self):
        Snw_pores = self.sat_nw_pores()
        Snw_tubes = self.sat_nw_tubes()
        return self.sat_pores_and_tubes(Snw_pores, Snw_tubes)

    def sat_nw_bb(self):
        Snw_pores = self.sat_nw_bb_pores()
        Snw_tubes = self.sat_nw_bb_tubes()
        return self.sat_pores_and_tubes(Snw_pores, Snw_tubes)

    def sat_nw_conn(self):
        Snw_pores = self.sat_nw_conn_pores()
        Snw_tubes = self.sat_nw_conn_tubes()
        return self.sat_pores_and_tubes(Snw_pores, Snw_tubes)

    def sat_w_film(self):
        Sw_pores = self.sat_w_film_pores()
        Sw_tubes = self.sat_w_film_tubes()
        return self.sat_pores_and_tubes(Sw_pores, Sw_tubes)

    def sat_pores_and_tubes(self, sat_pores, sat_tubes):
        """Computes the total network saturation given the saturation in pores and tubes.
        Args:
            sat_pores: Saturation of pores.
            sat_tubes: Saturation of tubes.
        """
        total_tube_vol = self.network.total_throat_vol
        total_pore_vol = self.network.total_pore_vol
        x = total_tube_vol / (total_pore_vol + total_tube_vol)
        return x * sat_tubes + (1 - x) * sat_pores

    def sat_nw_pores(self):
        return self.sat_nw_generic(self.network.pores, self.network.pores.invaded)

    def sat_nw_bb_pores(self):
        return self.sat_nw_generic(self.network.pores, self.network.pores.bbone)

    def sat_nw_conn_pores(self):
        return self.sat_nw_generic(self.network.pores, self.network.pores.connected)

    def sat_w_film_pores(self):
        pores = self.network.pores
        total_sat = np.sum(pores.invaded*pores.vol)/np.sum(pores.vol)
        nw_sat = self.sat_nw_generic(pores, pores.invaded)
        return total_sat - nw_sat

    def sat_nw_tubes(self):
        return self.sat_nw_generic(self.network.tubes, self.network.tubes.invaded)

    def sat_nw_bb_tubes(self):
        return self.sat_nw_generic(self.network.tubes, self.network.tubes.bbone)

    def sat_nw_conn_tubes(self):
        return self.sat_nw_generic(self.network.tubes, self.network.tubes.connected)

    def sat_w_film_tubes(self):
        tubes = self.network.tubes
        total_sat = np.sum(tubes.invaded*tubes.vol)/np.sum(tubes.vol)
        nw_sat = self.sat_nw_generic(tubes, tubes.invaded)
        return total_sat - nw_sat

    def sat_nw_generic(self, element_array, marker):
        """
        Generic function to compute saturation.
        Args: 
            element_array: Array of elements, either pores or tubes
            marker: marker array (1 and 0) of entries in "element_array" which
                    are considered for the saturation
        """
        assert (len(marker) == len(element_array.vol))
        eps = 1e-14
        return np.sum(element_array.vol[marker == 1]) / (np.sum(element_array.vol)+eps)

    def get_pore_sat_nw_contribution(self, pi):
        return self.network.pores.vol[pi] / self.network.total_pore_vol

    def get_tube_sat_nw_contribution(self, ti):
        eps = 1e-14
        return self.network.tubes.vol[ti] / (self.network.total_throat_vol + eps)

    def receive_update(self):
        self.compute()

    def compute(self):
        self.network.pores.sat[:] = self.compute_loc_sat(self.network.pores)
        self.network.tubes.sat[:] = self.compute_loc_sat(self.network.tubes)

    def compute_loc_sat(self, elements, pc_array=None):
        if pc_array is None:
            pc_array = elements.p_c
        gamma = sim_settings['fluid_properties']['gamma']

        sat_nw = self.model.pc_to_sat_func(r=elements.r, p_c=pc_array, gamma=gamma, G=elements.G, A_tot=elements.A_tot)

        return sat_nw


class QuasiStaticSaturationComputer(SaturationComputer):
    def __init__(self, network, model=ToraModel):
        super(QuasiStaticSaturationComputer, self).__init__(network)
        self.pc_model = model

    def sat_nw_generic(self, element_array, marker, pc_array=None):
        """
        Generic function to compute saturation.
        Args:
            volume_array: Array of volumes
            marker: marker array of entries in volume_array contributing to the saturation
            pc_array: Capillary pressure array. If None then element_array.p_c is assumed to exist
        """
        if pc_array is None:
            pc_array = element_array.p_c

        assert (np.all(pc_array[marker == 1] > 0.0))
        Sat_nw_loc = self.compute_loc_sat(element_array, pc_array)[marker == 1]

        assert (len(marker) == len(element_array.vol))
        eps = 1e-14
        return np.sum(Sat_nw_loc * element_array.vol[marker == 1]) / (np.sum(element_array.vol)+eps)

    def get_element_sat_nw(self, list_elements, ind, p_c):
        if p_c is None:
            p_c = list_elements.p_c[ind]

        assert (p_c > 0.0)
        gamma = sim_settings['fluid_properties']['gamma']

        sat_nw_loc = self.pc_model.pc_to_sat_func(p_c=p_c, G=list_elements.G[ind], gamma=gamma,
                                                  A_tot=list_elements.A_tot[ind], r=list_elements.r)

        assert (sat_nw_loc >= 0)
        return sat_nw_loc

    def get_pore_sat_nw_contribution(self, pi, p_c=None):
        sat_element = self.get_element_sat_nw(self.network.pores, pi, p_c)
        return sat_element * self.network.pores.vol[pi] / self.network.total_pore_vol

    def get_tube_sat_nw_contribution(self, ti, p_c=None):
        sat_element = self.get_element_sat_nw(self.network.tubes, ti, p_c)
        return sat_element * self.network.tubes.vol[ti] / self.network.total_throat_vol


class DynamicSaturationComputer(QuasiStaticSaturationComputer):
    """
    A class to compute and update the pore saturations for a dynamic pore-network simulation

    Parameters
    ----------

    network: The pore network object.
    mask_accounted_pores: bool ndarray, optional
        pores with a corresponding value of False in this array will be ignored when computing time-steps.
    """
    def __init__(self, network, mask_accounted_pores=None):
        self.network = network

        if mask_accounted_pores is None:
            mask_accounted_pores = np.ones(network.nr_p, dtype=np.bool)

        self.accounted_pores = mask_accounted_pores

    def update_saturation(self, flux_n, dt, source_nonwett=0.0):
        network = self.network
        pores = network.pores
        mask = self.accounted_pores
        assert (np.all(pores.vol[mask]) > 0.0)
        dvn_dt = source_nonwett - flux_n

        pores.sat[mask] += (dvn_dt[mask]) / pores.vol[mask] * dt
        pores.sat[(pores.sat > -1e-10) & (pores.sat < 0.0)] = 0.0

        assert np.all(pores.sat >= 0.0)

        assert (not np.isnan(np.sum(pores.sat[mask])))

    def sat_nw_bb(self):
        Snw_pores = self.sat_nw_bb_pores()
        return Snw_pores

    def sat_nw(self):
        network = self.network
        pores = network.pores
        return np.sum(pores.sat * pores.vol) / np.sum(pores.vol)

    def sat_nw_conn(self):
        Snw_pores = self.sat_nw_conn_pores()
        return Snw_pores

    def sat_nw_generic(self, element_array, marker):
        Sat_nw_loc = element_array.sat[marker == 1]
        assert (len(marker) == len(element_array.vol))
        return np.sum(Sat_nw_loc * element_array.vol[marker == 1]) / np.sum(element_array.vol)