import numpy as np
from pypnm.porenetwork.constants import *
from sim_settings import *

from pore_element_models import BasePEModel


class EntryPressureComputer(object):
    def __init__(self):
        self.gamma = sim_settings['fluid_properties']['gamma']

    def piston_all_tubes(self, network):
        assert np.all(network.tubes.G > 0.0)
        assert np.all(network.tubes.r > 0.0)
        return BasePEModel.piston_entry_pressure(r=network.tubes.r, G=network.tubes.G, gamma=self.gamma)

    def piston_tube(self, network, ti):
        """
        Returns the threshold capillary pressure for piston displacement pressure for network tubes ti
        Computed according to relation in Tora 2012
        Parameters
        ----------
        network: PoreNetwork
        ti: int, ndarray(int), slice

        Returns
        -------
        entry_pressure: ndarray
            Array of entry pressures for network tubes with indices ti

        """
        assert np.all(network.tubes.G > 0.0)
        assert np.all(network.tubes.r > 0.0)
        return BasePEModel.piston_entry_pressure(r=network.tubes.r[ti], G=network.tubes.G[ti], gamma=self.gamma)

    def snap_off_all_tubes(self, network):
        return BasePEModel.snap_off_pressure(r=network.tubes.r, gamma=self.gamma)

    def coop_pore(self, network, pi):
        ngh_tubes = network.ngh_tubes[pi]

        z = np.count_nonzero(network.tubes.invaded[ngh_tubes] == NWETT)
        assert (z > 0)

        # Formula for cooperative filling.
        p_b_eff = z * network.pores.r[pi]

        gamma = self.gamma

        pc_p = (1.0 + 2.0 * (np.pi * network.pores.G[pi]) ** 0.5) * gamma / (p_b_eff)
        return pc_p
