import numpy as np

from fluxes import flux_into_pores
from pypnm.attribute_calculators.conductance_calc import ConductanceCalc
from pypnm.linalg.linear_system_solver import LinearSystemStandard
from pypnm.percolation import graph_algs
from pypnm.porenetwork.constants import *
from sim_settings import *


class _PressureSolver(object):
    def __init__(self, network):
        self.network = network
        self.ls = LinearSystemStandard(network)
        self.sol = None

    def solve_pressure(self, FLAG, pi_in, pi_out, mask_ignored_tubes=None):
        """ Solves for pressure over a specific field over nodes of pore-network
        Args:
            FLAG: NWETT or WETT
        """
        network = self.network

        if mask_ignored_tubes is None:
            mask_ignored_tubes = np.zeros(self.network.nr_t, dtype=np.bool)

        if FLAG == NWETT:
            conductance = np.copy(network.tubes.k_n)
            conductance[mask_ignored_tubes] = 0.0

        if FLAG == WETT:
            conductance = np.copy(network.tubes.k_w)
            conductance[mask_ignored_tubes] = 0.0

        self.ls.fill_matrix(conductance)
        self.ls.set_dirichlet_pores(pi_in, 1.0)
        self.ls.set_dirichlet_pores(pi_out, 0.0)
        self.sol = self.ls.solve("AMG")


class RelPermComputer(object):
    def __init__(self, network):
        self.network = network

        self.ls = _PressureSolver(self.network)

        self.total_flux_one_phase = None
        self.kr_n = np.zeros(3)
        self.kr_w = np.zeros(3)

        self.K = 1.0e-8
        self.is_isotropic = True

    def compute_isotropic(self):
        kr_n, kr_w = self.compute_relperms(self.ls, self.network.pi_list_face[WEST], self.network.pi_list_face[EAST])
        self.kr_n[:] = kr_n
        self.kr_w[:] = kr_w

        assert (kr_n <= 1.0001), kr_n
        assert (kr_w <= 1.0001), kr_w

    def compute_anisotropic(self):
        self.kr_n[0], self.kr_w[0] = self.compute_relperms(self.ls, self.network.pi_list_face[WEST], self.network.pi_list_face[EAST])
        self.kr_n[1], self.kr_w[1] = self.compute_relperms(self.ls, self.network.pi_list_face[SOUTH], self.network.pi_list_face[NORTH])
        self.kr_n[2], self.kr_w[2] = self.compute_relperms(self.ls, self.network.pi_list_face[BOTTOM], self.network.pi_list_face[TOP])

        assert np.all(self.kr_n <= 18.0)
        assert np.all(self.kr_w <= 18.0)

    def compute(self):
        raise NotImplementedError("Method not implemented")


class SimpleRelPermComputer(RelPermComputer):
    def __init__(self, network):
        super(SimpleRelPermComputer, self).__init__(network)

        # Mask of tubes and pores not connected to the inlet, which makes the matrix singular
        p_conn_inlet, t_conn_inlet = graph_algs.get_pore_and_tube_connectivity_to_pore_list(self.network, self.network.pi_list_face[WEST])
        self.mask_ignored_tubes = (t_conn_inlet == 0)

        self.cond_computer = ConductanceCalc(self.network)
        self.compute_absolute_permeability()

    def compute_absolute_permeability(self):
        self.cond_computer.compute_fully_wetting()
        ls = _PressureSolver(self.network)
        ls.solve_pressure(FLAG=WETT, pi_in=self.network.pi_list_face[WEST], pi_out=self.network.pi_list_face[EAST], mask_ignored_tubes=self.mask_ignored_tubes)
        self.total_flux_one_phase = flux_into_pores(self.network, ls.sol, self.network.tubes.k_w, self.network.pi_list_face[WEST])

        assert np.all(self.network.tubes.k_w > 0.0)
        assert np.all(self.network.tubes.k_w < 0.1)

        mu_w = sim_settings['fluid_properties']['mu_w']

        # TODO: This is only generally valid for CUBIC or SQUARE isotropic pore-networks!
        self.K = mu_w*np.abs(self.total_flux_one_phase) * (np.max(self.network.pores.x) - np.min(self.network.pores.x))
        assert self.K > 0.0

    def compute_relperms(self, ls, pi_in, pi_out):
        network = self.network
        self.cond_computer.compute()

        ls.solve_pressure(WETT, pi_in, pi_out, self.mask_ignored_tubes)
        total_flux_wett = flux_into_pores(self.network, ls.sol, self.network.tubes.k_w, pi_in)
        kr_w = total_flux_wett / self.total_flux_one_phase

        is_nwett_pore_inlet = np.any(network.pores.invaded[pi_in] == 1)
        is_nwett_pore_outlet = np.any(network.pores.invaded[pi_out] == 1)

        if not (is_nwett_pore_inlet and is_nwett_pore_outlet):
            kr_n = 0.0
        else:
            p_conn_inlet, t_conn_inlet = graph_algs.get_pore_and_tube_nonwetting_connectivity_to_pore_list(network, pi_in)
            is_nwett_pore_outlet_conn_to_inlet = np.any(p_conn_inlet[pi_out] == 1)
            if not is_nwett_pore_outlet_conn_to_inlet:
                kr_n = 0.0
            else:
                ls.solve_pressure(NWETT, pi_in, pi_out, t_conn_inlet==0)

                total_flux_nwett = flux_into_pores(self.network, ls.sol, self.network.tubes.k_n, pi_in)

                mu_n = sim_settings['fluid_properties']['mu_n']
                mu_w = sim_settings['fluid_properties']['mu_w']
                kr_n = total_flux_nwett / self.total_flux_one_phase * mu_n / mu_w

        if kr_n < -10.0e-9:
            print kr_n

        eps = 1.e-3

        assert (kr_n >= 0.0), "kr_n:%g kr_w:%g"%(kr_n, kr_w)
        assert (kr_w >= 0.0), "kr_n:%g kr_w:%g"%(kr_n, kr_w)
        assert (kr_w <= 1.0+eps), "kr_n:%e kr_w:%e"%(kr_n, kr_w)
        assert (kr_n <= 1.0+eps), "kr_n:%e kr_w:%e"%(kr_n, kr_w)
        kr_n = abs(kr_n)
        kr_w = abs(kr_w)

        return kr_n, kr_w

    def compute(self):
        if self.is_isotropic:
            self.compute_isotropic()
        else:
            self.compute_anisotropic()