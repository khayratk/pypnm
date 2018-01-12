import numpy as np
from pypnm.linalg.laplacianmatrix import laplacian_from_network
from fluxes import flux_into_pores
from pypnm.attribute_calculators.conductance_calc import ConductanceCalc
from pypnm.percolation import graph_algs
from pypnm.porenetwork.constants import *
from pypnm.linalg.petsc_interface import petsc_solve


class SimpleRelPermComputer(object):
    def __init__(self, network, fluid_properties, pores_have_conductance=False):
        self.network = network
        self.total_flux_one_phase = None
        self.kr_n = np.zeros(3)
        self.kr_w = np.zeros(3)
        self.K = 1.0e-8 * np.ones(3)
        self.is_isotropic = True

        # Mask of tubes and pores not connected to the inlet, which makes the matrix singular
        p_conn_inlet, t_conn_inlet = graph_algs.get_pore_and_tube_connectivity_to_pore_list(self.network, self.network.pi_list_face[WEST])
        self.mask_ignored_tubes = (t_conn_inlet == 0)

        self.fluid_properties = fluid_properties

        self.cond_computer = ConductanceCalc(self.network, fluid_properties, pores_have_conductance)
        self.compute_absolute_permeability()

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


    def _general_absolute_permeability(self, coord, pi_inlet, pi_outlet):
        network = self.network
        tube_k_w = self.cond_computer.conductances_fully_wetting()

        pi_dirichlet = np.union1d(pi_inlet, pi_outlet)
        A = laplacian_from_network(network, weights=tube_k_w, ind_dirichlet=pi_dirichlet)
        rhs = np.zeros(network.nr_p)
        rhs[pi_inlet] = 1.0

        pressure = petsc_solve(A*1e20, rhs*1e20, tol=1e-12)

        mu_w = self.fluid_properties["mu_w"]
        self.total_flux_one_phase = flux_into_pores(network, pressure, tube_k_w, pi_inlet)

        x_max = np.mean(coord[pi_outlet])
        x_min = np.mean(coord[pi_inlet])

        return mu_w * np.abs(self.total_flux_one_phase) * (x_max-x_min)

    def compute_absolute_permeability(self):
        network = self.network

        self.K[0] = self._general_absolute_permeability(network.pores.x,
                                                        network.pi_list_face[WEST], network.pi_list_face[EAST])

        self.K[1] = self._general_absolute_permeability(network.pores.y,
                                                        network.pi_list_face[SOUTH], network.pi_list_face[NORTH])

        self.K[2] = self._general_absolute_permeability(network.pores.z,
                                                        network.pi_list_face[BOTTOM], network.pi_list_face[TOP])

        assert np.all(self.K > 0.0), self.K

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

                mu_n = self.fluid_properties['mu_n']
                mu_w = self.fluid_properties['mu_w']
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
        self.compute_isotropic()