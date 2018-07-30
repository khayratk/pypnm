from functools import reduce

import numpy as np
from fluxes import flux_into_pores
from pypnm.attribute_calculators.conductance_calc import ConductanceCalc
from pypnm.linalg.laplacianmatrix import laplacian_from_network
from pypnm.linalg.petsc_interface import petsc_solve
from pypnm.percolation import graph_algs
from pypnm.porenetwork.constants import *
from pypnm.porenetwork.component import tube_list_x_plane, tube_list_y_plane, tube_list_z_plane


def average_flux(network, press, cond, dir=0, ti_indices=Ellipsis):
    assert dir in [0, 1, 2]

    l_x, l_y, l_z = network.dim

    if dir == 0:
        ti_center = tube_list_x_plane(network, np.min(network.pores.x) + l_x / 2)

    if dir == 1:
        ti_center = tube_list_y_plane(network, np.min(network.pores.y) + l_y / 2)

    if dir == 2:
        ti_center = tube_list_z_plane(network, np.min(network.pores.z) + l_z / 2)

    tubes = network.tubes

    A_tot = np.zeros_like(tubes.A_tot)
    A_tot[ti_indices] = tubes.A_tot[ti_indices]

    orientation = network.edge_orientations.T[dir]
    pi_1, pi_2 = network.edgelist[:, 0], network.edgelist[:, 1]
    flux_12 = -(press[pi_2] - press[pi_1]) * cond
    dvdu = flux_12 * orientation * tubes.l_tot
    integral = np.sum(dvdu[ti_indices]) / np.sum((tubes.A_tot * np.abs(orientation) * tubes.l_tot)[ti_indices]
                                                 ) * np.sum(A_tot[ti_center])
    return integral


def work_done_per_second(network, press, cond, ti_indices=Ellipsis):
    pi_1, pi_2 = network.edgelist[:, 0], network.edgelist[:, 1]
    energy_dissipated = (press[pi_2] - press[pi_1])**2*cond
    total_energy_dissipated = np.sum(energy_dissipated[ti_indices])
    return total_energy_dissipated


def press_drop_from_energy(network, press, cond, total_flux, ti_indices=Ellipsis):
    total_energy_dissipated = work_done_per_second(network, press, cond, ti_indices)
    pressure_drop = total_energy_dissipated /(total_flux)
    return pressure_drop


def flux_by_pressure_drop_three_directions(network, fluid_properties, q_tot, ti_indices=Ellipsis):

    cond_computer = ConductanceCalc(network, fluid_properties)
    tube_k_w = cond_computer.conductances_fully_wetting()

    def _general_q_by_p(pi_inlet, pi_outlet, dir):
        pi_dirichlet = pi_outlet
        A = laplacian_from_network(network, weights=tube_k_w, ind_dirichlet=pi_dirichlet)
        rhs = np.zeros(network.nr_p)
        rhs[pi_inlet] = q_tot/len(pi_inlet)

        pressure = petsc_solve(A * 1e20, rhs * 1e20, tol=1e-12)
        total_flux_one_phase = average_flux(network, pressure, tube_k_w, dir=dir, ti_indices=ti_indices)
        press_drop = press_drop_from_energy(network, pressure, tube_k_w, total_flux_one_phase, ti_indices)
        return total_flux_one_phase/press_drop

    q_by_p = np.zeros(3)

    q_by_p[0] = _general_q_by_p(network.pi_list_face[WEST], network.pi_list_face[EAST], 0)
    q_by_p[1] = _general_q_by_p(network.pi_list_face[SOUTH], network.pi_list_face[NORTH], 1)
    q_by_p[2] = _general_q_by_p(network.pi_list_face[BOTTOM], network.pi_list_face[TOP], 2)

    return q_by_p


class SimpleRelPermComputer(object):
    def __init__(self, network, fluid_properties, pores_have_conductance=False):
        self.network = network
        l_x, l_y, l_z = network.dim
        self.network_vol = l_x * l_y * l_z

        self.total_flux_one_phase = None
        self.kr_n = np.zeros(3)
        self.kr_w = np.zeros(3)
        self.is_isotropic = True

        # Mask of tubes and pores not connected to the inlet, which makes the matrix singular
        p_conn_inlet, t_conn_inlet = graph_algs.get_pore_and_tube_connectivity_to_pore_list(self.network,
                                                                                            self.network.pi_list_face[
                                                                                                WEST])
        self.mask_ignored_tubes = (t_conn_inlet == 0)

        self.fluid_properties = fluid_properties

        self.cond_computer = ConductanceCalc(self.network, fluid_properties, pores_have_conductance)
        self.absolute_permeability()

    def _general_absolute_permeability(self, len_dir, pi_inlet, pi_outlet):
        network = self.network
        tube_k_w = self.cond_computer.conductances_fully_wetting()

        pi_dirichlet = np.union1d(pi_inlet, pi_outlet)
        A = laplacian_from_network(network, weights=tube_k_w, ind_dirichlet=pi_dirichlet)
        rhs = np.zeros(network.nr_p)
        rhs[pi_inlet] = 1.0

        pressure = petsc_solve(A * 1e20, rhs * 1e20, tol=1e-12)

        mu_w = self.fluid_properties["mu_w"]
        self.total_flux_one_phase = flux_into_pores(network, pressure, tube_k_w, pi_inlet)

        return mu_w * np.abs(self.total_flux_one_phase) * len_dir ** 2 / self.network_vol

    def _general_effective_wetting_permeability(self, len_dir, pi_inlet, pi_outlet):
        network = self.network
        tube_k_n, tube_k_w = self.cond_computer.conductances_two_phase()

        pi_dirichlet = np.union1d(pi_inlet, pi_outlet)
        A = laplacian_from_network(network, weights=tube_k_w, ind_dirichlet=pi_dirichlet)
        rhs = np.zeros(network.nr_p)
        rhs[pi_inlet] = 1.0

        pressure = petsc_solve(A * 1e20, rhs * 1e20, tol=1e-12)

        mu_w = self.fluid_properties["mu_w"]
        self.total_flux_one_phase = flux_into_pores(network, pressure, tube_k_w, pi_inlet)

        return mu_w * np.abs(self.total_flux_one_phase) * len_dir ** 2 / self.network_vol

    def _general_effective_nonwetting_permeability(self, len_dir, pi_inlet, pi_outlet):
        network = self.network
        tube_k_n, tube_k_w = self.cond_computer.conductances_two_phase()

        p_conn_inlet, t_conn_inlet = graph_algs.get_pore_and_tube_nonwetting_connectivity_to_pore_list(network,
                                                                                                       pi_inlet)
        is_nwett_pore_outlet_conn_to_inlet = np.any(p_conn_inlet[pi_outlet] == 1)

        if not is_nwett_pore_outlet_conn_to_inlet:
            kr_n_eff = 0.0
        else:
            pi_dirichlet = reduce(np.union1d, (pi_inlet, pi_outlet, (p_conn_inlet == 0).nonzero()[0]))
            A = laplacian_from_network(network, weights=tube_k_n, ind_dirichlet=pi_dirichlet)

            rhs = np.zeros(network.nr_p)
            rhs[pi_inlet] = 1.0

            pressure = petsc_solve(A * 1e20, rhs * 1e20, tol=1e-12)

            mu_n = self.fluid_properties["mu_n"]
            total_flux_nwett = flux_into_pores(network, pressure, tube_k_n, pi_inlet)

            kr_n_eff = mu_n * np.abs(total_flux_nwett) * len_dir ** 2 / self.network_vol

        return kr_n_eff

    def effective_nonwetting_permeability(self):
        network = self.network
        l_x, l_y, l_z = network.dim

        K_n = np.zeros(3)

        K_n[0] = self._general_effective_nonwetting_permeability(l_x,
                                                                 network.pi_list_face[WEST], network.pi_list_face[EAST])

        K_n[1] = self._general_effective_nonwetting_permeability(l_y,
                                                                 network.pi_list_face[SOUTH],
                                                                 network.pi_list_face[NORTH])

        K_n[2] = self._general_effective_nonwetting_permeability(l_z,
                                                                 network.pi_list_face[BOTTOM],
                                                                 network.pi_list_face[TOP])

        return np.asarray(K_n)

    def effective_wetting_permeability(self):
        network = self.network
        l_x, l_y, l_z = network.dim

        K_n = np.zeros(3)

        K_n[0] = self._general_effective_wetting_permeability(l_x,
                                                                 network.pi_list_face[WEST], network.pi_list_face[EAST])

        K_n[1] = self._general_effective_wetting_permeability(l_y,
                                                                 network.pi_list_face[SOUTH],
                                                                 network.pi_list_face[NORTH])

        K_n[2] = self._general_effective_wetting_permeability(l_z,
                                                                 network.pi_list_face[BOTTOM],
                                                                 network.pi_list_face[TOP])

        return np.asarray(K_n)

    def absolute_permeability(self):
        network = self.network
        K = np.zeros(3)

        l_x, l_y, l_z = network.dim

        K[0] = self._general_absolute_permeability(l_x,
                                                   network.pi_list_face[WEST], network.pi_list_face[EAST])

        K[1] = self._general_absolute_permeability(l_y,
                                                   network.pi_list_face[SOUTH], network.pi_list_face[NORTH])

        K[2] = self._general_absolute_permeability(l_z,
                                                   network.pi_list_face[BOTTOM], network.pi_list_face[TOP])

        assert np.all(K > 0.0), K

        return np.asarray(K)

        flux_inlet = intrinsic_average_flux_x(network, press, network.tubes.k_w)
        pressure_drop = press_drop_from_energy(network, press, network.tubes.k_w, flux_inlet)
        effective_permeability = flux_inlet * l_x / (l_y * l_z * pressure_drop)
