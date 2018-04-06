import numpy as np
from pypnm.porenetwork.component import neighboring_edges_to_vertices
import numexpr as ne


class ConductanceCalc(object):
    """ 
    Computes wetting and non-wetting conductances of all tubes.
    Equations used follow G.Tora et al. (2012).
    """
    def __init__(self, network, fluid_properties, pores_have_conductance=False):
        self.pores_have_conductance = pores_have_conductance
        self.fluid_properties = fluid_properties
        self.network = network
        self.tubes_ngh_to_pi_inout = neighboring_edges_to_vertices(self.network,
                                                                   np.union1d(self.network.pi_in, self.network.pi_out))

    @staticmethod
    def _wetting_area(r_w, G):
        # Eq. 2 in G.Tora
        if len(r_w) > 2.e4:
            pi = np.pi
            A_w = ne.evaluate("(r_w ** 2) * (1. / (4 * G) - pi)")
        else:
            A_w = (r_w ** 2) * (1. / (4 * G) - np.pi)

        return A_w

    @staticmethod
    def _compute_intfc_curvature(gamma, p_c, pi_nwett):
        assert np.all(p_c[pi_nwett] > 0.0)
        r_w = np.zeros_like(p_c)
        r_w[pi_nwett] = gamma / p_c[pi_nwett]

        return r_w

    @staticmethod
    def _harm_avg(a, b, c):
        eps = np.finfo(np.float64).tiny
        return 1.0 / (1.0 / (a+eps) + 1.0 / (b+eps) + 1.0 / (c+eps))

    def _harmonic_mean_tube_conductivities(self, tube_k_n, tube_k_w, pore_k_n, pore_k_w, invasion_status):
        k_n_avg = np.zeros_like(tube_k_n)
        k_w_avg = np.zeros_like(tube_k_n)

        ti_nw = (invasion_status == 1).nonzero()[0]

        pore_indices_1 = self.network.edgelist[ti_nw, 0]
        pore_indices_2 = self.network.edgelist[ti_nw, 1]

        k_n_avg[ti_nw] = self._harm_avg(tube_k_n[ti_nw], 2 * pore_k_n[pore_indices_1], 2 * pore_k_n[pore_indices_2])

        pore_indices_1 = self.network.edgelist[:, 0]
        pore_indices_2 = self.network.edgelist[:, 1]
        k_w_avg[:] = self._harm_avg(tube_k_w, 2 * pore_k_w[pore_indices_1], 2 * pore_k_w[pore_indices_2])

        # Do not harmonic average at inlet and outlet
        k_w_avg[self.tubes_ngh_to_pi_inout] = tube_k_w[self.tubes_ngh_to_pi_inout]
        k_n_avg[self.tubes_ngh_to_pi_inout] = tube_k_n[self.tubes_ngh_to_pi_inout]

        assert np.all(k_n_avg >= 0.0)
        assert np.all(k_w_avg > 0.0)

        return k_n_avg, k_w_avg

    @staticmethod
    def compute_conductances(fluid_properties, el_rad, el_len, el_G, el_A_tot, el_pc, invasion_status, beta=5.3):
        # Default flow resistance factor beta from  Ransohoff and Radke 1988 for square cross-section
        pi_nwett = (invasion_status == 1).nonzero()[0]
        pi_wett = (invasion_status == 0).nonzero()[0]

        mu_n = fluid_properties['mu_n']
        mu_w = fluid_properties['mu_w']
        gamma = fluid_properties['gamma']

        assert np.all(el_pc[pi_nwett] > 1.0)

        r_w = ConductanceCalc._compute_intfc_curvature(gamma, el_pc, pi_nwett)

        A_w = ConductanceCalc._wetting_area(r_w, el_G)

        A_n = np.zeros_like(A_w)

        A_n[pi_nwett] = (el_A_tot - A_w)[pi_nwett]

        # Mean hydraulic radii
        # R_n = 0.5 * (el_rad + np.sqrt(A_n / np.pi))
        R_w = 0.5 * (el_rad + np.sqrt(el_A_tot / np.pi))

        assert np.all(el_pc[pi_nwett] > gamma / el_rad[pi_nwett])
        assert np.all(A_n >= 0)

        k_n = np.zeros_like(el_rad)
        k_w = np.zeros_like(el_rad)

        temp = el_A_tot * (R_w**2) / el_len / 8.
        # Eq. 6,7,8  in G.Tora (2012)
        k_n[pi_nwett] = (A_n / el_A_tot * temp / mu_n)[pi_nwett]

        # Eq. 5 in G.Tora (2012)
        k_w[pi_nwett] = ((r_w ** 2) * A_w / el_len / (8 * mu_w * beta))[pi_nwett]

        # Eq. 6,7,8  in G.Tora (2012)
        k_w[pi_wett] = (temp /mu_w)[pi_wett]
        
        assert np.all(R_w[pi_wett] > 0.0)
        assert np.all(k_n[pi_nwett] > 0.0)
        assert np.all(k_n >= 0.0)
        assert np.all(k_w > 0.0)

        return k_n, k_w

    def compute(self):
        tubes = self.network.tubes
        tubes.k_n[:], tubes.k_w[:] = self._conductances(tubes.invaded)

    def conductances_two_phase(self):
        tubes = self.network.tubes
        return self._conductances(tubes.invaded)

    def conductances_fully_wetting(self):
        invasion_status = np.zeros(self.network.nr_t, dtype=np.int32)
        return self._conductances(invasion_status)[1]

    def _conductances(self, invasion_status):
        tubes = self.network.tubes
        tube_k_n, tube_k_w = self.compute_conductances(self.fluid_properties, tubes.r, tubes.l, tubes.G, tubes.A_tot,
                                                       tubes.p_c, invasion_status)

        if self.pores_have_conductance:
            pores = self.network.pores
            invasion_status = np.zeros(self.network.nr_p, dtype=np.int32)
            pore_k_n, pores_k_w = self.compute_conductances(self.fluid_properties, pores.r, pores.l,
                                                                   pores.G, pores.A_tot, pores.p_c, invasion_status)
            tube_k_n, tube_k_w = self._harmonic_mean_tube_conductivities(tube_k_n, tube_k_w, pore_k_n, pores_k_w,
                                                                         invasion_status)

        return tube_k_n, tube_k_w
