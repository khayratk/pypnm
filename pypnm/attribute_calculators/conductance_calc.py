import numpy as np
from pypnm.porenetwork.constants import *
import math

try:
    from sim_settings import *
except ImportError:
    sim_settings = dict()
    sim_settings["fluid_properties"] = dict()
    sim_settings["fluid_properties"]['mu_n'] = 1.0
    sim_settings["fluid_properties"]['mu_w'] = 1.0
    sim_settings["fluid_properties"]['gamma'] = 1.0

from pypnm.porenetwork.component import neighboring_edges_to_vertices


class ConductanceCalc(object):
    """ 
    Computes wetting and non-wetting conductances of all tubes.
    Equations used follow G.Tora et al. (2012).
    """
    def __init__(self, network):
        self.network = network
        self.tubes_ngh_to_pi_inout = neighboring_edges_to_vertices(self.network,
                                                                   np.union1d(self.network.pi_in, self.network.pi_out))

    @staticmethod
    def _wetting_area(r_w, G):
        # Eq. 2 in G.Tora
        assert len(r_w) == len(G)
        A_w = (r_w ** 2) * (1. / (4 * G) - np.pi)
        assert np.all((1. / (4 * G) - np.pi) > 0.0)
        assert np.all(A_w >= 0)
        return A_w

    @staticmethod
    def _compute_intfc_curvature(gamma, p_c, mask):
        if len(mask) != len(p_c):
            raise ValueError()

        assert np.all(p_c[mask] > 0.0)
        r_w = np.zeros_like(p_c)
        r_w[mask] = gamma / p_c[mask]
        assert np.all(r_w[mask] > 0.0)
        return r_w

    @staticmethod
    def _harm_avg(a, b, c):
        return 1.0 / (1.0 / a + 1.0 / b + 1.0 / c)

    def _harmonic_mean_tube_conductivities(self):
        tubes = self.network.tubes
        pores = self.network.pores

        nw_mask_tubes = (tubes.invaded == 1)
        w_mask_tubes = (tubes.invaded == 0)

        ti_nwett = nw_mask_tubes.nonzero()[0]
        ti_wett = w_mask_tubes.nonzero()[0]

        kw_tubes_old = tubes.k_w.copy()
        kn_tubes_old = tubes.k_n.copy()
        assert np.all(kn_tubes_old >= 0.0)
        assert np.all(kw_tubes_old > 0.0)

        pore_indices_1 = self.network.edgelist[ti_nwett, 0]
        pore_indices_2 = self.network.edgelist[ti_nwett, 1]

        tubes.k_n[nw_mask_tubes] = self._harm_avg(tubes.k_n[ti_nwett], 2 * pores.k_n[pore_indices_1],
                                                  2 * pores.k_n[pore_indices_2])
        tubes.k_w[nw_mask_tubes] = self._harm_avg(tubes.k_w[ti_nwett], 2 * pores.k_w[pore_indices_1],
                                                  2 * pores.k_w[pore_indices_2])

        pore_indices_1 = self.network.edgelist[ti_wett, 0]
        pore_indices_2 = self.network.edgelist[ti_wett, 1]

        tubes.k_w[w_mask_tubes] = self._harm_avg(tubes.k_w[ti_wett], 2 * pores.k_w[pore_indices_1],
                                                 2 * pores.k_w[pore_indices_2])

        # Do not harmonic average at inlet and outlet
        tubes.k_w[self.tubes_ngh_to_pi_inout] = kw_tubes_old[self.tubes_ngh_to_pi_inout]
        tubes.k_n[self.tubes_ngh_to_pi_inout] = kn_tubes_old[self.tubes_ngh_to_pi_inout]

        assert np.all(tubes.k_n >= 0.0)
        assert np.all(tubes.k_w > 0.0)

    @staticmethod
    def compute_conductances(fluid_properties, el_rad, el_len, el_G, el_A_tot, el_pc, invasion_status, beta=5.3):
        # Default flow resistance factor beta from  Ransohoff and Radke 1988 for square cross-section
        nw_mask = (invasion_status == 1)
        w_mask = (invasion_status == 0)

        mu_n = fluid_properties['mu_n']
        mu_w = fluid_properties['mu_w']
        gamma = fluid_properties['gamma']

        assert np.all(el_len > 0)
        assert np.all(el_G > 0)
        assert np.all(el_pc[nw_mask] > 1.0)

        assert len(nw_mask) == len(el_pc)
        r_w = ConductanceCalc._compute_intfc_curvature(gamma, el_pc, nw_mask)

        A_w = ConductanceCalc._wetting_area(r_w, el_G)

        A_n = np.zeros_like(A_w)

        A_n[nw_mask] = (el_A_tot - A_w)[nw_mask]

        # Mean hydraulic radii
        R_n = 0.5 * (el_rad + np.sqrt(A_n / math.pi))
        R_w = 0.5 * (el_rad + np.sqrt(el_A_tot / math.pi))
        assert np.all(R_n >= 0)
        assert np.all(R_w >= 0)
        assert np.all(el_pc[nw_mask] > gamma / el_rad[nw_mask])
        assert np.all(A_n >= 0)

        k_n = np.zeros_like(el_rad)
        k_w = np.zeros_like(el_rad)

        # Eq. 6,7,8  in G.Tora (2012)
        k_n[nw_mask] = (A_n / el_A_tot * ((R_w**2) * el_A_tot / el_len / (8 * mu_n)))[nw_mask]

        # Eq. 5 in G.Tora (2012)
        k_w[nw_mask] = ((r_w ** 2) * A_w / el_len / (8 * mu_w * beta))[nw_mask]

        # Eq. 6,7,8  in G.Tora (2012)
        k_w[w_mask] = ((R_w**2) * el_A_tot / el_len / (8 * mu_w))[w_mask]

        assert np.all(R_w[w_mask] > 0.0)
        assert np.all(el_A_tot[w_mask] > 0.0)
        assert np.all(k_w[w_mask] > 0.0)
        assert np.all(k_n[nw_mask] > 0.0)
        assert np.all(k_n >= 0.0)
        assert np.all(k_w > 0.0)

        return k_n, k_w

    def compute(self):
        tubes = self.network.tubes
        pores = self.network.pores
        tubes.k_n[:], tubes.k_w[:] = self.compute_conductances(sim_settings['fluid_properties'], tubes.r, tubes.l,
                                                               tubes.G, tubes.A_tot, tubes.p_c, tubes.invaded)

        if sim_settings['simulation']['pores_have_conductances']:
            nw_mask_pores = ((self.network.pores.invaded == 1) & (self.network.pore_domain_type == DOMAIN))
            invasion_status = np.zeros(self.network.nr_p, dtype=np.int32)
            invasion_status[nw_mask_pores] = 1

            pores.k_n[:], pores.k_w[:] = self.compute_conductances(sim_settings['fluid_properties'], pores.r, pores.l,
                                                                   pores.G, pores.A_tot, pores.p_c, invasion_status)
            self._harmonic_mean_tube_conductivities()

    def compute_fully_wetting(self):
        invasion_status = np.zeros(self.network.nr_t, dtype=np.int32)
        tubes = self.network.tubes
        tubes.k_n[:], tubes.k_w[:] = self.compute_conductances(sim_settings['fluid_properties'], tubes.r, tubes.l, tubes.G,
                                       tubes.A_tot, tubes.p_c, invasion_status)

        if sim_settings['simulation']['pores_have_conductances']:
            pores = self.network.pores
            invasion_status = np.zeros(self.network.nr_p, dtype=np.int32)
            pores.k_n[:], pores.k_w[:] = self.compute_conductances(sim_settings['fluid_properties'], pores.r, pores.l,
                                                                   pores.G, pores.A_tot, pores.p_c, invasion_status)
            self._harmonic_mean_tube_conductivities()

