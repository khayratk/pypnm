import numpy as np


class BasePEModel(object):
    @classmethod
    def piston_entry_pressure(cls, **kwargs):
        G = kwargs['G']
        r = kwargs['r']
        gamma = kwargs['gamma']

        assert np.all(G > 0.0)
        assert np.all(r > 0.0)

        # Taken from Tora 2012 Eq.(1)
        entry_pressure = gamma / r * (1.0 + 2.0 * (np.pi * G) ** 0.5)
        assert np.all(entry_pressure > 0.0)
        return entry_pressure

    @classmethod
    def snap_off_pressure(cls, **kwargs):
        gamma = kwargs['gamma']
        r = kwargs['r']

        return gamma / r


class JNModel(BasePEModel):
    """
    Implementation of local pc-sat pc_model as given in Joekar-Niasar JFM (2010)
    """
    @classmethod
    def pc_to_sat_func(cls, **kwargs):
        pore_rad = kwargs['r']
        p_c = kwargs['p_c']
        gamma = kwargs['gamma']

        exp_part = 1. - 2. * gamma / (p_c * pore_rad)
        sat = 1 + np.log(exp_part) / 6.83

        assert np.all(p_c > 0.0)
        assert np.all(exp_part > 0.0)
        assert np.all(sat <= 1.0)
        assert np.all(sat >= 0.0)

        return sat

    @classmethod
    def sat_to_pc_func(cls, **kwargs):
        # Joekar-Niasar JFM (2010) Eq. 3.7
        sat = kwargs['sat']
        pore_rad = kwargs['r']
        gamma = kwargs['gamma']

        exp_part = np.exp(-6.83 * (1. - sat))
        pc = 2 * gamma / (pore_rad * (1.0 - exp_part))
        assert (not np.isnan(np.sum(exp_part)))
        assert np.all(exp_part < 1.0)
        assert np.all(sat < 1.0)
        assert np.all(sat >= 0.0)

        np.all(pc >= 0.0)

        return pc


class ToraModel(BasePEModel):
    """
    Implementation of local pc-sat pc_model as given in Tora JFM (2012)
    """
    @classmethod
    def pc_to_sat_func(cls, **kwargs):
        p_c = kwargs['p_c']
        G = kwargs['G']
        gamma = kwargs['gamma']
        A_tot = kwargs['A_tot']

        r_w = gamma / p_c

        if isinstance(r_w, np.ndarray):
            r_w[p_c == 0] = 0.0

        A_w = (r_w ** 2) * (1. / (4 * G) - np.pi)
        A_nw = A_tot - A_w
        Sat_nw_loc = A_nw / A_tot

        if isinstance(Sat_nw_loc, np.ndarray):
            Sat_nw_loc[p_c < 0.01] = 0.0

        assert np.all(Sat_nw_loc >= 0.0), "Capillary pressure values are not consistant"

        return Sat_nw_loc


def throat_diameter_acharya(network, p_1, p_2, l_tot, n):
    r1, r2 = network.pores.r[p_1]/l_tot, network.pores.r[p_2]/l_tot
    sin = np.sin(np.pi/4.0)
    cos = np.cos(np.pi/4.0)

    rho1 = r1*sin/(1-r1*cos)**n
    rho2 = r2*sin/(1-r2*cos)**n

    r_t_norm = rho1*rho2*(rho1**(1./n) + rho2**(1./n))**(-n)

    return r_t_norm*l_tot