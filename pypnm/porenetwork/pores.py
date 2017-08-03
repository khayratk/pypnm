import numpy as np
from network_elements import NetworkElements


class Pores(NetworkElements):
    """ Stores geometry and position of pores"""

    def __init__(self, nr_p):
        self.nr = nr_p

        self.var_names_float = ["x", "y", "z", "r", "l", "vol", "G", "A_tot", "p_n", "p_w", "p_c", "k_w", "k_n", "sat"]
        self.var_names_int = ["invaded", "connected", "bbone"]
        self.var_names = self.var_names_float + self.var_names_int

        self._dict_float = {key: None for key in self.var_names_float}
        self._dict_int = {key: None for key in self.var_names_int}

        self.G[:] = 1.0 / 16.0  # Default shape parameter for a square

        self.init_field_groups()

    def init_field_groups(self):
        self.location_fields = [self.x, self.y, self.z]
        self.geometry_fields = [self.r, self.vol, self.G, self.A_tot, self.l]
        self.geom_and_loc_fields = self.geometry_fields + self.location_fields
        self.status_fields = [self.invaded, self.connected, self.bbone]
        self.pressure_fields = [self.p_n, self.p_w, self.p_c]

    def init_vol(self):
        self.vol[:] = self.compute_volume(self.r)

    @staticmethod
    def compute_volume(r):
        return 8 * r ** 3

    def append_pores(self, x, y, z, r, G=None):
        nr_new_pores = len(x)

        if G is None:
            G = np.mean(self.G)

        for var_name in self.var_names:
            if (var_name in self._dict_float) and (self._dict_float[var_name] is not None):
                self._dict_float[var_name] = np.append(self._dict_float[var_name],
                                                    np.zeros(nr_new_pores, dtype=self._dict_float[var_name].dtype))

            if (var_name in self._dict_int) and (self._dict_int[var_name] is not None):
                self._dict_int[var_name] = np.append(self._dict_int[var_name],
                                                    np.zeros(nr_new_pores, dtype=self._dict_int[var_name].dtype))

        pi_new_pores = np.arange(self.nr, self.nr + nr_new_pores, dtype=np.int32)

        self.x[pi_new_pores] = x
        self.y[pi_new_pores] = y
        self.z[pi_new_pores] = z
        self.r[pi_new_pores] = r
        self.l[pi_new_pores] = r
        self.G[pi_new_pores] = G

        self.A_tot[pi_new_pores] = self.compute_area(self.r[pi_new_pores], self.G[pi_new_pores])
        self.vol[pi_new_pores] = self.compute_volume(self.r[pi_new_pores])
        self.init_field_groups()

        self.nr += nr_new_pores

    def remove_pores(self, pore_indices):
        for var_name in self.var_names:
            if (var_name in self._dict_float) and (self._dict_float[var_name] is not None):
                self._dict_float[var_name] = np.delete(self._dict_float[var_name], pore_indices)

            if (var_name in self._dict_int) and (self._dict_int[var_name] is not None):
                self._dict_int[var_name] = np.delete(self._dict_int[var_name], pore_indices)

        self.init_field_groups()

    def select_pores(self, pi_list):
        """
        Returns a new Pores object with the selected ti_list
        """
        pores_new = Pores(len(pi_list))

        for name in self._dict_float:
            if self._dict_float[name] is not None:
                getattr(pores_new, name)[:] = getattr(self, name)[pi_list]

        for name in self._dict_int:
            if self._dict_int[name] is not None:
                getattr(pores_new, name)[:] = getattr(self, name)[pi_list]

        return pores_new