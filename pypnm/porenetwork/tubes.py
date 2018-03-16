import numpy as np
from network_elements import NetworkElements


class Tubes(NetworkElements):
    """ Stores geometry and position of tubes"""

    def __init__(self, nr_t):
        self.nr = nr_t

        # List of allowed tube (i.e. edge) attributes
        self.var_names_float = ["r", "l", "vol", "G", "A_tot", "p_c", "k_w", "k_n", "sat", "l_tot", "l_p1", "l_p2"]
        self.var_names_int = ["invaded", "connected", "bbone"]
        self.var_names = self.var_names_float + self.var_names_int

        self._dict_float = {key: None for key in self.var_names_float}
        self._dict_int = {key: None for key in self.var_names_int}

        self.G[:] = 1.0 / 16.0  # Default shape parameter for a square

        self._init_field_groups()

    def _init_field_groups(self):
        self.geometry_fields = [self.r, self.l, self.vol, self.G, self.A_tot, self.l_tot]
        self.status_fields = [self.invaded, self.connected, self.bbone]
        self.conductance_fields = [self.k_w, self.k_n]

    def init_vol(self):
        """
        Compute tubes volumes based on area and length.
        """
        if len(self.l) > 0:
            assert (np.sum(self.l) > 0)
            assert (np.sum(self.A_tot) > 0)
            self.vol[:] = self._compute_volume(self.A_tot[:], self.l[:])

    @staticmethod
    def _compute_volume(a_tot, len):
        return a_tot * len

    def append_tubes(self, l, r, G=None, l_tot=None):
        """
        Adds tubes to Tubes object

        Parameters
        ----------
        l: ndarray
            length of added tubes$
        r: ndarray
            radii of added tubes
        G: ndarray, optional
            shape factor of added tubes
        l_tot: ndarray
            pore to pore distance of added tubes
        """
        nr_new_tubes = len(l)

        for var_name in self.var_names:
            if (var_name in self._dict_float) and (self._dict_float[var_name] is not None):
                self._dict_float[var_name] = np.append(self._dict_float[var_name],
                                                    np.zeros(nr_new_tubes, dtype=self._dict_float[var_name].dtype))

            if (var_name in self._dict_int) and (self._dict_int[var_name] is not None):
                self._dict_int[var_name] = np.append(self._dict_int[var_name],
                                                    np.zeros(nr_new_tubes, dtype=self._dict_int[var_name].dtype))

        ti_new_tubes = np.arange(self.nr, self.nr + nr_new_tubes, dtype=np.int32)

        self.l[ti_new_tubes] = l
        self.r[ti_new_tubes] = r

        if G is None:
            self.G[ti_new_tubes] = np.mean(self.G)
        else:
            self.G[ti_new_tubes] = G

        if l_tot is None:
            self.l_tot[ti_new_tubes] = self.l[:]
        else:
            self.l_tot[ti_new_tubes] = l_tot

        self.A_tot[ti_new_tubes] = self.compute_area(self.r[ti_new_tubes], self.G[ti_new_tubes])
        self.vol[ti_new_tubes] = self._compute_volume(self.A_tot[ti_new_tubes], self.l[ti_new_tubes])
        self._init_field_groups()

        self.nr += nr_new_tubes

    def remove_tubes(self, ti_list):
        """
        Remove selected tubes from Tubes object

        Parameters
        ----------
        ti_list: ndarray
            indices of tubes to delete

        """
        for var_name in self.var_names:
            if (var_name in self._dict_float) and (self._dict_float[var_name] is not None):
                self._dict_float[var_name] = np.delete(self._dict_float[var_name], ti_list)

            if (var_name in self._dict_int) and (self._dict_int[var_name] is not None):
                self._dict_int[var_name] = np.delete(self._dict_int[var_name], ti_list)

        self.nr -= len(ti_list)
        self._init_field_groups()

    def select_tubes(self, ti_list):
        """
        Returns a new Tubes object with the selected ti_list

        Parameters
        ----------
        ti_list: ndarray
            indices of tubes to select
        """
        tubes_new = Tubes(len(ti_list))

        for name in self._dict_float:
            if self._dict_float[name] is not None:
                getattr(tubes_new, name)[:] = getattr(self, name)[ti_list]

        for name in self._dict_int:
            if self._dict_int[name] is not None:
                getattr(tubes_new, name)[:] = getattr(self, name)[ti_list]

        return tubes_new