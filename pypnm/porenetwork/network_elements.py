import numpy as np


class NetworkElements(object):
    @staticmethod
    def compute_area(r, G):
        return r ** 2 / (4 * G)

    def init_area(self):
        assert np.all(self.r > 0)
        self.A_tot[:] = self.compute_area(self.r, self.G)
        assert np.all(self.A_tot > 0.0)

    def init_radii_beta(self, alpha, beta, r_min, r_max):
        """
        Initialize pore radii according to a  beta distribution

        Parameters
        ----------
        alpha: float
            alpha parameter in beta distribution
        beta: float
            beta parameter in beta distribution
        r_min: float
            Minimum radius
        r_max: float
            Maximum radius
        """

        self.r[:] = np.random.beta(alpha, beta, size=self.nr)
        self.r[:] = self.r * (r_max - r_min) + r_min

    def init_radii_bimodal_beta(self, alpha_1, beta_1, r_min_1, r_max_1, alpha_2, beta_2, r_min_2, r_max_2, cont_factor):
        """
        Initialize pore radii according to a bimodal distribution created by adding two  beta distributions

        Parameters
        ----------
        alpha_1, alpha_2: float
            alpha parameters in beta distributions
        beta_1, beta_2: float
            beta parameters in beta distributions
        r_min_1, r_min_2: float
            Minimum radii
        r_max: float
            Maximum radius
        """
        cut_off_ind = int(cont_factor*self.nr)

        self.r[0:cut_off_ind] = np.random.beta(alpha_1, beta_1, size=cut_off_ind)
        self.r[0:cut_off_ind] = self.r[0:cut_off_ind]*(r_max_1 - r_min_1) + r_min_1

        self.r[cut_off_ind:self.nr] = np.random.beta(alpha_2, beta_2, size=self.nr - cut_off_ind)
        self.r[cut_off_ind:self.nr] = self.r[cut_off_ind:self.nr]*(r_max_2 - r_min_2) + r_min_2

        np.random.shuffle(self.r)

    def __getattr__(self, name):
        if name in ["_dict_float", "_dict_int"]:
            raise AttributeError()

        try:
            if name in self._dict_float:
                if self._dict_float[name] is None:
                    self._dict_float[name] = np.zeros(self.nr)
                return self._dict_float[name]

            if name in self._dict_int:
                if self._dict_int[name] is None:
                    self._dict_int[name] = np.zeros(self.nr, dtype=np.int8)
                return self._dict_int[name]

            raise KeyError

        except KeyError:
            msg = "'{0}' object has no attribute '{1}'"
            raise AttributeError(msg.format(type(self).__name__, name))
