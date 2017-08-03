import numpy as np


def equal_length_args(func):
    def new_func(self, a, b):
        assert len(a) == len(b)
        func(self, a, b)
    return new_func


def unique_first_argument(func):
    def new_func(self, a, b):
        assert len(a) == len(np.unique(a)), "Values in first argument have to be unique"
        func(self, a, b)
    return new_func


class SimulationBoundaryCondition(object):
    """
    This class holds the boundary conditions for the dynamic network simulations.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.pi_list_nw_source = []
        self.q_list_nw_source = []

        self.pi_list_nw_sink = []
        self.q_list_nw_sink = []

        self.pi_list_w_source = []
        self.q_list_w_source = []

        self.pi_list_w_sink = []
        self.q_list_w_sink = []

        self.pi_list_inlet = []
        self.press_inlet_w = []
        self.press_inlet_nw = []

        self.pi_list_outlet = []
        self.press_outlet_w = []

        self.no_dirichlet = True

    def __repr__(self):
        return ("self.pi_list_nw_source \n" + str(self.pi_list_nw_source) + "\n"
                + "self.pi_list_nw_sink \n" + str(self.pi_list_nw_sink) + "\n"
                + "self.pi_list_w_source \n" + str(self.pi_list_w_source) + "\n"
                + "self.q_list_w_source \n" + str(self.q_list_w_source) + str(np.sum(self.q_list_w_source)) + "\n"
                + "self.pi_list_w_sink \n" + str(self.pi_list_w_sink) + "\n"
                + "self.q_list_w_sink \n" + str(self.q_list_w_sink) + str(self.q_list_w_sink.sum()) + "\n"
                + "self.pi_list_inlet \n" + str(self.pi_list_inlet) + "\n"
                + "self.press_inlet_w \n" + str(self.press_inlet_w) + "\n"
                + "self.press_inlet_nw \n" + str(self.press_inlet_nw) + "\n"
                + "self.pi_list_outlet \n" + str(self.pi_list_outlet) + "\n"
                + "self.press_outlet_w \n" + str(self.press_outlet_w) + "\n")

    def set_pressure_inlet(self, pi_list, p_wett, p_nwett):
        assert len(pi_list) == len(np.unique(pi_list))
        self.pi_list_inlet = np.copy(pi_list).astype(np.int)
        self.press_inlet_w = p_wett
        self.press_inlet_nw = p_nwett
        self.no_dirichlet = False

    def set_pressure_outlet(self, pi_list, p_wett):
        assert len(pi_list) == len(np.unique(pi_list))
        self.pi_list_outlet = np.copy(pi_list).astype(np.int)
        self.press_outlet_w = p_wett
        self.no_dirichlet = False

    @equal_length_args
    @unique_first_argument
    def set_nonwetting_sink(self, pi_list, q_list):
        assert np.all(q_list <= 0.0), str(q_list)
        self.pi_list_nw_sink = np.copy(pi_list).astype(np.int)
        self.q_list_nw_sink = np.copy(q_list)
        self.check_sink_source_nonoverlap()

    @equal_length_args
    @unique_first_argument
    def set_nonwetting_source(self, pi_list, q_list):
        assert np.all(q_list >= 0.0)
        self.pi_list_nw_source = np.copy(pi_list).astype(np.int)
        self.q_list_nw_source = np.copy(q_list)
        self.check_sink_source_nonoverlap()

    @equal_length_args
    @unique_first_argument
    def set_wetting_sink(self, pi_list, q_list):
        assert np.all(np.asarray(q_list) <= 0.0)
        self.pi_list_w_sink = np.copy(pi_list).astype(np.int)
        self.q_list_w_sink = np.copy(q_list)
        self.check_sink_source_nonoverlap()

    @equal_length_args
    @unique_first_argument
    def set_wetting_source(self, pi_list, q_list):
        assert np.all(q_list >= 0.0)
        self.pi_list_w_source = np.copy(pi_list).astype(np.int)
        self.q_list_w_source = np.copy(q_list)
        self.check_sink_source_nonoverlap()

    def check_sink_source_nonoverlap(self):
        assert len(np.intersect1d(self.pi_list_nw_source, self.pi_list_nw_sink)) == 0
        assert len(np.intersect1d(self.pi_list_w_source, self.pi_list_w_sink)) == 0

    def mass_balance(self):
        return np.sum(self.q_list_w_sink) + np.sum(self.q_list_w_source) + np.sum(self.q_list_nw_sink) + np.sum(self.q_list_nw_source)