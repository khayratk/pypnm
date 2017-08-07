from pypnm.porenetwork.network_factory import *
from pypnm.porenetwork.relperm import SimpleRelPermComputer
from scipy.stats.mstats import hdquantiles


def test_relperm_xy():
    network = cube_network(10)
    relperm_calculator = SimpleRelPermComputer(network)
    relperm_calculator.compute()
    print np.max(network.tubes.k_w)
    print np.min(network.tubes.k_w)
    print hdquantiles(network.tubes.k_w, prob=[0.01, 0.5, 0.99])


if __name__ == "__main__":
    test_relperm_xy()
