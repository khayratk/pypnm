from pypnm.porenetwork.network_factory import *
from pypnm.porenetwork.relperm import SimpleRelPermComputer


def test_permeability_tensor():
    network = cube_network(10)
    fluid_properties = {"mu_n": 1e-8, "mu_w": 1000, "gamma": 1e-3}
    relperm_calculator = SimpleRelPermComputer(network, fluid_properties)
    K_int = relperm_calculator.absolute_permeability()
    print K_int


if __name__ == "__main__":
    test_permeability_tensor()
