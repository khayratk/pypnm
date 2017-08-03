from pypnm.porenetwork.network_factory import *
import numpy as np


def test_distribute_throat_volume():
    N = 10
    network = cube_network(N=N)
    total_vol_before = network.total_vol
    network.distribute_throat_volume_to_neighboring_pores()
    assert np.isclose(network.total_vol, total_vol_before, atol=1.e-20, rtol=1.e-15)
    assert network.total_throat_vol != network.total_pore_vol
    assert network.total_throat_vol == 0.0
