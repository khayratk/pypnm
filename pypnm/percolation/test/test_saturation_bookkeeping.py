from pypnm.attribute_calculators.pc_computer import CapillaryPressureComputer
from pypnm.percolation.invasion_percolation import InvasionPercolator
from pypnm.porenetwork.entry_press_computer import EntryPressureComputer
from pypnm.porenetwork.network_factory import cube_network
from pypnm.porenetwork.saturation_computer import QuasiStaticSaturationComputer
import numpy as np
from pypnm.porenetwork.constants import *


def test_saturation_bookkeeping():
    """
    Keeps Capillary pressure constant during drainage and imbibition.
    """
    network = cube_network(N=10)
    network.set_inlet_pores_invaded_and_connected()

    network.set_zero_volume_pores(network.pi_in)

    sat_comp = QuasiStaticSaturationComputer(network)
    pe_comp = EntryPressureComputer()

    pc_comp = CapillaryPressureComputer(network)
    pc_comp.p_c = 5000000.0
    pc_comp.compute()

    ip = InvasionPercolator(network, pe_comp, sat_comp, pc_comp)

    sat_target_ip = 0.5
    sat_after_ip = ip.invasion_percolation_drainage(sat_target_ip)
    print sat_target_ip, sat_after_ip, sat_comp.sat_nw_conn()
    assert (np.isclose(sat_after_ip, sat_comp.sat_nw_conn(), rtol=1.e-2)), "%f %f"%(sat_after_ip, sat_comp.sat_nw_conn())

    sat_target_ip = 0.99
    sat_after_ip = ip.invasion_percolation_drainage(sat_target_ip)
    print sat_target_ip, sat_after_ip, sat_comp.sat_nw_conn()
    assert (np.isclose(sat_after_ip, sat_comp.sat_nw_conn(), rtol=1.e-2)), "%f %f"%(sat_after_ip, sat_comp.sat_nw_conn())

    sat_target_ip = 0.1
    sat_after_ip = ip.invasion_percolation_imbibition(sat_target_ip)
    print sat_target_ip, sat_after_ip, sat_comp.sat_nw_conn()

    assert (np.isclose(sat_after_ip, sat_comp.sat_nw_conn(), rtol=1.e-2)), "%f %f"%(sat_after_ip, sat_comp.sat_nw_conn())

    sat_target_ip = 0.1
    sat_after_ip = ip.invasion_percolation_imbibition(sat_target_ip)
    print sat_target_ip, sat_after_ip, sat_comp.sat_nw_conn()
    assert (np.isclose(sat_after_ip, sat_comp.sat_nw_conn(), rtol=1.e-2)), "%f %f"%(sat_after_ip, sat_comp.sat_nw_conn())

    pc_comp.p_c = 5000000.0
    pc_comp.compute()

    sat_target_ip = 0.40
    sat_after_ip = ip.invasion_percolation_drainage(sat_target_ip)
    print sat_target_ip, sat_after_ip, sat_comp.sat_nw_conn()
    assert (np.isclose(sat_after_ip, sat_comp.sat_nw_conn(), rtol=1.e-2)), "%f %f"%(sat_after_ip, sat_comp.sat_nw_conn())

    pc_comp.p_c = 5000000.0
    pc_comp.compute()

    sat_target_ip = 0.95
    sat_after_ip = ip.invasion_percolation_drainage(sat_target_ip, target_type=INVADED)
    print sat_target_ip, sat_after_ip, sat_comp.sat_nw()
    assert (np.isclose(sat_after_ip, sat_comp.sat_nw(), rtol=1.e-2))