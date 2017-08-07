from pypnm.attribute_calculators.pc_computer import CapillaryPressureComputer
from pypnm.percolation.invasion_percolation import InvasionPercolator
from pypnm.porenetwork.entry_press_computer import EntryPressureComputer
from pypnm.porenetwork.network_factory import cube_network
from pypnm.porenetwork.saturation_computer import QuasiStaticSaturationComputer
import numpy as np


def test_saturation_bookkeeping():
    """
    Keeps Capillary pressure constant during drainage and imbibition.

    """
    network = cube_network(N=25)
    network.set_inlet_pores_invaded_and_connected()

    sat_comp = QuasiStaticSaturationComputer(network)
    pe_comp = EntryPressureComputer()

    pc_start = max(network.tubes.p_c)
    assert np.all(network.tubes.p_c == pc_start)

    pc_comp = CapillaryPressureComputer(network)
    pc_comp.p_c = 5000000.0
    pc_comp.compute()

    ip = InvasionPercolator(network, pe_comp, sat_comp, pc_comp)

    sat_target_ip = 0.5
    sat_after_ip = ip.invasion_percolation_drainage(sat_target_ip)
    assert (np.isclose(sat_after_ip, sat_comp.sat_nw_conn())), "%f %f"%(sat_after_ip, sat_comp.sat_nw_conn())

    sat_target_ip = 0.99
    sat_after_ip = ip.invasion_percolation_drainage(sat_target_ip)
    assert (np.isclose(sat_after_ip, sat_comp.sat_nw_conn())), "%f %f"%(sat_after_ip, sat_comp.sat_nw_conn())

    sat_target_ip = 0.25
    sat_after_ip = ip.invasion_percolation_imbibition(sat_target_ip)
    assert (np.isclose(sat_after_ip, sat_comp.sat_nw_conn())), "%f %f"%(sat_after_ip, sat_comp.sat_nw_conn())

    pc_comp.p_c = 5000000.0
    pc_comp.compute()

    sat_target_ip = 0.40
    sat_after_ip = ip.invasion_percolation_drainage(sat_target_ip)
    print sat_after_ip, sat_comp.sat_nw_conn()
    assert (np.isclose(sat_after_ip, sat_comp.sat_nw_conn())), "%f %f"%(sat_after_ip, sat_comp.sat_nw_conn())

    pc_comp.p_c = 5000000.0
    pc_comp.compute()

    sat_target_ip = 0.77
    sat_after_ip = ip.invasion_percolation_drainage(sat_target_ip)
    print sat_after_ip, sat_comp.sat_nw_conn()
    assert (np.isclose(sat_after_ip, sat_comp.sat_nw_conn()))