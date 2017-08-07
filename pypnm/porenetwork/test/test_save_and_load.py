from pypnm.attribute_calculators.pc_computer import CapillaryPressureComputer
from pypnm.percolation.invasion_percolation import InvasionPercolator
from pypnm.porenetwork.entry_press_computer import EntryPressureComputer
from pypnm.porenetwork.network_factory import *
from pypnm.porenetwork.saturation_computer import QuasiStaticSaturationComputer


def test_save_and_load_network():
    network = cube_network_27(N=20)
    network.set_inlet_pores_invaded_and_connected()

    sat_comp = QuasiStaticSaturationComputer(network)
    pe_comp = EntryPressureComputer()
    pc_comp = CapillaryPressureComputer(network)
    pc_comp.compute()
    ip = InvasionPercolator(network, pe_comp, sat_comp, pc_comp)

    sat_target_ip = 0.40
    ip.invasion_percolation_drainage(sat_target_ip)
    pc_comp.compute()
    assert sat_comp.sat_nw_conn() > 0.4

    network.save("my_network.pkl")

    network_2 = network.load("my_network.pkl")
    sat_comp_2 = QuasiStaticSaturationComputer(network_2)
    pe_comp_2 = EntryPressureComputer()
    pc_comp_2 = CapillaryPressureComputer(network_2)

    ip_2 = InvasionPercolator(network_2, pe_comp_2, sat_comp_2, pc_comp_2)

    assert sat_comp_2.sat_nw_conn() > 0.4

    sat_target_ip = 0.70
    ip.invasion_percolation_drainage(sat_target_ip)
    assert sat_comp.sat_nw_conn() > 0.7

    assert not np.all(network.pores.invaded == network_2.pores.invaded)
    assert sat_comp_2.sat_nw_conn() < 0.5

    sat_target_ip = 0.70
    ip_2.invasion_percolation_drainage(sat_target_ip)
    assert np.all(network.pores.invaded == network_2.pores.invaded)


