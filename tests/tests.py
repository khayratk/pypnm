from pypnm.attribute_calculators.pc_computer import CapillaryPressureComputer
from pypnm.percolation.invasion_percolation import InvasionPercolator
from pypnm.porenetwork import component
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

def test_set_tubes_and_pore_radii():
    network = cube_network_27(N=10)
    network.set_radius_pores(network.pi_in, np.max(network.pores.r))
    ti_list = component.tubes_within_pore_set(network, network.pi_in)
    network.set_radius_tubes(ti_list, np.max(network.tubes.r))


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
