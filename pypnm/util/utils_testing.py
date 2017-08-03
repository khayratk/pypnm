import numpy as np
from pypnm.attribute_calculators.pc_computer import CapillaryPressureComputer
from pypnm.percolation.invasion_percolation import InvasionPercolator
from pypnm.porenetwork.entry_press_computer import EntryPressureComputer
from pypnm.porenetwork.saturation_computer import QuasiStaticSaturationComputer


def run_ip_algorithm(network, sat):
    network.set_inlet_pores_invaded_and_connected()

    pc_comp = CapillaryPressureComputer(network)
    sat_comp = QuasiStaticSaturationComputer(network)
    pe_comp = EntryPressureComputer()

    ip = InvasionPercolator(network, pe_comp, sat_comp, pc_comp)
    pc_comp.compute()

    ip.invasion_percolation_drainage(sat)
    pc_comp.compute()

    assert np.all(network.pores.invaded == network.pores.connected)
    assert np.all(network.pores.p_c[network.pores.invaded == 1] > 0.0)

    # Create gradient for capillary pressure
    network.pores.p_c[:] = 5*pc_comp.p_c - network.pores.x/np.max(network.pores.x)*pc_comp.p_c - \
                                           network.pores.y/np.max(network.pores.y)*pc_comp.p_c - \
                                           network.pores.z/np.max(network.pores.z)*pc_comp.p_c
    network.pores.p_c[network.pores.invaded != 1] = 0.0

    pc_comp.set_pc_invaded_tubes()

    assert np.all(network.pores.p_c[network.pores.connected == 1] > 0.0)
    assert np.all(network.tubes.p_c[network.tubes.invaded == 1] > 0.0)
