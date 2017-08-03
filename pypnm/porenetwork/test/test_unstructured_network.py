from scipy.stats import beta, randint

from pypnm.attribute_calculators.pc_computer import CapillaryPressureComputer
from pypnm.percolation.invasion_percolation import InvasionPercolator
from pypnm.porenetwork.entry_press_computer import EntryPressureComputer
from pypnm.porenetwork.saturation_computer import QuasiStaticSaturationComputer
from pypnm.porenetwork.unstructured_porenetwork import create_unstructured_network


def test_unstructured_network():
    """
    Generate an unstructured spatially random network using pdfs of pore-radius, tube_radius, coordination number,
    number of pores and domain dimension.
    """
    domain_length = 1e-4 * 20
    domain_size = [domain_length * 2, domain_length, domain_length]

    r_min, r_max = 20e-6, 75e-6
    pdf_pore_radius = beta(1.25, 1.5, loc=r_min, scale=(r_max - r_min))

    r_min, r_max = 1e-6, 25e-6
    pdf_tube_radius = beta(1.5, 2, loc=r_min, scale=(r_max - r_min))

    pdf_coord_number = randint(low=2, high=15)

    nr_pores = 5000

    network = create_unstructured_network(nr_pores, pdf_pore_radius=pdf_pore_radius,
                                          pdf_tube_radius=pdf_tube_radius,
                                          pdf_coord_number=pdf_coord_number,
                                          domain_size=domain_size)

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

    from pypnm.postprocessing.vtk_output import VtkWriter
    writer = VtkWriter(network)
    writer.write()