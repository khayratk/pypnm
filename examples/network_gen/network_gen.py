"""
Creates and saves a network, saving output to VTK to be viewed
"""

from pypnm.porenetwork.network_factory import unstructured_network_delaunay

network = unstructured_network_delaunay(nr_pores=100000, quasi_2d=False, body_throat_corr_param=0.7)
network.save("network.pkl")
network.export_to_vtk("network.vtp")
network.write_network_statistics()
