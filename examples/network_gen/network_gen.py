"""
Creates and saves a network, saving output to VTK to be viewed
"""

from pypnm.porenetwork.network_factory import unstructured_network_delaunay

network = unstructured_network_delaunay(nr_pores=100000, quasi_2d=False, body_throat_corr_param=0.7)
network.save("network.pkl")
network.export_to_vtk("network.vtp")
network.write_network_statistics()

import numpy as np
pore_rad_1 = network.pores.r[network.edgelist[:, 0]]
pore_rad_2 = network.pores.r[network.edgelist[:, 1]]

tube_rad = network.tubes.r
print "Correlation coefficient matrices for body-throat radii"
print np.corrcoef(np.vstack([pore_rad_1, tube_rad]))
print np.corrcoef(np.vstack([pore_rad_2, tube_rad]))
