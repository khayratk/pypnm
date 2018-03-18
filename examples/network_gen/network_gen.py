"""
Creates and saves a network, saving output to VTK to be viewed
"""

from pypnm.porenetwork.network_factory import unstructured_network_delaunay

network = unstructured_network_delaunay(nr_pores=400000, body_throat_corr_param=2.)
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

lx = max(network.pores.x) - min(network.pores.x)
ly = max(network.pores.y) - min(network.pores.y)
lz = max(network.pores.z) - min(network.pores.z)

vol = lx*ly*lz

print "Average pore radius", np.mean(network.pores.r), np.max(network.pores.r), np.min(network.pores.r)
print "Average tube radius", np.mean(network.tubes.r), np.max(network.tubes.r), np.min(network.tubes.r)
print "Volumes of pores and tubes", network.total_vol/vol, network.total_pore_vol/network.total_vol, network.total_throat_vol/network.total_vol

print "l_tot/l", np.mean(network.tubes.l_tot/network.tubes.l)