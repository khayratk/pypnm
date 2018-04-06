import numpy as np

from porenetwork import PoreNetwork

from pypnm.io_network.statoil_reader import StatoilReader

from pores import Pores
from tubes import Tubes
from pypnm.porenetwork.constants import *


class StatoilPoreNetwork(PoreNetwork):
    """ Class which generates networks from the Statoil file format"""

    def __init__(self, filename, fix_tubes=True):
        """
        Args:
            filename: filename of the statoil file without the "_node.dat" or "_link.dat" suffixes
        """
        print "Reading Statoil network from file...."
        statoil = StatoilReader(filename)
        print "Finished Reading Statoil network from file...."

        self.network_type = "unstructured"

        self.pores = Pores(statoil.nr_p)
        self.tubes = Tubes(statoil.nr_t)

        self.nr_p = statoil.nr_p
        self.nr_t = statoil.nr_t

        self.network_length = statoil.network_length
        self.network_width = statoil.network_width
        self.network_height = statoil.network_height

        self.periodic_tube_marker = np.zeros(self.nr_t).astype(int)

        self.pores.r[:] = statoil.pore_prop["r"]
        self.pores.l[:] = statoil.pore_prop["r"]

        assert np.all(self.pores.l > 0.0)

        self.pores.G[:] = statoil.pore_prop["G"]
        self.pores.vol[:] = statoil.pore_prop["vol"]
        self.pores.x[:], self.pores.y[:], self.pores.z[:] = statoil.pore_prop["x"], statoil.pore_prop["y"], statoil.pore_prop["z"]

        assert np.all(self.pores.vol > 0.0)

        self.edgelist = np.zeros_like(statoil.edgelist)
        self.pt_adj = np.zeros_like(statoil.pt_adj)
        self.p_adj = np.zeros_like(statoil.p_adj)

        self.edgelist[:] = statoil.edgelist
        self.pt_adj[:] = statoil.pt_adj
        self.p_adj[:] = statoil.p_adj

        assert np.all(self.edgelist[:, 0] > -1)
        assert np.all(self.edgelist[:, 1] > -1)

        self.tubes.r[:] = statoil.tube_prop["r"]
        self.tubes.G[:] = statoil.tube_prop["G"]
        self.tubes.l[:] = statoil.tube_prop["l"]
        self.tubes.l_tot[:] = statoil.tube_prop["l_tot"]
        self.tubes.vol[:] = statoil.tube_prop["vol"]
        self.tubes.l_p1[:] = statoil.tube_prop["l_p1"]
        self.tubes.l_p2[:] = statoil.tube_prop["l_p2"]

        self.pore_domain_type = np.zeros(self.nr_p, dtype=np.int8)
        self.pore_domain_type[:] = DOMAIN

        self.pi_in = np.copy(statoil.pi_in)
        self.pi_out = np.copy(statoil.pi_out)
        self.pi_domain = np.copy(statoil.pi_domain)

        self.inface = WEST
        self.outface = EAST

        self.pi_list_face = [None, None, None, None, None, None]
        self.pi_list_face[WEST] = self.pi_in[:]
        self.pi_list_face[EAST] = self.pi_out[:]

        self.pi_list_face[SOUTH] = np.sort(np.argsort(self.pores.y)[0:len(self.pi_in)])
        self.pi_list_face[NORTH] = np.sort(np.argsort(-self.pores.y)[0:len(self.pi_in)])
        self.pi_list_face[BOTTOM] = np.sort(np.argsort(self.pores.z)[0:len(self.pi_in)])
        self.pi_list_face[TOP] = np.sort(np.argsort(-self.pores.z)[0:len(self.pi_in)])

        assert np.all(self.pores.r > 0)
        assert np.all(self.pores.G > 0)
        assert np.all(self.pores.vol > 0)

        # check if there are pores with G=0 and replace with a mean G
        self._fix_tube_property(self.tubes.G, "G")
        self._fix_tube_property(self.tubes.l, "l")
        self._fix_tube_property(self.tubes.r, "r")
        self._fix_tube_property(self.tubes.l_tot, "l_tot")

        self.tubes.G[self.tubes.G <= 0.001 * np.mean(self.tubes.G)] = np.mean(self.tubes.G)
        self.tubes.r[self.tubes.r <= 0.001 * np.mean(self.tubes.r)] = np.mean(self.tubes.r)
        self.tubes.l[self.tubes.l <= 0.0] = np.mean(self.tubes.l)

        assert np.all(self.tubes.G > 0)
        assert np.all(self.tubes.l > 0)
        assert np.all(self.tubes.r > 0)

        self.pores.init_area()
        self.tubes.init_area()

        self._create_ngh_elements_arrays()

        if fix_tubes:
            self._fix_tubes_larger_than_ngh_pores()
        self._create_helper_properties()
        self.indices = np.asarray(range(self.nr_p))

        del statoil

