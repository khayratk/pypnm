from xml.etree import ElementTree as _ET

import igraph as ig
import numpy as np
from pypnm.porenetwork.constants import *
from pypnm.porenetwork.porenetwork import PoreNetwork
from pypnm.porenetwork.pores import Pores
from pypnm.porenetwork.tubes import Tubes


class OpenPnmReader(object):
    r"""
    Class for reading a Vtp file written by OpenPNM
    Copyright (c) 2014 OpenPNM Team
    """

    @classmethod
    def load(cls, filename):
        r"""
        Read in pore and throat data from a saved VTK file.
        Parameters
        ----------
        filename : string (optional)
            The name of the file containing the data to import.  The formatting
            of this file is outlined below.
        Returns
        -------
        If no Network object is supplied then one will be created and returned.
        If return_geometry is True, then a tuple is returned containing both
        the network and a geometry object.
        """
        net = {}

        filename = filename.rsplit('.')[0]
        tree = _ET.parse(filename + '.vtp')
        piece_node = tree.find('PolyData').find('Piece')

        # Extract connectivity
        conn_element = piece_node.find('Lines').find('DataArray')
        array = cls._element_to_array(conn_element, 2)
        net.update({'throat.conns': array})
        # Extract coordinates
        coord_element = piece_node.find('Points').find('DataArray')
        array = cls._element_to_array(coord_element, 3)
        net.update({'pore.coords': array})

        # Extract pore data
        for item in piece_node.find('PointData').iter('DataArray'):
            key = item.get('Name')
            element = key.split('.')[0]
            array = cls._element_to_array(item)
            propname = key.split('.')[1]
            net.update({element + '.' + propname: array})
        # Extract throat data
        for item in piece_node.find('CellData').iter('DataArray'):
            key = item.get('Name')
            element = key.split('.')[0]
            array = cls._element_to_array(item)
            propname = key.split('.')[1]
            net.update({element + '.' + propname: array})

        return net

    @staticmethod
    def _element_to_array(element, n=1):
        string = element.text
        dtype = element.get("type")
        array = np.fromstring(string, sep='\t')
        array = array.astype(dtype)
        if n is not 1:
            array = array.reshape(array.size // n, n)
        return array


class OpenPnmPoreNetwork(PoreNetwork):

    def __init__(self, filename, fix_tubes=True):
        """
        Args:
            filename: filename of the statoil file without the "_node.dat" or "_link.dat" suffixes
        """
        net = OpenPnmReader.load(filename)

        self.network_type = "unstructured"

        self.pores = Pores(len(net["pore.all"]))
        self.tubes = Tubes(len(net["throat.all"]))

        self.nr_p = self.pores.nr
        self.nr_t = self.tubes.nr

        self.periodic_tube_marker = np.zeros(self.nr_t).astype(int)

        self.pores.r[:] = net["pore.GenericNetwork_h9Iy1_equivalent_diameter"][:] / 2. * 1.e-6
        self.pores.l[:] = net["pore.GenericNetwork_h9Iy1_equivalent_diameter"][:] / 2. * 1.e-6

        assert np.all(self.pores.l > 0.0)

        self.pores.G[:] = 1. / 16.
        self.pores.vol[:] = net["pore.GenericNetwork_h9Iy1_volume"] * 1.e-18
        self.pores.x[:], self.pores.y[:], self.pores.z[:] = net["pore.coords"][:, 0] * 1.e-6, net["pore.coords"][:,
                                                                                              1] * 1.e-6, net[
                                                                                                              "pore.coords"][
                                                                                                          :, 2] * 1.e-6

        assert np.all(self.pores.vol > 0.0)

        self.edgelist = net["throat.conns"]

        G = ig.Graph(self.nr_p)
        G.add_edges(self.edgelist)
        self.pt_adj = [np.asarray(x) for x in G.get_inclist()]
        self.p_adj = [np.asarray(x) for x in G.get_adjlist()]

        assert len(self.pores.r) == self.nr_p

        assert np.all(self.edgelist[:, 0] > -1)
        assert np.all(self.edgelist[:, 1] > -1)

        self.tubes.r[:] = net["throat.GenericNetwork_h9Iy1_diameter"] / 2.0 * 1.e-6
        self.tubes.G[:] = 1. / 16.
        self.tubes.l[:] = net["throat.GenericNetwork_h9Iy1_length"] * 1.e-6
        self.tubes.l_tot[:] = net["throat.GenericNetwork_h9Iy1_total_length"] * 1.e-6
        self.tubes.vol[:] = np.pi * self.tubes.r ** 2 * self.tubes.l

        self.pore_domain_type = np.zeros(self.nr_p, dtype=np.int8)
        self.pore_domain_type[:] = DOMAIN

        n_bnd_pores = int(self.nr_p ** (2. / 3.))

        self.pi_list_face = [None, None, None, None, None, None]
        self.pi_list_face[WEST] = np.sort(np.argsort(self.pores.x)[0:n_bnd_pores])
        self.pi_list_face[EAST] = np.sort(np.argsort(-self.pores.x)[0:n_bnd_pores])
        self.pi_list_face[SOUTH] = np.sort(np.argsort(self.pores.y)[0:n_bnd_pores])
        self.pi_list_face[NORTH] = np.sort(np.argsort(-self.pores.y)[0:n_bnd_pores])
        self.pi_list_face[BOTTOM] = np.sort(np.argsort(self.pores.z)[0:n_bnd_pores])
        self.pi_list_face[TOP] = np.sort(np.argsort(-self.pores.z)[0:n_bnd_pores])

        self.pi_in = self.pi_list_face[WEST]
        self.pi_out = self.pi_list_face[EAST]
        self.inface = WEST
        self.outface = EAST

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
