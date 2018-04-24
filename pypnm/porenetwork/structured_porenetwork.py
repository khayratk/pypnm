from numpy.ctypeslib import ndpointer
from ctypes import POINTER
import ctypes
import os

from pypnm.porenetwork.constants import *
from pores import Pores
from tubes import Tubes
from porenetwork import PoreNetwork
import numpy as np

path = os.path.dirname(os.path.abspath(__file__))
os.system('gcc -fPIC -shared -o ' + path + '/pn_lib.so ' + path + '/pn_lib.c')

try:
    lib = ctypes.cdll.LoadLibrary(path + '/pn_lib.so')
except:
    path = os.path.dirname(os.path.abspath(__file__))
    os.system('gcc -fPIC -shared -o ' + path + '/pn_lib.so ' + path + '/pn_lib.c')
    lib = ctypes.cdll.LoadLibrary(path + '/pn_lib.so')

init_t_adj_c = lib.init_t_adj
init_t_adj_c.restype = None
init_t_adj_c.argtypes = [ndpointer(ctypes.c_int32), ndpointer(ctypes.c_int32), ndpointer(ctypes.c_int32),
                         ctypes.c_int32, POINTER(ctypes.c_int32)]


class StructuredPoreNetwork(PoreNetwork):
    """Class which generates regular structured 3D pore networks"""

    def __init__(self, nr_pores_vec, dist_p2p, media_type="bimodal", periodic=False):
        """
         Args:
            nr_pores_vec: 3-element array with number of pores along each direction
            dist_p2p: Pore-center to pore-center distance
        """
        self.nr_p = nr_pores_vec[0] * nr_pores_vec[1] * nr_pores_vec[2]
        self.network_type = "structured"

        self.pore_domain_type = np.zeros(self.nr_p, dtype=np.int8)
        self.indices = np.asarray(range(self.nr_p))

        self.dist_p2p = dist_p2p

        self.pores = Pores(self.nr_p)

        self.Nx = nr_pores_vec[0]
        self.Ny = nr_pores_vec[1]
        self.Nz = nr_pores_vec[2]

        self.Lx = self.Nx * dist_p2p
        self.Ly = self.Ny * dist_p2p
        self.Lz = self.Nz * dist_p2p

        self.inface = WEST
        self.outface = EAST

        self.init_pore_topology(periodic)

        self._init_pore_inout_type(inface=self.inface, outface=self.outface)

        self.media_type = media_type

        self.nr_t = self.count_num_tubes()
        self.tubes = Tubes(self.nr_t)

        self.init_element_radii()

        self.pores.init_vol()
        self.tubes.init_area()
        self.pores.init_area()

        self.init_tube_length()
        self.tubes.init_vol()

        # Initialize pores
        self.pores.invaded[:] = WETT
        self.tubes.invaded[:] = WETT

        self._create_ngh_elements_arrays()

        self._create_helper_properties()

        self._fix_tubes_larger_than_ngh_pores()

        if self.nr_p > 1:
            assert (len(self.pi_in) == len(self.pi_out))
        self.check_index_arrays()

    def check_index_arrays(self):
        for FACE in FACES:
            assert (len(self.pi_list_face[FACE]) > 0.0)

        for FACE_1, FACE_2 in zip([WEST, SOUTH, BOTTOM], [EAST, NORTH, TOP]):
            assert (len(self.pi_list_face[FACE_1]) == len(self.pi_list_face[FACE_2]))

    def init_pore_topology(self, periodic):
        self.init_pore_positions()
        self.init_adj(periodic)
        self.init_pore_face()

    def init_pore_positions(self):
        """ Initializes pore positions """
        for k in range(self.Nz):
            for j in range(self.Ny):
                for i in range(self.Nx):
                    ind = self.ijk_to_ind(i, j, k)
                    self.pores.x[ind] = float(i) * self.dist_p2p
                    self.pores.y[ind] = float(j) * self.dist_p2p
                    self.pores.z[ind] = float(k) * self.dist_p2p

    def init_pore_face(self):
        """ Initialize the face location of each pore. A pore is either
        located on north, south, east, west, top or bottom face of the domain
        OR is not on the face of the domain """

        pore_face = np.zeros(self.nr_p, dtype=np.int32)
        pore_face[:] = DOMAIN

        self.pi_list_face = dict()

        for FACE, ind in zip([WEST, EAST], [0, self.Nx-1]):
            self.set_array_along_x_plane(pore_face, FACE, i=ind)
            self.pi_list_face[FACE] = (pore_face == FACE).nonzero()[0]

        for FACE, ind in zip([SOUTH, NORTH], [0, self.Ny-1]):
            self.set_array_along_y_plane(pore_face, FACE, j=ind)
            self.pi_list_face[FACE] = (pore_face == FACE).nonzero()[0]

        for FACE, ind in zip([BOTTOM, TOP], [0, self.Nz-1]):
            self.set_array_along_z_plane(pore_face, FACE, k=ind)
            self.pi_list_face[FACE] = (pore_face == FACE).nonzero()[0]

        self.check_index_arrays()

    def init_element_radii(self):
        """ Initialize the radii of the tubes and pores """

        assert (self.media_type in ["consolidated", "unconsolidated", "bimodal"])

        if self.media_type == "consolidated":
            # Parameters chosen to match  Jerauld and Salter 1990 (Page 112)
            self.pores.init_radii_beta(0.25 + 1, 1.5 + 1, 20e-6, 75e-6)
            self.tubes.init_radii_beta(0.5 + 1, 1.0 + 1, 1e-6, 25e-6)
            self.pores.l[:] = self.pores.r[:]

        if self.media_type == "unconsolidated":
            # Parameters chosen to match  Jerauld and Salter 1990 (Page 112)
            self.pores.init_radii_beta(1.5 + 1, 0.5 + 1, 40e-6, 64e-6)
            self.tubes.init_radii_beta(0.5 + 1, 0.5 + 1, 15e-6, 40e-6)
            self.pores.l[:] = self.pores.r[:]

        if self.media_type == "bimodal":
            # Parameters chosen to match Tsakiroglou & Payatakes 2000
            self.pores.init_radii_bimodal_beta(1.5 + 1, 0.5 + 1, 20e-6, 29e-6, 1.5 + 1, 0.5 + 1, 40e-6, 64e-6, 0.5)
            self.tubes.init_radii_bimodal_beta(0.5 + 1, 0.5 + 1, 1e-6, 15e-6, 0.5 + 1, 0.5 + 1, 25e-6, 40e-6, 0.5)
            self.pores.l[:] = self.pores.r[:]

    def init_adj(self, periodic):
        """
        Initializes pore-to-pore, pore-to-tube and tube-to-pore adjacency list
        """
        self.init_p2p_adj(periodic)
        self.init_t_adj()

    def init_p2p_adj(self, periodic):
        """ Initializes pore-to-pore adjacency list. """
        Nx = self.Nx
        Ny = self.Ny
        Nz = self.Nz

        self.p_adj = -np.ones([self.pores.nr, 6], dtype=np.int32)

        for k in xrange(self.Nz):
            for j in xrange(self.Ny):
                for i in xrange(self.Nx):
                    ind = self.ijk_to_ind(i, j, k)

                    # Sets the indices of the neighboring pores.
                    # Non-existing connectivities at the boundaries are marked with -1.
                    if periodic:
                        self.p_adj[ind, :] = [(ind - Nx * Ny) * (k > 0) + (ind + Nx * (Ny) * (Nz - 1)) * (k == 0),
                                              (ind - Nx) * (j > 0) + (ind + Nx * (Ny - 1)) * (j == 0),
                                              (ind - 1) * (i > 0) - (i == 0),
                                              (ind + 1) * (i < (Nx - 1)) - (i == Nx - 1),
                                              (ind + Nx) * (j < (Ny - 1)) + (ind - Nx * (Ny - 1)) * (j == Ny - 1),
                                              (ind + Nx * Ny) * (k < (Nz - 1)) + (ind - Nx * (Ny) * (Nz - 1)) * (
                                                  k == Nz - 1)]

                    else:
                        self.p_adj[ind, :] = [(ind - Nx * Ny) * (k > 0) - (k == 0),
                                              (ind - Nx) * (j > 0) - (j == 0),
                                              (ind - 1) * (i > 0) - (i == 0),
                                              (ind + 1) * (i < (Nx - 1)) - (i == Nx - 1),
                                              (ind + Nx) * (j < (Ny - 1)) - (j == Ny - 1),
                                              (ind + Nx * Ny) * (k < (Nz - 1)) - (k == Nz - 1)]

                    #  Fix connectivities for 2-D or 1-D networks
                    if Ny == 1:
                        self.p_adj[ind, 1] = -1
                        self.p_adj[ind, 4] = -1

                    if Nx == 1:
                        self.p_adj[ind, 2] = -1
                        self.p_adj[ind, 3] = -1

                    if Nz == 1:
                        self.p_adj[ind, 0] = -1
                        self.p_adj[ind, 5] = -1

    def remove_connections_in_padj(self, i, j, k, indices_2):
        ind = np.array(self.ijk_to_ind(i, j, k), dtype=np.int)
        for ind2 in indices_2:
            self.p_adj[ind, ind2] = -1

    def init_t_adj(self):
        """
        Initializes pore-to-throat and throat-to-pore adjacency list.
        To be called after pore to pore adjacency list is initialized
        """

        self.p_adj.flags.writeable = False  # Lock p_adj array

        self.pt_adj = -np.ones([self.pores.nr, 6], dtype=np.int32)
        self.edgelist = -np.ones([self.count_num_tubes(), 2], dtype=np.int32)

        tube_count = ctypes.c_int32(0)
        init_t_adj_c(self.p_adj, self.pt_adj, self.edgelist, self.nr_p, ctypes.byref(tube_count))

        assert (tube_count.value == self.count_num_tubes())

    def set_array_along_z_plane(self, array, val, k):
        assert (self.Nz > k >= 0)
        for j in xrange(self.Ny):
            for i in xrange(self.Nx):
                ind = self.ijk_to_ind(i, j, k)
                array[ind] = val

    def set_array_along_y_plane(self, array, val, j):
        assert (self.Ny > j >= 0)
        for k in xrange(self.Nz):
            for i in xrange(self.Nx):
                ind = self.ijk_to_ind(i, j, k)
                array[ind] = val

    def set_array_along_x_plane(self, array, val, i):
        assert (self.Nx > i >= 0)
        for k in xrange(self.Nz):
            for j in xrange(self.Ny):
                ind = self.ijk_to_ind(i, j, k)
                array[ind] = val

    def ijk_to_ind(self, i, j, k):
        return i + self.Nx * j + self.Nx * self.Ny * k

    def init_tube_length(self):
        p_nghs_1, p_nghs_2 = self.edgelist[:, 0], self.edgelist[:, 1]

        self.tubes.l_tot[:] = np.copy(self.dist_p2p)
        self.tubes.l[:] = self.dist_p2p - self.pores.r[p_nghs_1] - self.pores.r[p_nghs_2]
        assert (np.all(self.tubes.l > 0.0))

    def count_num_tubes(self):
        return np.sum(self.p_adj > -1) / 2
