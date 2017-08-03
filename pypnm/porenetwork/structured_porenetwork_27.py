from structured_porenetwork import StructuredPoreNetwork
import igraph as ig
import numpy as np
from pores import Pores
from tubes import Tubes
from constants import *
import itertools


class StructuredPoreNetwork27(StructuredPoreNetwork):
    def __init__(self, nr_pores_vec, dist_p2p, media_type="bimodal", periodic=False):
        """
         Args:
            nr_pores_vec: 3-element array with number of pores along each direction
            dist_p2p: Pore-center to pore-center distance
        """
        self.network_type = "structured27"

        self.nr_p = nr_pores_vec[0] * nr_pores_vec[1] * nr_pores_vec[2]

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

        self.init_pore_positions()
        self.init_pore_face()

        self._init_pore_inout_type(inface=self.inface, outface=self.outface)

        self.media_type = media_type

        self.create_ngh_elements_arrays_27()

        self.nr_t = self.edgelist.shape[0]
        self.tubes = Tubes(self.nr_t)

        self.init_element_radii()

        self.pores.init_vol()
        self.tubes.init_area()
        self.pores.init_area()

        self.init_tube_length()
        self.tubes.init_vol()

        # Initiliaze pores
        self.pores.invaded[:] = WETT
        self.tubes.invaded[:] = WETT

        self._create_helper_properties()

        self._fix_tubes_larger_than_ngh_pores()

        if self.nr_p > 1:
            assert (len(self.pi_in) == len(self.pi_out))
        self.check_index_arrays()

    def get_igraph_representation(self, edgelist):
        G = ig.Graph()
        G.add_vertices(self.nr_p)
        G.add_edges(edgelist)
        return G

    def create_ngh_elements_arrays_27(self):
        nx = self.Nx
        ny = self.Ny
        nz = self.Nz

        self.ngh_pores = np.ones([self.nr_p], dtype=np.object)
        self.ngh_tubes = np.ones([self.nr_p], dtype=np.object)

        # Create ngh_tubes array
        for i, j, k in itertools.product(xrange(nx), xrange(ny), xrange(nz)):
            ngh_pores = list()

            ind = self.ijk_to_ind(i, j, k)
            for ip, jp, kp in itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]):
                i_ngh = i + ip
                j_ngh = j + jp
                k_ngh = k + kp

                i_ngh = max(i_ngh, 0)
                j_ngh = max(j_ngh, 0)
                k_ngh = max(k_ngh, 0)

                i_ngh = min(i_ngh, nx-1)
                j_ngh = min(j_ngh, ny-1)
                k_ngh = min(k_ngh, nz-1)

                if i_ngh == i and j_ngh == j and k_ngh == k:
                    continue

                ind_ngh = self.ijk_to_ind(i_ngh, j_ngh, k_ngh)

                if ind_ngh not in ngh_pores:
                    ngh_pores.append(ind_ngh)

            self.ngh_pores[ind] = np.array(ngh_pores, dtype=np.int32)

        self.nr_t = len(np.hstack(self.ngh_pores.flat))/2
        self.edgelist = -np.ones([self.nr_t, 2], dtype=np.int32)

        tube_index = 0
        for pi in xrange(self.nr_p):
            for pi_ngh in self.ngh_pores[pi]:
                if pi < pi_ngh:
                    self.edgelist[tube_index, 0] = pi
                    self.edgelist[tube_index, 1] = pi_ngh
                    tube_index += 1
        assert tube_index == self.nr_t

        G = self.get_igraph_representation(self.edgelist)

        inc_list = G.get_inclist()
        adj_list = G.get_adjlist()

        for pi in xrange(self.nr_p):
            self.ngh_pores[pi] = np.array(adj_list[pi], dtype=np.int32)
            self.ngh_tubes[pi] = np.array(inc_list[pi], dtype=np.int32)

