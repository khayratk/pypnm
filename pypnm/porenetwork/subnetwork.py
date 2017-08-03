import operator

import numpy as np

from pypnm.porenetwork.constants import *
from porenetwork import PoreNetwork
import component
from pypnm.util.bounding_box import BoundingBox


class SubNetwork(PoreNetwork):
    def __init__(self, sup_network, sup_pore_list):
        """
        Parameters
        ----------
        sup_network: PoreNetwork
            Parent network
        sup_pore_list: list
            List of pore indices in the parent network from which the subnetwork will be induced
        """
        self.network_type = sup_network.network_type

        self.pi_local_to_global, self.ti_local_to_global = self._local_to_global_maps(sup_network, sup_pore_list)

        self.pi_global_to_local, self.ti_global_to_local = self._global_to_local_maps(self.pi_local_to_global,
                                                                                      self.ti_local_to_global)

        if __debug__:
            self._check_maps()

        self.pores, self.tubes = self._create_pores_and_tubes(sup_network, self.pi_local_to_global,
                                                              self.ti_local_to_global)

        self.nr_p = self.pores.nr
        self.nr_t = self.tubes.nr

        self._create_network_topology(sup_network)
        self._create_nr_nghs_array()

        self._init_pi_list_face(sup_network)
        self._init_pore_inout_type(inface=WEST, outface=EAST)

        self._create_helper_properties()
        self._assert_tubes_smaller_than_ngh_pores()

        self.p_indices = np.asarray(range(self.nr_p), dtype=np.int8)

    @classmethod
    def from_bounding_box(cls, sup_network, bounding_box):
        sup_pore_list = component.pore_list_from_bbox(sup_network, bounding_box)
        return cls(sup_network, sup_pore_list)

    @staticmethod
    def _local_to_global_maps(sup_network, sup_pore_list):
        pi_local_to_global = np.array(sup_pore_list, dtype=np.int32)
        ti_local_to_global = component.tubes_within_pore_set(sup_network, pi_local_to_global)
        return pi_local_to_global, ti_local_to_global

    @staticmethod
    def _global_to_local_maps(pi_local_to_global, ti_local_to_global):
        pi_global_to_local = dict()
        ti_global_to_local = dict()

        for pi_local, pi_global in enumerate(pi_local_to_global):
            pi_global_to_local[pi_global] = pi_local

        for ti_local, ti_global in enumerate(ti_local_to_global):
            ti_global_to_local[ti_global] = ti_local

        return pi_global_to_local, ti_global_to_local

    @staticmethod
    def _create_pores_and_tubes(sup_network, pi_local_to_global, ti_local_to_global):
        pores = sup_network.pores.select_pores(pi_local_to_global)
        tubes = sup_network.tubes.select_tubes(ti_local_to_global)
        return pores, tubes

    def _check_maps(self):
        for pi in self.pi_global_to_local.keys():
            pi_local = self.pi_global_to_local[pi]
            pi_global = self.pi_local_to_global[pi_local]
            assert (pi == pi_global)

        for ti in self.ti_global_to_local.keys():
            ti_local = self.ti_global_to_local[ti]
            ti_global = self.ti_local_to_global[ti_local]
            assert (ti == ti_global)

        for pi_local_1, pi_global in enumerate(self.pi_local_to_global):
            pi_local_2 = self.pi_global_to_local[pi_global]
            assert (pi_local_1 == pi_local_2)

        for ti_local_1, ti_global in enumerate(self.ti_local_to_global):
            ti_local_2 = self.ti_global_to_local[ti_global]
            assert (ti_local_1 == ti_local_2)

    def _create_network_topology(self, sup_network):
        ngh_pores = np.ones([self.pores.nr], dtype=np.object)
        ngh_tubes = np.ones([self.pores.nr], dtype=np.object)

        for pi, pi_global in enumerate(self.pi_local_to_global):
            ngh_pores[pi] = np.asarray([self.pi_global_to_local[x] for x in sup_network.ngh_pores[pi_global]
                                        if x in self.pi_global_to_local], dtype=np.int32)

            ngh_tubes[pi] = np.asarray([self.ti_global_to_local[x] for x in sup_network.ngh_tubes[pi_global]
                                        if x in self.ti_global_to_local], dtype=np.int32)

        self.ngh_pores = ngh_pores
        self.ngh_tubes = ngh_tubes

        edgelist = -np.ones([self.nr_t, 2], dtype=np.int32)
        edgelist[:, :] = sup_network.edgelist[self.ti_local_to_global, :]

        for i in xrange(self.nr_t):
            edgelist[i, 0] = self.pi_global_to_local[edgelist[i, 0]]
            edgelist[i, 1] = self.pi_global_to_local[edgelist[i, 1]]

        self.edgelist = edgelist

    def _init_pi_list_face(self, sup_network):
        pore_face = np.zeros(self.nr_p, dtype=np.int32)
        pore_face[:] = DOMAIN

        self.pi_list_face = [None, None, None, None, None, None]

        marker_boundary_pores = np.zeros(self.nr_p, dtype=np.int)
        for FACE in FACES:
            pore_face = self._mark_face_from_sup_network(sup_network, FACE, pore_face)
            pore_face = self._mark_face_at_boundary(sup_network, FACE, pore_face)
            self.pi_list_face[FACE] = (pore_face == FACE).nonzero()[0]
            marker_boundary_pores[self.pi_list_face[FACE]] = 1

        nr_nghs_bnd = np.zeros(self.nr_p, dtype=np.int)

        for i in xrange(self.nr_p):
            nr_nghs_bnd = np.sum(marker_boundary_pores[self.ngh_pores[i]])

        # Mask of pores which only have face arrays as neighbours
        mask_new_pores = np.zeros(self.nr_p, dtype=np.int)
        mask_new_pores[self.nr_nghs == nr_nghs_bnd] = 1

        for FACE in FACES:
            pi_face_new = list()
            for i_face in self.pi_list_face[FACE]:
                pi_nghs = self.ngh_pores[i_face]
                if np.any(mask_new_pores[pi_nghs] == 1):
                    new_indices_added = pi_nghs[mask_new_pores[pi_nghs] == 1].tolist()
                    pi_face_new += new_indices_added

            self.pi_list_face[FACE] = np.union1d(self.pi_list_face[FACE], np.asarray(pi_face_new, dtype=np.int))

    def _mark_face_from_sup_network(self, sup_network, FACE, pore_face):
        pi_face_global = np.intersect1d(sup_network.pi_list_face[FACE], self.pi_local_to_global)
        pi_face_local = [self.pi_global_to_local[pi] for pi in pi_face_global]
        pore_face[pi_face_local] = FACE
        return pore_face

    def _mark_face_at_boundary(self, sup_network, FACE, pore_face):
        assert (FACE in FACES)

        pores = self.pores
        bounding_box = BoundingBox(min(pores.x), max(pores.x),
                                   min(pores.y), max(pores.y),
                                   min(pores.z), max(pores.z))

        pos, comp = None, None

        if FACE == WEST or FACE == EAST:
            pos = sup_network.pores.x

        if FACE == SOUTH or FACE == NORTH:
            pos = sup_network.pores.y

        if FACE == BOTTOM or FACE == TOP:
            pos = sup_network.pores.z

        if FACE in [WEST, SOUTH, BOTTOM]:
            comp = operator.lt
        if FACE in [EAST, NORTH, TOP]:
            comp = operator.gt

        # If neighboring pore is on the opposite side of a plane, then mark it with the corresponding face
        pi_local = 0
        for pi_global in self.pi_local_to_global:
            ngh_pores_indices_global = sup_network.ngh_pores[pi_global]

            dict_select = {WEST: "xmin", EAST: "xmax", SOUTH: "ymin", NORTH: "ymax", BOTTOM: "zmin", TOP: "zmax"}
            axis = dict_select[FACE]
            face_coordinate = bounding_box[axis]
            for pi_ngh_global in ngh_pores_indices_global:
                if comp(pos[pi_ngh_global], face_coordinate):
                    pore_face[pi_local] = FACE
            pi_local += 1

        return pore_face

    def pi_list_from_global(self, pi_global):
        if len(pi_global) > 0:
            return np.asarray([self.pi_global_to_local[x] for x in pi_global], dtype=np.int)
        else:
            return []


class SubNetworkTightlyCoupled(SubNetwork):
    def __init__(self, sup_network, sup_pore_list):
        super(SubNetworkTightlyCoupled, self).__init__(sup_network, sup_pore_list)
        self.sup_network = sup_network

    def copy_status_fields_from_super_network(self):
        sup_network = self.sup_network

        for field, sup_field in zip(self.pores.status_fields, sup_network.pores.status_fields):
            field[:] = sup_field[self.pi_local_to_global]

        for field, sup_field in zip(self.tubes.status_fields, sup_network.tubes.status_fields):
            field[:] = sup_field[self.ti_local_to_global]

    def copy_pc_from_super_network(self):
        sup_network = self.sup_network

        self.pores.p_c[:] = sup_network.pores.p_c[self.pi_local_to_global]
        self.tubes.p_c[:] = sup_network.tubes.p_c[self.ti_local_to_global]

    def copy_sat_from_super_network(self):
        sup_network = self.sup_network

        self.pores.sat[:] = sup_network.pores.sat[self.pi_local_to_global]
        self.tubes.sat[:] = sup_network.tubes.sat[self.ti_local_to_global]
