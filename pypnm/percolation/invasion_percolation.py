import pypnm.percolation.graph_algs
import numpy as np

from pypnm.porenetwork.constants import *
from heapq import *

sat_eps = 0.001


class InvasionMechanism(object):
    def __init__(self):
        self.pc = 0.0


class InvasionPercolator(object):
    def __init__(self, network, pe_comp, sat_comp, pc_comp=None):
        self.network = network
        self.pe_comp = pe_comp
        self.sat_comp = sat_comp
        self.pc_comp = pc_comp

        assert self.network == sat_comp.network
        # Capillary pressure for piston displacement during drainage
        self.pc_piston_drainage = pe_comp.piston_all_tubes(network)

        # Capillary pressures for snap_off and piston displacement during imbibition
        self.pc_snapoff = pe_comp.snap_off_all_tubes(network)
        self.pc_piston_imbibition = pe_comp.piston_all_tubes(network)

        self.marker = np.zeros(network.nr_t, dtype=np.int32)

    def push_ngh_wett_tubes_of_pore_to_drainage_heap(self, pi):
        network = self.network
        assert (network.pores.connected[pi] == 1)
        ngh_pores = network.ngh_pores[pi]
        ngh_tubes = network.ngh_tubes[pi]

        for pi_ngh, ti_ngh in zip(ngh_pores, ngh_tubes):
            if self.marker[ti_ngh] == 0 and network.tubes.invaded[ti_ngh] == WETT:
                heappush(self.intfc_tp, (self.pc_piston_drainage[ti_ngh], ti_ngh, pi_ngh))
                self.marker[ti_ngh] = 1

    def push_wett_interface_tubes_to_drainage_heap(self):
        network = self.network
        #Interface tubes are neighbors of connected pores which are filled with wetting phase
        for pi in np.flatnonzero(network.pores.connected == 1):
            self.push_ngh_wett_tubes_of_pore_to_drainage_heap(pi)

    def push_wett_interface_tubes_to_drainage_heap_from_list(self, pore_list):
        network = self.network
        #Interface tubes are neighbors of connected pores which are filled with wetting phase
        for pi in pore_list:
            if network.pores.connected[pi] == 1:
                self.push_ngh_wett_tubes_of_pore_to_drainage_heap(pi)

    def pop_largest_tube_and_ngh_pore(self):
        pc_l, ti_l, pi_l  = heappop(self.intfc_tp)
        return ti_l, pi_l, pc_l

    def set_pore_capillary_pressure(self, pi_l):
        network = self.network
        #pc_adj_tubes = max(network.tubes.p_c[network.ngh_tubes[pi_l]])
        network.pores.p_c[pi_l] = self.pc_comp.p_c # max(pc_adj_tubes, self.pc_comp.p_c)

    def set_tube_capillary_pressure(self, ti_l):
        network = self.network
        pc_adj_pores = max(network.pores.p_c[network.edgelist[ti_l]])
        network.tubes.p_c[ti_l] = max(pc_adj_pores, self.pc_comp.p_c, self.pe_comp.piston_tube(network,ti_l))

    def connect_ngh_nwett_tubes_of_pore(self, pi):
        network = self.network
        assert (network.pores.connected[pi] == 1)
        for ti_ngh in network.ngh_tubes[pi]:
            if network.tubes.invaded[ti_ngh] == NWETT and network.tubes.connected[ti_ngh] == 0:
                self.connect_tube(ti_ngh)
            self.set_tube_capillary_pressure(ti_ngh)

    def init_saturations(self):
        sat_comp = self.sat_comp
        self.Snw_pores = sat_comp.sat_nw_pores()
        self.Snw_tubes = sat_comp.sat_nw_tubes()
        self.Snwc_pores = sat_comp.sat_nw_conn_pores()
        self.Snwc_tubes = sat_comp.sat_nw_conn_tubes()

    def connect_disconnected_cluster_and_push_to_drainage_heap(self, pi_source):
        vertices = pypnm.percolation.graph_algs.bfs_vertices_nw_disconn(self.network, pi_source)
        assert (np.all(self.network.pores.connected[vertices] == 0))
        assert (np.all(self.network.pores.invaded[vertices] == 1))
        for pi in vertices:
            self.set_pore_capillary_pressure(pi)
            self.connect_pore(pi)
            self.connect_ngh_nwett_tubes_of_pore(pi)
            self.push_ngh_wett_tubes_of_pore_to_drainage_heap(pi)

    def invade_pore(self, pi):
        assert(self.network.pores.invaded[pi] == WETT)
        self.network.pores.invaded[pi] = NWETT
        self.Snw_pores += self.sat_comp.get_pore_sat_nw_contribution(pi)

    def connect_pore(self, pi):
        assert(self.network.pores.connected[pi] == 0)
        self.network.pores.connected[pi] = 1
        self.Snwc_pores += self.sat_comp.get_pore_sat_nw_contribution(pi)

    def invade_tube(self, ti):
        assert(self.network.tubes.invaded[ti] == WETT)
        self.network.tubes.invaded[ti] = NWETT
        self.Snw_tubes += self.sat_comp.get_tube_sat_nw_contribution(ti)

    def connect_tube(self, ti):
        assert(self.network.tubes.connected[ti] == 0)
        self.network.tubes.connected[ti] = 1
        self.Snwc_tubes += self.sat_comp.get_tube_sat_nw_contribution(ti)

    def invade_pore_with_nw(self, pi):
        network = self.network
        assert(network.pores.invaded[pi] == WETT)
        self.set_pore_capillary_pressure(pi)
        self.invade_pore(pi)
        self.connect_pore(pi)

    def invade_tube_with_nw(self, ti):
        network = self.network
        assert(network.tubes.invaded[ti] == WETT)
        self.set_tube_capillary_pressure(ti)
        self.invade_tube(ti)
        self.connect_tube(ti)

    def invade_pore_with_wett(self, pi):
        pores = self.network.pores
        assert (pores.invaded[pi] == 1)
        assert (pores.connected[pi] == 1)
        pores.invaded[pi] = WETT
        pores.connected[pi] = 0

        self.Snwc_pores -= self.sat_comp.get_pore_sat_nw_contribution(pi)
        self.Snw_pores -= self.sat_comp.get_pore_sat_nw_contribution(pi)

    def invade_tube_with_wett(self, ti):
        tubes = self.network.tubes
        assert (tubes.invaded[ti] == 1)
        assert (tubes.connected[ti] == 1)
        tubes.invaded[ti] = WETT
        tubes.connected[ti] = 0

        self.Snwc_tubes -= self.sat_comp.get_tube_sat_nw_contribution(ti)
        self.Snw_tubes -= self.sat_comp.get_tube_sat_nw_contribution(ti)

    def push_nwconn_tubes_to_snapoff_heap(self):
        network = self.network
        for ti in xrange(network.nr_t):
            if network.tubes.connected[ti] == 1:
                assert (network.tubes.invaded[ti] == NWETT)
                p1, p2 = network.edgelist[ti]

                if network.pores.connected[p1] == 1 and network.pores.connected[p2] == 1:
                    heappush(self.intfc_t_snapoff, (-self.pc_snapoff[ti], ti))

    def invasion_percolation_drainage(self, sat_max, restart=True, target_type=CONNECTED):
        """
            Assumptions: Invasion percolation starting from nw-connected pores
        """
        sat_comp = self.sat_comp

        network = self.network
        Snw = sat_comp.sat_nw()
        Snwc = sat_comp.sat_nw_conn()

        self.init_saturations()

        if restart:
            self.intfc_tp = []  # Heaped list of tuples storing (entry pressure, tube index, ngh pore index)
            self.marker[:] = 0
            self.push_wett_interface_tubes_to_drainage_heap()

        sat_max_reached = False
        while (not sat_max_reached) and self.intfc_tp:
            ti, pi, pc = self.pop_largest_tube_and_ngh_pore()

            self.pc_comp.p_c = max(pc, self.pc_comp.p_c)

            if network.tubes.invaded[ti] == WETT:
                self.invade_tube_with_nw(ti)

            if network.pores.invaded[pi] == WETT:
                self.invade_pore_with_nw(pi)

            is_reconnected_nw_pore = network.is_pore_invaded_and_disconnected(pi)
            if is_reconnected_nw_pore:
                self.connect_disconnected_cluster_and_push_to_drainage_heap(pi)

            mask_untrapping = (network.tubes.connected[network.ngh_tubes[pi]]==0) & (network.tubes.invaded[network.ngh_tubes[pi]]==1)
            for pi_ngh in network.ngh_pores[pi][mask_untrapping]:
                print "reconnecting !!"
                self.connect_disconnected_cluster_and_push_to_drainage_heap(pi_ngh)

            Snwc = sat_comp.sat_pores_and_tubes(self.Snwc_pores, self.Snwc_tubes)
            Snw = sat_comp.sat_pores_and_tubes(self.Snw_pores, self.Snw_tubes)

            if (Snwc >= sat_max) and (target_type == CONNECTED):
                sat_max_reached = True

            if (Snw >= sat_max) and (target_type == INVADED):
                sat_max_reached = True

            self.push_ngh_wett_tubes_of_pore_to_drainage_heap(pi)

        return Snwc

    def invasion_percolation_imbibition(self, sat_min, restart=True, target_type=CONNECTED):
        """ Runs the invasion percolation algorithm  with trapping and snap-offs

        Args:
            network: Instance of PoreNetwork.
            sat_min: Non-wetting saturation at which the simulation ends
        """
        network = self.network
        sat_comp = self.sat_comp
        pe_comp = self.pe_comp

        assert(self.pc_comp.p_c > 0.0)

        self.init_saturations()

        Snw = sat_comp.sat_nw()
        Snwc = sat_comp.sat_nw_conn()

        if Snw < sat_min:
            return

        def update_connectivity_from_pores(network, pore_list):
            """Updates connectivity of pores and tubes given a list
            of potentially disconnected pores that are currently marked as connected pores.

            Args:
                network: Instance of PoreNetwork.
                pore_list: list of pore indices. These must be marked as connected.
            """
            assert len(pore_list) > 0
            assert np.all(network.pores.connected[pore_list] == 1)

            tubes = network.tubes

            for x in pore_list:  # Go through pore list
                is_connected_pore = pypnm.percolation.graph_algs.pore_connected_to_inlet(network, x)
                if not is_connected_pore:

                    # Update connectivities all pores(vertices) linked to this pore
                    vertices_disconn = pypnm.percolation.graph_algs.bfs_vertices_nw_conn(network, x)
                    for pi in vertices_disconn:
                        if network.pores.connected[pi] == 1:
                            network.pores.connected[pi] = 0
                            self.Snwc_pores -= sat_comp.get_pore_sat_nw_contribution(pi)

                        # Update connectivities of all tubes linked to the disconnected pores(vertices)
                        ngh_tubes = network.ngh_tubes[pi]
                        for ti in ngh_tubes:
                            if tubes.connected[ti] == 1:
                                tubes.connected[ti] = 0
                                self.Snwc_tubes -= sat_comp.get_tube_sat_nw_contribution(ti)
                    # If only two pores are in the list, and they are adjacent
                    # then it is a snap-off event and only one pore can be disconnected
                    if (len(pore_list) == 2) and (pore_list[1] in network.ngh_pores[pore_list[0]]):
                        break

        def push_nwconn_tubes_to_piston_heap():
            for ti in xrange(network.nr_t):
                if network.tubes.connected[ti] == 1:
                    assert (network.tubes.invaded[ti] == NWETT)

                    p1, p2 = network.edgelist[ti]
                    if network.pores.connected[p1] != network.pores.connected[p2]:
                        assert (network.pores.invaded[p1] != network.pores.invaded[p2])
                        heappush(self.intfc_t_piston, (-self.pc_piston_imbibition[ti], ti))

        def push_nwconn_pore_to_coop_heap(pi):
            """
            Add pore index to heap if:
            1) It is connected to the inlet.
            2) It is at an interface between two fluids.
            TODO: Improve handling of boundaries
            ARGS:
            pi - pore index.
            """
            if network.pores.connected[pi] == 1:
                assert (network.pores.invaded[pi] == NWETT)
                ngh_tubes = network.ngh_tubes[pi]
                is_ngh_tubes_wett = np.any(network.tubes.invaded[ngh_tubes] == WETT)
                is_ngh_tubes_nwett = np.any(network.tubes.invaded[ngh_tubes] == NWETT)

                # Find number of tubes filled with non-wetting phase
                if network.pore_domain_type[pi] != INLET and is_ngh_tubes_wett and is_ngh_tubes_nwett:
                    pc_p = pe_comp.coop_pore(network, pi)
                    heappush(self.intfc_p_coop, (-pc_p, pi))

        def push_nwconn_pores_to_coop_heap():
            for pi in xrange(network.nr_p):
                push_nwconn_pore_to_coop_heap(pi)

        def push_ngh_pores_of_tube_to_coop_heap(ti):
            p1, p2 = network.edgelist[ti]
            push_nwconn_pore_to_coop_heap(p1)
            push_nwconn_pore_to_coop_heap(p2)

        def push_ngh_tubes_of_pore_to_piston_heap(pi):
            ngh_tubes = network.ngh_tubes[pi]

            for ti in ngh_tubes:
                if network.tubes.connected[ti] == 1:
                    heappush(self.intfc_t_piston, (-self.pc_piston_imbibition[ti], ti))

        def is_trapping_event_after_pore_coop_fill(pi):
            is_trapping_event = True
            network = self.network
            ngh_pores_conn_to_pore = network._get_ngh_pores_conn_to_pore(pi)

            if len(ngh_pores_conn_to_pore) == 1:
                is_trapping_event = False
            else:
                source = ngh_pores_conn_to_pore[0]
                ngh_pores_conn_to_pore = set(ngh_pores_conn_to_pore)
                nearest_pores = pypnm.percolation.graph_algs.bfs_vertices_nw(network, source, max_level=20, max_count=100)

                if ngh_pores_conn_to_pore.intersection(nearest_pores) == ngh_pores_conn_to_pore:
                    is_trapping_event = False

            return is_trapping_event

        def is_trapping_event_after_tube_snapoff(ti):

            is_trapping_event = True

            source = network.edgelist[ti][0]

            ngh_pores = set(network.edgelist[ti])

            nearest_pores = pypnm.percolation.graph_algs.bfs_vertices_nw(network, source, 20, 100)
            if ngh_pores.intersection(nearest_pores) == ngh_pores:
                is_trapping_event = False

            return is_trapping_event

        def remove_pores_from_heap():
            while self.intfc_p_coop:
                # Get pore index from heap
                a = self.intfc_p_coop[0]
                pi = a[1]

                # If pore has been already trapped get another pore from heap
                if network.pores.connected[pi] == 0:
                    heappop(self.intfc_p_coop)
                else:
                    break

        def remove_tubes_from_heap():
            while self.intfc_t_snapoff:
                # Get tube index from heap
                a = self.intfc_t_snapoff[0]
                ti = a[1]

                # If tube has been already trapped get another tube from heap
                if network.tubes.connected[ti] == 0:
                    heappop(self.intfc_t_snapoff)
                else:
                    break

            while self.intfc_t_piston:
                # Get tube index from heap
                a = self.intfc_t_piston[0]
                ti = a[1]

                # If tube has been already trapped get another tube from heap
                if network.tubes.connected[ti] == 0:
                    heappop(self.intfc_t_piston)
                else:
                    break

        tube_snapoff = InvasionMechanism()
        tube_piston = InvasionMechanism()
        pore_coop = InvasionMechanism()

        #Update connectivity of pores
        #graph_algs.update_pore_and_tube_nw_connectivity_to_inlet(network)

        if restart:
            self.intfc_t_snapoff = []  # Heaped list of tuples storing (Pc_snap_off, tube index)
            self.intfc_t_piston = []  # Heaped list of tuples storing (pc_piston , tube index)
            self.intfc_p_coop = []  # Heaped list of tuples storing (pc_cooperative_filling, pore index)

            push_nwconn_pores_to_coop_heap()
            self.push_nwconn_tubes_to_snapoff_heap()
            push_nwconn_tubes_to_piston_heap()

        tubes = self.network.tubes
        pores = self.network.pores

        sat_min_reached = False
        while not sat_min_reached:

            remove_pores_from_heap()
            remove_tubes_from_heap()

            if self.intfc_p_coop:
                pore_coop.pc = -self.intfc_p_coop[0][0]
                pore_coop.pi = self.intfc_p_coop[0][1]
            else:
                pore_coop.pc = 0.0

            if self.intfc_t_snapoff:
                tube_snapoff.pc = -self.intfc_t_snapoff[0][0]
                tube_snapoff.ti = self.intfc_t_snapoff[0][1]
            else:
                tube_snapoff.pc = 0.0

            if self.intfc_t_piston:
                tube_piston.pc = -self.intfc_t_piston[0][0]
                tube_piston.ti = self.intfc_t_piston[0][1]
            else:
                tube_piston.pc = 0.0

            imbibe_mechanism_pcs = [pore_coop.pc, tube_piston.pc, tube_snapoff.pc]
            imbibe_mechanism = np.argmax(imbibe_mechanism_pcs)
            imbibe_mechanism_pc = np.max(imbibe_mechanism_pcs)

            COOP_FILLING = 0
            TUBE_PISTON = 1
            TUBE_SNAPOFF = 2
            NONE = -1

            # Break loop if no more imbibition mechanisms exists
            if imbibe_mechanism_pc == 0:
                break

            self.pc_comp.p_c = min(imbibe_mechanism_pc, self.pc_comp.p_c)

            assert(self.pc_comp.p_c > 0.0)

            if len(self.intfc_p_coop) == 0 and len(self.intfc_t_piston) == 0 and len(self.intfc_t_snapoff) == 0:
                imbibe_mechanism = NONE
                sat_min_reached = True

            if imbibe_mechanism == COOP_FILLING:
                pi = pore_coop.pi

                self.invade_pore_with_wett(pi)
                push_ngh_tubes_of_pore_to_piston_heap(pi)

                is_trapping_event = is_trapping_event_after_pore_coop_fill(pi)
                if is_trapping_event:
                    ngh_pores_conn_to_pore = network._get_ngh_pores_conn_to_pore(pi)
                    update_connectivity_from_pores(network, ngh_pores_conn_to_pore)

                pore_coop.pc = 0.0

            if imbibe_mechanism == TUBE_PISTON:
                ti = tube_piston.ti
                assert (ti > -1)

                self.invade_tube_with_wett(ti)

                push_ngh_pores_of_tube_to_coop_heap(tube_piston.ti)

                tube_piston.pc = 0.0

            if imbibe_mechanism == TUBE_SNAPOFF:
                ti = tube_snapoff.ti
                assert (ti > -1)

                self.invade_tube_with_wett(ti)

                is_trapping_event = is_trapping_event_after_tube_snapoff(ti)

                if is_trapping_event:
                    ngh_pores = network.edgelist[ti]
                    update_connectivity_from_pores(network, ngh_pores)

                if (not is_trapping_event) and __debug__:
                    pore1, pore2 = network.edgelist[ti]
                    assert (pypnm.percolation.graph_algs.pore_connected_to_inlet(network, pore1))
                    assert (pypnm.percolation.graph_algs.pore_connected_to_inlet(network, pore2))

                push_ngh_pores_of_tube_to_coop_heap(ti)

                tube_snapoff.pc = 0.0

            Snwc = sat_comp.sat_pores_and_tubes(self.Snwc_pores, self.Snwc_tubes)
            Snw = sat_comp.sat_pores_and_tubes(self.Snw_pores, self.Snw_tubes)

            if (Snwc < sat_min) and (target_type == CONNECTED):
                sat_min_reached = True

            if (Snw < sat_min) and (target_type == INVADED):
                sat_min_reached = True

            remove_tubes_from_heap()
            remove_pores_from_heap()

            # In case two successive events have equal critical capillary pressure.
            if self.intfc_t_snapoff and (self.pc_comp.p_c == -self.intfc_t_snapoff[0][0]):
                sat_min_reached = False

        return Snwc

