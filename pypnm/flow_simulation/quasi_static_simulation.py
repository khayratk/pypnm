import numpy as np

from pypnm.attribute_calculators.pc_computer import CapillaryPressureComputer
from pypnm.percolation import graph_algs
from pypnm.percolation.invasion_percolation import InvasionPercolator
from pypnm.porenetwork.constants import DRAINAGE, IMBIBITION, DOMAIN
from pypnm.porenetwork.saturation_computer import QuasiStaticSaturationComputer
from pypnm.flow_simulation.simulation import Simulation


class QuasiStaticSimulation(Simulation):
    def __init__(self, network, fluid_properties):
        super(QuasiStaticSimulation, self).__init__(network, fluid_properties)
        self.SatComputer = QuasiStaticSaturationComputer
        self.sat_comp = self.SatComputer(network)
        self.pc_comp = CapillaryPressureComputer(network)
        self.ip = InvasionPercolator(self.network, self.pe_comp, self.sat_comp,
                                     self.pc_comp)  # Warning: Modifies self.pc_comp.p_c implicitly!
        self.previous_mode = None

    def apply_initial_conditions(self):
        # Need to initialize the status and capillary pressure of the pores
        self.network.set_inlet_pores_invaded_and_connected()
        self.pc_comp.initialize_pc_from_invasion_status()
        assert np.all(self.network.pores.p_c[self.network.pores.invaded == 1] > 0.0)

    def update_saturation_nw(self, sat):
        self.__update_saturation_generic(sat, self.sat_comp.sat_nw)

    def update_saturation_conn(self, sat_target):
        self.__update_saturation_generic(sat_target, self.sat_comp.sat_nw_conn)

    def __update_saturation_generic(self, sat_target, sat_comp_func):
        network = self.network
        snw_start = sat_comp_func()
        is_restart = True

        # Imbibition
        if snw_start > sat_target:
            if self.previous_mode == DRAINAGE:
                is_restart = True
            if self.previous_mode == IMBIBITION:
                is_restart = False
            self.previous_mode = IMBIBITION

            self.ip.invasion_percolation_imbibition(sat_target, restart=is_restart)

            if __debug__:
                self.__check_connected_consistency()

            self.pc_comp.compute()

            sat_current = sat_comp_func()
            assert (sat_current <= sat_target)

        # Drainage
        if snw_start < sat_target:
            if self.previous_mode == IMBIBITION:
                is_restart = True
            if self.previous_mode == DRAINAGE:
                is_restart = False
            self.previous_mode = DRAINAGE

            self.ip.invasion_percolation_drainage(sat_target, restart=is_restart)
            self.pc_comp.compute()

            if __debug__:
                self.__check_connected_consistency()

            assert np.all(np.max(network.pores.p_c[network.pores.connected == 1]) ==
                          network.pores.p_c[network.pores.connected == 1])

            sat_current = sat_comp_func()
            assert(sat_current >= sat_target)

    def __check_connected_consistency(self):
        network = self.network
        pores_connected_before = np.copy(network.pores.connected)
        tubes_connected_before = np.copy(network.tubes.connected)

        graph_algs.update_pore_and_tube_nw_connectivity_to_inlet(network)

        pi_domain = (network.pore_domain_type == DOMAIN).nonzero()[0]
        assert np.all(pores_connected_before[pi_domain] == network.pores.connected[pi_domain])
        assert np.all(tubes_connected_before== network.tubes.connected)