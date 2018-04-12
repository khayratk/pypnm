import copy
import logging

import numpy as np
from numpy.linalg import norm

from pypnm.attribute_calculators.conductance_calc import ConductanceCalc
from pypnm.attribute_calculators.pc_computer import DynamicCapillaryPressureComputer
from pypnm.flow_simulation.pn_algorithms import update_pore_status, update_tube_piston_w, snapoff_all_tubes, \
    get_piston_disp_tubes_nonwett, invade_tube_nw, get_piston_disp_tubes_wett, invade_tube_w
from pypnm.flow_simulation.simulation import Simulation
from pypnm.flow_simulation.simulation_bc import SimulationBoundaryCondition
from pypnm.linalg.laplacianmatrix import laplacian_from_network
from pypnm.linalg.linear_system_solver import PressureSolverDynamicDirichlet
from pypnm.linalg.petsc_interface import get_petsc_ksp, petsc_solve_from_ksp
from pypnm.porenetwork.constants import NWETT, WETT
from pypnm.porenetwork.pore_element_models import JNModel
from pypnm.porenetwork.saturation_computer import DynamicSaturationComputer
from time_stepper import DynamicTimeStepper

logger = logging.getLogger('pypnm')


class DynamicSimulation(Simulation):
    def __init__(self, network, fluid_properties, explicit=True, delta_pc=0.01, ptol=1e-6):
        super(DynamicSimulation, self).__init__(network, fluid_properties)

        if np.any(network.pores.vol <= 0.0):
            raise ValueError("All pore volumes have to be positive")

        if np.any(network.tubes.vol != 0.0):
            raise ValueError("Network throats have to all have zero volume for dynamic flow solver")

        self.press_solve_tol = ptol
        self.explicit = explicit
        self.fluid_properties = fluid_properties
        self.SatComputer = DynamicSaturationComputer
        self.bool_accounted_pores = np.ones(network.nr_p, dtype=np.bool)  # Saturation is updated only in these pores
        self.sat_comp = self.SatComputer(network, self.bool_accounted_pores)
        self.time_stepper = DynamicTimeStepper(network, self.bool_accounted_pores, delta_pc=delta_pc)

        self.press_solver = PressureSolverDynamicDirichlet(self.network)
        self.press_solver_type = "petsc"

        self.pc_comp = DynamicCapillaryPressureComputer(network)
        self.k_comp = ConductanceCalc(network, self.fluid_properties)

        self.q_n = np.zeros(network.nr_p)  # nonwetting fluid source/sink
        self.q_w = np.zeros(network.nr_p)  # wetting fluid source/sink

        gamma = self.fluid_properties['gamma']
        self.snap_off_press = 1.001*JNModel.snap_off_pressure(gamma=gamma, r=network.tubes.r)
        self.piston_entry = JNModel.piston_entry_pressure(r=network.tubes.r, gamma=gamma, G=network.tubes.G)

        self.bc = SimulationBoundaryCondition()

        self.total_sink_nonwett = 0.0
        self.total_sink_wett = 0.0

        self.time = 0.0

        self.accumulated_saturation = 0.0
        self.saturation_wett_inflow = 0.0

        self.freeze_sink_nw = dict()

        self.stop_time = None
        self.flux_n = np.zeros(network.nr_p)  # Out-fluxes of the nonwetting phase from each pore
        self.flux_w = np.zeros(network.nr_p)  # Out-fluxes of the wetting phase from each pore

        self.sat_prev = np.copy(network.pores.sat)
        self.tube_invaded_prev = np.copy(network.tubes.invaded)
        self.pores_invaded_prev = np.copy(network.pores.invaded)

        self.ti_freeze_displacement = dict()

        self.cum_flux_tubes = np.zeros(network.nr_t)

    def reset_status(self):
        """
        Resets the saturation in all pores as well as invasion state of all pores and tubes to those that were previously
        saved with a call to save_status
        """
        self.network.pores.sat[:] = self.sat_prev
        self.network.tubes.invaded[:] = self.tube_invaded_prev
        self.network.pores.invaded[:] = self.pores_invaded_prev
        self.__update_capillary_pressure()
        self.set_boundary_conditions(self.bc_prev)

        self.network.pores.p_w[:] = 0.0
        self.network.pores.p_n[:] = 0.0
        self.time = self.time_prev

    def save_status(self):
        """
        Saves the saturation in all pores as well as invasion states of all pores and tubes
        """
        self.sat_prev[:] = np.copy(self.network.pores.sat)
        self.tube_invaded_prev[:] = np.copy(self.network.tubes.invaded)
        self.pores_invaded_prev[:] = np.copy(self.network.pores.invaded)

        self.time_prev = self.time
        self.bc_prev = copy.deepcopy(self.bc)

    def set_boundary_conditions(self, bc):
        """
        Set boundary condition for simulation

        Parameters
        ----------
        bc: SimulationBoundaryCondition

        """
        self.bc = copy.deepcopy(bc)
        self.__set_rhs_source_arrays(self.bc)
        self.total_sink_nonwett = self.__compute_total_sink(NWETT)
        self.total_sink_wett = self.__compute_total_sink(WETT)
        self.total_source_nonwett = self.__compute_total_source(NWETT)
        self.total_source_wett = self.__compute_total_source(WETT)

        self.bool_accounted_pores = np.ones(self.network.nr_p, dtype=np.bool)
        self.bool_accounted_pores[self.bc.pi_list_inlet] = 0  # Ignore saturation for pressure boundary conditions
        self.bool_accounted_pores[self.bc.pi_list_outlet] = 0  # Ignore saturation for pressure boundary conditions

        self.sat_comp = DynamicSaturationComputer(self.network, self.bool_accounted_pores)

        assert len(self.bc.pi_list_w_sink) == len(np.unique(self.bc.pi_list_w_sink))

        self.pi_nghs_of_w_sinks_interior = {}
        for pi in self.bc.pi_list_w_sink:
            ngh_pores_interior = np.setdiff1d(self.network.ngh_pores[pi], self.bc.pi_list_w_sink, assume_unique=True)
            if len(ngh_pores_interior) == 0:
                ngh_pores_interior = self.network.ngh_pores[pi]

            self.pi_nghs_of_w_sinks_interior[pi] = ngh_pores_interior

    def advance_in_sat(self, delta_s, callback=None):
        """
        Advances the simulation to a specified time. Boundary conditions should be set before calling this function.

        Parameters
        ----------
        delta_s: float
            sat difference between initial state and final state of the simulation.

        """
        logger.debug("Starting simulation with time criterion")

        self.stop_sat = self.sat_comp.sat_nw() + delta_s

        def stop_criterion():
            return self.sat_comp.sat_nw() >= self.stop_sat

        self.save_status()

        return self.__advance(stop_criterion, callback)

    def advance_in_time(self, delta_t, callback=None):
        """
        Advances the simulation to a specified time. Boundary conditions should be set before calling this function.

        Parameters
        ----------
        delta_t: float
            Time difference between initial state and final state of the simulation.

        """
        logger.debug("Starting simulation with time criterion")

        self.stop_time = self.time + delta_t

        def stop_criterion():
            return self.time >= self.stop_time

        self.save_status()

        return self.__advance(stop_criterion, callback)

    def __set_rhs_source_arrays(self, bc):
        """
        Sets (or resets) source arrays from provided boundary conditions
        """
        self.q_n[:] = 0.0
        self.q_w[:] = 0.0

        self.q_n[bc.pi_list_nw_source] = bc.q_list_nw_source
        self.q_n[bc.pi_list_nw_sink] = bc.q_list_nw_sink

        self.q_w[bc.pi_list_w_source] = bc.q_list_w_source
        self.q_w[bc.pi_list_w_sink] = bc.q_list_w_sink

    def __compute_total_sink(self, FLUID):
        if FLUID == WETT:
            total_sink = np.sum(self.q_w[self.bc.pi_list_w_sink])
        if FLUID == NWETT:
            total_sink = np.sum(self.q_n[self.bc.pi_list_nw_sink])
        return total_sink

    def __compute_total_source(self, FLUID):
        if FLUID == WETT:
            total_source = np.sum(self.q_w[self.bc.pi_list_w_source])
        if FLUID == NWETT:
            total_source = np.sum(self.q_n[self.bc.pi_list_nw_source])
        return total_source

    def __solve_linear_system(self):
        network = self.network
        logger.debug("Start of __solve_linear_system")

        press_solver = self.press_solver

        logger.debug("Setting linear system")

        press_solver.setup_linear_system(k_n=self.network.tubes.k_n,
                                         k_w=self.network.tubes.k_w,
                                         p_c=self.network.pores.p_c)
        press_solver.add_source_rhs(self.q_n + self.q_w)

        logger.debug("Fixing boundary conditions")
        press_solver.set_dirichlet_pores(pi_list=self.bc.pi_list_inlet, value=self.bc.press_inlet_w)
        press_solver.set_dirichlet_pores(pi_list=self.bc.pi_list_outlet, value=self.bc.press_outlet_w)

        if self.bc.no_dirichlet:
            # Choose any pore to fix pressure to zero
            pi_dirichlet = ((self.q_n + self.q_w) == 0.0).nonzero()[0][0]
            press_solver.set_dirichlet_pores(pi_list=[pi_dirichlet], value=0.0)

        logger.debug("Solving Pressure with " + self.press_solver_type)
        network.pores.p_w[:] = press_solver.solve(self.press_solver_type, x0=network.pores.p_w,
                                                  tol=self.press_solve_tol)
        network.pores.p_n[:] = network.pores.p_w + network.pores.p_c

        logger.debug("Computing nonwetting and wetting fluxes")
        self.flux_n[:] = press_solver.compute_nonwetting_flux()
        self.flux_w[:] = press_solver.compute_wetting_flux()

    def __update_saturation_implicit(self, dt):
        eps_sat = 1.e-4
        logger.debug("Solving fully implicit")
        network = self.network
        print "solving fully implicit"

        def residual_saturation(p_w, p_c, sat, dt):
            p_n = p_w + p_c
            residual = (sat - network.pores.sat) + (A_n * p_n - self.q_n) * dt / network.pores.vol
            return residual

        def residual_pressure(p_w, p_c):
            rhs = -A_n * p_c + self.q_n + self.q_w
            rhs[pi_dirichlet] = 0.0
            ref_residual = norm(A * (rhs / A.diagonal()) - rhs, ord=np.inf)
            residual_normalized = (A*p_w - rhs)/ref_residual
            return residual_normalized

        pi_dirichlet = ((self.q_n + self.q_w) == 0.0).nonzero()[0][0]

        # Assume that the conductances do not change and are fixed
        A = laplacian_from_network(network, weights=network.tubes.k_n + network.tubes.k_w, ind_dirichlet=pi_dirichlet)
        A_n = laplacian_from_network(network, weights=network.tubes.k_n)

        p_w = np.copy(network.pores.p_w)
        sat = np.copy(network.pores.sat)

        ksp = get_petsc_ksp(A=A * 1e20, ksptype="minres", max_it=10000, tol=1e-9)

        logger.debug("Starting iteration")

        p_c = DynamicCapillaryPressureComputer.sat_to_pc_func(network.pores.sat, network.pores.r)

        while True:
            for iter in xrange(200):
                # Solve for pressure

                rhs = -A_n * p_c + self.q_n + self.q_w
                rhs[pi_dirichlet] = 0.0

                p_w[:] = petsc_solve_from_ksp(ksp, rhs*1e20, x=p_w, tol=1e-9)

                p_n = p_w + p_c
                # Solve for saturation
                if iter == 0:
                    damping = 1.
                if iter > 0:
                    damping = 0.4
                if iter > 10:
                    damping = 0.1
                if iter > 20:
                    damping = 0.055
                sat = (1-damping)*sat + damping * (network.pores.sat + (self.q_n - A_n * p_n) * dt / network.pores.vol)
                if np.min(sat) < 0.0:
                    print "saturation undershoot decreasing time-step slightly"
                    dt *= 0.5

                if np.max(sat) > 1.0:
                    print "saturation overshoot"

                sat = np.maximum(sat, 0.0)
                sat = np.minimum(sat, 0.99999999999)

                p_c = DynamicCapillaryPressureComputer.sat_to_pc_func(sat, network.pores.r)
                res_pw = residual_pressure(p_w, p_c)
                res_sat = residual_saturation(p_w, p_c, sat, dt)

                linf_res_pw = norm(res_pw, ord=np.inf)
                linf_res_sat = norm(res_sat, ord=np.inf)

                if iter % 10 == 0:
                    print "iteration %d \t sat res: %g \t press res %g"%(iter, linf_res_sat, linf_res_pw)

                if iter > 3 and linf_res_sat > 1000:
                    break

                if iter > 10 and linf_res_sat > 1.0:
                    break

                if linf_res_sat < eps_sat and linf_res_pw < 1e-5:
                    break

            if linf_res_sat < eps_sat and linf_res_pw < 1e-5:
                print "Iteration converged"
                print "iteration %d \t sat res: %g \t press res %g" % (iter, linf_res_sat, linf_res_pw)
                network.pores.sat[:] = sat
                network.pores.p_w[:] = p_w
                network.pores.p_n[:] = p_n
                network.pores.p_c[:] = p_c
                assert np.all(sat <= 1.0)
                print "Leaving implicit loop"
                break
            else:
                p_w = np.copy(network.pores.p_w)
                sat = np.copy(network.pores.sat)
                p_c = DynamicCapillaryPressureComputer.sat_to_pc_func(sat, network.pores.r)
                dt *= 0.5
                print "decreasing timestep to", dt

        print "timestep is", dt
        return dt

    def __check_mass_conservation(self):
        if self.bc.no_dirichlet:
            total_flux = np.sum(self.q_w) + np.sum(self.q_n)
            source_max = np.max(np.abs(self.q_n)) + np.max(np.abs(self.q_w))
            assert np.abs(total_flux) <= source_max * 1.e-4, "total flux is %e. Maximum source is %e" % (
            total_flux, source_max)

    def __adjust_magnitude_wetting_sink_pores(self):
        """
        Adjusts the magnitude of the sink pores when their status changes so that the total sink remains constant.
        If no more sink pores are available, the source pores are adjusted.
        """
        network = self.network
        p_c = network.pores.p_c
        sat = network.pores.sat
        pi_list_sink = self.bc.pi_list_w_sink
        pi_list_source = self.bc.pi_list_w_source
        rhs_source_wett = self.q_w
        pi_nghs_of_w_sinks_interior = self.pi_nghs_of_w_sinks_interior

        # TODO: Above a certain threshold block a wetting sink using self.rhs_source_wett[pi] = 0.0
        for pi in pi_list_sink:
            pi_nghs_interior = pi_nghs_of_w_sinks_interior[pi]
            pc_max_ngh = max(p_c.take(pi_nghs_interior))

            if (sat[pi] > 0.9) and (p_c[pi] > 1.5 * pc_max_ngh):  # Heuristic that can be improved
                rhs_source_wett[pi] = 0.0
                logger.debug("freezing W sink %d. p_c: %g. Max pc_ngh: %g. sat: %g", pi, p_c[pi], pc_max_ngh,  sat[pi])

        total_after_blockage = np.sum(self.q_w[pi_list_sink])
        assert total_after_blockage <= 0.0

        # Scale wetting sinks so that their total is equal to self.q_w_tot_sink
        if total_after_blockage < 0.0:
            self.q_w[pi_list_sink] = self.q_w[pi_list_sink] * self.q_w_tot_sink / total_after_blockage
            assert np.all(self.q_w[pi_list_sink] <= 0.0)

        elif total_after_blockage == 0.0 and self.q_w_tot_sink < 0.0:
            # If there is net imbibition and the simulation boundary conditions are defined
            # by sources then scale the wetting sources.

            if ((self.q_w_tot_source + self.q_w_tot_sink) > 0.0) and self.bc.no_dirichlet:
                self.q_w[pi_list_source] = (self.q_w[pi_list_source] *
                                            (self.q_w_tot_source + self.q_w_tot_sink) / self.q_w_tot_source)
            # Otherwise exit simulation
            else:
                logger.warning("Total sink before blockage is %e")
                logger.warning("Cannot adjust wetting sink")
                return -1

        assert np.all(self.q_w[pi_list_sink] <= 0.0)

        self.__check_mass_conservation()

    def __adjust_magnitude_nonwetting_sink_pores(self):
        """
        Adjusts the magnitude of the sink pores when their status changes so that the total sink remains constant.
        If no more sink pores are available, the source pores are adjusted.
        """
        network = self.network
        sat = network.pores.sat
        pi_list_sink = self.bc.pi_list_nw_sink
        pi_list_source = self.bc.pi_list_nw_source

        rhs_source = self.q_n

        if np.sum(self.q_n_tot_sink) == 0.0:
            return

        freeze_sink_nw = self.freeze_sink_nw

        for pi in pi_list_sink:

            # Set a nonwetting sink to be frozen if it is completely filled with the wetting fluid.
            if network.pores.invaded[pi] == WETT:
                rhs_source[pi] = 0.0
                logger.debug("freezing NW sink %d", pi)
                freeze_sink_nw[pi] = 1

            # Set a nonwetting sink to be unfrozen if it surpasses a certain nonwetting threshold
            if (freeze_sink_nw.setdefault(pi, 0) == 1) & (sat[pi] > 0.5):
                logger.debug("Unfreezing NW sink %d", pi)
                freeze_sink_nw[pi] = 0

            # Freeze a nonwetting sink.
            if freeze_sink_nw.setdefault(pi, 0) == 1:
                logger.debug("Setting NW sink to zero since it is marked as frozen. Its Saturation is %e" % sat[pi])
                rhs_source[pi] = 0.0

        total_after_blockage = np.sum(self.q_n[pi_list_sink])

        assert total_after_blockage <= 0.0

        if total_after_blockage < 0.0:
            rhs_source[pi_list_sink] = rhs_source[pi_list_sink] * self.q_n_tot_sink / total_after_blockage
            assert np.all(rhs_source[pi_list_sink] <= 0.0)

        elif total_after_blockage == 0.0 and self.q_n_tot_sink < 0.0:

            # If drainage and the simulation is defined solely by sources then scale the nonwetting sources
            if ((self.q_n_tot_source + self.q_n_tot_sink) > 0.0) and self.bc.no_dirichlet:
                rhs_source[pi_list_source] = rhs_source[pi_list_source] * (self.q_n_tot_source + self.q_n_tot_sink) / self.q_n_tot_source
            else:
                logger.warning("Cannot adjust nonwetting sink")
                return -1

        self.__check_mass_conservation()

    def __adjust_magnitude_sink_pores(self, FLUID):
        """
        Adjusts the magnitude of the sink pores when their status changes so that the total sink remains constant.
        If no more sink pores are available, the source pores are adjusted.
        """

        if FLUID == WETT:
            return self.__adjust_magnitude_wetting_sink_pores()

        if FLUID == NWETT:
            return self.__adjust_magnitude_nonwetting_sink_pores()

    def __invade_nonwetting_source_pores(self):
        self.network.pores.invaded[self.bc.pi_list_nw_source] = NWETT
        self.network.pores.invaded[self.bc.pi_list_inlet] = NWETT

    def __compute_time_step(self):
        assert np.all(self.network.pores.invaded[self.q_n > 0.0] == 1)
        dt, dt_details = self.time_stepper.timestep(flux_n=self.flux_n, source_nonwett=self.q_n)

        STOP_FLAG = False

        iter = 0
        while (dt == 0) and (iter < 4):
            logger.debug("Time step is zero. Attempting correction")
            logger.debug("Starting inner iteration" + str(iter))

            self.__invade_nonwetting_source_pores()

            update_pore_status(self.network, flux_n=self.flux_n, flux_w=self.flux_w, source_nonwett=self.q_n,
                               source_wett=self.q_w, bool_accounted_pores=self.bool_accounted_pores)

            self.__set_rhs_source_arrays(self.bc)
            ierr = self.__adjust_magnitude_sink_pores(WETT)
            if ierr == -1:
                STOP_FLAG = True
                break

            ierr = self.__adjust_magnitude_sink_pores(NWETT)
            if ierr == -1:
                STOP_FLAG = True
                break

            self.__update_capillary_pressure()

            self.k_comp.compute()  # Side effect- Computes network.tubes.k_n and k_w

            if np.sum(self.q_w) == 0 and np.sum(self.q_n) == 0:
                logger.info("Exiting loop because sources are zero")
                STOP_FLAG = True
                break

            logger.debug("Solving linear system")
            self.__solve_linear_system()

            iter += 1

        if STOP_FLAG is True:
            dt = 0.0
        else:
            dt, dt_details = self.time_stepper.timestep(flux_n=self.flux_n, source_nonwett=self.q_n)

        assert np.all(self.network.pores.invaded[self.q_n > 0.0] == 1)

        if dt == 0.0:
            logger.debug("Time step is zero after attempting correction")

        # Compute time-step for a stop_time criteria
        if self.stop_time is not None:
            assert self.stop_time >= self.time
            dt = min(self.stop_time - self.time, dt)

        return dt, dt_details

    def __solve_pressure_and_pore_status(self):
        network = self.network
        k_comp = self.k_comp
        pe_comp = self.pe_comp

        def interior_loop():
            self.__invade_nonwetting_source_pores()
            self.__update_capillary_pressure()

            logger.debug("Snapping off Tubes")
            snapoff_all_tubes(network, pe_comp)

            logger.debug("Computing Conductances")
            k_comp.compute()  # Side effect - Computes network.tubes.k_n and k_w
            # Set source and sink arrays
            self.__set_rhs_source_arrays(self.bc)

            ierr = self.__adjust_magnitude_sink_pores(WETT)
            if ierr == -1:
                print "cannot adjust wetting sinks"
                return ierr

            ierr = self.__adjust_magnitude_sink_pores(NWETT)
            if ierr == -1:
                print "cannot adjust nonwetting sinks"
                return ierr

            self.__solve_linear_system()
            self.__invade_nonwetting_source_pores()
            self.__update_capillary_pressure()

            logger.debug("Done with interior loop")

            return 0

        ierr = interior_loop()

        if ierr == -1:
            return ierr

        update_pore_status(self.network, flux_n=self.flux_n, flux_w=self.flux_w, source_nonwett=self.q_n,
                           source_wett=self.q_w, bool_accounted_pores=self.bool_accounted_pores)

        ierr = interior_loop()

        if ierr == -1:
            return ierr

        for ti in self.ti_freeze_displacement:
            self.ti_freeze_displacement[ti] += 1

        for key in list(self.ti_freeze_displacement.keys()):
            if self.ti_freeze_displacement[key] > 20:
                del self.ti_freeze_displacement[key]

        logger.debug("frozen tubes are currently %s", self.ti_freeze_displacement.keys())

        for iter in xrange(10000):
            is_event = update_tube_piston_w(network, self.piston_entry, self.flux_w, self.q_w)
            ierr = interior_loop()
            if ierr == -1:
                return -1
            if not is_event:
                break

        ti_piston_nonwett = get_piston_disp_tubes_nonwett(network, self.piston_entry, self.flux_n, self.q_n)
        ti_piston_nonwett = set(ti_piston_nonwett) - set(self.ti_freeze_displacement.keys())

        if len(ti_piston_nonwett) == 0:
            ti_piston_nonwett = get_piston_disp_tubes_nonwett(network, self.piston_entry, self.flux_n, self.q_n)

        for ti_nonwett in ti_piston_nonwett:
            invade_tube_nw(network, ti_nonwett)
            ierr = interior_loop()

            if ierr == -1:
                return -1

            ti_piston_wetting = get_piston_disp_tubes_wett(network, self.piston_entry, self.flux_w, self.q_w)

            for ti_wett in ti_piston_wetting:
                if ti_wett == ti_nonwett:
                    self.ti_freeze_displacement[ti_nonwett] = 1
                invade_tube_w(network, ti_wett)
                ierr = interior_loop()
                if ierr == -1:
                    return ierr

        if ierr == -1:
            return ierr

        update_pore_status(self.network, flux_n=self.flux_n, flux_w=self.flux_w, source_nonwett=self.q_n,
                           source_wett=self.q_w, bool_accounted_pores=self.bool_accounted_pores)

        ierr = interior_loop()
        if ierr == -1:
            return -1

        return 0

    def __update_capillary_pressure(self):
        self.pc_comp.compute()

        if len(self.bc.pi_list_inlet) > 0:
            self.network.pores.p_c[self.bc.pi_list_inlet] = self.bc.press_inlet_nw - self.bc.press_inlet_w

        if len(self.bc.pi_list_outlet) > 0:
            self.network.pores.p_c[self.bc.pi_list_outlet] = 0.001

    def __advance(self, stop_criterion, callback):
        self.network.pores.p_w[:] = 0.0
        self.network.pores.p_n[:] = 0.0

        network = self.network

        # Reset dictionary used to track frozen sink pores
        self.freeze_sink_nw = dict()

        self.q_w_tot_source = np.sum(self.q_w[self.bc.pi_list_w_source])
        self.q_w_tot_sink = np.sum(self.q_w[self.bc.pi_list_w_sink])
        self.q_w_tot = self.q_w_tot_source + self.q_w_tot_sink

        self.q_n_tot_source = np.sum(self.q_n[self.bc.pi_list_nw_source])
        self.q_n_tot_sink = np.sum(self.q_n[self.bc.pi_list_nw_sink])
        self.q_n_tot = self.q_n_tot_source + self.q_n_tot_sink

        logger.debug("Simulation Status. Time: %f , Saturation: %f", self.time, self.sat_comp.sat_nw())

        network.pores.invaded[self.bc.pi_list_nw_source] = NWETT

        #  Notes: During loop, self.q_n_tot and self.q_w_tot are NOT modified. But self.rhs_source_* are updated
        counter = 0
        time_init = self.time
        while True:
            _plist = self.bc.pi_list_inlet
            if len(_plist) > 0:
                self.network.pores.p_c[_plist] = self.bc.press_inlet_nw - self.bc.press_inlet_w
                self.network.pores.sat[_plist] = self.pc_comp.pc_to_sat_func(self.network.pores.r[_plist], self.network.pores.p_c[_plist])
            logger.debug("Simulation Status. Time: %f , Saturation: %f", self.time, self.sat_comp.sat_nw())

            assert np.all(network.pores.invaded[self.q_n > 0.0] == 1)

            logger.debug("Solving Pressure")
            self.__update_capillary_pressure()

            ierr = self.__solve_pressure_and_pore_status()
            self.__update_capillary_pressure()

            if ierr == -1:
                logger.warning("Exiting solver because nonwetting sinks or wetting sinks cannot be adjusted anymore")
                logger.warning("Time elapsed  Time elapsed is %e", self.time - time_init)
                logger.warning("Initial nonwetting source was  %e", self.q_n_tot_source)
                logger.warning("Initial nonwetting sink was  %e", self.q_n_tot_sink)

                break

            if self.bc.no_dirichlet:
                assert np.isclose(self.q_n_tot, np.sum(self.q_n), atol=max(abs(self.q_n_tot) * 1e-5, 1e-14)), "%g %g %d" % (self.q_n_tot, np.sum(self.q_n), counter)

            if self.bc.no_dirichlet:
                assert np.isclose(self.q_w_tot, np.sum(self.q_w), atol=max(abs(self.q_w_tot) * 1e-5, 1e-14)), "%g %g %d" % (self.q_w_tot, np.sum(self.q_w), counter)

            logger.debug("Computing time step")
            dt, dt_details = self.__compute_time_step()

            dt_ratio = dt_details["pc_crit_drain"]/dt_details["sat_n_double"]

            logger.debug("Updating saturation")
            if self.explicit:
                self.sat_comp.update_saturation(flux_n=self.flux_n, dt=dt, source_nonwett=self.q_n)
            else:
                dt = self.__update_saturation_implicit(dt=dt)

            pi_1 = network.edgelist[:,0]
            pi_2 = network.edgelist[:,1]
            self.cum_flux_tubes += (network.pores.p_n[pi_1] - network.pores.p_n[pi_2]) * network.tubes.k_n * dt

            _plist = self.bc.pi_list_inlet
            if len(_plist) > 0:
                self.network.pores.p_c[_plist] = self.bc.press_inlet_nw - self.bc.press_inlet_w
                self.network.pores.sat[_plist] = self.pc_comp.pc_to_sat_func(self.network.pores.r[_plist], self.network.pores.p_c[_plist])

            pi_sink_all = np.union1d(self.bc.pi_list_w_sink, self.bc.pi_list_nw_sink).astype(np.int)

            flux_nw_inlet = np.sum(self.q_n[self.bc.pi_list_nw_source])
            flux_nw_outlet = np.sum(self.q_n[pi_sink_all])

            flux_w_inlet = np.sum(self.q_w[self.bc.pi_list_w_source])
            flux_w_outlet = np.sum(self.q_w[pi_sink_all])

            self.accumulated_saturation += (flux_nw_inlet + flux_nw_outlet)*dt/network.total_pore_vol
            self.time += dt

            self.dt = dt
            pn_max = np.max(network.pores.p_n)
            pn_min = np.min(network.pores.p_n)

            pw_max = np.max(network.pores.p_w)
            pw_min = np.min(network.pores.p_w)

            logger.debug("")
            logger.debug("="*80)
            logger.debug("Nonwetting influx: %e,  outflux: %e", flux_nw_inlet, flux_nw_outlet)
            logger.debug("Wetting influx: %e,  outflux: %e", flux_w_inlet, flux_w_outlet)

            logger.debug("Max Pressures. NW: %e, W: %e", pn_max, pw_max)
            logger.debug("Min Pressures. NW: %e, W: %e", pn_min, pw_min)
            logger.debug("Max PC %e", np.max(network.pores.p_n-network.pores.p_w))

            logger.debug("Accumulated Nonwetting Saturation %e", self.accumulated_saturation)
            logger.debug("Time %e after timestep %e", self.time, dt)
            logger.debug("Time ratio %e", dt_ratio)
            logger.debug("="*80)
            logger.debug("")

            if dt_ratio < 1e-4 and counter > 1000:
                logger.warning("Exiting solver because time step too slow.")
                break

            if stop_criterion():
                logger.debug("Exiting solver because stopping criterion has been successfully reached")
                break

            # If either of the initial nonwetting sink and source is nonzero and they are now, then stop the simulation
            if (self.q_n_tot_source != 0.0) or (self.q_n_tot_sink != 0.0):

                if (flux_nw_inlet == 0.0) and (flux_nw_outlet == 0.0):
                    logger.warning("Exiting solver because both flux_nw_inlet and outlet are zero. Time elapsed is %e", self.time - time_init)
                    logger.warning("Initial nonwetting source was  %e", self.q_n_tot_source)
                    logger.warning("Initial nonwetting sink was  %e", self.q_n_tot_sink)
                    break

            if callback is not None:
                callback(self)

            counter += 1

        return self.time - time_init
