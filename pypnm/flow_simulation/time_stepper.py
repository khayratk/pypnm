import numpy as np
from pypnm.porenetwork.pore_element_models import JNModel
from pypnm.util.logging_pypnm import logger
from pypnm.porenetwork.constants import *
import sys

#  TODO: Remove sim_setting dependence
try:
    from sim_settings import *
except ImportError:
    sim_settings = dict()
    sim_settings["fluid_properties"] = dict()
    sim_settings["fluid_properties"]['gamma'] = 1.0


class DynamicTimeStepper(object):
    """
    A class to compute the time step for the dynamic pore-network simulation

    Parameters
    ----------

    network: The pore network object.
    mask_accounted_pores: bool ndarray, optional
        pores with a corresponding value of False in this array will be ignored when computing time-steps.
    pc_model: PC-SAT Class
        class specifying the pc_model of a the pc-saturation relationship
    """
    def __init__(self, network, mask_accounted_pores=None, pc_model=JNModel, delta_pc=0.01):
        self.network = network

        if mask_accounted_pores is None:
            mask_accounted_pores = np.ones(network.nr_p, dtype=np.bool)

        self.accounted_pores = mask_accounted_pores
        self.pc_model = pc_model
        self.delta_pc = delta_pc

    def __timestep_imbibition(self, dvn_dt):
        """
        Compute time-step for pores undergoing imbibition locally based on three criteria.
        These are:
            time for capillary pressure to change by delta_pc;
            and time for pore to be completely imbibed
        """
        network = self.network
        pores = network.pores
        dt_pc_diff_imbibe = dt_sat_w_double = dt_imbibe_pore = 1.0

        mask_imb = (dvn_dt < 0.0) & self.accounted_pores
        delta_pc = self.delta_pc

        if np.any(mask_imb):
            # Time step for emptying pore of nonwetting phase
            nwett_vol_imb = pores.vol[mask_imb] * pores.sat[mask_imb]
            dt_imbibe_pore = np.min(nwett_vol_imb / -dvn_dt[mask_imb])
            assert dt_imbibe_pore >= 0.0

            # Time step to doubling wetting volume
            wett_vol = pores.vol[mask_imb] - nwett_vol_imb
            dt_sat_w_double = np.min(wett_vol/-dvn_dt[mask_imb])
            assert dt_sat_w_double >= 0.0

        dt_sat_w_double = 1.0

        mask_pc_diff_criteria = (pores.sat > 0.75) & mask_imb

        if np.any(mask_pc_diff_criteria):
            # Criterion: capillary pressure is decreased by delta_pc
            sat_min_pc_diff_criteria = np.maximum(
                self.pc_model.pc_to_sat_func(r=pores.r[mask_pc_diff_criteria],
                                             p_c=pores.p_c[mask_pc_diff_criteria]*(1.-delta_pc),
                                             gamma=sim_settings['fluid_properties']['gamma'],
                                             G=pores.G,
                                             A_tot=pores.A_tot),
                0.0)

            nwett_vol_pc_diff = pores.vol[mask_pc_diff_criteria] * (pores.sat[mask_pc_diff_criteria] -
                                                                    sat_min_pc_diff_criteria)
            dt_pc_diff_imbibe = np.min(nwett_vol_pc_diff / -dvn_dt[mask_pc_diff_criteria])
            assert dt_pc_diff_imbibe >= 0.0

        dt_imb = min(dt_imbibe_pore, dt_sat_w_double, dt_pc_diff_imbibe)

        logger.debug("Imbibition time steps are: dt_pore_imbibe: %e, dt_sat_w_double: %e, pc_crit_imbibe: %e",
                     dt_imbibe_pore, dt_sat_w_double, dt_pc_diff_imbibe)

        # If time step is zero, then there is pore with zero nw-sat which is getting imbibed.
        # Print information for debugging.
        if dt_imb == 0.0:
            p_inds_dt_zero = (nwett_vol_imb == 0.0).nonzero()[0]
            logger.debug("indices with zero dt" + np.array_str(p_inds_dt_zero))
            for i in p_inds_dt_zero:
                network._log_pore_info(i)

            logger.debug("Time Step for imbibition is Zero")

        return dt_imb, {"pore_imbibe": dt_imbibe_pore,
                        "dt_sat_w_double": dt_sat_w_double,
                        "dt_pc_diff_imbibe": dt_pc_diff_imbibe}

    def __timestep_drainage(self, dvn_dt):
        """
        Compute time-step for pores undergoing drainage locally based on three criteria.
        These are:
            time for the nonwetting saturation to double;
            time for capillary pressure to change by delta_pc;
            and time for pore to be completely drained
        """
        delta_pc = self.delta_pc
        dt_pore_drain = dt_sat_n_double = dt_sat_n_max = 1.0

        network = self.network
        pores = network.pores

        pi_drain = ((dvn_dt > 0.0) & self.accounted_pores).nonzero()[0]

        assert np.all(network.pores.invaded[pi_drain] == NWETT)

        sat_max_pc_increment = self.pc_model.pc_to_sat_func(r=network.pores.r[pi_drain],
                                                            p_c=network.pores.p_c[pi_drain] * (1+delta_pc),
                                                            gamma=sim_settings['fluid_properties']['gamma'],
                                                            G=network.pores.G,
                                                            A_tot=network.pores.A_tot)

        sat_max = np.maximum(sat_max_pc_increment, 0.1)

        if len(pi_drain)>0:
            wett_vol = pores.vol[pi_drain] * (1. - pores.sat[pi_drain])
            nwett_vol_double = pores.vol[pi_drain] * np.maximum(pores.sat[pi_drain], 0.2)
            nwett_vol_max = pores.vol[pi_drain] * (sat_max - pores.sat[pi_drain])

            dt_pore_drain = np.min(wett_vol / dvn_dt[pi_drain])
            dt_sat_n_double = np.min(nwett_vol_double / dvn_dt[pi_drain])

            dt_sat_n_max = np.min(nwett_vol_max / dvn_dt[pi_drain])

            assert dt_pore_drain > dt_sat_n_max

            logger.debug("Drainage time steps are: dt_pore_drain: %e, dt_sat_n_double: %e, pc_crit_drain: %e",
                         dt_pore_drain, dt_sat_n_double, dt_sat_n_max)

        dt_drain = min(dt_pore_drain, dt_sat_n_double, dt_sat_n_max)

        assert dt_drain >= 0, "Drainage time steps are: dt_pore_drain: %e, dt_sat_n_double: %e, pc_crit_drain: %e"%(dt_pore_drain, dt_sat_n_double, dt_sat_n_max)

        if dt_drain == 0.0:
            sys.stderr.write("Time Step for drainage is zero!!! \n")

        return dt_drain, {"pore_drain": dt_pore_drain, "sat_n_double": dt_sat_n_double, "pc_crit_drain": dt_sat_n_max}

    def timestep(self, flux_n, source_nonwett=0.0):
        dvn_dt = source_nonwett - flux_n

        dt_drain, dt_drain_details = self.__timestep_drainage(dvn_dt)
        dt_imb, dt_imb_details = self.__timestep_imbibition(dvn_dt)

        dt = min(dt_drain, dt_imb)

        dt_details = dt_drain_details.copy()
        dt_details.update(dt_imb_details)

        assert dt >= 0.0, "Time step is negative. dt_drain: %e dt_imb %e"%(dt_drain, dt_imb)

        logger.debug("Time step is %e. dt_drain: %e, dt_imb: %e", dt, dt_drain, dt_imb)

        return dt, dt_details