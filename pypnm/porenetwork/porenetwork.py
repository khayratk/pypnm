import cPickle as pickle
import logging
import os

import h5py
import numpy as np

import component
from pypnm.porenetwork.constants import *
from pypnm.postprocessing.vtk_output import VtkWriter
from pypnm.util.bounding_box import BoundingBox

logger = logging.getLogger('pypnm.porenetwork')
from pypnm.postprocessing.plot_functions import plot_save_histogram


class PoreNetwork(object):
    @property
    def total_pore_vol(self):
        """
        Returns
        _______
        out: float
            Total volume of pores
        """
        return np.sum(self.pores.vol)

    @property
    def total_throat_vol(self):
        """
        Returns
        _______
        out: float
            Total volume of throats
        """
        return np.sum(self.tubes.vol)

    @property
    def total_vol(self):
        """
        Returns
        _______
        out: float
            Total void volume
        """
        return self.total_pore_vol + self.total_throat_vol

    def distribute_throat_volume_to_neighboring_pores(self):
        """
        Distributes volume in throats equally among the neighboring pores (proportional according to pore volumes).
        """
        pores_1 = self.edgelist[:, 0]
        pores_2 = self.edgelist[:, 1]
        vol_increment_pore_1 = self.tubes.vol*self.pores.vol[pores_1]/(self.pores.vol[pores_1]+self.pores.vol[pores_2])
        vol_increment_pore_2 = self.tubes.vol - vol_increment_pore_1
        np.add.at(self.pores.vol, pores_1, vol_increment_pore_1)
        np.add.at(self.pores.vol, pores_2, vol_increment_pore_2)
        self.tubes.vol[:] = 0

    def _fix_tubes_larger_than_ngh_pores(self):
        """
        Adjusts the radius of the tubes, if needed, to be at most 99% that of the smallest neighboring pore
        """
        pi_1 = self.edgelist[:, 0]
        pi_2 = self.edgelist[:, 1]
        mask_large_tubes = (self.tubes.r >= self.pores.r[pi_1]) | (self.tubes.r >= self.pores.r[pi_2])
        if np.sum(mask_large_tubes) > 0:
            logger.warn("Had to decrease radius of %d tubes so that they are smaller than neighboring pores",
                        np.sum(mask_large_tubes))

        self.tubes.r[mask_large_tubes] = 0.99 * np.minimum(self.pores.r[pi_1][mask_large_tubes],
                                                           self.pores.r[pi_2][mask_large_tubes])
        self._assert_tubes_smaller_than_ngh_pores()

    def _assert_tubes_smaller_than_ngh_pores(self):
        pi_1 = self.edgelist[:, 0]
        pi_2 = self.edgelist[:, 1]
        mask_large_tubes = (self.tubes.r >= self.pores.r[pi_1]) | (self.tubes.r >= self.pores.r[pi_2])
        assert not np.any(mask_large_tubes), "Some tubes are larger than neighboring pores"

    def _log_pore_info(self, i):
        """
        Logs diagnostic information of pores. Useful for debugging

        Parameters
        ----------
        i: int
            pore index

        """
        logger.debug("=" * 40)
        logger.debug("Information on Pore: %d", i)
        logger.debug("Pore type %e", self.pore_domain_type[i])
        logger.debug("Pore invasion status %e", self.pores.invaded[i])
        logger.debug("Ngh pores " + np.array_str(self.ngh_pores[i]))
        logger.debug("Ngh pore types " + np.array_str(self.pore_domain_type[self.ngh_pores[i]]))
        logger.debug("Ngh pore invasion status" + np.array_str(self.pores.invaded[self.ngh_pores[i]]))
        logger.debug("Ngh pore saturations" + np.array_str(self.pores.sat[self.ngh_pores[i]]))
        logger.debug("Ngh tubes" + np.array_str(self.ngh_tubes[i]))
        logger.debug("Ngh tubes invasion status" + np.array_str(self.tubes.invaded[self.ngh_tubes[i]]))
        logger.debug("Pore saturation %e", self.pores.sat[i])
        logger.debug("Pore capillary pressure %e", self.pores.p_c[i])
        logger.debug("Pore Nonwetting Pressure %e", self.pores.p_n[i])
        logger.debug("Pore Wetting Pressure %e", self.pores.p_w[i])
        logger.debug("Ngh pore Nonwetting Pressure" + np.array_str(self.pores.p_n[self.ngh_pores[i]]))
        logger.debug("Ngh pore Wetting Pressure" + np.array_str(self.pores.p_w[self.ngh_pores[i]]))
        logger.debug("Ngh pore Capillary Pressure" + np.array_str(self.pores.p_c[self.ngh_pores[i]]))
        logger.debug("Ngh tube Nonwetting conductances" + np.array_str(self.tubes.k_n[self.ngh_tubes[i]]))
        logger.debug("Ngh tube Wetting conductances" + np.array_str(self.tubes.k_w[self.ngh_tubes[i]]))
        logger.debug("=" * 40)

    def set_zero_volume_at_inout(self):
        self.set_zero_volume_pores(self.pi_in)
        self.set_zero_volume_pores(self.pi_out)

    def set_zero_volume_pores(self, pi_list=None):
        if pi_list is None:
            self.pores.vol[:] = self.pores.vol*1e-10
        else:
            self.pores.vol[pi_list] = self.pores.vol[pi_list]*1e-10

    def set_zero_volume_all_tubes(self):
        self.tubes.vol[:] = 0.0

    def set_radius_pores(self, pi_list, r):
        self.pores.r[pi_list] = r
        self.pores.init_area()
        self.pores.init_vol()

    def set_radius_tubes(self, ti_list, r):
        self.tubes.r[ti_list] = r
        self.tubes.init_area()
        self.tubes.init_vol()

    def set_inlet_pores_invaded_and_connected(self):
        self.pores.invaded[self.pi_in] = NWETT
        self.pores.connected[self.pi_in] = 1

    def set_face_pores_nonwetting(self, FACE):
        self.pores.invaded[self.pi_list_face[FACE]] = NWETT
        self.pores.connected[self.pi_list_face[FACE]] = 1

    def is_pore_invaded_and_disconnected(self, pi):
        return (self.pores.invaded[pi] == NWETT) and (self.pores.connected[pi] == 0)

    def _create_nr_nghs_array(self):
        self.nr_nghs = np.zeros(self.pores.nr, dtype=np.int32)

        nr_nghs = self.nr_nghs
        ngh_pores = self.ngh_pores
        ngh_tubes = self.ngh_tubes

        for s in xrange(self.nr_p):
            nr_nghs[s] = ngh_pores[s].size
            assert (ngh_pores[s].size == ngh_tubes[s].size)

    def _create_ngh_elements_arrays(self):
        self.ngh_pores = np.ones([self.pores.nr], dtype=np.object)
        self.ngh_tubes = np.ones([self.pores.nr], dtype=np.object)

        for s in xrange(self.nr_p):
            self.ngh_pores[s] = (self.p_adj[s][self.p_adj[s] > -1])
            self.ngh_tubes[s] = (self.pt_adj[s][self.pt_adj[s] > -1])

        del self.pt_adj
        del self.p_adj

    def _sort_ngh_pores(self):
        for s in xrange(self.nr_p):
            argsort_indices = np.argsort(self.ngh_pores[s])
            self.ngh_pores[s] = self.ngh_pores[s][argsort_indices]
            self.ngh_tubes[s] = self.ngh_tubes[s][argsort_indices]

    def _create_helper_properties(self):
        self._sort_ngh_pores()
        self._create_nr_nghs_array()
        self._init_pore_inout_type(self.inface, self.outface)

    def _init_pore_inout_type(self, inface, outface):
        """ Initialize the types of each pore. A pore is either
        an inlet pore, and outlet pore or a domain pore
        Args:
            inface: The ID of the face which is taken to be the inlet
            outface: The ID of the face which is taken to be the outlet"""

        assert (inface in FACES)
        assert (outface in FACES)
        assert (inface != outface)

        self.inface = inface
        self.outface = outface

        self.pore_domain_type = np.zeros(self.nr_p, dtype=np.int8)
        self.pore_domain_type[:] = DOMAIN
        self.pore_domain_type[self.pi_list_face[inface]] = INLET
        self.pore_domain_type[self.pi_list_face[outface]] = OUTLET

        self._init_inlet_pores_list()
        self._init_outlet_pores_list()
        self._init_domain_pores_list()

    def _init_inlet_pores_list(self):
        self.pi_in = (self.pore_domain_type == INLET).nonzero()[0]
        self.pi_in = self.pi_in.astype(dtype=np.int32)

    def _init_outlet_pores_list(self):
        self.pi_out = (self.pore_domain_type == OUTLET).nonzero()[0]
        self.pi_out = self.pi_out.astype(dtype=np.int32)

    def _init_domain_pores_list(self):
        self.pi_domain = (self.pore_domain_type == DOMAIN).nonzero()[0]
        self.pi_domain = self.pi_domain.astype(dtype=np.int32)

    def _set_zero_vol_outside_bbox(self, bbox):
        mask_pores = component.pore_mask_outside_bbox(self, bbox)
        self.pores.vol[mask_pores] = 0.0

        self.tubes.x = (self.pores.x[self.edgelist[:, 0]] + self.pores.x[self.edgelist[:, 1]]) / 2
        self.tubes.y = (self.pores.y[self.edgelist[:, 0]] + self.pores.y[self.edgelist[:, 1]]) / 2
        self.tubes.z = (self.pores.z[self.edgelist[:, 0]] + self.pores.z[self.edgelist[:, 1]]) / 2

        mask_tubes = component.tube_mask_outside_bbox(self, bbox)
        self.tubes.vol[mask_tubes] = 0.0

    def _fix_tube_property(self, array, name):
        mean_val = array.mean(axis=0)
        idx = np.where(array == 0.0)
        array[idx] = mean_val
        lenidx = len(idx[0])
        if lenidx > 0:
            ratio = 100 * lenidx / self.nr_t
            print "had to fix ", name, "for ", lenidx, "tubes,", ratio, "%"

    def _get_ngh_pores_conn_to_pore(self, pi):
        ngh_pores = self.ngh_pores[pi]
        ngh_tubes = self.ngh_tubes[pi]
        ngh_pores_conn_to_pore = ngh_pores[
            (self.pores.connected[ngh_pores] == 1) & (self.tubes.connected[ngh_tubes] == 1)]
        return ngh_pores_conn_to_pore

    def add_pores(self, pos_x, pos_y, pos_z, radii, G=None):
        """
        Adds pores to the pore-network
        :param pos_x: Array of X positions of the added pores
        :param pos_y: Array of Y positions of the added pores
        :param pos_z: Array of Z positions of the added pores
        :param radii: Array of radii of the added pores
        """
        assert len(pos_x) == len(pos_y)
        assert len(pos_y) == len(pos_z)
        nr_new_pores = len(pos_x)

        if G is None:
            G = np.ones(nr_new_pores) * np.mean(self.pores.G)

        self.nr_p += nr_new_pores
        ngh_pores_new = np.ones([nr_new_pores], dtype=np.object)
        ngh_tubes_new = np.ones([nr_new_pores], dtype=np.object)

        # Append ngh_pores and ngh_tubes arrays
        for s in xrange(nr_new_pores):
            ngh_pores_new[s] = np.array([], dtype=np.int32)
            ngh_tubes_new[s] = np.array([], dtype=np.int32)

        self.ngh_pores = np.hstack([self.ngh_pores, ngh_pores_new])
        self.ngh_tubes = np.hstack([self.ngh_tubes, ngh_tubes_new])

        self.pores.append_pores(pos_x, pos_y, pos_z, radii, G)

        self._create_helper_properties()

    def add_throats(self, edgelist, r=None, l=None, l_tot = None, G=None):
        """
        Adds throats to network

        Parameters
        ----------
        edgelist: ndarray
            Edge list array of size :math:`N_t \\times 2` where :math:`N_t` is the number of throats to be added.
        r: ndarray
            Array of size :math:`N_t`  containing the radii of the added throats
        l: ndarray
            Array of size :math:`N_t`  containing the lengths of the added throats
        l_tot: ndarray
            Array of size :math:`N_t`  containing the pore to pore distance of the added throats
        G: ndarray
            Array of size :math:`N_t`  containing the shape factors of the added throats
        """
        nr_new_tubes = np.shape(edgelist)[0]

        if r is None:
            r = np.ones(nr_new_tubes) * np.mean(self.tubes.r)

        if l is None:
            l = np.ones(nr_new_tubes) * np.mean(self.tubes.l)

        if l_tot is None:
            l_tot = l + self.pores.r[edgelist[:, 0]] + self.pores.r[edgelist[:, 1]]

        if G is None:
            G = np.ones(nr_new_tubes) * np.mean(self.tubes.G)

        self.tubes.append_tubes(r=r, l=l, G=G, l_tot = l_tot)

        self.edgelist = np.vstack((self.edgelist, edgelist))

        # Append ngh_pores and ngh_tubes arrays
        for x in xrange(nr_new_tubes):
            pi_1, pi_2 = edgelist[x, :]

            assert (pi_2 not in self.ngh_pores[pi_1])
            assert (pi_1 not in self.ngh_pores[pi_2])
            self.ngh_pores[pi_1] = np.append(self.ngh_pores[pi_1], pi_2)
            self.ngh_pores[pi_2] = np.append(self.ngh_pores[pi_2], pi_1)

            self.ngh_tubes[pi_1] = np.append(self.ngh_tubes[pi_1], self.nr_t + x)
            self.ngh_tubes[pi_2] = np.append(self.ngh_tubes[pi_2], self.nr_t + x)

            self.nr_nghs[pi_1] += 1
            self.nr_nghs[pi_2] += 1

        self.nr_t += nr_new_tubes

        self._create_helper_properties()

    def remove_throats(self, ti_list_delete):
        """
        Deletes throats from pore network

        Parameters
        ----------
        ti_list_delete: intarray
            Indices of throats to be deleted

        Notes
        ----------
        Throat indices will be adjust to remain continguous after deletion


        """
        assert len(np.unique(ti_list_delete)) == len(ti_list_delete)
        ti_list_old = np.arange(self.nr_t)
        ti_new_to_old = np.delete(ti_list_old, ti_list_delete)
        ti_old_to_new = {ti_new_to_old[i]: i for i in xrange(len(ti_new_to_old))}
        assert np.max(ti_old_to_new.values()) < self.nr_t - len(ti_list_delete)

        # Remove entries in ngh_pores and ngh_tubes arrays corresponding to deleted tubes
        for ti in ti_list_delete:
            pi_1, pi_2 = self.edgelist[ti, :]

            assert ti in self.ngh_tubes[pi_1]
            assert ti in self.ngh_tubes[pi_2]

            mask_pi_1 = self.ngh_tubes[pi_1] != ti
            mask_pi_2 = self.ngh_tubes[pi_2] != ti

            self.ngh_pores[pi_1] = self.ngh_pores[pi_1][mask_pi_1]
            self.ngh_pores[pi_2] = self.ngh_pores[pi_2][mask_pi_2]

            self.ngh_tubes[pi_1] = self.ngh_tubes[pi_1][mask_pi_1]
            self.ngh_tubes[pi_2] = self.ngh_tubes[pi_2][mask_pi_2]

            self.nr_nghs[pi_1] -= 1
            self.nr_nghs[pi_2] -= 1
            self.nr_t -= 1

        assert self.nr_t == len(ti_new_to_old)

        # Change indices of tubes in ngh_tubes arrays
        for pi in xrange(self.nr_p):
            new_ngh_tubes_pi = [ti_old_to_new[self.ngh_tubes[pi][x]] for x in xrange(self.nr_nghs[pi])]
            self.ngh_tubes[pi] = np.asarray(new_ngh_tubes_pi, dtype=np.int32)
            if len(new_ngh_tubes_pi) > 0:
                assert np.max(new_ngh_tubes_pi) < self.nr_t

        self.edgelist = np.delete(self.edgelist, ti_list_delete, 0)

        self.tubes.remove_tubes(ti_list_delete)

        assert self.edgelist.shape[0] == self.nr_t
        assert self.edgelist.shape[1] == 2

        self._create_helper_properties()

    def rotate_around_z_axis(self, angle):
        x, y, z = self.pores.x, self.pores.y, self.pores.z
        x_center, y_center, z_center = np.mean(x), np.mean(y), np.mean(z)

        x_r = x_center + (x - x_center) * np.cos(angle) - (y - y_center) * np.sin(angle)
        y_r = y_center + (x - x_center) * np.sin(angle) + (y - y_center) * np.cos(angle)

        self.pores.x[:] = x_r
        self.pores.y[:] = y_r

    def restrict_volume(self, boundingbox_percent):
        """
        Sets volume of pores and throats outside a bounding box to zero

        Parameters
        ----------
        boundingbox_percent: tuple
            bounding box in form of :math:`\\frac{x_{min}}{x_{max}-x_{min}}, \\frac{x_{max}}{x_{max}-x_{min}}, \\frac{y_{min}}{y_{max}-y_{min}}
            ,\\frac{y_{max}}{y_{max}-y_{min}},\\frac{z_{min}}{z_{max}-z_{min}},\\frac{z_{max}}{z_{max}-z_{min}}`

        Returns
        -------

        """
        bounding_box = BoundingBox(min(self.pores.x) + (max(self.pores.x) - min(self.pores.x)) * boundingbox_percent[0],
                                   min(self.pores.x) + (max(self.pores.x) - min(self.pores.x)) * boundingbox_percent[1],
                                   min(self.pores.y) + (max(self.pores.y) - min(self.pores.y)) * boundingbox_percent[2],
                                   min(self.pores.y) + (max(self.pores.y) - min(self.pores.y)) * boundingbox_percent[3],
                                   min(self.pores.z) + (max(self.pores.z) - min(self.pores.z)) * boundingbox_percent[4],
                                   min(self.pores.z) + (max(self.pores.z) - min(self.pores.z)) * boundingbox_percent[5])
        self._set_zero_vol_outside_bbox(bounding_box)

    def export_to_vtk(self, filename, folder_name="network_vtk"):
        """
        Exports network to vtk file

        Parameters
        ----------
        filename: str
            Name of vtk file
        folder_name: str
            Name of folder in which the files will be stored

        """
        vtkwriter = VtkWriter(self, folder_name)
        vtkwriter.write_vtk_binary_file(filename)

    def export_to_hdf(self, filename):
        """
        Exports network to an hdf file

        Parameters
        ----------
        filename: str
            Name of hdf file

        """
        with h5py.File(filename, 'w') as f:
            f.require_group("pores")
            f['/pores/index'] = np.arange(self.nr_p).astype(np.int)
            f['/pores/radius'] = self.pores.r
            f['/pores/shape_factor'] = self.pores.G
            f['/pores/vol'] = self.pores.vol
            f['/pores/clay_vol'] = self.pores.vol*0.01
            f['/pores/x'] = self.pores.x
            f['/pores/y'] = self.pores.y
            f['/pores/z'] = self.pores.z

            f.require_group("tubes")
            f['/tubes/index'] = np.arange(self.nr_t).astype(np.int)
            f['/tubes/pore_1'] = self.edgelist[:, 0]
            f['/tubes/pore_2'] = self.edgelist[:, 1]
            f['/tubes/radius'] = self.tubes.r
            f['/tubes/shape_factor'] = self.tubes.G
            f['/tubes/length_total'] = self.tubes.l_tot
            f['/tubes/length_pore_1'] = self.pores.l[self.edgelist[:, 0]]
            f['/tubes/length_pore_2'] = self.pores.l[self.edgelist[:, 1]]
            f['/tubes/length_throat'] = self.tubes.l
            f['/tubes/vol'] = self.tubes.vol
            f['/tubes/clay_vol'] = self.tubes.vol*0.01

    def write_network_statistics(self, folder_name="network_statistics"):
        """
        Writes plots displaying statistics of the porenetwork into a folder

        Parameters
        ----------
        folder_name: str
            Name of folder in which the plots will be stored

        """
        if folder_name[-1] != "/":
            folder_name += "/"

        os.system("rm -r " + folder_name)
        os.system("mkdir " + folder_name)
        plot_save_histogram(folder_name + "coord_number", [self.nr_nghs])
        plot_save_histogram(folder_name + "rad_tubes_hist", [self.tubes.r])
        plot_save_histogram(folder_name + "rad_pores_hist", [self.pores.r])
        plot_save_histogram(folder_name + "shape_factor", [self.pores.G, self.tubes.G])
        plot_save_histogram(folder_name + "Area", [self.pores.A_tot, self.tubes.A_tot])
        plot_save_histogram(folder_name + "length", [self.pores.l, self.tubes.l])
        plot_save_histogram(folder_name + "length_total", [self.tubes.l_tot])

    def save(self, filename):
        """
        Saves network to a pkl file

        Parameters
        ----------
        filename: str

        """
        import os
        directory = os.path.dirname(filename)


        try:
            if not directory == '':
                os.makedirs(directory)
        except OSError:
            if not os.path.isdir(directory):
                raise

        output_file = open(filename, 'wb')
        pickle.dump(self, output_file, protocol=pickle.HIGHEST_PROTOCOL)
        output_file.close()

    @classmethod
    def load(cls, filename):
        """
        loads network from a pkl file

        Parameters
        ----------
        filename: str

        """
        input_file = open(filename, 'rb')
        network = pickle.load(input_file)
        return network

