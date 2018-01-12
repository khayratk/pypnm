from pypnm.porenetwork.network_factory import structured_network
from pypnm.porenetwork.constants import WEST, EAST
from pypnm.linalg.laplacianmatrix import laplacian_from_network, LaplacianMatrix
from pypnm.linalg.petsc_interface import petsc_solve_lu, petsc_solve
from pypnm.linalg.trilinos_interface import solve_pressure_trilinos_from_scipy, solve_pressure_mltrilinos_from_scipy
from pypnm.linalg.linear_system_solver import solve_pyamg
from pypnm.linalg.trilinos_interface import matrix_scipy_to_epetra, vector_numpy_to_epetra
import numpy as np
from numpy.random import choice

import cProfile
import pstats
import time

from pypnm.util.hd5_output import add_field_to_hdf_file
from pypnm.util.utils import require_path


def write_to_hdf(network, label, folder_name):
    require_path(folder_name)
    filename = folder_name + "/hdf_net.h5"

    add_field_to_hdf_file(filename, label, "p_n", network.pores.p_n)
    add_field_to_hdf_file(filename, label, "pore_invaded", network.pores.invaded)
    add_field_to_hdf_file(filename, label, "tube_invaded", network.tubes.invaded)

    add_field_to_hdf_file(filename, 0, "G", network.pores.G)
    add_field_to_hdf_file(filename, 0, "pore_r", network.pores.r)

    add_field_to_hdf_file(filename, 0, "pore_x", network.pores.x)
    add_field_to_hdf_file(filename, 0, "pore_y", network.pores.y)
    add_field_to_hdf_file(filename, 0, "pore_z", network.pores.z)

    add_field_to_hdf_file(filename, 0, "tube_r", network.tubes.r)
    add_field_to_hdf_file(filename, 0, "tube_l", network.tubes.l)
    add_field_to_hdf_file(filename, 0, "pore_vol", network.pores.vol)

def get_interface_invasion_throats(network, pressure, entry_pressure, tube_conductances):
    pore_list_1 = network.edgelist[:, 0]
    pore_list_2 = network.edgelist[:, 1]

    pores_invaded_1 = network.pores.invaded[pore_list_1]
    pores_invaded_2 = network.pores.invaded[pore_list_2]

    press_diff = pressure[pore_list_1] - pressure[pore_list_2]

    intfc_tubes_mask = (pores_invaded_1 | pores_invaded_2) & (network.tubes.invaded == 0)
    intfc_tubes = intfc_tubes_mask.nonzero()[0]

    pressure_drop_1 = (pores_invaded_1 * press_diff - entry_pressure)
    pressure_drop_2 = (pores_invaded_2 * (-press_diff) - entry_pressure)
    pressure_drop = np.maximum(pressure_drop_1, pressure_drop_2) * tube_conductances
    displacing_tubes = intfc_tubes[pressure_drop[intfc_tubes] > 0]

    prob = pressure_drop[displacing_tubes] / np.sum(pressure_drop[displacing_tubes])

    if len(displacing_tubes) == 0:
        displacing_tubes = [intfc_tubes[np.argmax(network.tubes.r[intfc_tubes])]]
        prob = [1.0]

    return displacing_tubes, prob


def run():
    network = structured_network(50, 50, 50)
    mu_w = 1.0
    mu_n = 0.1
    gamma = 1.0
    network.pores.invaded[network.pi_list_face[WEST]] = 1
    pressure = np.zeros(network.nr_p)
    entry_pressure = 2 * gamma / network.tubes.r

    sf = 1.0e20

    tube_conductances = np.pi * (network.tubes.r ** 4) / (8. * network.tubes.l * mu_w)

    A = laplacian_from_network(network, weights=tube_conductances, ind_dirichlet=network.pi_list_face[EAST])
    source = np.zeros(network.nr_p)
    source[network.pi_list_face[WEST]] = 1.e-9
    outlet_pore_set = set(network.pi_list_face[EAST])
    try:

        for niter in xrange(100000):
            # start = time.time()

            pressure = petsc_solve(A * sf, source * sf, x0=pressure, ksptype="minres", tol=1e-10)
            # pressure = solve_pressure_trilinos_from_scipy(A*sf, source*sf, x0=pressure, tol=1e-7)
            # pressure = solve_pyamg(A*sf, source*sf, x0=pressure, tol=1e-10)
            # pressure = solve_pressure_mltrilinos_from_scipy(A*sf, source*sf, x0=pressure, tol=1e-10)

            # end = time.time()
            # print(end - start)

            displacing_tubes, prob = get_interface_invasion_throats(network, pressure, entry_pressure,
                                                                    tube_conductances)

            ti_displaced = choice(displacing_tubes, 1, p=prob)[0]
            network.tubes.invaded[ti_displaced] = 1
            network.pores.invaded[network.edgelist[ti_displaced]] = 1

            # Update CSR matrix
            g_12_new = np.pi * (network.tubes.r[ti_displaced] ** 4) / (8. * network.tubes.l[ti_displaced] * mu_n)
            p1, p2 = network.edgelist[ti_displaced]
            if p1 not in outlet_pore_set:
                A[p1, p1], A[p1, p2] = A[p1, p1] + A[p1, p2] + g_12_new, -g_12_new
            if p2 not in outlet_pore_set:
                A[p2, p2], A[p2, p1] = A[p2, p2] + A[p2, p1] + g_12_new, -g_12_new

            tube_conductances[ti_displaced] = g_12_new

            if niter % 100 == 0:
                network.pores.p_n[:] = pressure
                network.export_to_vtk("output_viscous" + str(niter).zfill(4))

                print "Number of throats invaded:", niter
                write_to_hdf(network, niter, "hdf5_output")


    except KeyboardInterrupt:
        pass


def print_profiling_info(filename):
    p = pstats.Stats(filename)
    p.sort_stats('cumulative').print_stats(50)
    p.sort_stats('time').print_stats(50)


if __name__ == "__main__":
    exec_string = 'run()'

    cProfile.run(exec_string, 'restats')
    print_profiling_info('restats')


