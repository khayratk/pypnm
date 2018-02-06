"""
Implementation of gradient governed percolation from scratch using pypnm
"""

import sys
import petsc4py
from mpi4py import MPI

petsc4py.init(sys.argv)
from petsc4py import PETSc

import cProfile
import pstats

import numpy as np
from numpy.random import choice

from pypnm.linalg.laplacianmatrix import laplacian_from_network
from pypnm.linalg.petsc_interface import  scipy_to_petsc_matrix
from pypnm.porenetwork.constants import WEST, EAST
from pypnm.porenetwork.network_factory import unstructured_network_delaunay
from pypnm.porenetwork.porenetwork import PoreNetwork


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
    comm = MPI.COMM_SELF

    try:
        network = PoreNetwork.load("benchmark_network.pkl")

    except IOError:
        network = unstructured_network_delaunay(200000, quasi_2d=True)
        network.save("benchmark_network.pkl")

    mu_w = 1.0
    mu_n = 0.01
    gamma = 1.0
    network.pores.invaded[network.pi_list_face[WEST]] = 1
    pressure = np.zeros(network.nr_p)
    entry_pressure = 2 * gamma / network.tubes.r

    ds_out = 0.01
    s_target = 0.01
    n_out = 0

    sf = 1.0e20

    tube_conductances = np.pi * (network.tubes.r ** 4) / (8. * network.tubes.l * mu_w)

    A = laplacian_from_network(network, weights=tube_conductances, ind_dirichlet=network.pi_list_face[EAST])
    source = np.zeros(network.nr_p)
    source[network.pi_list_face[WEST]] = 1.e-9
    outlet_pore_set = set(network.pi_list_face[EAST])

    ksp = PETSc.KSP().create(comm=comm)
    ksp.setTolerances(rtol=1e-8, max_it=10000)
    ksp.setType("minres")
    pc = ksp.getPC()
    pc.setType("gamg")
    ksp.setInitialGuessNonzero(True)
    ksp.setFromOptions()
    petsc_mat = scipy_to_petsc_matrix(A * sf)
    ksp.setOperators(A=petsc_mat)

    change_list = []
    try:
        for niter in xrange(4000000):
            petsc_mat = scipy_to_petsc_matrix(A*sf)
            _, P = ksp.getOperators()

            if niter %10 == 0:
                P = petsc_mat

            ksp.setOperators(A=petsc_mat, P=P)
            petsc_rhs = PETSc.Vec().createWithArray(source * sf, comm=comm)
            petsc_sol = PETSc.Vec().createWithArray(pressure, comm=comm)
            ksp.solve(petsc_rhs, petsc_sol)

            pressure = petsc_sol.getArray()

            displacing_tubes, prob = get_interface_invasion_throats(network, pressure, entry_pressure,
                                                                    tube_conductances)

            ti_displaced = choice(displacing_tubes, 1, p=prob)[0]
            network.tubes.invaded[ti_displaced] = 1
            network.pores.invaded[network.edgelist[ti_displaced]] = 1

            # Update CSR matrix
            g_12_new = np.pi * (network.tubes.r[ti_displaced] ** 4) / (8. * network.tubes.l[ti_displaced] * mu_n)
            p1, p2 = network.edgelist[ti_displaced]
            change_list.append(g_12_new + A[p1, p2])

            if p1 not in outlet_pore_set:
                A[p1, p1], A[p1, p2] = A[p1, p1] + A[p1, p2] + g_12_new, -g_12_new
            if p2 not in outlet_pore_set:
                A[p2, p2], A[p2, p1] = A[p2, p2] + A[p2, p1] + g_12_new, -g_12_new

            tube_conductances[ti_displaced] = g_12_new

            saturation = np.sum(network.tubes.vol[network.tubes.invaded == 1]) / network.total_vol + np.sum(
                network.pores.vol[network.pores.invaded == 1]) / network.total_vol

            if saturation > s_target:
                network.pores.p_n[:] = pressure
                print "Number of throats invaded:", niter
                network.save("network_history/network"+str(n_out).zfill(5)+".pkl")
                s_target += ds_out
                n_out += 1

            if niter % 10 == 0:

                print "Number of throats invaded:", niter
                print "saturation is:", saturation

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
