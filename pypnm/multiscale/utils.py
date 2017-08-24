from itertools import izip

import numpy as np
import pandas as pd
from PyTrilinos import Epetra, EpetraExt
from scipy.sparse import coo_matrix, diags

from pypnm.linalg.trilinos_interface import solve_aztec, sum_of_columns
from pypnm.porenetwork.pore_element_models import JNModel
from pypnm.flow_simulation.simulation_bc import SimulationBoundaryCondition


def create_inter_processor_edgelist(graph, my_subgraph_ids, my_id):
    proc_id = np.asarray(graph.vs["proc_id"], dtype=np.int32)
    subgraph_id = graph.vs["subgraph_id"]
    global_id = np.asarray(graph.vs["global_id"], dtype=np.int32)

    inter_processor_edgelist = [(u, v) if proc_id[u] == my_id else (v, u) for (u, v) in graph.get_edgelist()
                             if (
                                 ((proc_id[u] == my_id and proc_id[v] != my_id) or
                                  (proc_id[u] != my_id and proc_id[v] == my_id)) and
                                 ((subgraph_id[u] in my_subgraph_ids) or (subgraph_id[v] in my_subgraph_ids))
                             )
                             ]

    eids = graph.get_eids(pairs=inter_processor_edgelist)

    inter_processor_edgelist = np.asarray(inter_processor_edgelist)

    inter_processor_edges = dict()

    for attr in ["G", "global_id", "l", "A_tot", "r"]:
        inter_processor_edges[attr] = np.asarray(graph.es[eids][attr])

    inter_processor_edges["invaded"] = np.zeros(len(inter_processor_edges["r"]), dtype=np.int32)

    if len(inter_processor_edgelist) == 0:
            inter_processor_edges["edgelist"] = [[], []]
    else:

        vertices_1 = global_id[inter_processor_edgelist.T[0]]
        vertices_2 = global_id[inter_processor_edgelist.T[1]]

        proc_1 = proc_id[inter_processor_edgelist.T[0]]
        proc_2 = proc_id[inter_processor_edgelist.T[1]]

        assert np.all(proc_1 != proc_2)

        inter_processor_edges["edgelist"] = [vertices_1, vertices_2]

    return inter_processor_edges


def update_inter_invasion_status_snap_off(inter_edges, p_c):
    inter_edgelist_local_1 = np.asarray([p_c.Map().LID(i) for i in inter_edges['edgelist'][0]], dtype=np.int32)
    inter_edgelist_local_2 = np.asarray([p_c.Map().LID(i) for i in inter_edges['edgelist'][1]], dtype=np.int32)
    assert np.all(inter_edgelist_local_1>=0)
    assert np.all(inter_edgelist_local_2>=0)

    inter_pc_edge = np.maximum(p_c[inter_edgelist_local_1], p_c[inter_edgelist_local_2])

    gamma = 1.0
    snap_off_press = 1.001 * JNModel.snap_off_pressure(gamma=gamma, r=inter_edges["r"])

    for eid in xrange(len(inter_pc_edge)):
        cond1 = (inter_pc_edge[eid] < snap_off_press[eid])
        cond2 = (inter_edges["invaded"][eid] == 1)
        if cond1 and cond2:
            print "Inter subgraph snap-off at global edge id %d"%inter_edges["global_id"][eid]
            inter_edges["invaded"][eid] = 0
    return inter_edges


def update_inter_invasion_status_piston(inter_edges, p_w, p_c, sat, outflux_n, source_nonwett):
    pores_1 = np.asarray([p_c.Map().LID(i) for i in inter_edges['edgelist'][0]], dtype=np.int32)  # local id
    pores_2 = np.asarray([p_c.Map().LID(i) for i in inter_edges['edgelist'][1]], dtype=np.int32)  # local id
    sat_crit = 0.5
    gamma = 1.0
    entry_pressure = 1.001 * JNModel.piston_entry_pressure(G=inter_edges["G"], r=inter_edges["r"], gamma=gamma)

    sat_above_crit_1 = (sat[pores_1] > sat_crit)
    sat_above_crit_2 = (sat[pores_2] > sat_crit)

    tubes_wett = (inter_edges["invaded"] == 0)

    pc_above_pe_1 = p_c[pores_1] >= entry_pressure
    pc_above_pe_2 = p_c[pores_2] >= entry_pressure

    condition1 = pc_above_pe_1 & sat_above_crit_1
    condition2 = pc_above_pe_2 & sat_above_crit_2

    condition = (condition1 | condition2) & tubes_wett

    for eid in xrange(len(condition)):
        if condition[eid]:
            print "Inter subgraph nonwetting piston displacement at global edge id %d"%inter_edges["global_id"][eid]
            inter_edges["invaded"][eid] = 1

    return inter_edges


def update_inter_invasion_status_piston_wetting(inter_edges, p_w, p_c, sat):
    p_n = p_w + p_c
    inter_edgelist_local_1 = np.asarray([p_c.Map().LID(i) for i in inter_edges['edgelist'][0]], dtype=np.int32)
    inter_edgelist_local_2 = np.asarray([p_c.Map().LID(i) for i in inter_edges['edgelist'][1]], dtype=np.int32)

    sat_crit = 0.001  # Important. This has to be the same as the sat_crit in dynamic simulator
    sat_below_crit_1 = (sat[inter_edgelist_local_1] < sat_crit)
    sat_below_crit_2 = (sat[inter_edgelist_local_2] < sat_crit)

    condition1 = sat_below_crit_1 & (p_n[inter_edgelist_local_1] > p_n[inter_edgelist_local_2])
    condition2 = sat_below_crit_2 & (p_n[inter_edgelist_local_2] > p_n[inter_edgelist_local_1])

    tubes_nwett = (inter_edges["invaded"] == 1)
    condition = (condition1 | condition2) & tubes_nwett

    for eid in xrange(len(condition)):
        if condition[eid]:
            print "Inter subgraph wetting piston displacement at global edge id %d"%inter_edges["global_id"][eid]
            inter_edges["invaded"][eid] = 0

    return inter_edges


def create_matrix(unique_map, edge_attributes, subnetworks=None, inter_processor_edges=None, inter_subgraph_edges=None, matformat="trilinos"):
    A = Epetra.CrsMatrix(Epetra.Copy, unique_map, 30)
    my_global_elements_set = set(unique_map.MyGlobalElements())

    row_lists = []
    col_lists = []
    val_lists = []

    if inter_processor_edges is not None:
        vertices_1 = inter_processor_edges['edgelist'][0]
        vertices_2 = inter_processor_edges['edgelist'][1]

        if len(vertices_1) > 0:
            assert set(vertices_1) <= my_global_elements_set, inter_processor_edges
            assert not set(vertices_2) <= my_global_elements_set, inter_processor_edges

            for attr in edge_attributes:
                row_lists.append(vertices_1)
                col_lists.append(vertices_2)
                val_lists.append(inter_processor_edges[attr])

    if inter_subgraph_edges is not None:
        vertices_1 = inter_subgraph_edges['edgelist'][0]
        vertices_2 = inter_subgraph_edges['edgelist'][1]

        assert set(vertices_1) <= my_global_elements_set, inter_subgraph_edges
        assert set(vertices_2) <= my_global_elements_set, inter_subgraph_edges

        for attr in edge_attributes:
            row_lists.append(vertices_1)
            col_lists.append(vertices_2)
            val_lists.append(inter_subgraph_edges[attr])
            row_lists.append(vertices_2)
            col_lists.append(vertices_1)
            val_lists.append(inter_subgraph_edges[attr])

    if subnetworks is not None:
        for i in subnetworks:
            vertices_1_local = subnetworks[i].edgelist[:, 0]
            vertices_2_local = subnetworks[i].edgelist[:, 1]
            vertices_1_global = subnetworks[i].pi_local_to_global[vertices_1_local]
            vertices_2_global = subnetworks[i].pi_local_to_global[vertices_2_local]

            assert set(vertices_1_global) <= my_global_elements_set
            assert set(vertices_2_global) <= my_global_elements_set

            cond = np.zeros(subnetworks[i].tubes.nr)
            for attr in edge_attributes:
                cond += getattr(subnetworks[i].tubes, attr)

            row_lists.append(vertices_1_global)
            col_lists.append(vertices_2_global)
            val_lists.append(cond)
            row_lists.append(vertices_2_global)
            col_lists.append(vertices_1_global)
            val_lists.append(cond)

    if matformat == "trilinos":
        for row, col, val in izip(row_lists, col_lists, val_lists):
            ierr = A.InsertGlobalValues(row, col, val)
            assert ierr == 0, ierr

        A.FillComplete()

        ones = Epetra.Vector(unique_map)
        ones.PutScalar(1.0)

        x = Epetra.Vector(unique_map)
        A.Multiply(False, ones, x)

        D = Epetra.CrsMatrix(Epetra.Copy, unique_map, 1)
        row_inds = D.Map().MyGlobalElements()
        D.InsertGlobalValues(row_inds, row_inds, x)

        EpetraExt.Add(A, False, -1.0, D, 1.0)
        D.FillComplete()
        check = sum_of_columns(D)
        if check:
            error = np.max(np.abs(check))
            assert error < 1.e-14, error

        return D

    if matformat == "scipy":
        N = unique_map.NumGlobalElements()
        row = np.concatenate(row_lists)
        col = np.concatenate(col_lists)
        val = np.concatenate(val_lists)
        A = coo_matrix((val, (row, col)), shape=(N, N))
        ones = np.ones(N)
        x = A*ones
        D = diags(x)
        A = D-A
        error = np.max(np.abs(A*ones))
        assert error < 1.e-14, error
        return A


def create_subnetwork_boundary_conditions(source_n, source_w, my_subnetworks, subgraph_ids,
                                          pressure_inlet=None, pressure_outlet=None):
    """
    Parameters
    ----------
    source_n: Epetra Vector
        Nonwetting volume source
    source_w: Epetra Vector
        Wetting volume source
    my_subnetworks: dict
        Dictionary containing references to subnetworks handled by current processor
    subgraph_ids: Epetra Vector
        The subgraph id to which each pore belongs to
    pressure_inlet: tuple
        specifies the inlet pressure boundary condition
    pressure_outlet: tuple
        specifies the outlet pressure boundary condition
    Returns
    -------
    bc: dictionary of boundary conditions for each subnetwork to be used in a dynamic simulation.

    Notes
    ______
    source_n, source_w, and subgraph_ids have to be the same size N = sum of pores of all subnetworks across all processors.

    """
    eps = 1e-30

    mask_w = np.abs(source_w) > eps

    assert (len(mask_w) > np.sum(mask_w)) or (len(mask_w) == 0)

    coarse_ids_boundaries_w = subgraph_ids[mask_w]

    flux_boundaries_w = source_w[mask_w]
    global_id_boundaries_w = source_w.Map().MyGlobalElements()[mask_w]

    df_w = pd.DataFrame(
        data={'pores_id': global_id_boundaries_w, 'network_id': coarse_ids_boundaries_w,
              'source_wett': flux_boundaries_w, 'source_nonwett': flux_boundaries_w*0.0})

    mask_n = np.abs(source_n) > eps

    assert (len(mask_n) > np.sum(mask_n)) or (len(mask_n) == 0)

    coarse_ids_boundaries_n = subgraph_ids[mask_n]

    flux_boundaries_n = source_n[mask_n]
    global_id_boundaries_n = source_n.Map().MyGlobalElements()[mask_n]

    df_n = pd.DataFrame(
        data={'pores_id': global_id_boundaries_n, 'network_id': coarse_ids_boundaries_n,
              'source_wett': flux_boundaries_n*0.0, 'source_nonwett': flux_boundaries_n})

    df = df_w.append(df_n)

    pi_list_sink_wett = dict()
    q_list_sink_wett = dict()
    pi_list_source_wett = dict()
    q_list_source_wett = dict()
    pi_list_sink_nonwett = dict()
    q_list_sink_nonwett = dict()
    pi_list_source_nonwett = dict()
    q_list_source_nonwett = dict()
    df_indexed_by_network_id = df.set_index('network_id').sort_index()

    bc = dict()
    for i in my_subnetworks:
        dataset = df_indexed_by_network_id.loc[i][['pores_id', 'source_wett', 'source_nonwett']].values.T
        pi_list = dataset[0].astype(np.int)
        q_list_wett = dataset[1].astype(np.float)
        q_list_nonwett = dataset[2].astype(np.float)

        subnetwork = my_subnetworks[i]

        pi_list_sink_wett[i] = subnetwork.pi_list_from_global(pi_list[q_list_wett < 0])
        q_list_sink_wett[i] = q_list_wett[q_list_wett < 0]

        pi_list_source_wett[i] = subnetwork.pi_list_from_global(pi_list[q_list_wett > 0])
        q_list_source_wett[i] = q_list_wett[q_list_wett > 0]

        pi_list_source_nonwett[i] = subnetwork.pi_list_from_global(pi_list[q_list_nonwett > 0])
        q_list_source_nonwett[i] = q_list_nonwett[q_list_nonwett > 0]

        pi_list_sink_nonwett[i] = subnetwork.pi_list_from_global(pi_list[q_list_nonwett < 0])
        q_list_sink_nonwett[i] = q_list_nonwett[q_list_nonwett < 0]

        bc[i] = SimulationBoundaryCondition()
        bc[i].set_nonwetting_source(pi_list_source_nonwett[i], q_list_source_nonwett[i])
        bc[i].set_wetting_source(pi_list_source_wett[i], q_list_source_wett[i])
        bc[i].set_nonwetting_sink(pi_list_sink_nonwett[i], q_list_sink_nonwett[i])
        bc[i].set_wetting_sink(pi_list_sink_wett[i], q_list_sink_wett[i])

        pi_list = np.intersect1d(pressure_inlet["pi_list"], subnetwork.pi_local_to_global)

        if len(pi_list) > 0:
            pi_list_local = subnetwork.pi_list_from_global(pi_list)
            bc[i].set_pressure_inlet(pi_list_local, pressure_inlet["wett"], pressure_inlet["nwett"])

        pi_list = np.intersect1d(pressure_outlet["pi_list"], subnetwork.pi_local_to_global)
        if len(pi_list) > 0:
            pi_list_local = subnetwork.pi_list_from_global(pi_list)
            bc[i].set_pressure_outlet(pi_list_local, pressure_outlet["wett"])

        if bc[i].no_dirichlet:
            mass_balance = bc[i].mass_balance()
            total_sources = np.sum(q_list_source_nonwett[i]) + np.sum(q_list_source_wett[i])
            assert abs(mass_balance) < total_sources * 1.e-8, "Mass balance: %e, Total sources: %e" % (mass_balance, total_sources)

    return bc


def create_rhs(unique_map, my_subnetworks, inter_processor_edges, inter_subgraph_edges, p_c, global_source_wett,
                 global_source_nonwett, vecformat="trilinos"):
    A_nw = create_matrix(unique_map, ["k_n"], my_subnetworks, inter_processor_edges, inter_subgraph_edges)

    source_capillary = Epetra.Vector(unique_map)
    ierr = A_nw.Multiply(False, p_c, source_capillary)
    assert ierr == 0
    b = (global_source_wett + global_source_nonwett) - source_capillary

    if vecformat == "trilinos":
        return b
    if vecformat == "numpy":
        return np.copy(b)


def solve_multiscale(ms, A, b, p_c, p_w=None, smooth_prolongator=True, tol=1.0e-5):
    ms.A = A
    if smooth_prolongator:
        ms.smooth_prolongation_operator(10)
    if p_w is None:
        p_w = Epetra.Vector(A.RangeMap())
    p_w = ms.iterative_solve(b, p_w, tol=tol)
    p_n = p_w + p_c
    return p_n, p_w