from pypnm.porenetwork.structured_porenetwork import StructuredPoreNetwork
from pypnm.porenetwork.structured_porenetwork_27 import StructuredPoreNetwork27
from pypnm.porenetwork.statoil_porenetwork import StatoilPoreNetwork
from pypnm.porenetwork.network_manipulation import prune_network, reorder_network
from pypnm.porenetwork.unstructured_porenetwork import create_unstructured_network
from pypnm.porenetwork.delaunay_network import create_delaunay_network
from scipy.stats import beta, randint
import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from pypnm.util.sphere_packing import sphere_packing
from pypnm.porenetwork.coordination_number import choose_edges_for_target_coord_num
from pypnm.porenetwork import component


def cube_network(N, media_type="consolidated"):
    dist_p2p = 160e-6
    network = StructuredPoreNetwork([N, N, N], dist_p2p, media_type=media_type)
    return network


def cube_network_27(N):
    dist_p2p = 160e-6
    network = StructuredPoreNetwork27([N, N, N], dist_p2p)
    return network


def square_network(N):
    dist_p2p = 160e-6
    network = StructuredPoreNetwork([N, N, 1], dist_p2p)
    return network


def structured_network(Nx, Ny, Nz, media_type="consolidated", periodic=False):
    dist_p2p = 160e-6
    network = StructuredPoreNetwork([Nx, Ny, Nz], dist_p2p, media_type=media_type, periodic=periodic)
    network = reorder_network(network)  # Important for efficiency

    return network


def structured_network_27(Nx, Ny, Nz, media_type="consolidated", periodic=False):
    dist_p2p = 160e-6
    network = StructuredPoreNetwork27([Nx, Ny, Nz], dist_p2p, media_type=media_type, periodic=periodic)
    return network


def structured_network_uniform_coord_number(Nx, Ny, Nz, ngh_min=8, ngh_max=12,
                                            media_type="consolidated", periodic=False):
    network = structured_network_27(Nx, Ny, Nz, media_type=media_type, periodic=periodic)

    target_coord_number = np.minimum(network.nr_nghs-1, np.random.randint(ngh_min, ngh_max+1))

    ti_list = choose_edges_for_target_coord_num(network, target_coord_number)
    ti_list_remove = component.complement_tube_set(network, ti_list)
    network.remove_throats(ti_list_remove)
    network = prune_network(network, [-0.1, 1.1, -0.1, 1.1, -0.1, 1.1])

    return network

def unstructured_network(nr_pores, domain_size=None, quasi_2d=False):
    r_min, r_max = 20e-6, 75e-6
    pdf_pore_radius = beta(1.25, 1.5, loc=r_min, scale=(r_max - r_min))

    r_min, r_max = 1e-6, 25e-6
    pdf_tube_radius = beta(1.5, 2, loc=r_min, scale=(r_max - r_min))

    pdf_coord_number = randint(low=4, high=12)

    if domain_size is None:
        if quasi_2d:
            r_max = 75e-6
            domain_length = (4. / 3. * np.pi * r_max ** 3 * nr_pores) ** (1. / 3.)
            domain_size = [2.5 * domain_length, 2.5*domain_length, domain_length/8.]
        else:
            r_max = 75e-6
            domain_length = (4. / 3. * np.pi * r_max ** 3 * nr_pores) ** (1. / 3.)
            domain_size = [2 * domain_length, domain_length, domain_length]

    network = create_unstructured_network(nr_pores, pdf_pore_radius=pdf_pore_radius,
                                          pdf_tube_radius=pdf_tube_radius,
                                          pdf_coord_number=pdf_coord_number,
                                          domain_size=domain_size, is_2d=False)
    network = reorder_network(network)  # Important for efficiency
    return network


def unstructured_network_delaunay(nr_pores, domain_size=None, quasi_2d=False, body_throat_corr_param=None):
    r_min, r_max = 20e-6, 75e-6
    pdf_pore_radius = beta(1.25, 1.5, loc=r_min, scale=(r_max - r_min))

    r_min, r_max = 1e-6, 25e-6
    pdf_tube_radius = beta(1.5, 2, loc=r_min, scale=(r_max - r_min))

    if domain_size is None:
        if quasi_2d:
            r_max = 75e-6
            domain_length = (4. / 3. * np.pi * r_max ** 3 * nr_pores) ** (1. / 3.)
            domain_size = [2.5 * domain_length, 2.5*domain_length, domain_length/8.]
        else:
            r_max = 75e-6
            domain_length = (4. / 3. * np.pi * r_max ** 3 * nr_pores) ** (1. / 3.)
            domain_size = [domain_length, domain_length, domain_length]

    if body_throat_corr_param is not None:
        network = create_delaunay_network(nr_pores, pdf_pore_radius=pdf_pore_radius,
                                          body_throat_corr_param=body_throat_corr_param, domain_size=domain_size)
    else:
        network = create_delaunay_network(nr_pores, pdf_pore_radius=pdf_pore_radius,
                                          pdf_tube_radius=pdf_tube_radius, domain_size=domain_size)

    network = reorder_network(network)  # Important for efficiency
    return network


def unstructured_network_periodic_y(nr_pores, domain_size=None, quasi_2d=False):
    r_min, r_max = 20e-6, 75e-6
    pdf_pore_radius = beta(1.25, 1.5, loc=r_min, scale=(r_max - r_min))

    r_min, r_max = 1e-6, 25e-6
    pdf_tube_radius = beta(1.5, 2, loc=r_min, scale=(r_max - r_min))

    if domain_size is None:
        if quasi_2d:
            r_max = 75e-6
            domain_length = (4. / 3. * np.pi * r_max ** 3 * nr_pores) ** (1. / 3.)
            domain_size = [2 * domain_length, 2*domain_length, domain_length/4.]
        else:
            r_max = 75e-6
            domain_length = (4. / 3. * np.pi * r_max ** 3 * nr_pores) ** (1. / 3.)
            domain_size = [2 * domain_length, domain_length, domain_length]

    rad_generated = pdf_pore_radius.rvs(size=nr_pores)
    x_coord, y_coord, z_coord, rad = sphere_packing(rad=rad_generated, domain_size=domain_size)

    x_coord_dup = np.tile(x_coord, 3)
    y_coord_dup = np.hstack([y_coord, y_coord + domain_size[1], y_coord - domain_size[1]])
    z_coord_dup = np.tile(z_coord, 3)
    origin_id = np.hstack([np.arange(nr_pores), np.arange(nr_pores), np.arange(nr_pores)])
    marker = np.hstack([np.ones(nr_pores), np.zeros(nr_pores), np.zeros(nr_pores)]).astype(np.int)

    points = zip(x_coord_dup, y_coord_dup, z_coord_dup)
    tri = Delaunay(points)

    indices, indptr = tri.vertex_neighbor_vertices
    csr_mat = csr_matrix((np.ones(len(indptr)), np.asarray(indptr), np.asarray(indices)),
                         shape=(nr_pores * 3, nr_pores * 3))
    coo_mat = csr_mat.tocoo()

    edgelist_1 = coo_mat.row[coo_mat.row < coo_mat.col]
    edgelist_2 = coo_mat.col[coo_mat.row < coo_mat.col]
    length = np.sqrt((x_coord_dup[edgelist_1] - x_coord_dup[edgelist_2]) ** 2 +
                     (y_coord_dup[edgelist_1] - y_coord_dup[edgelist_2]) ** 2 +
                     (z_coord_dup[edgelist_1] - z_coord_dup[edgelist_2]) ** 2)

    edge_ids_short = (length < np.max(rad) * 4).nonzero()[0]

    edgelist_1_short = edgelist_1[edge_ids_short]
    edgelist_2_short = edgelist_2[edge_ids_short]
    length = np.sqrt((x_coord_dup[edgelist_1_short] - x_coord_dup[edgelist_2_short]) ** 2 +
                     (y_coord_dup[edgelist_1_short] - y_coord_dup[edgelist_2_short]) ** 2 +
                     (z_coord_dup[edgelist_1_short] - z_coord_dup[edgelist_2_short]) ** 2)
    minlen_edge = min(length)

    edge_ids_truncated = (marker[edgelist_1_short] | marker[edgelist_2_short]).nonzero()[0]
    # Remove edges not leading in original domain
    row = edgelist_1_short[edge_ids_truncated]
    col = edgelist_2_short[edge_ids_truncated]

    # Convert ids of edges
    row = origin_id[row]
    col = origin_id[col]

    edgelist_1 = row[row < col]
    edgelist_2 = col[row < col]
    edgelist = np.vstack([edgelist_1, edgelist_2]).T

    y_diff = np.minimum(abs(x_coord[edgelist_1] - x_coord[edgelist_2]),
                        domain_size[0] - y_coord[edgelist_1] + y_coord[edgelist_2])
    y_diff = np.minimum(y_diff, domain_size[0] - y_coord[edgelist_2] + y_coord[edgelist_1])

    length = np.sqrt((x_coord[edgelist_1] - x_coord[edgelist_2]) ** 2 +
                     (y_diff) ** 2 +
                     (z_coord[edgelist_1] - z_coord[edgelist_2]) ** 2)
    length = np.maximum(length, minlen_edge)
    rad_tubes = pdf_tube_radius.rvs(size=len(edgelist_1))
    network = StructuredPoreNetwork([1, 1, 1], 1.0e-16)
    network.add_pores(x_coord, y_coord, z_coord, rad)
    network.add_throats(edgelist + 1, r=rad_tubes, l=length, G=np.ones(len(edgelist_1)) / 16.0)
    network._fix_tubes_larger_than_ngh_pores()
    network = prune_network(network, [0.1, 0.9, -0.1, 1.1, 0.1, 0.9])
    network = reorder_network(network)

    return network

def network_from_statoil_file(filename, prune_window=(0.05, 0.95, 0.05, 0.95, 0.05, 0.95)):
    network = StatoilPoreNetwork(filename)
    network = prune_network(network, prune_window)
    return network


def network_from_statoil_file_no_prune(filename):
    network = StatoilPoreNetwork(filename)
    return network

