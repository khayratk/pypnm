import numpy as np
from pypnm.porenetwork.network_manipulation import prune_network
from pypnm.porenetwork.structured_porenetwork import StructuredPoreNetwork
from pypnm.porenetwork.pore_element_models import throat_diameter_acharya
from pypnm.util.sphere_packing import sphere_packing
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix


def create_delaunay_network(nr_pores, pdf_pore_radius, domain_size, is_2d=False, pdf_tube_radius=None, body_throat_corr_param = None):
    """
    Creates an unstructured network with pore bodies randomly distributed in space. A Delaunay tesselation is used
    to connect the pores together with throats.

    Parameters
    ----------
    nr_pores: int
        number of pore bodies

    pdf_pore_radius: rv
        scipy random variable specifying the pdf from which the pore radii are sampled from

    pdf_tube_radius: rv
        scipy random variable specifying the pdf from which the throat radii are sampled from

    domain_size: 2 or 3-tuple
        tuple specifying dimensions of domain in meters

    is_2d: bool
        if True, a 2d network is generated

    Returns
    -------
    network: PoreNetwork

    """

    if pdf_tube_radius is None and body_throat_corr_param is None:
        raise ValueError("either pdf_tube_radius or body_throat_corr_param need to be specified")

    nr_new_pores = nr_pores - 1

    rad_generated = pdf_pore_radius.rvs(size=nr_new_pores)
    x_coord, y_coord, z_coord, rad = sphere_packing(rad=rad_generated, domain_size=domain_size, is_2d=is_2d)

    network = StructuredPoreNetwork([1, 1, 1], 1.0e-16)  # Pore-Network with only one pore as seed

    network.add_pores(x_coord, y_coord, z_coord, rad)

    points = zip(x_coord, y_coord, z_coord)
    tri = Delaunay(points)

    indices, indptr = tri.vertex_neighbor_vertices
    coo_mat = csr_matrix((np.ones(len(indptr)), np.asarray(indptr), np.asarray(indices)),
                         shape=(nr_new_pores, nr_new_pores)).tocoo()

    edgelist_1 = coo_mat.row[coo_mat.row < coo_mat.col] + 1
    edgelist_2 = coo_mat.col[coo_mat.row < coo_mat.col] + 1
    edgelist = np.vstack([edgelist_1, edgelist_2]).T

    l_total = np.sqrt((network.pores.x[edgelist_1] - network.pores.x[edgelist_2]) ** 2 +
                     (network.pores.y[edgelist_1] - network.pores.y[edgelist_2]) ** 2 +
                     (network.pores.z[edgelist_1] - network.pores.z[edgelist_2]) ** 2)

    length = l_total - network.pores.r[edgelist_1] - network.pores.r[edgelist_2]

    assert np.all(length > 0.0)

    if pdf_tube_radius is not None:
        rad_tubes = pdf_tube_radius.rvs(size=len(edgelist_1))

    if body_throat_corr_param is not None:
        rad_tubes = throat_diameter_acharya(network, edgelist_1, edgelist_2, l_total, body_throat_corr_param)

    network.add_throats(edgelist, r=rad_tubes, l=length, G=np.ones(len(edgelist_1)) / 16.0)

    network._fix_tubes_larger_than_ngh_pores()
    network = prune_network(network, [0.1, 0.9, 0.1, 0.9, 0.1, 0.9])
    network.network_type = "unstructured"

    return network
