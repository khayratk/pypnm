from pypnm.porenetwork.structured_porenetwork import StructuredPoreNetwork
from pypnm.porenetwork.structured_porenetwork_27 import StructuredPoreNetwork27
from pypnm.porenetwork.statoil_porenetwork import StatoilPoreNetwork
from pypnm.porenetwork.network_manipulation import prune_network
from pypnm.porenetwork.unstructured_porenetwork import create_unstructured_network
from pypnm.porenetwork.delaunay_network import create_delaunay_network
from scipy.stats import beta, randint
import numpy as np


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
    return network


def structured_network_27(Nx, Ny, Nz, media_type="consolidated", periodic=False):
    dist_p2p = 160e-6
    network = StructuredPoreNetwork27([Nx, Ny, Nz], dist_p2p, media_type=media_type, periodic=periodic)
    return network


def unstructured_network(nr_pores, domain_size=None, is_2d=False):
    r_min, r_max = 20e-6, 75e-6
    pdf_pore_radius = beta(1.25, 1.5, loc=r_min, scale=(r_max - r_min))

    r_min, r_max = 1e-6, 25e-6
    pdf_tube_radius = beta(1.5, 2, loc=r_min, scale=(r_max - r_min))

    pdf_coord_number = randint(low=4, high=12)

    if domain_size is None:
        if is_2d:
            r_max = 75e-6
            domain_length = (np.pi * r_max ** 2 * nr_pores) ** (1. / 2.)
            domain_size = [2 * domain_length, domain_length, 0.0]
        else:
            r_max = 75e-6
            domain_length = (4. / 3. * np.pi * r_max ** 3 * nr_pores) ** (1. / 3.)
            domain_size = [2 * domain_length, domain_length, domain_length]

    network = create_unstructured_network(nr_pores, pdf_pore_radius=pdf_pore_radius,
                                          pdf_tube_radius=pdf_tube_radius,
                                          pdf_coord_number=pdf_coord_number,
                                          domain_size=domain_size, is_2d=is_2d)

    return network


def unstructured_network_delaunay(nr_pores, domain_size=None, is_2d=False):
    r_min, r_max = 20e-6, 75e-6
    pdf_pore_radius = beta(1.25, 1.5, loc=r_min, scale=(r_max - r_min))

    r_min, r_max = 1e-6, 25e-6
    pdf_tube_radius = beta(1.5, 2, loc=r_min, scale=(r_max - r_min))

    if domain_size is None:
        if is_2d:
            r_max = 75e-6
            domain_length = (np.pi * r_max ** 2 * nr_pores) ** (1. / 2.)
            domain_size = [2 * domain_length, domain_length, 0.0]
        else:
            r_max = 75e-6
            domain_length = (4. / 3. * np.pi * r_max ** 3 * nr_pores) ** (1. / 3.)
            domain_size = [2 * domain_length, domain_length, domain_length]

    network = create_delaunay_network(nr_pores, pdf_pore_radius=pdf_pore_radius,
                                      pdf_tube_radius=pdf_tube_radius, domain_size=domain_size, is_2d=is_2d)
    return network



def network_from_statoil_file(filename, prune_window=[0.05, 0.95, 0.05, 0.95, 0.05, 0.95]):
    network = StatoilPoreNetwork(filename)
    network = prune_network(network, prune_window)
    return network


def network_from_statoil_file_no_prune(filename):
    network = StatoilPoreNetwork(filename)
    return network

