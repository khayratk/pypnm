import numpy as np
from pypnm.porenetwork import component


def create_wire_basket(network, nx, ny, nz):
    """
    Returns a "wire basket" coloring of a network which is divided into overlapping subnetworks.

    Returns
    ________

    create_wire_basket : Numpy array
        A numpy integer array of size network.nr_p storing the wire-basket values.
        The values of this array are: 0 for interior nodes, 1 for faces, 2 for edges, and 3 for nodes.

    Notes
    ------
    Assumes that the pore indices at the faces overlap. Also assumes that there is exactly one corner node
    for every 3 adjacent faces
    """

    min_x, max_x = np.min(network.pores.x), np.max(network.pores.x)
    min_y, max_y = np.min(network.pores.y), np.max(network.pores.y)
    min_z, max_z = np.min(network.pores.z), np.max(network.pores.z)

    len_x = max_x - min_x
    len_y = max_y - min_y
    len_z = max_z - min_z

    eps = 1.e-8  # perturbation to make sure no pore exactly intersects the plane

    x_coords = np.linspace(min_x + 0.5 * len_x / nx - eps, max_x - 0.5 * len_x / nx, nx)
    y_coords = np.linspace(min_y + 0.5 * len_y / ny - eps, max_y - 0.5 * len_y / ny, ny)
    z_coords = np.linspace(min_z + 0.5 * len_z / nz - eps, max_z - 0.5 * len_z / nz, nz)

    wire_basket = np.zeros(network.nr_p, dtype=np.int)

    def fix_wire_basket(wire_basket):
        wire_basket_temp = np.copy(wire_basket)
        for pi in xrange(network.nr_p):
            if wire_basket_temp[pi] == 0:
                wire_basket[pi] = np.min(wire_basket_temp[network.ngh_pores[pi]])
        return wire_basket

    for i in xrange(nx):
        pi_plane = component.pore_list_x_plane_plus(network, x_coords[i])
        wire_basket[pi_plane] += 1

    for j in xrange(ny):
        pi_plane = component.pore_list_y_plane_plus(network, y_coords[j])
        wire_basket[pi_plane] += 1

    for k in xrange(nz):
        pi_plane = component.pore_list_z_plane_plus(network, z_coords[k])
        wire_basket[pi_plane] += 1

    assert len((wire_basket == 3).nonzero()[0]) == nx*ny*nz, len((wire_basket == 3).nonzero()[0])

    wire_basket = fix_wire_basket(wire_basket)

    assert len((wire_basket == 3).nonzero()[0]) == nx*ny*nz

    assert np.max(wire_basket) == 3
    assert np.min(wire_basket) == 0

    return wire_basket