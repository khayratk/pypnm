from itertools import product, izip

from pypnm.porenetwork import gridder
from pypnm.porenetwork.network_factory import *
from pypnm.util.utils import all_unique


def test_grid3d_of_bounding_boxes():
    nr_p = 27
    network = cube_network(N=nr_p)
    nx, ny, nz = 8, 22, 12

    bboxes = gridder.grid3d_of_bounding_boxes(network, nx, ny, nz)

    pores = network.pores
    list_of_coords = [pores.x, pores.x, pores.y, pores.y, pores.z, pores.z]

    # Check that there is no intersection with pores
    for i, j, k in product(xrange(nx), xrange(ny), xrange(nz)):
        for FACE, pore_coords in izip(["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"] , list_of_coords):
            assert np.all(bboxes[i, j, k][FACE] != pore_coords)

    # Check if neighbouring boxes touch
    for i, j, k in product(xrange(nx-1), xrange(ny), xrange(nz)):
        assert bboxes[i, j, k].xmax == bboxes[i+1, j, k].xmin

    for i, j, k in product(xrange(nx), xrange(ny-1), xrange(nz)):
        assert bboxes[i, j, k].ymax == bboxes[i, j+1, k].ymin

    for i, j, k in product(xrange(nx), xrange(ny), xrange(nz-1)):
        assert bboxes[i, j, k].zmax == bboxes[i, j, k+1].zmin


def test_grid3d_of_pore_lists():
    nr_p = 27
    network = cube_network(N=nr_p)
    nx, ny, nz = 8, 22, 12

    pi_lists = gridder.grid3d_of_pore_lists(network, nx, ny, nz)

    pi_all = []
    for i, j, k in product(xrange(nx), xrange(ny), xrange(nz)):
        pi_all.extend(pi_lists[i, j, k])

    assert all_unique(pi_all)

    assert max(pi_all) == (network.nr_p - 1)
    assert min(pi_all) == 0