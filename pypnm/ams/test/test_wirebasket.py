from itertools import permutations

from pypnm.ams.wire_basket import create_wire_basket
from pypnm.porenetwork.network_factory import structured_network


def test_wire_basket():
    for nx, ny, nz, n_fine_per_cell in permutations([4, 5, 6, 7], 4):
        print nx, ny, nz, n_fine_per_cell
        network = structured_network(Nx=n_fine_per_cell * nx, Ny=n_fine_per_cell * ny, Nz=n_fine_per_cell * nz)
        wb = create_wire_basket(network, nx, ny, nz)
        assert len((wb == 3).nonzero()[0]) == nx*ny*nz

