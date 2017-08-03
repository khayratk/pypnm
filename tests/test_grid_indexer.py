from pypnm.util.indexing import GridIndexer3D


def test_indexer_3d():
    nx = 7
    ny = 13
    nz = 23
    gi = GridIndexer3D(nx, ny, nz)

    assert gi.get_index(0, 0, 0) == 0
    assert gi.get_index(6, 0, 0) == 6
    assert gi.get_index(nx-1, ny-1, nz-1) == nx*ny*nz - 1
    assert gi.ind_to_ijk(gi.get_index(4, 4, 4)) == (4, 4, 4)
    assert gi.ind_to_ijk(gi.get_index(6, 12, 17)) == (6, 12, 17)
