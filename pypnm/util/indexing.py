class GridIndexer3D(object):
    def __init__(self, nx, ny, nz):
        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.get_index_face = [self.get_index_west, self.get_index_east,
                               self.get_index_south, self.get_index_north,
                               self.get_index_bottom, self.get_index_top]

    def get_index(self, i, j, k):
        return (self.nx * self.ny) * k + self.nx * j + i

    def get_index_east(self, i, j, k):
        return self.get_index(i + 1, j, k)

    def get_index_west(self, i, j, k):
        return self.get_index(i - 1, j, k)

    def get_index_north(self, i, j, k):
        return self.get_index(i, j + 1, k)

    def get_index_south(self, i, j, k):
        return self.get_index(i, j - 1, k)

    def get_index_top(self, i, j, k):
        return self.get_index(i, j, k + 1)

    def get_index_bottom(self, i, j, k):
        return self.get_index(i, j, k - 1)

    def ind_to_ijk(self, ind):
        k = int(ind / (self.nx * self.ny))
        j = int((ind - k * (self.nx * self.ny)) / self.nx)
        i = ind - self.nx * j - (self.nx * self.ny) * k
        return i, j, k
