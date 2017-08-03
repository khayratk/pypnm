import numpy as np


class BoundingBox(dict):
    def __getattr__(self, key):
        i = self.get(key)
        if i is None:
            raise AttributeError
        return self.get(key)

    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax):
        eps = 10e-20
        if xmin == xmax:
            xmin -= eps
            xmax += eps

        if ymin == ymax:
            ymin -= eps
            ymax += eps

        if zmin == zmax:
            zmin -= eps
            zmax += eps

        self.update({"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "zmin": zmin, "zmax": zmax})
        self.update({0: xmin, 1: xmax, 2: ymin, 3: ymax, 4: zmin, 5: zmax})

    @classmethod
    def from_coords(cls, x_coords, y_coords, z_coords):
        return cls(np.min(x_coords), np.max(x_coords),
                   np.min(y_coords), np.max(y_coords),
                   np.min(z_coords), np.max(z_coords))
