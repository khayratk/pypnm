import numpy as np
from mayavi import mlab
from itertools import product


def draw_network(network, show_outline=True, pi_list=None, ti_list=None):
    if ti_list is not None:
        connections = network.edgelist[ti_list]
    else:
        connections = network.edgelist[:]

    x = np.copy(network.pores.x)
    y = np.copy(network.pores.y)
    z = np.copy(network.pores.z)

    scalars = np.ones_like(network.pores.r) * 1e-19
    if pi_list is not None:
        scalars[pi_list] = network.pores.r[pi_list]
    else:
        scalars[:] = network.pores.r

    mlab.clf()
    fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0., 0., 0.))

    pts = mlab.points3d(x, y, z, scalars, scale_factor=1, resolution=10, color=(0.4, 0.4, 0.4))
    pts.mlab_source.dataset.lines = np.array(connections)
    pts.mlab_source.update()

    tube = mlab.pipeline.tube(pts)
    tube.filter.radius_factor = 1.
    tube.filter.vary_radius = 'vary_radius_by_scalar'
    tube.filter.radius = np.mean(network.tubes.r)
    mlab.pipeline.surface(tube, color=(0.4, 0.4, 0.4))
    dist = np.max(network.dim)
    mlab.view(-95, 35, distance=dist * 2.2)
    if show_outline:
        mlab.outline(extent=[np.min(x), np.max(x), np.min(y), np.max(y), np.min(z), np.max(z)], line_width=10)

    return fig


def draw_network_invaded(network, show_outline=True):
    return draw_network(network, show_outline=show_outline,
                        pi_list=(network.pores.invaded == 1).nonzero()[0],
                        ti_list=(network.tubes.invaded == 1).nonzero()[0])


def mlab_plot_bbox(network, angle=0.0, color=(0, 0, 0)):
    x = np.copy(network.pores.x)
    y = np.copy(network.pores.y)
    z = np.copy(network.pores.z)
    xyz = np.asarray(list(product([np.min(x), np.max(x)], [np.min(y), np.max(y)], [np.min(z), np.max(z)])))

    theta = float(angle) / 180. * np.pi
    x_center, y_center = np.mean(xyz[:, 0]), np.mean(xyz[:, 1])

    x_r = x_center + (xyz[:, 0] - x_center) * np.cos(theta) - (xyz[:, 1] - y_center) * np.sin(theta)
    y_r = y_center + (xyz[:, 0] - x_center) * np.sin(theta) + (xyz[:, 1] - y_center) * np.cos(theta)

    xyz[:, 0] = x_r
    xyz[:, 1] = y_r

    seq = [0, 1, 3, 2, 0, 4, 5, 1]
    mlab.plot3d(xyz[seq, 0], xyz[seq, 1], xyz[seq, 2], tube_radius=5e-5, color=color)
    seq = [7, 6, 4, 5, 7, 3, 2, 6]
    mlab.plot3d(xyz[seq, 0], xyz[seq, 1], xyz[seq, 2], tube_radius=5e-5, color=color)
