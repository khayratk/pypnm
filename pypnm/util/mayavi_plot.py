import numpy as np
from mayavi import mlab

def fig_mayavi(network):
    connections = network.edgelist
    x = np.copy(network.pores.x)
    y = np.copy(network.pores.y)
    z = np.copy(network.pores.z)
    scalars = np.copy(network.pores.r)
    
    mlab.clf()
    fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0., 0., 0.))

    pts = mlab.points3d(x, y, z, scalars, scale_factor=1, resolution=10, )
    pts.mlab_source.dataset.lines = np.array(connections)
    pts.mlab_source.update()

    tube = mlab.pipeline.tube(pts)
    tube.filter.radius_factor = 1.
    tube.filter.vary_radius = 'vary_radius_by_scalar'
    tube.filter.radius = np.mean(network.tubes.r)
    mlab.pipeline.surface(tube, color=(0.8, 0.8, 0))
    mlab.view(-95, 35, distance=0.006)
    mlab.outline(extent=[np.min(x), np.max(x), np.min(y), np.max(y), np.min(z), np.max(z)])

    return fig