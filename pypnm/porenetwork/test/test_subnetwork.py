from pypnm.porenetwork import component
from pypnm.porenetwork.network_factory import *
from pypnm.porenetwork.subnetwork import SubNetworkTightlyCoupled
from pypnm.util.bounding_box import BoundingBox


def bbox_whole_network(network):
    eps = 0.001
    return BoundingBox(0 - eps, max(network.pores.x) + eps, 0 - eps, max(network.pores.y) + eps, 0 - eps,
                       max(network.pores.z) + eps)


def check_saturation_close(sat_comp, sat_target, tol):
    saturation = sat_comp.sat_nw()
    print saturation, sat_target
    assert saturation < sat_target + tol
    assert saturation >= sat_target


def test_subnetwork_whole_network():
    network = cube_network(N=10)
    network.set_inlet_pores_invaded_and_connected()
    bounding_box = bbox_whole_network(network)

    sub_pore_list = component.pore_list_from_bbox(network, bounding_box)

    sub_network = SubNetworkTightlyCoupled(network, sub_pore_list)
    sub_network.set_inlet_pores_invaded_and_connected()

    assert (sub_network.nr_p == network.nr_p)
    assert (sub_network.nr_t == network.nr_t)

    assert (np.all(sub_network.pores.vol == network.pores.vol))
    assert (np.all(sub_network.tubes.vol == network.tubes.vol))
    assert (np.all(sub_network.tubes.r == network.tubes.r))

    assert (np.all(sub_network.pi_local_to_global == np.arange(network.nr_p)))
    assert (np.all(sub_network.ti_local_to_global == np.arange(network.nr_t)))

    for pi in xrange(sub_network.nr_p):
        assert np.all(sub_network.ngh_pores[pi] == network.ngh_pores[pi])

    for ti in xrange(sub_network.nr_t):
        assert np.all(sub_network.edgelist[ti] == network.edgelist[ti])


def test_four_subnetworks():
    network = cube_network(N=20)
    network.set_inlet_pores_invaded_and_connected()

    len_x = max(network.pores.x)
    len_y = max(network.pores.y)
    len_z = max(network.pores.z)
    x_max1 = len_x / 3.0
    y_max1 = len_y / 3.0

    x_max2 = len_x
    y_max2 = len_y

    bounding_box1 = BoundingBox(0.0, x_max1, 0.0, y_max1, 0.0, len_z)
    bounding_box2 = BoundingBox(0.0, x_max1, y_max1, len_y, 0.0, len_z)
    bounding_box3 = BoundingBox(x_max1, x_max2, 0.0, y_max1, 0.0, len_z)
    bounding_box4 = BoundingBox(x_max1, y_max2, y_max1, len_y, 0.0, len_z)

    bounding_boxes = [bounding_box1, bounding_box2, bounding_box3, bounding_box4]

    sub_networks = list()

    for bounding_box in bounding_boxes:
        sub_networks.append(SubNetworkTightlyCoupled.from_bounding_box(network, bounding_box))