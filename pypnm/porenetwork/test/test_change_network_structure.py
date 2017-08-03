from scipy import spatial

from pypnm.porenetwork import component
from pypnm.porenetwork.coordination_number import choose_edges_for_target_coord_num
from pypnm.porenetwork.network_factory import *


def test_change_coordination_number():
    N = 10
    network = cube_network_27(N=N)
    target_coord_number = np.ones(network.nr_p, dtype=np.int)*6

    ti_list = choose_edges_for_target_coord_num(network, target_coord_number)
    ti_list_remove = component.complement_tube_set(network, ti_list)
    network.remove_tubes(ti_list_remove)


def test_delete_tubes():
    N = 5
    network = cube_network(N=N)

    nr_tubes_prev = network.nr_t

    removed_tubes = [0, 1, 2, 8]

    network.remove_tubes(removed_tubes)

    for field in [network.tubes.r, network.tubes.l, network.tubes.G]:
        assert (len(field) == nr_tubes_prev - len(removed_tubes))


def test_add_tubes_to_network():
    N = 13
    network = cube_network(N=N)
    nr_new_tubes = 8
    edgelist = -np.ones([nr_new_tubes, 2], dtype=np.int32)
    pi_1 = [1, 7, 14, 21, 29, 32, 39, 150]
    pi_2 = [10, 12, 44, 55, 66, 87, 48, 29]

    edgelist[:, 0] = pi_1
    edgelist[:, 1] = pi_2
    network.add_throats(edgelist)


def test_add_pores_to_network():
    N = 13
    network = cube_network(N=N)
    nr_new_pores = 7
    new_pores_x = np.ones(nr_new_pores)
    new_pores_y = np.ones(nr_new_pores)
    new_pores_z = np.ones(nr_new_pores)
    new_pores_r = np.ones(nr_new_pores)

    new_pores_x[:] = np.mean(network.pores.x)
    new_pores_y[:] = np.mean(network.pores.y)
    new_pores_z[:] = np.mean(network.pores.z) * np.arange(0.0, 1.0, nr_new_pores)
    new_pores_r[:] = np.mean(network.pores.r)

    network.add_pores(new_pores_x, new_pores_y, new_pores_z, new_pores_r)

    assert network.nr_p == (N ** 3 + nr_new_pores)


def test_add_pores_and_tubes_to_network():
    N = 13
    network = cube_network(N=N)
    nr_p_old = network.nr_p
    nr_new_pores = 7
    nr_new_tubes = 7

    new_pores_x = np.ones(nr_new_pores)
    new_pores_y = np.ones(nr_new_pores)
    new_pores_z = np.ones(nr_new_pores)
    new_pores_r = np.ones(nr_new_pores)

    new_pores_x[:] = 0.0 - np.mean(network.pores.x)
    new_pores_y[:] = np.mean(network.pores.y)
    new_pores_z[:] = np.mean(network.pores.z) * np.linspace(0.0, 1.0, nr_new_pores)
    new_pores_r[:] = np.mean(network.pores.r)

    network.add_pores(new_pores_x, new_pores_y, new_pores_z, new_pores_r)

    edgelist = -np.ones([nr_new_tubes, 2], dtype=np.int32)
    pi_1 = [1, 1, 1, 1, 1, 1, 1]
    pi_2 = np.arange(nr_p_old, nr_new_pores + nr_p_old)

    edgelist[:, 0] = pi_1
    edgelist[:, 1] = pi_2
    network.add_throats(edgelist)

    assert network.nr_p == (N ** 3 + nr_new_pores)


def test_generate_arbitrary_network_pore():
    N = 1
    network = cube_network(N=1)
    nr_p_old = network.nr_p
    nr_new_pores = 10
    nr_new_tubes = 10

    new_pores_x = np.ones(nr_new_pores)
    new_pores_y = np.ones(nr_new_pores)
    new_pores_z = np.ones(nr_new_pores)
    new_pores_r = np.ones(nr_new_pores)

    new_pores_x[:] = np.random.rand(nr_new_pores)
    new_pores_y[:] = np.random.rand(nr_new_pores)
    new_pores_z[:] = np.random.rand(nr_new_pores)
    new_pores_r[:] = np.random.rand(nr_new_pores)*0.1

    points = zip(new_pores_x, new_pores_y, new_pores_z)
    tree = spatial.KDTree(points)
    print tree.data
    network.add_pores(new_pores_x, new_pores_y, new_pores_z, new_pores_r)

    print tree.query([new_pores_x[0], new_pores_y[0], new_pores_z[0]], 2)

    edgelist = -np.ones([nr_new_tubes, 2], dtype=np.int32)
    pi_1 = [1, 2, 3, 4, 5, 7, 8, 6, 8, 1]
    pi_2 = [3, 7, 2, 3, 6, 3, 2, 8, 5, 8]

    edgelist[:, 0] = pi_1
    edgelist[:, 1] = pi_2
    edgelist += nr_p_old
    network.add_throats(edgelist)

    assert network.nr_p == (N ** 3 + nr_new_pores)

