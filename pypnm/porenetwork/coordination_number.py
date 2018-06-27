import numpy as np
from pqdict import maxpq


def choose_edges_for_target_coord_num(network, n_coord):
    """
    Given a network and an array of required coordination number for each vertex, returns a list of edges that would
    (approximately) satisfy the given coordination number.

    Parameters
    ----------
    network: PoreNetwork
    n_coord: ndarray

    Returns
    -------
    out: ndarray
        Index array of the chosen throats

    Notes
    _____
    Implementation of algorithm as described in ROBERT M. SOK ET AL. 2002 Section 4.1.2

    """

    WHITE = 0
    GRAY = 1
    BLACK = 2

    tube_marker = np.ones(network.nr_t, dtype=np.int)*GRAY
    n_avail = network.nr_nghs
    n_white = np.zeros(network.nr_p, dtype=np.int)
    assert np.all(n_coord < network.nr_nghs)

    def priority_tube(ti):
        pi_1, pi_2 = network.edgelist[ti, :]

        ns_1 = n_coord[pi_1] - n_white[pi_1]
        ns_2 = n_coord[pi_2] - n_white[pi_2]

        fs_1 = n_avail[pi_1] - n_white[pi_1]
        fs_2 = n_avail[pi_2] - n_white[pi_2]

        Fs_1 = 1. - float(ns_1) / float(fs_1)
        Fs_2 = 1. - float(ns_2) / float(fs_2)

        return 1. / (1. + Fs_1 * Fs_2)

    # Initialize priority queue

    pq = maxpq()
    for ti in np.random.permutation(xrange(network.nr_t)):
        pq[ti] = priority_tube(ti)

    while pq:
        # Pop tube and mark it as white
        ti, _ = pq.popitem()
        tube_marker[ti] = WHITE

        # update n_white
        pi_1, pi_2 = network.edgelist[ti, :]
        n_white[[pi_1, pi_2]] += 1

        # If the target coordination number is reached, mark other tubes as black and delete
        for pi in [pi_1, pi_2]:
            if n_white[pi] == n_coord[pi]:
                for ti in network.ngh_tubes[pi]:
                    if tube_marker[ti] == GRAY:
                        tube_marker[ti] = BLACK
                        del pq[ti]

        # Update priorities of adj GRAY tubes
        for pi in network.edgelist[ti]:
            for ti in network.ngh_tubes[pi]:
                if tube_marker[ti] == GRAY:
                    pq[ti] = priority_tube(ti)

    assert np.all(n_white <= n_coord)

    print "Unsuccessful tubes", np.sum(n_coord - n_white)
    print "Total number of tubes", np.sum(n_white)

    return (tube_marker == WHITE).nonzero()[0]
