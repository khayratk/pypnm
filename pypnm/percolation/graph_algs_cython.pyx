from collections import deque
from libcpp.deque cimport deque
from libcpp.map cimport map
from libcpp.vector cimport vector
cimport cython
from libcpp cimport bool
import numpy as np
cimport numpy as np
from pypnm.porenetwork.constants import *

DTYPE = np.int8
ctypedef np.int8_t DTYPE_t

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
def bfs_vertices(network, int source, np.ndarray[DTYPE_t, ndim=1] flag_pore, np.ndarray[DTYPE_t, ndim=1] flag_tube, int max_level, int max_count):

    cdef int k, count, i, s
    cdef deque[int] queue
    visited = {}
    
    s = source
    
    queue.push_back(s)
    visited[s]=0
    cdef np.ndarray[np.int32_t, ndim=1] nr_nghs = network.nr_nghs

    cdef np.ndarray[np.int32_t, ndim=1] pi_nghs, ti_nghs
    count = 0

    while not queue.empty():
        s = queue.front()
        queue.pop_front()
        pi_nghs = network.ngh_pores[s]
        ti_nghs = network.ngh_tubes[s]     

        if visited[s]<max_level:
            # Add neighbouring vertices to queue and mark them
            for k in xrange(nr_nghs[s]):
                i = pi_nghs[k]
                if flag_pore[i]==1 and flag_tube[ti_nghs[k]]==1 and  (i not in visited):
                    queue.push_back(i)
                    visited[i] = visited[s]+1
                    count+=1

        if count > max_count:
            break

    return visited.keys()           

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
def pore_connected_to_inlet(network, int source):
    """ Depth first search in order to find if a pore is connected to the inlet"""


    cdef vector[int] stack
    visited = {}


    cdef int s, pi, ti, x
    cdef int inlet = INLET
    cdef bool has_neighbours, is_connected
    cdef np.ndarray[DTYPE_t, ndim=1] flag_pore = network.pores.connected
    cdef np.ndarray[DTYPE_t, ndim=1] flag_tube = network.tubes.connected
    cdef np.ndarray[np.int32_t, ndim=1] pi_nghs, ti_nghs
    cdef np.ndarray[np.long_t, ndim=1] sorted_indices
    cdef np.ndarray[np.int32_t, ndim=1] nr_nghs = network.nr_nghs
    cdef np.ndarray[DTYPE_t, ndim=1] pore_type = network.pore_domain_type
    cdef np.ndarray[np.double_t, ndim=1] dist_to_inlet

    if network.inface == WEST:
        dist_to_inlet = network.pores.x
    if network.inface == SOUTH:
        dist_to_inlet = network.pores.y
    if network.inface == BOTTOM:
        dist_to_inlet = network.pores.z

    s = source
    visited[s] = 0
    stack.push_back(s)

    is_connected = False

    if pore_type[s] == inlet:
        return True

    while stack.size() > 0:
        s = stack.back()

        pi_nghs = network.ngh_pores[s]
        ti_nghs = network.ngh_tubes[s]

        sorted_indices = np.argsort(dist_to_inlet[pi_nghs])

        has_neighbors = False

        for x in xrange(nr_nghs[s]):
            pi = pi_nghs[sorted_indices[x]]
            ti = ti_nghs[sorted_indices[x]]
            if (flag_pore[pi] == 1) and (flag_tube[ti] == 1) and not(pi in visited):
                has_neighbors = True
                visited[pi] = 0
                stack.push_back(pi)
                if pore_type[pi] == inlet:
                    is_connected = True
                break

        if not has_neighbors:
            stack.pop_back()
    
        if is_connected:
            break

    return is_connected


@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
def get_nr_nghs_mask(network, np.ndarray[DTYPE_t ,ndim=1] tube_mask):
    cdef np.ndarray[DTYPE_t ,ndim=1] nr_nghs_mask
    cdef np.ndarray[DTYPE_t ,ndim=1] nr_nghs = network.nr_nghs
    cdef np.ndarray[np.int32_t ,ndim=1] ti_nghs
    
    cdef int i, k
    cdef int nr_p = network.nr_p

    nr_nghs_mask = np.zeros(network.nr_p, dtype=np.int8)
    for i in xrange(nr_p):
        ti_nghs = network.ngh_tubes[i]	
        for k in xrange(nr_nghs[i]):
            nr_nghs_mask[i] += tube_mask[ti_nghs[k]]

    return nr_nghs_mask
    
