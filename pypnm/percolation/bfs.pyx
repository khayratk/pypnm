from collections import deque
from libcpp.deque cimport deque
from libcpp.map cimport map
cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.int32
ctypedef np.int32_t DTYPE_t


@cython.boundscheck(False) # turn of bounds-checking for entire function
def bfs_vertices(network, int source, np.ndarray[DTYPE_t ,ndim=1] flag_pore, np.ndarray[DTYPE_t ,ndim=1] flag_tube, int max_level, int max_count):

    cdef int k, count, i, s
    cdef deque[int] queue
    cdef map[int, int] visited
    
    s = source
    
    queue.push_back(s)
    visited[s]=0
    cdef np.ndarray[DTYPE_t ,ndim=1] nr_nghs = network.nr_nghs

    cdef np.ndarray[DTYPE_t ,ndim=1] pi_nghs, ti_nghs
    count = 0
 
    while not queue.empty():
        s = queue.front()
        queue.pop_front()
        pi_nghs = network.ngh_pores[s]
        ti_nghs = network.ngh_tubes[s]     

        if visited[s] < max_level:
            # Add neighbouring vertices to queue and mark them
            for k in xrange(nr_nghs[s]):
                i = pi_nghs[k]
                if (flag_pore[i]==1) and (flag_tube[ti_nghs[k]]==1) and  (not (visited.count(i)>0)):
                    queue.push_back(i)
                    visited[i]=visited[s]+1
                    count+=1

        if count>max_count:
            break

    return visited.keys()