import numpy as np
from scipy.sparse import csr_matrix, dia_matrix
from scipy.sparse.csgraph import connected_components
from collections import Counter

from scipy.sparse.linalg import spsolve
from pypnm.ams.msrb_funcs import solve_with_msrsb


def read_matrix(filepath):
    with open(filepath) as f:
        mlab_text = f.read()
        mlab_text = mlab_text.splitlines()
        opening_indices = [i for i, s in enumerate(mlab_text) if "[" in s]
        closing_indices = [i for i, s in enumerate(mlab_text) if "]" in s]
        for ind in opening_indices:
            mlab_text[ind] = mlab_text[ind].split("[")[-1]

        ia = [int(line) for line in mlab_text[opening_indices[0]:closing_indices[0]]]
        ja = [int(line) for line in mlab_text[opening_indices[1]:closing_indices[1]]]
        a = [float(line) for line in mlab_text[opening_indices[2]:closing_indices[2]]]
        b = [float(line) for line in mlab_text[opening_indices[3]:closing_indices[3]]]

        ia = np.asarray(ia) - 1
        ja = np.asarray(ja) - 1
        a = np.asarray(a)
        b = np.asarray(b)

        return ia, ja, a, b


cut_off = 500

ia, ja, a, b = read_matrix("compressible/DUMP_Ab.m")
A = csr_matrix((a, ja, ia))

diagonal = dia_matrix(((A + A.T) * np.ones(A.shape[0]), [0]), shape=A.shape)
A_b = 0.5 * (A + A.T - diagonal)

num_components, v2components = connected_components(A_b)
component_sizes = Counter(v2components)
large_components = {component for component in component_sizes if component_sizes[component] > cut_off}

inds_in_large_components = list()
inds_in_small_components = list()
for v, label in enumerate(v2components):
    if label in large_components:
        inds_in_large_components.append(v)
    else:
        inds_in_small_components.append(v)

inds_in_large_components = np.asarray(inds_in_large_components)
inds_in_small_components = np.asarray(inds_in_small_components)

A_trunc = A[:, inds_in_large_components][inds_in_large_components, :]

rhs = b[inds_in_large_components]


sol, history = solve_with_msrsb(A_trunc, rhs, tol=1e-7, smoother="gmres", v_per_subdomain=1000, conv_history=True,
                                    with_multiscale=True, max_iter=10000, tol_basis=1e-3, n_smooth=80,
                                    adapt_smoothing=False, verbose=True)

print "converged in", len(history["residual"])

sol_ref = spsolve(A_trunc, rhs)

error = np.max((sol - sol_ref)/(sol_ref+1.e-20))
print "error is", error