import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from pypnm.ams.msrb_funcs import solve_with_msrsb_compressible


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


ia, ja, a, b = read_matrix("compressible/DUMP_Ab.m")

sol_msrsb, inds_in_large_components = solve_with_msrsb_compressible(ia, ja, a, b, v_per_subdomain=1000,
                                                                    cut_off=200, tol=1.e-8, return_inds=True)

A = csr_matrix((a, ja, ia))
sol_scipy = spsolve(A[:, inds_in_large_components][inds_in_large_components, :], b[inds_in_large_components])

sol_ref = np.zeros_like(sol_msrsb)
sol_ref[inds_in_large_components] = sol_scipy

eps = 1e-10
print sol_ref
print sol_msrsb

error = (sol_ref-sol_msrsb)/(sol_ref+eps)
print np.max(error)
