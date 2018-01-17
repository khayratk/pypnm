from scipy.sparse import dia_matrix
import numpy as np
from scipy.sparse.linalg import spsolve
from pypnm.ams.msrb_funcs import get_supports
from scipy.sparse.csgraph import connected_components
from collections import Counter
from pypnm.ams.msrsb import MSRSB
from pypnm.linalg.trilinos_interface import matrix_scipy_to_epetra
from scipy.sparse import csr_matrix
from PyTrilinos import Epetra
from pypnm.linalg.trilinos_interface import solve_direct

f = open("compressible/DUMP_Ab.m")
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

A = csr_matrix((a, ja, ia))
sol_ref = spsolve(A, b)

diagonal = dia_matrix(((A+A.T)*np.ones(A.shape[0]),[0]), shape = A.shape )
A_b = 0.5*(A+A.T - diagonal)

cut_off = 200
num_components, v2components = connected_components(A_b)
component_sizes = Counter(v2components)
large_components = {component for component in component_sizes if component_sizes[component]>cut_off}

inds_in_large_components = list()
inds_in_small_components = list()
for v, label in enumerate(v2components):
    if label in large_components:
        inds_in_large_components.append(v)
    else:
        inds_in_small_components.append(v)

inds_in_large_components = np.asarray(inds_in_large_components)
inds_in_small_components = np.asarray(inds_in_small_components)


A_trunc = A[:, inds_in_large_components][inds_in_large_components,:]
A_trunc_b = A_b[:, inds_in_large_components][inds_in_large_components,:]
n_components, labels = connected_components(A_trunc_b, directed=False)
print n_components

my_restriction_supports, my_basis_support = get_supports(A_trunc_b)

A_trunc_epetra_b = matrix_scipy_to_epetra(A_trunc_b)
A_trunc_epetra = matrix_scipy_to_epetra(A_trunc)

ms = MSRSB(A_trunc_epetra_b, my_restriction_supports, my_basis_support)

ms.smooth_prolongation_operator(niter= 100)

sol = Epetra.Vector(ms.A.RangeMap())
rhs = Epetra.Vector(ms.A.RangeMap())

rhs[:] = b[inds_in_large_components]
sol[:] = 0.0

sol = ms.iterative_solve(rhs, sol, tol=1.e-10, max_iter=100, A=A_trunc_epetra)
sol_ref = spsolve(A_trunc, b[inds_in_large_components])

sol_ref_epetra = solve_direct(A_trunc_epetra, rhs)
error = (sol_ref[:] - sol[:]) / sol_ref[:]
print max(error)



