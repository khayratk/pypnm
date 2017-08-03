import sys


try:
    import petsc4py
    from mpi4py import MPI

    petsc4py.init(sys.argv)
    from petsc4py import PETSc
except ImportError:
    pass


def get_petsc_ksp(A, pctype="ilu", ksptype="gmres", tol=1e-5, max_it=10000):
    petsc_mat = scipy_to_petsc_matrix(A)
    ksp = PETSc.KSP().create()
    ksp.setOperators(petsc_mat)
    ksp.setType(ksptype)
    ksp.setTolerances(rtol=tol, max_it=max_it)
    pc = ksp.getPC()
    pc.setType(pctype)
    return ksp


def petsc_solve_ilu(A, rhs, x0=None, tol=1e-5, max_it=5):
    petsc_rhs = PETSc.Vec().createWithArray(rhs)
    if x0 is None:
        petsc_sol = petsc_rhs.duplicate()
    else:
        petsc_sol = PETSc.Vec().createWithArray(x0)

    ksp = get_petsc_ksp(A, pctype="ilu", ksptype="richardson", tol=tol, max_it=max_it)
    ksp.solve(petsc_rhs, petsc_sol)
    x = petsc_sol.getArray()
    return x


def petsc_solve_lu(A, rhs, x0=None):
    petsc_rhs = PETSc.Vec().createWithArray(rhs)
    if x0 is None:
        petsc_sol = petsc_rhs.duplicate()
    else:
        petsc_sol = PETSc.Vec().createWithArray(x0)

    ksp = get_petsc_ksp(A, pctype="lu", ksptype="richardson", max_it=1)
    ksp.solve(petsc_rhs, petsc_sol)
    x = petsc_sol.getArray()
    return x


def petsc_solve(A, b, x0=None, tol=1e-5, ksptype="gmres"):
    """
    Solves Ax=b using Petsc krylov-type iterative solver

    Parameters
    ----------
    A: scipy matrix
    b: ndarray
    x0: ndarray, optional
        initial guess
    tol: int, optional
        convergence tolerance
    ksptype: string, optional
        type of Krylov accelerator. eg. "bcgs", "gmres"

    Returns
    -------
    out: ndarray
        solution x to equation Ax=b

    """
    comm = MPI.COMM_SELF  # Only works in serial

    ksp = get_petsc_ksp(A, pctype="ilu", ksptype=ksptype, tol=tol)

    petsc_rhs = PETSc.Vec().createWithArray(b, comm=comm)

    if x0 is None:
        petsc_sol = petsc_rhs.duplicate()
    else:
        petsc_sol = PETSc.Vec().createWithArray(x0, comm=comm)

    ksp.setInitialGuessNonzero(True)
    ksp.solve(petsc_rhs, petsc_sol)

    return petsc_sol.getArray()


def petsc_solve_from_ksp(ksp, rhs, x=None, tol=1e-5):
    ksp.setTolerances(rtol=tol)

    petsc_rhs = PETSc.Vec().createWithArray(rhs)

    if x is None:
        petsc_sol = petsc_rhs.duplicate()
    else:
        petsc_sol = PETSc.Vec().createWithArray(x)
    ksp.setInitialGuessNonzero(True)
    ksp.solve(petsc_rhs, petsc_sol)

    return petsc_sol.getArray()


def scipy_to_petsc_matrix(A):
    comm = MPI.COMM_SELF
    A = A.tocsr()
    petsc_mat = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data), comm=comm)
    petsc_mat.setUp()
    return petsc_mat