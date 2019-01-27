import sys


try:
    import petsc4py
    from mpi4py import MPI

    petsc4py.init(sys.argv)
    from petsc4py import PETSc

    WITH_PETSC = True
except ImportError:
    WITH_PETSC = False

ksp_existing = dict()
pc_existing = dict()


def get_petsc_ksp(A, pctype="ilu", ksptype="gmres", tol=1e-5, max_it=10000):
    """
    Parameters
    ----------
    A: cs_matrix
        Scipy sparse matrix

    pctype: string, optional
        preconditioner type. Defult is "ilu"

    ksptype: string, optional
        Petsc's krylov subspace solver (KSP) object type. Defuilt is "gmres"

    tol: float, optional
        tolerance level for KSP object

    max_it: int, optional
        Maximum number of iteration

    Returns
    -------
    out: ksp
        Petsc KSP object

    """
    comm = MPI.COMM_SELF
    petsc_mat = scipy_to_petsc_matrix(A)
    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOperators(A=petsc_mat)
    ksp.setType(ksptype)
    ksp.setTolerances(rtol=tol, max_it=max_it)
    ksp.setFromOptions()
    pc = ksp.getPC()
    pc.setType(pctype)
    return ksp


def petsc_solve_ilu(A, b, x0=None, tol=1e-5, max_it=5):
    """
    Solves Ax=b using richardson iteration preconditioned with ilu

    Parameters
    ----------
    A: scipy matrix
    b: ndarray
        right hand side of Ax=b
    x0: ndarray, optional
        initial guess for x
    tol: float, optional
        convergence tolerance
    max_it: int, optional
        Maximum number of iteration

    Returns
    -------
    out: ndarray
        solution x to equation Ax=b
    """
    petsc_rhs = PETSc.Vec().createWithArray(b)
    if x0 is None:
        petsc_sol = petsc_rhs.duplicate()
    else:
        petsc_sol = PETSc.Vec().createWithArray(x0)

    ksp = get_petsc_ksp(A, pctype="ilu", ksptype="richardson", tol=tol, max_it=max_it)
    ksp.solve(petsc_rhs, petsc_sol)
    x = petsc_sol.getArray()
    return x


def petsc_solve_lu(A, b, x0=None):
    """
    Solves Ax=b directly using LU decomposition

    Parameters
    ----------
    A: scipy matrix
    b: ndarray
        right hand side of Ax=b

    Returns
    -------
    out: ndarray
        solution x to equation Ax=b
    """
    petsc_rhs = PETSc.Vec().createWithArray(b)
    petsc_sol = petsc_rhs.duplicate()

    ksp = get_petsc_ksp(A, pctype="lu", ksptype="richardson", max_it=1)
    ksp.solve(petsc_rhs, petsc_sol)
    x = petsc_sol.getArray()
    return x


def petsc_solve(A, b, x0=None, tol=1.e-5, ksptype="minres", pctype="ilu"):
    """
    Solves Ax=b using Petsc krylov-type iterative solver

    Parameters
    ----------
    A: scipy matrix
    b: ndarray
        right hand side of Ax=b
    x0: ndarray, optional
        initial guess for x
    tol: float, optional
        convergence tolerance
    ksptype: string, optional
        type of Krylov accelerator. eg. "bcgs", "gmres"

    Returns
    -------
    out: ndarray
        solution x to equation Ax=b
    """
    comm = MPI.COMM_SELF  # Only works in serial

    ksp = get_petsc_ksp(A, pctype=pctype, ksptype=ksptype, tol=tol)

    petsc_rhs = PETSc.Vec().createWithArray(b, comm=comm)

    if x0 is None:
        petsc_sol = petsc_rhs.duplicate()
    else:
        petsc_sol = PETSc.Vec().createWithArray(x0, comm=comm)

    ksp.setInitialGuessNonzero(True)
    ksp.setFromOptions()
    ksp.solve(petsc_rhs, petsc_sol)

    return petsc_sol.getArray()


def petsc_solve_from_ksp(ksp, b, x0=None, tol=1e-5):
    """
    Solves Ax=b using Petsc KSP object

    Parameters
    ----------
    ksp: petsc KSP object
        Krylov subspace solber object
    b: ndarray:
        right hand side of Ax=b
    x0: ndarray, optional
        initial guess for x
    tol: float, optional
        convergence tolerance

    Returns
    -------
    out: ndarray
        solution x to equation Ax=b

    """
    ksp.setTolerances(rtol=tol)

    comm = MPI.COMM_SELF  # Only works in serial
    petsc_rhs = PETSc.Vec().createWithArray(b, comm=comm)

    if x0 is None:
        petsc_sol = petsc_rhs.duplicate()
    else:
        petsc_sol = PETSc.Vec().createWithArray(x0, comm=comm)

    if not ksp.getType() == "preonly":
        ksp.setInitialGuessNonzero(True)

    ksp.solve(petsc_rhs, petsc_sol)

    return petsc_sol.getArray()


def scipy_to_petsc_matrix(A):
    """
    Converts scipy sparse matrix to petsc matrix
    """
    comm = MPI.COMM_SELF
    A = A.tocsr()
    petsc_mat = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data), comm=comm)
    petsc_mat.setUp()
    return petsc_mat