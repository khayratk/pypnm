import numpy as np
from PyTrilinos import Epetra, AztecOO, ML, EpetraExt, Amesos
from mpi4py import MPI
from scipy.sparse import coo_matrix

def epetra_set_matrow_to_zero(A, row):
    if row in A.Map().MyGlobalElements():
        values, cols = A.ExtractGlobalRowCopy(row)
        n_entries = len(cols)
        ierr = A.ReplaceGlobalValues([row] * n_entries, cols, np.zeros_like(values))
        assert ierr == 0
    return A


def epetra_set_vecrow_to_zero(v, row):
    if row in v.Map().MyGlobalElements():
        ierr = v.ReplaceGlobalValues(np.asarray([0.0]), np.asarray([row], dtype=np.int32))
        assert ierr == 0
    return v


def transpose_matrix(A):
    """
    Transposes a matrix
    Parameters
    ----------
    A: Epetra Matrix

    Returns
    -------
    C: transpose of matrix A
    """
    V = Epetra.Vector(A.RangeMap())
    V.PutScalar(1.0)
    I = diagonal_matrix_from_vector(V, A.RangeMap())
    C = Epetra.CrsMatrix(Epetra.Copy, A.DomainMap(), 10)
    EpetraExt.Multiply(A, True, I, False, C)
    C.FillComplete()
    return C


def mat_multiply(A, B, transpose_1=False, transpose_2=False, fill=30):
    """
    Matrix product of two matrices
    Parameters
    ----------
    A: Epetra Matrix
    B: Epetra Matrix
    transpose_1: bool, optional
        If true, then A is transposed before multiplication
    transpose_2: bool, optional
        If true, then B is transposed before multiplication
    fill: int, optional
        estimate of how many nonzeros per row

    Returns
    -------
    C: Epetra Matrix
        C = A*B
    """
    if transpose_1:
        C = Epetra.CrsMatrix(Epetra.Copy, A.DomainMap(), fill)
    else:
        C = Epetra.CrsMatrix(Epetra.Copy, A.RangeMap(), fill)

    ierr = EpetraExt.Multiply(A, transpose_1, B, transpose_2, C)
    assert ierr == 0

    if transpose_1:
        C.FillComplete(B.DomainMap(), A.DomainMap())
    else:
        C.FillComplete()

    if not (transpose_1 or transpose_2):
        assert (C.NumGlobalRows(), C.NumGlobalCols()) == (A.NumGlobalRows(), B.NumGlobalCols())

    if transpose_1 and not transpose_2:
        assert (C.NumGlobalRows(), C.NumGlobalCols()) == (A.NumGlobalCols(), B.NumGlobalCols())

    if not transpose_1 and transpose_2:
        assert (C.NumGlobalRows(), C.NumGlobalCols()) == (A.NumGlobalRows(), B.NumGlobalRows())

    return C


def matrix_scipy_to_epetra(A_scipy, A_epetra=None):
    """
    Converts scipy matrix to a local (non-distributed) epetra matrix.
    Parameters
    ----------
    A_scipy: Scipy Matrix
    A_epetra: Epetra Matrix, optional
        An existing epetra matrix which has the same structure as the scipy matrix

    Returns
    -------
    A_epetra: Epetra Matrix
        If an existing Epetra matrix is passed, the values are replaced

    """
    comm = Epetra.MpiComm(MPI.COMM_SELF)
    map = Epetra.Map(A_scipy.shape[0], 0, comm)
    B = A_scipy.tocoo()

    if A_epetra is None:
        A_epetra = Epetra.CrsMatrix(Epetra.Copy, map, A_scipy.getnnz(axis=1), True)
        A_epetra.InsertGlobalValues(B.row, B.col, B.data)
        ierr = A_epetra.FillComplete()
    else:
        ierr = A_epetra.ReplaceMyValues(B.row, B.col, B.data)

    assert ierr == 0
    return A_epetra


def matrix_epetra_to_scipy(A_epetra):
    values = [A_epetra.ExtractGlobalRowCopy(i)[0] for i in xrange(A_epetra.NumGlobalRows())]
    columns = [A_epetra.ExtractGlobalRowCopy(i)[1] for i in xrange(A_epetra.NumGlobalRows())]
    rows = [i * np.ones(len(vals)) for i, vals in enumerate(values)]
    values = np.hstack(values)
    columns = np.hstack(columns).astype(np.int)
    rows = np.hstack(rows).astype(np.int)
    assert len(rows) == len(columns) == len(values)

    solve_pressure_trilinos_from_scipy

    return coo_matrix((values, (rows, columns)))

def vector_numpy_to_epetra(v):
    """

    Parameters
    ----------
    v

    Returns
    -------

    """
    comm = Epetra.MpiComm(MPI.COMM_SELF)
    map = Epetra.Map(len(v), 0, comm)
    vect = Epetra.Vector(map)
    vect[:] = v[:]
    return vect


def trilinos_ml_prec(A_epetra):
    mlList = {"max levels": 10,
              "output": 0,
              "smoother: pre or post": "both",
              "smoother: type": "Aztec"
              }

    prec = ML.MultiLevelPreconditioner(A_epetra)
    return prec


def trilinos_solve(A, rhs, prec, x=None, tol=1e-5):
    solver = AztecOO.AztecOO(A, x, rhs)
    solver.SetPrecOperator(prec)
    solver.SetAztecOption(AztecOO.AZ_solver, AztecOO.AZ_gmres)
    solver.SetAztecOption(AztecOO.AZ_conv, AztecOO.AZ_rhs)
    solver.SetAztecOption(AztecOO.AZ_output, 0)

    solver.Iterate(3000, tol)

    return x


def solve_aztec(A, rhs, x=None, tol=1e-5, plist=None):
    if x is None:
        x = Epetra.Vector(A.RangeMap())

    if plist is None:
        plist = {"Solver": "GMRES",
                 "Precond": "Dom_Decomp",
                 "Output": 0
                 }

    solver = AztecOO.AztecOO(A, x, rhs)
    solver.SetParameters(plist, True)
    solver.SetAztecOption(AztecOO.AZ_output, 0)
    solver.SetAztecOption(AztecOO.AZ_conv, AztecOO.AZ_rhs)

    ierr = solver.Iterate(300000, tol)
    assert (ierr == 0) or (ierr == -3), ierr
    return x


def solve_pressure_trilinos_from_scipy(A, rhs, x=None, tol=1e-15, plist=None):
    A_epetra = matrix_scipy_to_epetra(A)

    if x is None:
        x_epetra = vector_numpy_to_epetra(x)
    else:
        x_epetra = Epetra.Vector(A_epetra.RangeMap())

    rhs_epetra = vector_numpy_to_epetra(rhs)
    return solve_aztec(A_epetra, rhs_epetra, x_epetra, tol, plist)

def solve_direct(A, rhs, x=None):
    if x is None:
        x = Epetra.Vector(A.RangeMap())

    problem = Epetra.LinearProblem(A, x, rhs)
    solver = Amesos.Klu(problem)
    ierr = solver.Solve()
    assert (ierr == 0), ierr

    return x


def scale_cols_matrix(P):
    sums = sum_of_columns(P)
    v_sums_inv = Epetra.Vector(sums)
    v_sums_inv[:] = 1. / sums[:]
    assert np.all(v_sums_inv >= 0.0)
    P.LeftScale(v_sums_inv)


def sum_of_columns(A):
    X = Epetra.Vector(A.DomainMap())
    X.PutScalar(1.0)
    Y = Epetra.Vector(A.RangeMap())
    A.Multiply(False, X, Y)
    return Y


def diagonal_inverse(A, scalar=1.0):
    map = A.RangeMap()
    v_diagonal = Epetra.Vector(map)
    A.ExtractDiagonalCopy(v_diagonal)
    v_diagonal_inv = 1./v_diagonal[:]

    D_inv = Epetra.CrsMatrix(Epetra.Copy, map, 1)
    row_inds = D_inv.Map().MyGlobalElements()
    D_inv.InsertGlobalValues(row_inds, row_inds, scalar*v_diagonal_inv)
    D_inv.FillComplete()
    return D_inv


def identity_matrix(map):
    I = Epetra.CrsMatrix(Epetra.Copy, map, 1)
    row_inds = map.MyGlobalElements()
    I.InsertGlobalValues(row_inds, row_inds, np.ones(len(row_inds), dtype=np.float))
    I.FillComplete()
    return I


def diagonal_matrix_from_vector(V, row_map):
    C = Epetra.CrsMatrix(Epetra.Copy, row_map, 1)
    row_inds = C.Map().MyGlobalElements()
    C.InsertGlobalValues(row_inds, row_inds, V)
    C.FillComplete()
    return C


def DinvA(A, omega=1.0):
    D_inv = diagonal_inverse(A, scalar=omega)
    J = mat_multiply(D_inv, A)
    return J


def G_Jacobi(A, omega=1.0):
    """
    Computes the product G = (I- omega * Dinv A)
    Parameters
    ----------
    A: CsrMatrix

    Returns
    -------
    G: CsrMatrix
    """
    map = A.RangeMap()
    D_inv = diagonal_inverse(A, scalar=-omega)
    J = mat_multiply(D_inv, A)
    row_inds = map.MyGlobalElements()
    ierr = J.SumIntoGlobalValues(row_inds, row_inds, np.ones(len(row_inds), dtype=np.float))
    assert ierr == 0
    return J