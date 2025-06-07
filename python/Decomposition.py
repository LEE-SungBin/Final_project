import numpy as np
import numpy.typing as npt
import scipy as sp
import sys
from sklearn.decomposition import TruncatedSVD
from copy import deepcopy
import time
# from threadpoolctl import threadpool_limits as tl

from python.utils import print_traceback, round_sig
from python.Backend import Backend

# thread_limit=8

def SVD(
    matrix: npt.NDArray,
    full_SVD: bool = False,
    Nkeep: int | None = None,
    Skeep: float | None = None,
    bk: Backend = Backend('auto')
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    
    """
    Parameters
    ----------
    matrix : array_like
        A real or complex array with a.ndim >= 2.
    full_SVD : bool, optional
        If True, compute full SVD. If False, compute reduced SVD.
    Nkeep : int, optional
        Number of singular values to keep.
    Skeep : float, optional
        Threshold for singular values to keep.
    
    Returns
    -------
    U : ndarray
        Unitary matrix having left singular vectors as columns.
    S : ndarray
        The singular values, sorted in non-increasing order.
    Vh : ndarray
        Unitary matrix having right singular vectors as rows.
    """
    
    try:
        U, S, Vh = bk.svd(matrix, full_matrices=full_SVD)
    except Exception as e:
        print(e)
        
        U, S, Vh = random_SVD(matrix)
    
    if Nkeep is not None:
        U = U[:, :Nkeep]
        S = S[:Nkeep]
        Vh = Vh[:Nkeep, :]
    
    if Skeep is not None:
        mask = S > Skeep
        U = U[:, mask]
        S = S[mask]
        Vh = Vh[mask, :]
    
    return U, S, Vh


def exact_SVD(
    matrix, stable_matrix,
    Nkeep: int | None = None, Skeep: float | None = None,
    threshold: float = 1.e-8,
    max_attempts: int = 2,
    bk: Backend = Backend('auto')
):
    attempts = 0
    while attempts < max_attempts:
        try:
            U, S, Vh = bk.svd(stable_matrix, full_matrices=False)
            S[S < 0] = 0

            if Nkeep is not None:
                U = U[:, :Nkeep]
                Vh = Vh[:Nkeep, :]
                S = S[:Nkeep]

            return U, S, Vh

        except Exception as e:
            print(f"exact SVD error at attempt {attempts+1}\n{e}\n{matrix.shape=}\n{matrix=}")
            print_traceback(e)
            attempts += 1

    print(f"SVD failed to converge after {max_attempts} attempts. Trying random SVD")

    try:
        U, S, Vh = random_SVD(stable_matrix, bk=bk)
        return U, S, Vh
    
    except Exception as e:
        print(f"Random SVD error\n{e}\nnp.max(matrix)={round_sig(bk.max(bk.abs(stable_matrix)))} np.min(matrix)={round_sig(bk.min(bk.abs(stable_matrix)))}")
        print_traceback(e)
        sys.exit()


def random_SVD(
    stable_matrix,
    Oversampling: int | None = None,
    Iteration: int = 5,
    Nkeep: int | None = None,
    bk: Backend = Backend('auto')
):

    M, N = stable_matrix.shape

    if Oversampling is None:
        Oversampling = int(
            max(min(max(M**(3/4), N**(3/4)), 1000),10))
        # # print(f"{Oversampling=}")

    if Nkeep is None:
        Nkeep = min(M, N)

    q = int(Iteration/2)

    if Iteration % 2 == 1:
        Y = stable_matrix @ bk.randn(N, Nkeep+Oversampling)
    else:
        Y = bk.randn(M, Nkeep+Oversampling)

    for _ in range(q):
        try:
            Y = stable_matrix @ stable_matrix.conj().T @ Y
        except RuntimeWarning as e:
            print(f"random SVD warning\n{e}\nnp.max(matrix)={round_sig(bk.max(bk.abs(stable_matrix)))} np.min(matrix)={round_sig(bk.min(bk.abs(stable_matrix)))}\nnp.max(Y)={round_sig(bk.max(bk.abs(Y)))} np.min(Y)={round_sig(bk.min(bk.abs(Y)))}")
            print_traceback(e)

    Q, _ = bk.qr(Y)

    U, S, Vh = bk.svd(
        Q.conj().T @ stable_matrix, full_matrices=False)
    S[S < 0] = 0
    U = Q @ U

    U = U[:, :Nkeep]
    Vh = Vh[:Nkeep, :]
    S = S[:Nkeep]

    return U, S, Vh


def EIGH(
    matrix: npt.NDArray,
    bk: Backend = Backend('auto')
) -> tuple[npt.NDArray, npt.NDArray]:
    
    """
    Parameters
    ----------
    matrix : array_like
        A real or complex array with a.ndim >= 2.
    
    Returns
    -------
    eigenvalues : ndarray
        The eigenvalues, sorted in non-increasing order.
    eigenvectors : ndarray
        The eigenvectors, sorted in non-increasing order.
    """
    # Convert matrix to the correct backend type and complex dtype
    matrix = bk.to_device(bk.array(matrix, dtype=bk.complex))

    # Get eigenvalues and eigenvectors
    eigvals, eigvecs = bk.eigh(matrix)

    return eigvals, eigvecs


def exact_EIGH(
    matrix, stable_matrix,
    Nkeep: int | None = None, Skeep: float | None = None,
    threshold=1.e-8,
    max_attempts: int = 2,
    bk: Backend = Backend('auto')
):
    attempts = 0
    while attempts < max_attempts:
        try:
            eigvals, eigvecs = bk.eigh(stable_matrix)
            eigvals = eigvals[::-1]
            eigvecs = eigvecs[:, ::-1]

            if Nkeep is not None:
                eigvals = eigvals[:Nkeep]
                eigvecs = eigvecs[:, :Nkeep]

            return eigvals, eigvecs

        except Exception as e:
            print(f"exact EIGH error\n{e}\n{matrix.shape=}\n{matrix=}")
            print_traceback(e)
            attempts += 1

    print("EIGH failed to converge after maximum number of attempts.")
    return None


def lanczos(A, k, n_iter=10, bk: Backend = Backend('auto')):
    """
    Perform the Lanczos algorithm on matrix A to find k eigenvalues.

    Parameters:
    A (numpy.ndarray): The input matrix.
    k (int): The number of eigenvalues to compute.
    n_iter (int): The number of iterations.

    Returns:
    numpy.ndarray: Approximated eigenvalues.
    numpy.ndarray: Corresponding eigenvectors.
    """
    n = A.shape[0]
    Q = bk.zeros((n, n_iter + 1))
    alpha = bk.zeros(n_iter)
    beta = bk.zeros(n_iter + 1)
    Q[:, 0] = bk.randn(n)
    Q[:, 0] = Q[:, 0] / bk.norm(Q[:, 0])

    for j in range(1, n_iter + 1):
        v = bk.matmul(A, Q[:, j - 1])
        alpha[j - 1] = bk.trace(bk.matmul(Q[:, j - 1].T, v))
        v = v - alpha[j - 1] * Q[:, j - 1] - beta[j - 1] * Q[:, j - 2]
        beta[j] = bk.norm(v)
        Q[:, j] = v / beta[j]

    T = bk.diag(alpha) + bk.diag(beta[1:n_iter], k=1) + bk.diag(beta[1:n_iter], k=-1)
    eigenvalues, eigenvectors = bk.eigh(T)
    
    idx = bk.argsort(eigenvalues)[-k:]

    return eigenvalues[idx], eigenvectors[:, idx]


def QR(
    matrix: npt.NDArray,
    bk: Backend = Backend('auto')
) -> tuple[npt.NDArray, npt.NDArray]:
    
    """
    Parameters
    ----------
    matrix : array_like
        A real or complex array with a.ndim >= 2.
    
    Returns
    -------
    Q : ndarray
        The orthogonal/unitary matrix.
    R : ndarray
        The upper triangular matrix.
    """
    
    return bk.qr(matrix)


def RQ(
    matrix: npt.NDArray,
    bk: Backend = Backend('auto')
):
    """
    Return R, Q
    
    For an arbitrary complex matrix with m <= n,
    matrix = R @ Q where
    Q is right isometry (Q @ Q.conj().T = I)
    R is upper triangular
    """

    assert isinstance(matrix, (np.ndarray, bk.xp.Tensor)), f"{type(matrix)=} != numpy.ndarray or torch.Tensor"
    assert len(matrix.shape) == 2, f"{len(matrix.shape)=} != 2"

    stable_matrix = matrix
    
    if bk.isfinite(stable_matrix).all():
        print("Data contains NaN or Inf values")
        stable_matrix = bk.nan_to_num(stable_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    if bk.prod(matrix.shape) * bk.min(matrix.shape) > 10**9:
        print(f"RQ big matrix, shape={matrix.shape}: ", end="")

    try:
        reversed_matrix = bk.flipud(bk.fliplr(stable_matrix))
        Q, R = bk.qr(reversed_matrix.T, mode="reduced")

        R = bk.flipud(bk.fliplr(R.T))
        Q = bk.flipud(bk.fliplr(Q.T))

    except Exception as e:
        print(f"RQ decomposition error\n{e}")
        print_traceback(e)
        sys.exit()
    
    return R, Q

