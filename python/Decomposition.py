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
    
    U, S, Vh = bk.svd(matrix, full_matrices=full_SVD)
    
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
    max_attempts: int = 2
):
    attempts = 0
    while attempts < max_attempts:
        try:
            U, S, Vh = np.linalg.svd(stable_matrix, full_matrices=False)
            S[S < 0] = 0

            # if not np.allclose(matrix, U @ np.diag(S) @ Vh, rtol=1.e-8, atol=1.e-8):
            # if not np.linalg.norm(matrix - U @ np.diag(S) @ Vh)/np.linalg.norm(matrix) < 1.e-6 and not np.linalg.norm(matrix - U @ np.diag(S) @ Vh) < 1.e-6:
            #     # print(f"{np.linalg.norm(matrix - U @ np.diag(S) @ Vh)/np.linalg.norm(matrix)=}\n{np.linalg.norm(matrix - U @ np.diag(S) @ Vh)=}\n{matrix=}\n{stable_matrix=}")
            #     raise Exception(f"SVD Error, Incorrect decomposition")

            if Nkeep is not None:
                U = U[:, :Nkeep]
                Vh = Vh[:Nkeep, :]
                S = S[:Nkeep]

            return U, S, Vh

        except np.linalg.LinAlgError as e:
            print(f"exact SVD error at attempt {attempts+1}\n{e}\n{matrix.shape=}\n{matrix=}")
            print_traceback(e)
            # raise e
            attempts += 1

    print(f"SVD failed to converge after {max_attempts} attempts. Trying random SVD")

    try:
        U, S, Vh = random_SVD(stable_matrix)
        return U, S, Vh
    
    except np.linalg.LinAlgError as e:
        print(f"Random SVD error\n{e}\nnp.max(matrix)={round_sig(np.max(np.abs(stable_matrix)))} np.min(matrix)={round_sig(np.min(np.abs(stable_matrix)))}")
        print_traceback(e)
        sys.exit()
        # U, S, Vh = exact_SVD(matrix, stable_matrix, Nkeep, Skeep)


def random_SVD(
    stable_matrix,
    Oversampling: int | None = None,
    Iteration: int = 5,
    Nkeep: int | None = None
):

    rng = np.random.default_rng()
    M, N = stable_matrix.shape

    if Oversampling is None:
        Oversampling = int(
            max(min(max(M**(3/4), N**(3/4)), 1000),10))
        # # print(f"{Oversampling=}")

    if Nkeep is None:
        Nkeep = min(M, N)

    q = int(Iteration/2)

    if Iteration % 2 == 1:
        Y = stable_matrix @ rng.random(
            size=(N, Nkeep+Oversampling))
    else:
        Y = rng.random(size=(M, Nkeep+Oversampling))

    for _ in range(q):
        try:
            Y = stable_matrix @ stable_matrix.conj().T @ Y
        except RuntimeWarning as e:
            print(f"random SVD warning\n{e}\nnp.max(matrix)={round_sig(np.max(np.abs(stable_matrix)))} np.min(matrix)={round_sig(np.min(np.abs(stable_matrix)))}\nnp.max(Y)={round_sig(np.max(np.abs(Y)))} np.min(Y)={round_sig(np.min(np.abs(Y)))}")
            print_traceback(e)

    Q, _ = np.linalg.qr(Y)

    U, S, Vh = np.linalg.svd(
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
    
    return bk.eigh(matrix)


def exact_EIGH(
    matrix, stable_matrix,
    Nkeep: int | None = None, Skeep: float | None = None,
    threshold=1.e-8,
    max_attempts: int = 2
):
    attempts = 0
    while attempts < max_attempts:
        try:
            eigvals, eigvecs = sp.linalg.eigh(stable_matrix)
            eigvals = eigvals[::-1]
            eigvecs = eigvecs[:, ::-1]
            # eigvals[eigvals < 0] = 0

            # if not np.allclose(matrix, eigvecs @ np.diag(eigvals) @ eigvecs.conj().T, rtol=1.e-8, atol=1.e-8):
            # if not np.linalg.norm(matrix - eigvecs @ np.diag(eigvals) @ eigvecs.conj().T)/np.linalg.norm(matrix) < 1.e-6 and not np.linalg.norm(matrix - eigvecs @ np.diag(eigvals) @ eigvecs.conj().T) < 1.e-6:
            #     # print(f"{np.linalg.norm(matrix - eigvecs @ np.diag(eigvals) @ eigvecs.conj().T)/np.linalg.norm(matrix)=}\n{np.linalg.norm(matrix - eigvecs @ np.diag(eigvals) @ eigvecs.conj().T)}\n{matrix=}\n{stable_matrix=}")
            #     raise Exception(
            #         f"EIGH Error, Incorrect decomposition")

            if Nkeep is not None:
                eigvals = eigvals[:Nkeep]
                eigvecs = eigvecs[:, :Nkeep]

            # eigvals[
            #     eigvals.max() - eigvals < eigvals.max() * threshold] = eigvals.max()

            return eigvals, eigvecs

        except np.linalg.LinAlgError as e:
            print(f"exact EIGH error\n{e}\n{matrix.shape=}\n{matrix=}")
            print_traceback(e)
            # raise e
            attempts += 1

    print("EIGH failed to converge after maximum number of attempts.")
    return None


def lanczos(A, k, n_iter=10):
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
    Q = np.zeros((n, n_iter + 1))
    alpha = np.zeros(n_iter)
    beta = np.zeros(n_iter + 1)
    Q[:, 0] = np.random.rand(n)
    Q[:, 0] = Q[:, 0] / np.linalg.norm(Q[:, 0])

    for j in range(1, n_iter + 1):
        v = A @ Q[:, j - 1]
        alpha[j - 1] = np.dot(Q[:, j - 1], v)
        v = v - alpha[j - 1] * Q[:, j - 1] - beta[j - 1] * Q[:, j - 2]
        beta[j] = np.linalg.norm(v)
        Q[:, j] = v / beta[j]

    T = np.diag(alpha) + np.diag(beta[1:n_iter], k=1) + np.diag(beta[1:n_iter], k=-1)
    eigenvalues, eigenvectors = np.linalg.eigh(T)
    
    # Select the k largest eigenvalues and corresponding eigenvectors.
    idx = np.argsort(eigenvalues)[-k:]

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
    matrix: npt.NDArray
):
    """
    Return R, Q
    
    For an arbitrary complex matrix with m <= n,
    matrix = R @ Q where
    Q is right isometry (Q @ Q.conj().T = I)
    R is upper triangular
    """

    # Ensure the input is a numpy array
    assert isinstance(matrix, np.ndarray), f"{type(matrix)=} != numpy.ndarray"
    assert len(matrix.shape) == 2, f"{len(matrix.shape)=} != 2"

    stable_matrix = matrix
    
    # Check for NaN or Inf values
    if np.isnan(stable_matrix).any() or np.isinf(stable_matrix).any():
        print("Data contains NaN or Inf values")
        stable_matrix = np.nan_to_num(stable_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # Warn if the matrix is very large
    if np.prod(matrix.shape) * np.min(matrix.shape) > 10**9:
        print(f"RQ big matrix, shape={matrix.shape}: ", end="")

    try:
        # Reverse the matrix
        reversed_matrix = np.flipud(np.fliplr(stable_matrix))
        
        # Perform QR decomposition on the reversed matrix
        Q, R = np.linalg.qr(reversed_matrix.T, mode="reduced")

        # Reverse the R and Q to obtain the RQ decomposition
        R = np.flipud(np.fliplr(R.T))
        Q = np.flipud(np.fliplr(Q.T))

    except np.linalg.LinAlgError as e:
        print(f"RQ decomposition error\n{e}")
        print_traceback(e)
        sys.exit()
    
    return R, Q

