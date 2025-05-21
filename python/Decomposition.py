import numpy as np
import numpy.typing as npt
import scipy as sp
import sys
from sklearn.decomposition import TruncatedSVD
from copy import deepcopy
import time
# from threadpoolctl import threadpool_limits as tl

from python.utils import print_traceback, round_sig

# thread_limit=8

def SVD(
    matrix: npt.NDArray,
    Nkeep: int | None = None,
    Skeep: float | None = None,
    norm_Keep: float | None = None,
    use_sklearn: bool = True,
    Oversampling: int | None = None,
    Iteration: int = 5,
    threshold: float = 1.e-8,
    ifloss: bool = False,
    full_SVD: bool = False,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray] | tuple[npt.NDArray, npt.NDArray, npt.NDArray, float]:
    
    """
    Return U, S, Vh
    
    For arbitary complex matrix,
    matrix = U @ np.diag(S) @ Vh where
    U is left isometry (U.conj().T @ U = I)
    S is real diagonal (singular values = abs(eigenvalues))
    Vh is right isometry (Vh @ Vh.conj().T = I)
    """

    assert type(matrix) == type(
        np.array([1])), f"{type(matrix)=} != numpy.ndarray"
    # assert np.isfinite(matrix).all(), f"Matrix not finite, {matrix=}"
    assert len(matrix.shape) == 2, f"{len(matrix.shape)=} != 2"
    # stable_matrix = deepcopy(matrix)
    stable_matrix = matrix
    
    if np.isnan(stable_matrix).any() or np.isinf(stable_matrix).any():
        print("Data contains NaN or Inf values")
        # Optionally, replace NaNs or Infs
        stable_matrix = np.nan_to_num(stable_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    if np.prod(matrix.shape) * np.min(matrix.shape) > 10**9:
        print(f"SVD big matrix, shape={matrix.shape}: ", end="")

    now = time.perf_counter()
    # print(f"SVD, {matrix.shape=} {Nkeep=} {Skeep=}", end=" ")

    # if np.linalg.norm(matrix-matrix.conj())/np.linalg.norm(matrix) < 1.e-8:

    if full_SVD or Nkeep == 0:
        try:
            U, S, Vh = np.linalg.svd(stable_matrix, full_matrices=False)
        except np.linalg.LinAlgError as e:
            print(f"np.linalg.svd error\n{e}")
            print_traceback(e)
            print(f"Trying random SVD")
            U, S, Vh = random_SVD(stable_matrix)
        # return U, S, Vh

    elif norm_Keep is not None:
        try:
            U, S, Vh = np.linalg.svd(stable_matrix, full_matrices=False)
        except np.linalg.LinAlgError as e:
            print(f"np.linalg.svd error\n{e}")
            print_traceback(e)
            print(f"Trying random SVD")
            U, S, Vh = random_SVD(stable_matrix)
        
        squares = S ** 2
        full_norm = squares.sum()
        # Cumulative sum from the end, then reverse
        # cumsum_from_end = np.sqrt(np.cumsum(squares[::-1])[::-1])
        cumsum_from_end = np.cumsum(squares[::-1])[::-1]
        
        # print(f"{squares=}")
        # print(f"{cumsum_from_end=}")
        
        # Find indices where cumsum exceeds threshold
        mask = cumsum_from_end > norm_Keep
        # mask = cumsum_from_end > norm_Keep * full_norm
        # If no value exceeds threshold, return empty array
        if not np.any(mask):
            pass
        
        else:
            last_idx = np.where(mask)[0][-1]
            # Return elements up to and including that index
            U = U[:, :last_idx + 1]
            Vh = Vh[:last_idx + 1, :]
            S = S[:last_idx + 1]

    elif Nkeep is not None and Nkeep < min(matrix.shape) * 0.1:
        if use_sklearn and np.linalg.norm(matrix-matrix.conj())/np.linalg.norm(stable_matrix) < threshold and min(stable_matrix.shape) > 1:
            try:
                svd = TruncatedSVD(n_components=min(
                    min(matrix.shape), Nkeep))

                stable_matrix[np.abs(matrix) < 1.e-100] = 0
                # stable_matrix[np.abs(matrix.max() - matrix) <
                #               np.abs(matrix.max()) * threshold] = matrix.max()
                # stable_matrix[np.abs(matrix - matrix.min()) <
                #               np.abs(matrix.min()) * threshold] = matrix.min()
                # * Reduced U matrix
                U = svd.fit_transform(
                    np.real(stable_matrix).astype(np.float64))
                S = svd.singular_values_  # * The top n_components singular values
                Vh = svd.components_

                U = U[:, S > 0]
                Vh = Vh[S > 0, :]
                S = S[S > 0]
                U = U / S

            except Exception as e:
                print(f"Sklearn SVD error\n{e}Trying random SVD")
                print_traceback(e)
                
                try:
                    U, S, Vh = random_SVD(stable_matrix, Nkeep=Nkeep)
                except np.linalg.LinAlgError as e:
                    print(f"Random SVD error\n{e}\nnp.max(matrix)={round_sig(np.max(np.abs(stable_matrix)))} np.min(matrix)={round_sig(np.min(np.abs(stable_matrix)))}\nTrying np.linalg.svd")
                    print_traceback(e)
                    U, S, Vh = exact_SVD(matrix, stable_matrix, Nkeep, Skeep)
                    
        else:
            try:
                U, S, Vh = random_SVD(stable_matrix, Nkeep=Nkeep)
    
            except np.linalg.LinAlgError as e:
                print(f"Random SVD error\n{e}\nnp.max(matrix)={round_sig(np.max(np.abs(stable_matrix)))} np.min(matrix)={round_sig(np.min(np.abs(stable_matrix)))}\nTrying np.linalg.svd")
                print_traceback(e)
                U, S, Vh = exact_SVD(matrix, stable_matrix, Nkeep, Skeep)

    else:
        try:
            U, S, Vh = exact_SVD(matrix, stable_matrix, Nkeep, Skeep)

        except np.linalg.LinAlgError as e:
            print(f"{matrix.shape=}\n{np.max(np.abs(matrix))=} {np.min(np.abs(matrix))}\n{e}")
            print_traceback(e)
            
            try:
                U, S, Vh = random_SVD(stable_matrix)
                return U, S, Vh
            
            except np.linalg.LinAlgError as e:
                print(f"Random SVD error\n{e}\nnp.max(matrix)={round_sig(np.max(np.abs(stable_matrix)))} np.min(matrix)={round_sig(np.min(np.abs(stable_matrix)))}")
                print_traceback(e)
                sys.exit()

    # S[S.max() - S < S.max() * threshold] = S.max()

    # U = U[:, S > 0]
    # Vh = Vh[S > 0, :]
    # S = S[S > 0]
    
    # print(f"Before: {U=} {S=} {Vh=} {Skeep=}")

    # Check if S is defined and not empty
    if S is not None and len(S) > 0:
        if Skeep is not None:
            if S[0] > Skeep:
                U = U[:, S > Skeep]
                Vh = Vh[S > Skeep, :]
                S = S[S > Skeep]
            else:
                U = U[:, :1]
                Vh = Vh[:1, :]
                S = S[:1]
    
    # print(f"After: {U=} {S=} {Vh=}")

# non_finite_mask = ~np.isfinite(array)

    assert np.isfinite(
        U).all(), f"SVD Error, Not finite {U.shape=}\n{~np.isfinite(U)=}\n{U[~np.isfinite(U)]=}\n{matrix.shape=}\n{~np.isfinite(matrix)=}\n{matrix[~np.isfinite(matrix)]}"
    assert np.isfinite(
        S).all(), f"SVD Error, Not finite {S.shape=}\n{~np.isfinite(S)=}\n{S[~np.isfinite(S)]=}\n{matrix.shape=}\n{~np.isfinite(matrix)=}\n{matrix[~np.isfinite(matrix)]}"
    assert np.isfinite(
        Vh).all(), f"SVD Error, Not finite {Vh.shape=}\n{~np.isfinite(Vh)=}\n{Vh[~np.isfinite(Vh)]=}\n{matrix.shape=}\n{~np.isfinite(matrix)=}\n{matrix[~np.isfinite(matrix)]}"

    if np.prod(matrix.shape) * np.min(matrix.shape) > 10**9:
        print(f"{round_sig(time.perf_counter()-now)}s")

    if ifloss:
        loss = np.linalg.norm(
            matrix - U @ np.diag(S) @ Vh)/np.linalg.norm(matrix)

        # print(f"Finished {round_sig(time.perf_counter()-now)}s")

        return U, S, Vh, loss

    else:
        # print(f"Finished {round_sig(time.perf_counter()-now)}s")
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
    Nkeep: int | None = None,
    Skeep: float | None = None,
    threshold: float = 1.e-8,
    ifloss: bool = False,
) -> tuple[npt.NDArray, npt.NDArray] | tuple[npt.NDArray, npt.NDArray, float]:
    
    
    """
    Return eigvals, eigvecs
    
    Only for hermitian matrix,
    matrix = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T where
    eigvecs is left isometry
    eigvals is real diagonal (eigenvalues real only for hermitian matrix)
    """

    assert type(matrix) == type(
        np.array([1])), f"{type(matrix)=} != numpy.ndarray"
    assert np.isfinite(matrix).all(), f"Matrix not finite, {matrix=}"
    assert len(matrix.shape) == 2, f"EIGH {len(matrix.shape)=} != 2"

    assert matrix.shape[0] == matrix.shape[1], f"{matrix.shape[0]=} != {matrix.shape[1]=}"
    # if not np.allclose(matrix, matrix.conj().T, rtol=1.e-8, atol=1.e-8):
    
    if matrix.shape[0] == 1:
        if ifloss:
            return matrix[0], np.diag([1]), 0.0
        else:
            return matrix[0], np.diag([1])

    try:
        hermitian_error = np.linalg.norm(
            matrix-matrix.conj().T)/np.linalg.norm(matrix)
    except RuntimeWarning as e:
        print(f"Hermitian error, {matrix.shape=}\n{e=}")
        print_traceback(e)
        print(f"{matrix=}")
    
    if hermitian_error > 1.e-6 or np.linalg.norm(matrix-matrix.conj().T) > 1.e-6:
        print(f"matrix not hermitian\n{hermitian_error=}\n\n{matrix.shape=}\n{matrix=}")
        # raise Exception(f"matrix not hermitian")
    # print(f"hermitian, {round_sig(time.perf_counter()-now)}s", end=" ")

    # stable_matrix = deepcopy(matrix)
    stable_matrix = matrix

    eigvals, eigvecs = exact_EIGH(matrix, stable_matrix, Nkeep=Nkeep, Skeep=Skeep)

    # if eigvals is not None and len(eigvals) > 0:
    #     if Skeep is not None:
    #         try:
    #             eigvecs = eigvecs[:, eigvals > Skeep]
    #             eigvals = eigvals[eigvals > Skeep]

    #         except Exception as e:
    #             print(f"{e=}")
    #             print_traceback(e)
    #             print(f"{matrix=} {eigvals=} {eigvecs=}")

    # eigvals[
    # eigvals.max() - eigvals < eigvals.max() * threshold] = eigvals.max()

    # print(f"eigh, {round_sig(time.perf_counter()-now)}s", end=" ")

    assert np.isfinite(eigvals).all(
    ), f"EIGH Error, Not finite {eigvals=}\n{matrix.shape=}\n{matrix=}"
    assert np.isfinite(eigvecs).all(
    ), f"EIGH Error, Not finite {eigvecs=}\n{matrix.shape=}\n{matrix=}"


    if ifloss:
        loss = np.linalg.norm(
            matrix - eigvecs @ np.diag(eigvals)
            @ eigvecs.conj().T)/np.linalg.norm(matrix)

        # print(f"Finished {round_sig(time.perf_counter()-now)}s")

        return eigvals, eigvecs, loss

    # print(f"Finished {round_sig(time.perf_counter()-now)}s")
    else:
        
        return eigvals, eigvecs


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
    matrix: npt.NDArray
):
    """
    Return Q, R
    
    For arbitary complex matrix with m >= n,
    matrix = Q @ R where
    Q is left isometry (Q.conj().T @ Q = I)
    R is right triangular
    """

    assert type(matrix) == type(
        np.array([1])), f"{type(matrix)=} != numpy.ndarray"
    # assert np.isfinite(matrix).all(), f"Matrix not finite, {matrix=}"
    assert len(matrix.shape) == 2, f"{len(matrix.shape)=} != 2"
    
    stable_matrix = matrix
    
    m, n = matrix.shape    
    # if m < n:
    #     print(f"{matrix.shape=}, m<n")
        
    #     R, Q = RQ(stable_matrix)
    #     return Q.T, R.T
    
    # stable_matrix = deepcopy(matrix)
    
    if np.isnan(stable_matrix).any() or np.isinf(stable_matrix).any():
        print("Data contains NaN or Inf values")
        # Optionally, replace NaNs or Infs
        stable_matrix = np.nan_to_num(stable_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    if np.prod(matrix.shape) * np.min(matrix.shape) > 10**9:
        print(f"QR big matrix, shape={matrix.shape}: ", end="")
    
    try:
        Q, R = np.linalg.qr(stable_matrix, mode="reduced")
    except np.linalg.LinAlgError as e:
        print(f"np.linalg.qr error\n{e}")
        print_traceback(e)
        sys.exit()
    
    return Q, R


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

