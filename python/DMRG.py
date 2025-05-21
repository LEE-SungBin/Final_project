import numpy as np
import numpy.typing as npt
# from python.Canonical_Form import move_site_left, move_site_right
from python.Contract import Contract
from python.Decomposition import SVD, EIGH, QR
# from python.Gauging import *
from python.initialization import random_initialization, Iterative_diagonalization
from python.utils import round_sig
from python.Zippers import MPS_MPO_MPS_overlap
import time

from copy import deepcopy


def DMRG(
     Hamiltonian: list[npt.NDArray],
     NKeep: int,
     NSweep: int,
     iterative_diag: bool = True,
     two_site: bool = True,
     Krylov_bases: int = 5,
     Lanczos_cutoff: float = 1e-4,
     verbose: bool = False,
) -> tuple[npt.NDArray, npt.NDArray, list]:

     """

     Hamiltonian

          1
          |
     2 --- --- 3
          |
          0

     MPS

     0 --- --- 1
          |
          2

     """
     n_sites = len(Hamiltonian)
     
     if verbose:
          print(f"L={n_sites} | {NKeep=} | {NSweep=} | iterative={iterative_diag} | {two_site=} | {Krylov_bases=} | {Lanczos_cutoff=}")

     """
     initial MPS is automatically normalized
     """
     
     if iterative_diag:
          MPS = Iterative_diagonalization(
               Hamiltonian=Hamiltonian, NKeep=NKeep
          )
          
     else:
          MPS = random_initialization(
               Hamiltonian=Hamiltonian, NKeep=NKeep
          )
     
     """
     Transpose to calculate overlap
     """
     
     MPO = [tensor.transpose(2, 3, 0, 1) for tensor in Hamiltonian]
     
     initial_energy = MPS_MPO_MPS_overlap(MPS, MPO, MPS)
     
     """
     Compute contract_list_left and right
     """
     
     contract_list_left: list[npt.NDArray] = [
          np.array([1.0]).reshape(1, 1, 1) for _ in range(n_sites+1)
     ]
     contract_list_right: list[npt.NDArray] = [
          np.array([1.0]).reshape(1, 1, 1) for _ in range(n_sites+1)
     ]

     for it in range(1, n_sites+1):
          contract_list_left[it] = Contract(
               "abc,aix,yxbj,cky->ijk",
               contract_list_left[it-1],
               MPS[it-1], Hamiltonian[it-1], MPS[it-1].conj()
          )
     
     """
     Iterate to get the ground state and ground state energy
     """
     
     times = [0.0]
     total_energies = [initial_energy.real]
     
     if verbose:
          print(f"iter=0 | energy={round_sig(total_energies[0], 8)} | time={round_sig(times[-1], 3)}s")
     
     for iter in range(NSweep):
          now = time.perf_counter()
          
          if two_site:
               energies, MPS = sweep_with_two_site_update(
                    MPS = MPS,
                    Hamiltonian = Hamiltonian,
                    contract_list_left = contract_list_left,
                    contract_list_right = contract_list_right,
                    NKeep = NKeep,
                    Krylov_bases = Krylov_bases,
                    Lanczos_cutoff = Lanczos_cutoff,
               )
          
          else:
               energies, MPS = sweep_with_single_site_update(
                    MPS = MPS,
                    Hamiltonian = Hamiltonian,
                    contract_list_left = contract_list_left,
                    contract_list_right = contract_list_right,
                    Krylov_bases = Krylov_bases,
                    Lanczos_cutoff = Lanczos_cutoff,
               )
          
          total_energies.extend(energies)
          if iter > 0:
               times.append(times[-1] + time.perf_counter()-now)
          else:
               times.append(time.perf_counter()-now)
          
          if verbose:
               print(f"iter={iter+1} | energy={round_sig(np.real(energies[-1]), 8)} | time={round_sig(times[-1], 3)}s")
     
     total_energies = np.array(total_energies)
     times = np.array(times)
     
     return total_energies, times, MPS


def sweep_with_single_site_update(
     MPS: list[npt.NDArray],
     Hamiltonian: list[npt.NDArray],
     contract_list_left: list[npt.NDArray],
     contract_list_right: list[npt.NDArray],
     Krylov_bases: int = 5,
     Lanczos_cutoff: float = 1e-4,
) -> tuple[float, list[npt.NDArray]]:
     
     n_sites = len(Hamiltonian)
     
     energies = []
     
     """
     Right to left
     """
     
     for it in range(n_sites-1):
          
          orthogonality_center = n_sites - it - 1
          left_loc = orthogonality_center
          right_loc = n_sites - orthogonality_center - 1
          
          initial_vector = deepcopy(MPS[orthogonality_center])
          
          contract_left = deepcopy(contract_list_left[left_loc])
          contract_center = deepcopy(Hamiltonian[orthogonality_center])
          contract_right = deepcopy(contract_list_right[right_loc])
          
          alphas, betas, left_isometries = lanczos_for_single_site(
               initial_vector = initial_vector,
               contract_left = contract_left,
               contract_center = contract_center,
               contract_right = contract_right,
               Krylov_bases = Krylov_bases,
               Lanczos_cutoff = Lanczos_cutoff,
          )
          
          energy, vector = get_eigen_vector_from_lanczos(
               alphas = alphas, betas = betas,
               left_isometries = left_isometries,
          )
          
          energies.append(energy)
                    
          matrix = vector.reshape(vector.shape[0], -1)
          U, S, Vh = SVD(matrix, full_SVD=True)
          Vh = Vh.reshape(-1, vector.shape[1], vector.shape[2])
          
          """
          renormalize
          """
          S = S / np.linalg.norm(S)
          
          """
          Update MPS
          """
          
          MPS[orthogonality_center] = Vh
          MPS[orthogonality_center-1] = Contract(
               "iak,ab,bj->ijk", MPS[orthogonality_center-1],
               U, np.diag(S)
          )
          
          """
          Update contract_list_right
          """
          
          contract_list_right[right_loc + 1] = Contract(
               "iax,yxjb,kcy,abc->ijk",
               MPS[orthogonality_center], Hamiltonian[orthogonality_center],
               MPS[orthogonality_center].conj(), contract_list_right[right_loc]
          )
     
     """
     Left to right
     """
     
     for it in range(n_sites-1):
          
          orthogonality_center = it
          left_loc = orthogonality_center
          right_loc = n_sites - orthogonality_center - 1
          
          initial_vector = deepcopy(MPS[orthogonality_center])
          
          contract_left = deepcopy(contract_list_left[left_loc])
          contract_center = deepcopy(Hamiltonian[orthogonality_center])
          contract_right = deepcopy(contract_list_right[right_loc])
          
          alphas, betas, left_isometries = lanczos_for_single_site(
               initial_vector = initial_vector,
               contract_left = contract_left,
               contract_center = contract_center,
               contract_right = contract_right,
               Krylov_bases = Krylov_bases,
               Lanczos_cutoff = Lanczos_cutoff,
          )
          
          energy, vector = get_eigen_vector_from_lanczos(
               alphas = alphas, betas = betas,
               left_isometries = left_isometries,
          )
          
          energies.append(energy)
                    
          matrix = vector.transpose(0, 2, 1).reshape(-1, vector.shape[1])
          U, S, Vh = SVD(matrix, full_SVD=True)
          U = U.reshape(vector.shape[0], vector.shape[2], -1).transpose(0, 2, 1)
          
          """
          renormalize
          """
          S = S / np.linalg.norm(S)
          
          """
          Update MPS
          """
          
          MPS[orthogonality_center] = U
          MPS[orthogonality_center + 1] = Contract(
               "ia,ab,bjk->ijk", np.diag(S), Vh,
               MPS[orthogonality_center+1],
          )
          
          """
          Update contract_list_left
          """
          
          contract_list_left[left_loc + 1] = Contract(
               "abc,aix,yxbj,cky->ijk",
               contract_list_left[left_loc], MPS[orthogonality_center],
               Hamiltonian[orthogonality_center], MPS[orthogonality_center].conj()
          )
     
     energies = np.array(energies)
     
     return energies, MPS


def get_eigen_vector_from_lanczos(
     alphas: npt.NDArray,
     betas: npt.NDArray,
     left_isometries: list[npt.NDArray]
) -> tuple[float, npt.NDArray]:
     
     Tridiagonal = get_tridiagonal_matrix(alphas, betas)
     
     # print(f"{Tridiagonal=}")
     
     eigvals, eigvecs = EIGH(Tridiagonal)
     
     # print(f"{alphas=} {betas=} {eigvals=}")
     
     """
     get eigenstate with the lowest eigenvalue
     """
     eigval = eigvals[-1]
     eigvec = eigvecs[:,-1]
     
     eigenvector = np.zeros(shape=left_isometries[0].shape)
     
     for coefficient, left_isometry in zip(eigvec, left_isometries):
          eigenvector = eigenvector + coefficient * left_isometry
     
     return eigval, eigenvector


def lanczos_for_single_site(
     initial_vector: npt.NDArray,
     contract_left: npt.NDArray,
     contract_center: npt.NDArray,
     contract_right: npt.NDArray,
     Krylov_bases: int = 5,
     Lanczos_cutoff: float = 1e-4,
) -> tuple[npt.NDArray, npt.NDArray, list[npt.NDArray]]:
     
     left_isometries = []
     alphas = []
     betas = []
     
     """
     Iteration 1
     """
     
     vector = deepcopy(initial_vector)
     norm = np.linalg.norm(vector)
     vector = vector / norm
     
     # print(f"{np.linalg.norm(vector)=}")
     
     left_isometries.append(vector)
     
     omega = matrix_vector_multi_for_single_site(
          vector,
          contract_left,
          contract_center,
          contract_right,
     )
     
     alpha = Contract(
          "ijk,ijk->", vector.conj(), omega
     )
     
     # assert np.abs(alpha.imag) > 1e-15 * np.abs(alpha.real), f"Hamiltonian not Hermitian"
     
     alphas.append(alpha)
     vector = orthogonalize(omega, left_isometries, two_site=False)
     
     for iter in range(1, Krylov_bases):
          beta = np.linalg.norm(vector)
          
          if beta < Lanczos_cutoff:
               break
          
          betas.append(beta)
          vector = vector / beta
          left_isometries.append(vector)

          omega = matrix_vector_multi_for_single_site(
               vector,
               contract_left,
               contract_center,
               contract_right,
          )
          
          alpha = Contract("ijk,ijk->", vector.conj(), omega)
          alphas.append(alpha)
          
          vector = orthogonalize(omega, left_isometries, two_site=False)
     
     alphas = np.array(alphas)
     betas = np.array(betas)
     
     return alphas, betas, left_isometries


def matrix_vector_multi_for_single_site(
     vector: npt.NDArray,
     contract_left: npt.NDArray,
     contract_center: npt.NDArray,
     contract_right: npt.NDArray,
) -> npt.NDArray:
     
     return Contract(
          "abi,acx,kxbd,cdj->ijk",
          contract_left, vector, contract_center, contract_right
     )


def orthogonalize(
     omega: npt.NDArray,
     left_isometries: list[npt.NDArray],
     two_site: bool = True,
) -> npt.NDArray:
     
     coefficients = []
     
     for left_isometry in left_isometries:
          if two_site:
               coefficient = Contract(
                    "ijkl,ijkl->", omega, left_isometry.conj()
               )
          else:
               coefficient = Contract(
                    "ijk,ijk->", omega, left_isometry.conj()
               )
          
          coefficients.append(coefficient)
     
     coefficients = np.array(coefficients)
     
     new_vector = omega
     
     for coefficient, left_isometry in zip(coefficients, left_isometries):
          new_vector = new_vector - coefficient * left_isometry
     
     return new_vector


def get_tridiagonal_matrix(
     alphas: npt.NDArray,
     betas: npt.NDArray,
) -> npt.NDArray:
     
     size = len(alphas)
     assert len(betas) == size - 1, f"{len(alphas)=} != {len(betas)+1=}"
     
     Tridiagonal = np.diag(alphas)
     
     for it, beta in enumerate(betas):
          Tridiagonal[it+1][it] = beta
          Tridiagonal[it][it+1] = beta
     
     return Tridiagonal


def sweep_with_two_site_update(
     MPS: list[npt.NDArray],
     Hamiltonian: list[npt.NDArray],
     contract_list_left: list[npt.NDArray],
     contract_list_right: list[npt.NDArray],
     NKeep: int,
     Krylov_bases: int = 5,
     Lanczos_cutoff: float = 1e-4,
) -> tuple[float, list[npt.NDArray]]:
     
     n_sites = len(Hamiltonian)
     
     energies = []
     
     """
     Right to left
     """
     
     for it in range(n_sites-1):
          
          orthogonality_center = n_sites - it - 1
          left_loc = orthogonality_center - 1
          right_loc = n_sites - orthogonality_center - 1
          
          initial_vector1 = deepcopy(MPS[orthogonality_center-1])
          initial_vector2 = deepcopy(MPS[orthogonality_center])
          
          initial_vector = Contract("iak,ajl->ijkl", initial_vector1, initial_vector2)
          
          contract_left = deepcopy(contract_list_left[left_loc])
          contract_center1 = deepcopy(Hamiltonian[orthogonality_center-1])
          contract_center2 = deepcopy(Hamiltonian[orthogonality_center])
          contract_right = deepcopy(contract_list_right[right_loc])
          
          alphas, betas, left_isometries = lanczos_for_two_site(
               initial_vector = initial_vector,
               contract_left = contract_left,
               contract_center1 = contract_center1,
               contract_center2 = contract_center2,
               contract_right = contract_right,
               Krylov_bases = Krylov_bases,
               Lanczos_cutoff = Lanczos_cutoff,
          )
          
          energy, vector = get_eigen_vector_from_lanczos(
               alphas = alphas, betas = betas,
               left_isometries = left_isometries,
          )
          
          energies.append(energy)
                    
          matrix = vector.transpose(0, 2, 1, 3).reshape(vector.shape[0] * vector.shape[2], -1)
          U, S, Vh = SVD(matrix, Nkeep=NKeep, Skeep=1.e-8)
          Vh = Vh.reshape(-1, vector.shape[1], vector.shape[3])
          U = U.reshape(vector.shape[0], vector.shape[2], -1).transpose(0, 2, 1)
          
          """
          renormalize
          """
          S = S / np.linalg.norm(S)
          
          """
          Update MPS
          """
          
          MPS[orthogonality_center] = Vh
          MPS[orthogonality_center-1] = Contract(
               "iak,aj->ijk", U, np.diag(S)
          )
          
          """
          Update contract_list_right
          """
          
          contract_list_right[right_loc + 1] = Contract(
               "iax,yxjb,kcy,abc->ijk",
               MPS[orthogonality_center], Hamiltonian[orthogonality_center],
               MPS[orthogonality_center].conj(), contract_list_right[right_loc]
          )
     
     """
     Left to right
     """
     
     for it in range(n_sites-1):
          
          orthogonality_center = it
          left_loc = orthogonality_center
          right_loc = n_sites - orthogonality_center - 2
          
          initial_vector1 = deepcopy(MPS[orthogonality_center])
          initial_vector2 = deepcopy(MPS[orthogonality_center+1])
          
          initial_vector = Contract("iak,ajl->ijkl", initial_vector1, initial_vector2)
          
          contract_left = deepcopy(contract_list_left[left_loc])
          contract_center1 = deepcopy(Hamiltonian[orthogonality_center])
          contract_center2 = deepcopy(Hamiltonian[orthogonality_center+1])
          contract_right = deepcopy(contract_list_right[right_loc])
          
          alphas, betas, left_isometries = lanczos_for_two_site(
               initial_vector = initial_vector,
               contract_left = contract_left,
               contract_center1 = contract_center1,
               contract_center2 = contract_center2,
               contract_right = contract_right,
               Krylov_bases = Krylov_bases,
               Lanczos_cutoff = Lanczos_cutoff,
          )
          
          energy, vector = get_eigen_vector_from_lanczos(
               alphas = alphas, betas = betas,
               left_isometries = left_isometries,
          )
          
          energies.append(energy)
                    
          matrix = vector.transpose(0, 2, 1, 3).reshape(vector.shape[0] * vector.shape[2], -1)
          U, S, Vh = SVD(matrix, Nkeep = NKeep, Skeep = 1e-8)
          U = U.reshape(vector.shape[0], vector.shape[2], -1).transpose(0, 2, 1)
          Vh = Vh.reshape(-1, vector.shape[1], vector.shape[3])
          
          """
          renormalize
          """
          S = S / np.linalg.norm(S)
          
          """
          Update MPS
          """
          
          MPS[orthogonality_center] = U
          MPS[orthogonality_center + 1] = Contract(
               "ia,ajk->ijk", np.diag(S), Vh,
          )
          
          """
          Update contract_list_left
          """
          
          contract_list_left[left_loc + 1] = Contract(
               "abc,aix,yxbj,cky->ijk",
               contract_list_left[left_loc], MPS[orthogonality_center],
               Hamiltonian[orthogonality_center], MPS[orthogonality_center].conj()
          )
     
     energies = np.array(energies)
     
     return energies, MPS


def lanczos_for_two_site(
     initial_vector: npt.NDArray,
     contract_left: npt.NDArray,
     contract_center1: npt.NDArray,
     contract_center2: npt.NDArray,
     contract_right: npt.NDArray,
     Krylov_bases: int = 5,
     Lanczos_cutoff: float = 1e-8,
) -> tuple[npt.NDArray, npt.NDArray, list[npt.NDArray]]:
     
     left_isometries = []
     alphas = []
     betas = []
     
     """
     Iteration 1
     """
     
     vector = deepcopy(initial_vector)
     left_isometries.append(vector)
     
     omega = matrix_vector_multi_for_two_site(
          vector,
          contract_left,
          contract_center1,
          contract_center2,
          contract_right,
     )
     
     alpha = Contract(
          "ijkl,ijkl->", vector.conj(), omega
     )
     
     # assert np.abs(alpha.imag) > 1e-15 * np.abs(alpha.real), f"Hamiltonian not Hermitian"
     
     alphas.append(alpha)
     vector = orthogonalize(omega, left_isometries, two_site=True)
     
     for iteratioin in range(1, Krylov_bases):
          beta = np.linalg.norm(vector)
          
          if beta < Lanczos_cutoff:
               break
          
          betas.append(beta)
          vector = vector / beta
          left_isometries.append(vector)

          omega = matrix_vector_multi_for_two_site(
               vector,
               contract_left,
               contract_center1,
               contract_center2,
               contract_right,
          )
          
          alpha = Contract("ijkl,ijkl->", vector.conj(), omega)
          alphas.append(alpha)
          
          vector = orthogonalize(omega, left_isometries, two_site=True)
     
     alphas = np.array(alphas)
     betas = np.array(betas)
     
     return alphas, betas, left_isometries


def matrix_vector_multi_for_two_site(
     vector: npt.NDArray,
     contract_left: npt.NDArray,
     contract_center1: npt.NDArray,
     contract_center2: npt.NDArray,
     contract_right: npt.NDArray,
) -> npt.NDArray:
     
     return Contract(
          "abi,aexy,kxbd,lydf,efj->ijkl",
          contract_left, vector, contract_center1, contract_center2, contract_right
     )

