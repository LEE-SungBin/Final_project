import numpy as np
import numpy.typing as npt
# from python.Canonical_Form import move_site_left, move_site_right
from python.Contract import Contract
from python.Decomposition import SVD, EIGH, QR, RQ
# from python.Gauging import *
from python.initialization import random_initialization, Iterative_diagonalization
from python.utils import round_sig
from python.Zippers import MPS_MPO_MPS_overlap, MPS_MPS_overlap
from python.Backend import Backend
import time

from copy import deepcopy


def get_multi_excited_state(
     Hamiltonian: list[npt.NDArray],
     NKeep: int,
     NSweep: int,
     iterative_diag: bool = True,
     two_site: bool = True,
     Krylov_bases: int = 5,
     Lanczos_cutoff: float = 1e-4,
     num_excited_state: int = 0,
     verbose: bool = False,
):
     
     return 0


def DMRG(
     Hamiltonian: list[npt.NDArray],
     NKeep: int,
     NSweep: int,
     iterative_diag: bool = True,
     two_site: bool = True,
     Krylov_bases: int = 5,
     Lanczos_cutoff: float = 1e-4,
     orthogonal_to_list_of_MPS: list[list[npt.NDArray]] | None = None,
     verbose: bool = False,
     tol: float = 1e-6,
     bk: Backend = Backend('auto')
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
               Hamiltonian=Hamiltonian, NKeep=NKeep, bk=bk
          )
          
     else:
          MPS = random_initialization(
               Hamiltonian=Hamiltonian, NKeep=NKeep, bk=bk
          )
     
     # Verify MPS normalization
     norm = MPS_MPS_overlap(MPS, MPS, bk=bk)
     if not bk.all(bk.isfinite(norm)) or abs(norm - 1.0) > 1e-10:
          raise ValueError(f"MPS not properly normalized: {norm}")
     
     """
     Transpose to calculate overlap
     """
     
     MPO = [bk.transpose(tensor, (2, 3, 0, 1)) for tensor in Hamiltonian]
     
     initial_energy = MPS_MPO_MPS_overlap(MPS, MPO, MPS, bk=bk)
     
     """
     Compute contract_list_left and right
     """
     
     contract_list_left: list = [
          bk.array([1.0]).reshape(1, 1, 1) for _ in range(n_sites+1)
     ]
     contract_list_right: list = [
          bk.array([1.0]).reshape(1, 1, 1) for _ in range(n_sites+1)
     ]

     for it in range(1, n_sites+1):
          contract_list_left[it] = Contract(
               "abc,aix,yxbj,cky->ijk",
               contract_list_left[it-1],
               MPS[it-1], Hamiltonian[it-1], bk.conj(MPS[it-1]),
               bk=bk
          )
     
     """
     Iterate to get the ground state and ground state energy
     """
     
     times = [0.0]
     total_energies = [bk.real(initial_energy)]
     
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
                    orthogonal_to_list_of_MPS = orthogonal_to_list_of_MPS,
                    bk = bk
               )
          
          else:
               energies, MPS = sweep_with_single_site_update(
                    MPS = MPS,
                    Hamiltonian = Hamiltonian,
                    contract_list_left = contract_list_left,
                    contract_list_right = contract_list_right,
                    Krylov_bases = Krylov_bases,
                    Lanczos_cutoff = Lanczos_cutoff,
                    orthogonal_to_list_of_MPS = orthogonal_to_list_of_MPS,
                    bk = bk
               )

          last_energy = total_energies[-1]
          
          total_energies.extend(energies)
          if iter > 0:
               times.append(times[-1] + time.perf_counter()-now)
          else:
               times.append(time.perf_counter()-now)

          if bk.xp.abs(last_energy.real - total_energies[-1].real) < tol:
               print(f"iter={iter+1} | energy={energies[-1]} | time={times[-1]}s")
               break

     total_energies = bk.array(total_energies)
     times = bk.array(times)
     
     return bk.to_cpu(total_energies), bk.to_cpu(times), MPS


def sweep_with_single_site_update(
     MPS: list[npt.NDArray],
     Hamiltonian: list[npt.NDArray],
     contract_list_left: list[npt.NDArray],
     contract_list_right: list[npt.NDArray],
     Krylov_bases: int = 5,
     Lanczos_cutoff: float = 1e-4,
     orthogonal_to_list_of_MPS: list[list[npt.NDArray]] | None = None,
     bk: Backend = Backend('auto')
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
          
          constraints = None
          
          if orthogonal_to_list_of_MPS is not None:
               
               constraints = []
               
               for orthogonal_MPS in orthogonal_to_list_of_MPS:
                    MPS_without_orthogonality_center = deepcopy(MPS)
                    MPS_without_orthogonality_center[orthogonality_center] = None
                    
                    constraint_mps = MPS_MPS_overlap(
                         orthogonal_MPS, MPS_without_orthogonality_center, bk=bk
                    )
                    
                    constraints.append(constraint_mps)
          
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
               constraints = constraints,
               bk = bk
          )
          
          energy, vector = get_eigen_vector_from_lanczos(
               alphas = alphas, betas = betas,
               left_isometries = left_isometries,
               bk = bk
          )
          
          energies.append(energy)
                    
          matrix = vector.reshape(vector.shape[0], -1)
          U, S, Vh = SVD(matrix, full_SVD=True, bk=bk)
          Vh = Vh.reshape(-1, vector.shape[1], vector.shape[2])
          
          """
          renormalize
          """
          S = S / bk.norm(S)
          
          """
          Update MPS
          """
          
          MPS[orthogonality_center] = Vh
          MPS[orthogonality_center-1] = Contract(
               "iak,ab,bj->ijk", MPS[orthogonality_center-1],
               U, bk.diag(S), bk=bk
          )
          
          """
          Update contract_list_right
          """
          
          contract_list_right[right_loc + 1] = Contract(
               "iax,yxjb,kcy,abc->ijk",
               MPS[orthogonality_center], Hamiltonian[orthogonality_center],
               bk.conj(MPS[orthogonality_center]), contract_list_right[right_loc], bk=bk
          )
     
     """
     Left to right
     """
     
     for it in range(n_sites-1):
          
          orthogonality_center = it
          left_loc = orthogonality_center
          right_loc = n_sites - orthogonality_center - 1
          
          initial_vector = deepcopy(MPS[orthogonality_center])
          
          constraints = None
          
          if orthogonal_to_list_of_MPS is not None:
               
               constraints = []
               
               for orthogonal_MPS in orthogonal_to_list_of_MPS:
                    MPS_without_orthogonality_center = deepcopy(MPS)
                    MPS_without_orthogonality_center[orthogonality_center] = None
                    
                    constraint_mps = MPS_MPS_overlap(
                         orthogonal_MPS, MPS_without_orthogonality_center, bk=bk
                    )
                    
                    constraints.append(constraint_mps)
          
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
               constraints = constraints,
               bk = bk
          )
          
          energy, vector = get_eigen_vector_from_lanczos(
               alphas = alphas, betas = betas,
               left_isometries = left_isometries,
               bk = bk
          )
          
          energies.append(energy)
                    
          matrix = bk.transpose(vector, (0, 2, 1)).reshape(-1, vector.shape[1])
          U, S, Vh = SVD(matrix, full_SVD=True, bk=bk)
          U = bk.transpose(U.reshape(vector.shape[0], vector.shape[2], -1), (0, 2, 1))
          
          """
          renormalize
          """
          S = S / bk.norm(S)
          
          """
          Update MPS
          """
          
          MPS[orthogonality_center] = U
          MPS[orthogonality_center + 1] = Contract(
               "ia,ab,bjk->ijk", bk.diag(S), Vh, bk=bk
          )
          
          """
          Update contract_list_left
          """
          
          contract_list_left[left_loc + 1] = Contract(
               "abc,aix,yxbj,cky->ijk",
               contract_list_left[left_loc], MPS[orthogonality_center],
               Hamiltonian[orthogonality_center], bk.conj(MPS[orthogonality_center]), bk=bk
          )
     
     energies = bk.array(energies)
     
     return energies, MPS


def get_eigen_vector_from_lanczos(
     alphas: npt.NDArray,
     betas: npt.NDArray,
     left_isometries: list[npt.NDArray],
     bk: Backend = Backend('auto')
) -> tuple[float, npt.NDArray]:
     
     Tridiagonal = get_tridiagonal_matrix(alphas, betas, bk=bk)
     
     eigvals, eigvecs = EIGH(Tridiagonal, bk=bk)
     
     # eigval = eigvals[-1]
     # eigvec = eigvecs[:,-1]
     
     eigval = eigvals[0]
     eigvec = eigvecs[:,0]
     
     # print(f"{eigval=}")
     
     eigenvector = bk.zeros(shape=left_isometries[0].shape)
     
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
     constraints: list[npt.NDArray] | None = None,
     bk: Backend = Backend('auto')
) -> tuple[npt.NDArray, npt.NDArray, list[npt.NDArray]]:
     
     left_isometries = []
     alphas = []
     betas = []
     
     """
     Iteration 1
     """
     
     if constraints is not None:
          for constraint in constraints:
               left_isometries.append(constraint / bk.norm(constraint))
     
     vector = deepcopy(initial_vector)
     norm = bk.norm(vector)
     vector = vector / norm
     
     vector = orthogonalize(vector, left_isometries, two_site=True, bk=bk)     
     left_isometries.append(vector)
     
     omega = matrix_vector_multi_for_single_site(
          vector,
          contract_left,
          contract_center,
          contract_right,
          bk = bk
     )
     
     alpha = Contract(
          "ijk,ijk->", vector.conj(), omega, bk=bk
     )
     
     # assert np.abs(alpha.imag) > 1e-15 * np.abs(alpha.real), f"Hamiltonian not Hermitian"
     
     alphas.append(alpha)
     vector = orthogonalize(omega, left_isometries, two_site=False, bk=bk)
     
     for iter in range(1, Krylov_bases):
          beta = bk.norm(vector)
          
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
               bk = bk
          )
          
          alpha = Contract("ijk,ijk->", vector.conj(), omega, bk=bk)
          alphas.append(alpha)
          
          vector = orthogonalize(omega, left_isometries, two_site=False, bk=bk)
     
     alphas = bk.array(alphas)
     betas = bk.array(betas)
     
     if constraints is not None:
          for _ in range(len(constraints)):
               left_isometries.pop(0)
     
     return alphas, betas, left_isometries


def matrix_vector_multi_for_single_site(
     vector: npt.NDArray,
     contract_left: npt.NDArray,
     contract_center: npt.NDArray,
     contract_right: npt.NDArray,
     bk: Backend = Backend('auto')
) -> npt.NDArray:
     
     return Contract(
          "abi,acx,kxbd,cdj->ijk",
          contract_left, vector, contract_center, contract_right, bk=bk
     )


def orthogonalize(
     omega: npt.NDArray,
     left_isometries: list[npt.NDArray],
     two_site: bool = True,
     bk: Backend = Backend('auto')
) -> npt.NDArray:
     
     coefficients = []
     
     for left_isometry in left_isometries:
          if two_site:
               coefficient = Contract(
                    "ijkl,ijkl->", omega, bk.conj(left_isometry), bk=bk
               )
          else:
               coefficient = Contract(
                    "ijk,ijk->", omega, bk.conj(left_isometry), bk=bk
               )
          
          coefficients.append(coefficient)
     
     coefficients = bk.array(coefficients)
     
     new_vector = omega
     
     for coefficient, left_isometry in zip(coefficients, left_isometries):
          new_vector = new_vector - coefficient * left_isometry
     
     return new_vector


def get_tridiagonal_matrix(
     alphas: npt.NDArray,
     betas: npt.NDArray,
     bk: Backend = Backend('auto')
) -> npt.NDArray:
     
     size = len(alphas)
     assert len(betas) == size - 1, f"{len(alphas)=} != {len(betas)+1=}"
     
     Tridiagonal = bk.diag(alphas)
     
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
     orthogonal_to_list_of_MPS: list[list[npt.NDArray]] | None = None,
     bk: Backend = Backend('auto')
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
          
          constraints = None
          
          if orthogonal_to_list_of_MPS is not None:
               
               constraints = []
               
               for orthogonal_MPS in orthogonal_to_list_of_MPS:
                    MPS_without_orthogonality_center = deepcopy(MPS)
                    
                    MPS_without_orthogonality_center[orthogonality_center-1] = None
                    MPS_without_orthogonality_center[orthogonality_center] = None
                    
                    constraint_mps = MPS_MPS_overlap(
                         orthogonal_MPS, MPS_without_orthogonality_center, bk=bk
                    )
                    
                    constraints.append(constraint_mps)
          
          initial_vector = Contract("iak,ajl->ijkl", initial_vector1, initial_vector2, bk=bk)
          
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
               constraints = constraints,
               bk = bk
          )
          
          energy, vector = get_eigen_vector_from_lanczos(
               alphas = alphas, betas = betas,
               left_isometries = left_isometries,
               bk = bk
          )
          
          energies.append(energy)
                    
          matrix = bk.transpose(vector, (0, 2, 1, 3)).reshape(vector.shape[0] * vector.shape[2], -1)
          U, S, Vh = SVD(matrix, Nkeep=NKeep, Skeep=1.e-8, bk=bk)
          Vh = Vh.reshape(-1, vector.shape[1], vector.shape[3])
          U = bk.transpose(U.reshape(vector.shape[0], vector.shape[2], -1), (0, 2, 1))
          
          """
          renormalize
          """
          S = S / bk.norm(S)
          
          """
          Update MPS
          """
          
          MPS[orthogonality_center] = Vh
          MPS[orthogonality_center-1] = Contract(
               "iak,aj->ijk", U, bk.diag(S), bk=bk
          )
          
          """
          Update contract_list_right
          """
          
          contract_list_right[right_loc + 1] = Contract(
               "iax,yxjb,kcy,abc->ijk",
               MPS[orthogonality_center], Hamiltonian[orthogonality_center],
               bk.conj(MPS[orthogonality_center]), contract_list_right[right_loc], bk=bk
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
          
          initial_vector = Contract("iak,ajl->ijkl", initial_vector1, initial_vector2, bk=bk)
          
          contract_left = deepcopy(contract_list_left[left_loc])
          contract_center1 = deepcopy(Hamiltonian[orthogonality_center])
          contract_center2 = deepcopy(Hamiltonian[orthogonality_center+1])
          contract_right = deepcopy(contract_list_right[right_loc])
          
          constraints = None
          
          if orthogonal_to_list_of_MPS is not None:
               
               constraints = []
               
               for orthogonal_MPS in orthogonal_to_list_of_MPS:
                    MPS_without_orthogonality_center = deepcopy(MPS)
                    
                    MPS_without_orthogonality_center[orthogonality_center] = None
                    MPS_without_orthogonality_center[orthogonality_center + 1] = None
                    
                    constraint_mps = MPS_MPS_overlap(
                         orthogonal_MPS, MPS_without_orthogonality_center, bk=bk
                    )
                    
                    constraints.append(constraint_mps)
          
          alphas, betas, left_isometries = lanczos_for_two_site(
               initial_vector = initial_vector,
               contract_left = contract_left,
               contract_center1 = contract_center1,
               contract_center2 = contract_center2,
               contract_right = contract_right,
               Krylov_bases = Krylov_bases,
               Lanczos_cutoff = Lanczos_cutoff,
               constraints = constraints,
               bk = bk
          )
          
          energy, vector = get_eigen_vector_from_lanczos(
               alphas = alphas, betas = betas,
               left_isometries = left_isometries,
               bk = bk
          )
          
          energies.append(energy)
                    
          matrix = bk.transpose(vector, (0, 2, 1, 3)).reshape(vector.shape[0] * vector.shape[2], -1)
          U, S, Vh = SVD(matrix, Nkeep = NKeep, Skeep = 1e-8, bk=bk)
          U = bk.transpose(U.reshape(vector.shape[0], vector.shape[2], -1), (0, 2, 1))
          Vh = Vh.reshape(-1, vector.shape[1], vector.shape[3])
          
          """
          renormalize
          """
          S = S / bk.norm(S)
          
          """
          Update MPS
          """
          
          MPS[orthogonality_center] = U
          MPS[orthogonality_center + 1] = Contract(
               "ia,ajk->ijk", bk.diag(S), Vh, bk=bk
          )
          
          """
          Update contract_list_left
          """
          
          contract_list_left[left_loc + 1] = Contract(
               "abc,aix,yxbj,cky->ijk",
               contract_list_left[left_loc], MPS[orthogonality_center],
               Hamiltonian[orthogonality_center], bk.conj(MPS[orthogonality_center]), bk=bk
          )
     
     energies = bk.array(energies)
     
     return energies, MPS


def lanczos_for_two_site(
     initial_vector: npt.NDArray,
     contract_left: npt.NDArray,
     contract_center1: npt.NDArray,
     contract_center2: npt.NDArray,
     contract_right: npt.NDArray,
     Krylov_bases: int = 5,
     Lanczos_cutoff: float = 1e-8,
     constraints: list[npt.NDArray] | None = None,
     bk: Backend = Backend('auto')
) -> tuple[npt.NDArray, npt.NDArray, list[npt.NDArray]]:
     
     left_isometries = []
     alphas = []
     betas = []
     
     if constraints is not None:
          for constraint in constraints:
               left_isometries.append(constraint / bk.norm(constraint))

     """
     Iteration 1
     """     
     
     vector = deepcopy(initial_vector)
     norm = bk.norm(vector)
     vector = vector / norm
     
     vector = orthogonalize(vector, left_isometries, two_site=True, bk=bk)
     left_isometries.append(vector)
     
     omega = matrix_vector_multi_for_two_site(
          vector,
          contract_left,
          contract_center1,
          contract_center2,
          contract_right,
          bk = bk
     )
     
     alpha = Contract(
          "ijkl,ijkl->", vector.conj(), omega, bk=bk
     )
     
     # assert np.abs(alpha.imag) > 1e-15 * np.abs(alpha.real), f"Hamiltonian not Hermitian"
     
     alphas.append(alpha)
     vector = orthogonalize(omega, left_isometries, two_site=True, bk=bk)
     
     for iteratioin in range(1, Krylov_bases):
          beta = bk.norm(vector)
          
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
               bk = bk
          )
          
          alpha = Contract("ijkl,ijkl->", vector.conj(), omega, bk=bk)
          alphas.append(alpha)
          
          vector = orthogonalize(omega, left_isometries, two_site=True, bk=bk)
     
     alphas = bk.array(alphas)
     betas = bk.array(betas)
     
     if constraints is not None:
          for _ in range(len(constraints)):
               left_isometries.pop(0)
     
     return alphas, betas, left_isometries


def matrix_vector_multi_for_two_site(
     vector: npt.NDArray,
     contract_left: npt.NDArray,
     contract_center1: npt.NDArray,
     contract_center2: npt.NDArray,
     contract_right: npt.NDArray,
     bk: Backend = Backend('auto')
) -> npt.NDArray:
     
     return Contract(
          "abi,aexy,kxbd,lydf,efj->ijkl",
          contract_left, vector, contract_center1, contract_center2, contract_right, bk=bk
     )

