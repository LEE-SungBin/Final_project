import numpy as np
import numpy.typing as npt
from python.Backend import Backend
from python.Decomposition import QR, SVD, EIGH
from python.Contract import Contract
from copy import deepcopy


def random_initialization(
    Hamiltonian: list[npt.NDArray],
    NKeep: int,
    bk: Backend = Backend('auto')
) -> list[npt.NDArray]:
    
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
    physical_bond = Hamiltonian[0].shape[0]

    init_MPS: list[npt.NDArray] = []

    """
    Initialization
    """

    for it in range(n_sites):
        
        if it == n_sites - 1:        
            bond_shape = [
                min(physical_bond**it, NKeep),
                1,
                physical_bond
            ]
        else:
            bond_shape = [
                min(physical_bond**it, NKeep),
                min(physical_bond**(it+1), NKeep),
                physical_bond
            ]
        
        tensor = bk.randn(*bond_shape)
        mps_at_site_it = bk.to_device(tensor)
        temp_matrix = bk.transpose(mps_at_site_it, (0, 2, 1)).reshape(-1, mps_at_site_it.shape[1])
        
        left_iso, _ = QR(temp_matrix, bk=bk)
        
        mps = bk.transpose(
            left_iso.reshape(
                mps_at_site_it.shape[0], mps_at_site_it.shape[2], mps_at_site_it.shape[1]
            ), (0, 2, 1)
        )

        init_MPS.append(mps)
    
    return init_MPS


def Iterative_diagonalization(
    Hamiltonian: list[npt.NDArray],
    NKeep: int,
    bk: Backend = Backend('auto')
) -> list[npt.NDArray]:
    
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
    
    MPS = []
    before_Hamiltonian = bk.array([[1.0]])
    before_tensor = bk.array([[[1.0]]])
    
    for it, local_Hamiltonian in enumerate(Hamiltonian):
        
        incoming_bond_dim = before_Hamiltonian.shape[0]
        physical_bond_dim = local_Hamiltonian.shape[0]
        
        full_dim = incoming_bond_dim * physical_bond_dim
        
        truncated_Hamiltonian = local_Hamiltonian[:,:,:,0]
        
        identity = bk.identity(full_dim).reshape(
            incoming_bond_dim, full_dim, physical_bond_dim
        )
        
        new_Hamiltonian = Contract(
            "ab,aix,bjx->ij", before_Hamiltonian, identity, identity, bk=bk
        )
        
        new_Hamiltonian = new_Hamiltonian + Contract(
            "abc,aix,yxb,cjy->ij", before_tensor,
            identity, truncated_Hamiltonian, identity, bk=bk
        )
        
        if it == len(Hamiltonian) - 1:
            Keep = 1
        else:
            Keep = NKeep
        
        eigvals, eigvecs = EIGH(new_Hamiltonian, bk=bk)
        
        """
        Get the lowest Keep eigvals and its corresponding eigvecs
        """
        # eigvals = eigvals[-Keep:]
        # eigvecs = eigvecs[:, -Keep:]
        
        eigvals = eigvals[:Keep]
        eigvecs = eigvecs[:, :Keep]
        
        # Ensure all tensors are in the correct backend type
        eigvecs = bk.to_device(eigvecs)
        new_Hamiltonian = bk.to_device(new_Hamiltonian)
        
        # Perform matrix multiplication using backend functions
        before_Hamiltonian = bk.matmul(bk.matmul(bk.conj(eigvecs).T, new_Hamiltonian), eigvecs)
        
        update_isometry = Contract(
            "iak,aj->ijk", identity, eigvecs.conj(), bk=bk
        )
        MPS.append(update_isometry)
        
        before_tensor = Contract(
            "abc,aix,yxbj,cky->ijk",
            before_tensor, update_isometry,
            local_Hamiltonian, update_isometry.conj(), bk=bk
        )
    
    return MPS

