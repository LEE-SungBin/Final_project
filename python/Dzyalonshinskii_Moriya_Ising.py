import numpy as np
import numpy.typing as npt
from copy import deepcopy
from python.Zippers import MPS_MPS_overlap, MPS_MPO_MPS_overlap


def DM_Ising(
    n_sites: int,
    J: float = 1.0,
    D: float = 1.0,
    magnetic_field: float = 0.0,
) -> list[npt.NDArray]:
    
    
    """

    Hamiltonian.shape = [2, 2, 5, 5]

         1
         |
    2 --- --- 3
         |
         0

    """
    
    Hamiltonian = []
    Hamiltonian_shape = [2, 2, 5, 5]
    
    identity = np.identity(2)
    S_x = 1 / 2 * np.array([
        [0, 1],
        [1, 0]
    ]
    )
    S_y = 1 / 2 * np.array([
        [0, -1j],
        [1j, 0]
    ])
    S_z = 1 / 2 * np.array([
        [1, 0],
        [0, -1]
    ])
    S_plus = S_x + 1j*S_y
    S_minus = S_x - 1j*S_y
    
    for it in range(n_sites):
        
        MPO = np.zeros(Hamiltonian_shape, dtype=complex)
        
        MPO[:,:,0,0] = identity
        MPO[:,:,1,0] = S_z
        MPO[:,:,2,0] = S_x
        MPO[:,:,3,0] = S_y
        MPO[:,:,4,0] = -1 * magnetic_field * S_z
        MPO[:,:,4,1] = J * S_z
        MPO[:,:,4,2] = -D * S_y
        MPO[:,:,4,3] = D * S_x
        MPO[:,:,4,4] = identity
        
        Hamiltonian.append(MPO)
    
    Hamiltonian[0] = Hamiltonian[0][:,:,-1,:].reshape(
        Hamiltonian_shape[0], Hamiltonian_shape[1], 1, Hamiltonian_shape[3]
    )
    Hamiltonian[-1] = Hamiltonian[-1][:,:,:,0].reshape(
        Hamiltonian_shape[0], Hamiltonian_shape[1], Hamiltonian_shape[2], 1
    )
    
    return Hamiltonian

