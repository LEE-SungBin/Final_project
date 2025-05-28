import numpy as np
import numpy.typing as npt
from copy import deepcopy
from python.Zippers import MPS_MPS_overlap, MPS_MPO_MPS_overlap
from python.Canonical_Form import site_to_bond_canonical_MPS, site_canonical_MPS
from python.Backend import Backend
from typing import List, Optional, Tuple, Union


def Hubbard_model(
    n_sites: int,
    hopping_t: float,
    interaction_U: float,
    bk: Backend = Backend('auto'),
) -> list[npt.NDArray]:

    """

    Hamiltonian.shape = [4, 4, 6, 6]

         1
         |
    2 --- --- 3
         |
         0

    """
    
    Hamiltonian = []
    Hamiltonian_shape = [4, 4, 6, 6]
    
    identity = bk.identity(4)
    Jordan_Wigner_string = bk.diag([1, -1, -1, 1])
    
    annihilation_up = bk.zeros([4, 4])
    annihilation_up[0,1] = 1.0
    annihilation_up[2,3] = -1.0
    
    annihilation_down = bk.zeros([4, 4])
    annihilation_down[0,2] = 1.0
    annihilation_down[1,3] = 1.0
    
    creation_up = annihilation_up.conj().T
    
    creation_down = annihilation_down.conj().T
    
    density_up = creation_up @ annihilation_up
    
    density_down = creation_down @ annihilation_down
    
    for it in range(n_sites):
        
        MPO = bk.zeros(Hamiltonian_shape)
        
        MPO[:,:,0,0] = identity
        MPO[:,:,1,0] = Jordan_Wigner_string @ annihilation_up
        MPO[:,:,2,0] = creation_up @ Jordan_Wigner_string
        MPO[:,:,3,0] = Jordan_Wigner_string @ annihilation_down
        MPO[:,:,4,0] = creation_down @ Jordan_Wigner_string
        MPO[:,:,5,0] = interaction_U / 2.0 * (density_up + density_down - identity) @ (density_up + density_down - identity)
        
        MPO[:,:,5,1] = hopping_t * creation_up
        MPO[:,:,5,2] = hopping_t * annihilation_up
        MPO[:,:,5,3] = hopping_t * creation_down
        MPO[:,:,5,4] = hopping_t * annihilation_down
        MPO[:,:,5,5] = identity

        Hamiltonian.append(MPO)
    
    Hamiltonian[0] = Hamiltonian[0][:,:,-1,:].reshape(4, 4, 1, 6)
    Hamiltonian[-1] = Hamiltonian[-1][:,:,:,0].reshape(4, 4, 6, 1)
    
    return Hamiltonian


def Hubbard_model_with_filling(
    n_sites: int,
    hopping_t: float,
    interaction_U: float,
    chemical_potential: float,
    bk: Backend = Backend('auto'),
) -> list[npt.NDArray]:

    """

    Hamiltonian.shape = [4, 4, 6, 6]

         1
         |
    2 --- --- 3
         |
         0

    """
    
    Hamiltonian = []
    Hamiltonian_shape = [4, 4, 6, 6]
    
    identity = bk.identity(4)
    Jordan_Wigner_string = bk.diag([1, -1, -1, 1])
    
    annihilation_up = bk.zeros([4, 4])
    annihilation_up[0,1] = 1.0
    annihilation_up[2,3] = -1.0
    
    annihilation_down = bk.zeros([4, 4])
    annihilation_down[0,2] = 1.0
    annihilation_down[1,3] = 1.0
    
    creation_up = annihilation_up.conj().T
    
    creation_down = annihilation_down.conj().T
    
    density_up = creation_up @ annihilation_up
    
    density_down = creation_down @ annihilation_down
    
    for it in range(n_sites):
        
        MPO = bk.zeros(Hamiltonian_shape)
        
        MPO[:,:,0,0] = identity
        MPO[:,:,1,0] = Jordan_Wigner_string @ annihilation_up
        MPO[:,:,2,0] = creation_up @ Jordan_Wigner_string
        MPO[:,:,3,0] = Jordan_Wigner_string @ annihilation_down
        MPO[:,:,4,0] = creation_down @ Jordan_Wigner_string
        MPO[:,:,5,0] = interaction_U * density_up @ density_down - chemical_potential * (density_up + density_down)
        
        MPO[:,:,5,1] = -1 * hopping_t * creation_up
        MPO[:,:,5,2] = -1 * hopping_t * annihilation_up
        MPO[:,:,5,3] = -1 * hopping_t * creation_down
        MPO[:,:,5,4] = -1 * hopping_t * annihilation_down
        MPO[:,:,5,5] = identity

        Hamiltonian.append(MPO)
    
    Hamiltonian[0] = Hamiltonian[0][:,:,-1,:].reshape(4, 4, 1, 6)
    Hamiltonian[-1] = Hamiltonian[-1][:,:,:,0].reshape(4, 4, 6, 1)
    
    return Hamiltonian


def Double_occupancy(
    MPS: list[npt.NDArray],
    bk: Backend = Backend('auto'),
):
    
    double_occupancy = []
    
    for it, mps in enumerate(MPS):
        
        double_occupied_MPS = deepcopy(MPS)
        double_occupied_MPS[it] = mps[:,:,3].reshape(mps.shape[0], mps.shape[1], 1)
        
        occupancy = MPS_MPS_overlap(double_occupied_MPS, double_occupied_MPS, bk=bk)
        
        double_occupancy.append(occupancy)
    
    double_occupancy = bk.array(double_occupancy)
    
    return double_occupancy


def get_filling(
    MPS: list[npt.NDArray],
    bk: Backend = Backend('auto'),
):
    
    filling = []
    
    for it, mps in enumerate(MPS):
        
        MPO = [bk.identity(4).reshape(1, 1, 4, 4) for _ in MPS]
        MPO[it][0,0] = bk.diag([0, 1, 1, 2])
        
        filled = MPS_MPO_MPS_overlap(MPS, MPO, MPS, bk=bk)
        
        filling.append(filled)
    
    filling = bk.array(filling)
    
    return filling


