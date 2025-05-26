import numpy as np
import numpy.typing as npt
from copy import deepcopy
from python.Zippers import MPS_MPS_overlap


def Spinless_fermions(
    hopping_amps: npt.NDArray,
):
    
    """

     Hamiltonian.shape = [2, 2, 4, 4]

          1
          |
     2 --- --- 3
          |
          0

     """

    Hamiltonian = []
    Hamiltonian_shape = [2, 2, 4, 4]
    
    identity = np.identity(2)
    Jordan_Wigner_string = np.diag([1.0, -1.0])
    
    annihilation = np.array([
        [0.0, 1.0],
        [0.0, 0.0]
    ])
    
    creation = np.array([
        [0.0, 0.0],
        [1.0, 0.0]
    ])
    
    for it, hopping_amp in enumerate(hopping_amps):
        
        MPO = np.zeros(shape = Hamiltonian_shape)
        
        MPO[:,:,0,0] = identity
        MPO[:,:,1,0] = Jordan_Wigner_string @ annihilation
        MPO[:,:,2,0] = creation @ Jordan_Wigner_string
        MPO[:,:,3,1] = hopping_amp * creation
        MPO[:,:,3,2] = hopping_amp * annihilation
        MPO[:,:,3,3] = identity
    
        Hamiltonian.append(MPO)
    
    Hamiltonian[0] = Hamiltonian[0][:,:,-1,:].reshape(2, 2, 1, 4)
    Hamiltonian[-1] = Hamiltonian[-1][:,:,:,0].reshape(2, 2, 4, 1)
    
    return Hamiltonian


def Spinless_fermion_single_ptcl_hamiltonian(
    hopping_amps: npt.NDArray,
):
    
    n_sites = len(hopping_amps)
    
    single_ptcl_hamiltonian = np.zeros(shape=(n_sites, n_sites))
    
    for it, hopping_amp in enumerate(hopping_amps[:-1]):
        single_ptcl_hamiltonian[it, it+1] = hopping_amp
        single_ptcl_hamiltonian[it+1, it] = hopping_amp
    
    return single_ptcl_hamiltonian


def get_exact_gs_energy_spinless_fermions(
    hopping_amps: npt.NDArray
):
    
    exact_hamiltonian = Spinless_fermion_single_ptcl_hamiltonian(hopping_amps)
    
    eigvals, eigvecs = np.linalg.eigh(exact_hamiltonian)
    
    return eigvals[:int(len(hopping_amps)/2)].sum()


def get_eigvals_single_ptcl(
    hopping_amps: npt.NDArray
):
    
    exact_hamiltonian = Spinless_fermion_single_ptcl_hamiltonian(hopping_amps)
    
    eigvals, eigvecs = np.linalg.eigh(exact_hamiltonian)
    
    return eigvals

