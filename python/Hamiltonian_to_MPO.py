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


def Hubbard_model(
    n_sites: int,
    hopping_t: float,
    interaction_U: float
):

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
    
    identity = np.identity(4)
    Jordan_Wigner_string = np.diag([1, -1, -1, 1])
    
    annihilation_up = np.zeros([4, 4])
    annihilation_up[0,1] = 1.0
    annihilation_up[2,3] = -1.0
    
    annihilation_down = np.zeros([4, 4])
    annihilation_down[0,2] = 1.0
    annihilation_down[1,3] = 1.0
    
    creation_up = annihilation_up.conj().T
    
    creation_down = annihilation_down.conj().T
    
    density_up = creation_up @ annihilation_up
    
    density_down = creation_down @ annihilation_down
    
    for it in range(n_sites):
        
        MPO = np.zeros(Hamiltonian_shape)
        
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


def Double_occupancy(
    MPS: list[npt.NDArray]
):
    
    double_occupancy = []
    
    for it, mps in enumerate(MPS):
        
        double_occupied_MPS = deepcopy(MPS)
        double_occupied_MPS[it] = mps[:,:,3].reshape(mps.shape[0], mps.shape[1], 1)
        
        occupancy = MPS_MPS_overlap(double_occupied_MPS, double_occupied_MPS)
        
        double_occupancy.append(occupancy)
    
    double_occupancy = np.array(double_occupancy)
    
    return double_occupancy

