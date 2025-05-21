import numpy as numpy
import numpy.typing as npt
from copy import deepcopy

from python.utils import rearrange_list_by_values
from python.Decomposition import *
from python.Contract import *


def MPS_MPS_overlap(
    MPS1: list[npt.NDArray],
    MPS2: list[npt.NDArray],
    conj: bool = True,
) -> npt.NDArray:
    
    assert len(MPS1) == len(MPS2), f"{len(MPS1)=} != {len(MPS2)=}"
    ord_MPS = len(MPS1)
    
    # for it, _ in enumerate(MPS1):
    #     print(f"MPS1[{it}].shape={MPS1[it].shape} MPS2[{it}].shape={MPS2[it].shape}")
    
    overlap = np.array([1]).reshape(1, 1)
    backward = True
    
    for i in range(ord_MPS):

        mps1 = MPS1[i]
        mps2 = MPS2[i]
    
        # print(f"{i}")
        if mps2 is None:
            # print(f"{mps1.shape=}, {mps2=}")
            overlap = mps_none(overlap, mps1)
            # print(f"{overlap.shape=}")
            backward = False
        else:
            # print(f"{mps1.shape=}, {mps2.shape=}")
            if conj:
                overlap = mps_mps(overlap, mps1, mps2.conj(), backward)
            else:
                overlap = mps_mps(overlap, mps1, mps2, backward)
            # # print(f"{overlap.shape=}")
            backward = True
        i += 1
    
    if backward:
        overlap = Tensordot(overlap, np.array([1]).reshape(1, 1), axes=[(0, 1), (0, 1)])
    
    else:
        overlap = Tensordot(overlap, np.array([1]).reshape(1, 1), axes=[(0), (0)])
        
        ord_overlap = len(overlap.shape)
        lst = [i for i in range(ord_overlap)]
        
        overlap = overlap.transpose(
            rearrange_list_by_values(lst, [ord_overlap-1], [1])
        )
    
    # print(f"{overlap.shape=}")
    
    return overlap


def mps_mps(
    tensor: npt.NDArray,
    mps1: npt.NDArray,
    mps2: npt.NDArray,
    backward: bool = True,
) -> npt.NDArray:
    
    new = mps_none(tensor, mps1)
    
    if backward:
        new = Tensordot(
            new, mps2, axes=[(1, len(new.shape)-1), (0, 2)]
        )
        
        ord_new = len(new.shape)
        lst = [i for i in range(ord_new)]
        
        new = new.transpose(
            rearrange_list_by_values(lst, [ord_new-1], [1])
        )
    
    else:
        new = Tensordot(
            new, mps2, axes=[(len(new.shape)-1), (2)]
        )
        
        ord_new = len(new.shape)
        lst = [i for i in range(ord_new)]
        
        new = new.transpose(
            rearrange_list_by_values(lst, [ord_new-1, ord_new-2], [1, 3])
        )
    
    return new


def mps_none(
    tensor: npt.NDArray,
    mps1: npt.NDArray
) -> npt.NDArray:
    
    new = Tensordot(tensor, mps1, axes=[(0), (0)])
    
    ord_new = len(new.shape)
    lst = [i for i in range(ord_new)]
    
    new = new.transpose(
        rearrange_list_by_values(lst, [ord_new-2], [0])
    )
    
    return new


def MPS_MPO_MPS_overlap(
    MPS1: list[npt.NDArray],
    MPO: list[npt.NDArray],
    MPS2: list[npt.NDArray],
    conj: bool = True
) -> npt.NDArray:
    
    """
    Overlap between MPS, MPO, and MPS
    
       2|         2|  3|
    0 -- -- 1, 0 -- --- -- 1
    
       3|
    0 -- -- 1
       2|
    """
    
    assert len(MPS1) == len(MPS2), f"{len(MPS1)=} != {len(MPS2)=}"
    assert len(MPS1) == len(MPO), f"{len(MPS1)=} != {len(MPO)=}"
    
    for i in range(len(MPS1)):
        
        assert MPS1[i].shape[2] == MPO[i].shape[3], f"MPS1[{i}].shape[2]={MPS1[i].shape[2]} != MPO[{i}].shape[3]={MPO[i].shape[3]}"
        
        if MPS2[i] is not None:
            assert MPS2[i].shape[2] == MPO[i].shape[2], f"MPS2[{i}].shape[2]={MPS2[i].shape[2]} != MPO[{i}].shape[2]={MPO[i].shape[2]}"
        
        # print(f"MPS1[{i}].shape={MPS1[i].shape} MPO[{i}].shape={MPO[i].shape} MPS2[{i}].shape={MPS2[i].shape}")
    
    overlap = np.array([1]).reshape(1, 1, 1)
    backward = True    
    
    i = 0
    for mps1, mpo, mps2 in zip(MPS1, MPO, MPS2):
        # print(f"Zippers {i}")
        if mps2 is None:
            # print(f"{mps1.shape=} {mpo.shape=} {mps2=}")
            overlap = mps_mpo_none(overlap, mps1, mpo)
            # print(f"{overlap.shape=}")
            backward = False
        else:
            # print(f"{mps1.shape=} {mpo.shape=} {mps2.shape=}")
            if conj:
                overlap = mps_mpo_mps(
                    overlap, mps1, mpo, mps2.conj(), backward)
            else:
                overlap = mps_mpo_mps(
                    overlap, mps1, mpo, mps2, backward)
            # print(f"{overlap.shape=}")
            backward = True
        i += 1
    
    if backward:
        overlap = Tensordot(overlap, np.array([1]).reshape(1, 1, 1), axes=[(0, 1, 2), (0, 1, 2)])
    
    else:
        overlap = Tensordot(overlap, np.array([1]).reshape(1, 1, 1), axes=[(0, 1), (0, 1)])
        
        ord_overlap = len(overlap.shape)
        lst = [i for i in range(ord_overlap)]
        
        overlap = overlap.transpose(
            rearrange_list_by_values(lst, [ord_overlap-1], [1])
        )
    
    # print(f"{overlap.shape=}")

    return overlap


def MPS_MPO_multiplication(
    MPS: list[npt.NDArray],
    MPO: list[npt.NDArray],
):
    
    """
    Overlap between MPS, MPO, and MPS
    
       2|         2|  3|
    0 -- -- 1, 0 -- --- -- 1
    
       3|
    0 -- -- 1
       2|
    """
    
    assert len(MPS) == len(MPO), f"{len(MPS)=} != {len(MPO)=}"
    
    for i in range(len(MPS)):
        
        assert MPS[i].shape[2] == MPO[i].shape[3], f"MPS1[{i}].shape[2]={MPS[i].shape[2]} != MPO[{i}].shape[3]={MPO[i].shape[3]}"
        
        # print(f"MPS1[{i}].shape={MPS1[i].shape} MPO[{i}].shape={MPO[i].shape} MPS2[{i}].shape={MPS2[i].shape}")
    
    overlap = np.array([1]).reshape(1, 1, 1)
    backward = True    
    
    i = 0
    for mps, mpo in zip(MPS, MPO):
        # print(f"Zippers {i}")
        # print(f"{mps1.shape=} {mpo.shape=} {mps2=}")
        overlap = mps_mpo_none(overlap, mps, mpo)
        i += 1
    
    overlap = Tensordot(overlap, np.array([1]).reshape(1, 1, 1), axes=[(0, 1), (0, 1)])   
    ord_overlap = len(overlap.shape)
    lst = [i for i in range(ord_overlap)]
    
    overlap = overlap.transpose(
        rearrange_list_by_values(lst, [ord_overlap-1], [1])
    )
    overlap = overlap.reshape(overlap.shape[2:])
    
    return overlap


def contract_MPS(MPS: list[npt.NDArray]) -> npt.NDArray:
    
    absolute = MPS[0]
    
    for it in range(1, len(MPS)):
        absolute = Tensordot(absolute, MPS[it], axes=[(1), (0)])
        ord_absolute = len(absolute.shape)
        lst = [i for i in range(ord_absolute)]
        absolute = absolute.transpose(
            rearrange_list_by_values(lst, [ord_absolute-2], [1])
        )
    
    absolute = Tensordot(absolute, np.identity(absolute.shape[0]), axes=[(0,1), (0,1)])

    assert len(absolute.shape) == len(MPS)

    return absolute


def Hamiltonian_to_MPO(
    Hamiltonian: npt.NDArray,
):
    
    tensor = deepcopy(Hamiltonian)

    total = len(tensor.shape)
    
    if total % 2 == 1:
        return None
    
    n_sites = int(total/2)
    
    transpose = []
    
    for it in range(n_sites):
        transpose.append(it)
        transpose.append(n_sites+it)
    
    tensor = tensor.transpose(tuple(transpose))
    tensor = tensor.reshape((1,)+tensor.shape+(1,))
    
    MPO = []
    
    for it in range(n_sites - 1):
        
        matrix = tensor.reshape(
            np.prod(tensor.shape[0:3]), -1
        )
        
        Q, R = QR(matrix)
        
        MPO.append(Q.reshape(tensor.shape[0:3]+(-1,)).transpose(0, 3, 1, 2))
        
        tensor = R.reshape((-1,)+tensor.shape[3:])
    
    MPO.append(tensor.transpose(0, 3, 1, 2))
    
    return MPO
    

def MPO_to_Hamiltonian(
    MPO: list[npt.NDArray]
):
    
    """
       3|
    0 -- -- 1
       2|
    """

    n_sites = len(MPO)
    
    rotate_MPO = [mpo.transpose(0, 2, 3, 1) for mpo in MPO]
    physical_dim = rotate_MPO[0].shape[1]

    Hamiltonian = rotate_MPO[0]
    
    for it in range(1, n_sites):
        Hamiltonian = Tensordot(Hamiltonian, rotate_MPO[it], axes=([2*it+1], [0]))

    Hamiltonian = Hamiltonian.reshape(Hamiltonian.shape[1:2*n_sites+1])

    transpose = []
    
    for it in range(n_sites):
        transpose.append(2*it)
    for it in range(n_sites):
        transpose.append(2*it+1)
        
    Hamiltonian = Hamiltonian.transpose(tuple(transpose))
    
    Hamiltonian = Hamiltonian.reshape(physical_dim**n_sites, physical_dim**n_sites)
    
    return Hamiltonian


def mps_mpo_mps(
    tensor: npt.NDArray,
    mps1: npt.NDArray,
    mpo: npt.NDArray,
    mps2: npt.NDArray,
    backward: bool = True,
):
    
    new = mps_mpo_none(tensor, mps1, mpo)
    
    if backward:
        new = Tensordot(new, mps2, axes=[(2, len(new.shape)-1), (0, 2)])
        
        ord_new = len(new.shape)
        lst = [i for i in range(ord_new)]
        
        new = new.transpose(
            rearrange_list_by_values(lst, [ord_new-1], [2])
        )
    
    else:
        new = Tensordot(new, mps2, axes=[(len(new.shape)-1), (2)])
        
        ord_new = len(new.shape)
        lst = [i for i in range(ord_new)]
        
        new = new.transpose(
            rearrange_list_by_values(lst, [ord_new-1, ord_new-2], [2, 4])
        )
    
    return new


def MPS_MPO_MPS_env(
    MPS1: list[npt.NDArray],
    MPO: list[npt.NDArray],
    MPS2: list[npt.NDArray,]
):
    
    """
    Overlap between MPS, MPO, and MPS
    
        2|         2|  3|
    0 -- -- 1, 0 -- --- -- 1

        3|
    0 -- -- 1
        2|
    """
    
    tensor = np.array([[[1.0]]])
    
    for mps1, mpo, mps2 in zip(MPS1, MPO, MPS2):
        tensor = mps_mpo_mps(tensor, mps1, mpo, mps2)
    
    return tensor


def mps_mpo_none(
    tensor: npt.NDArray,
    mps1: npt.NDArray,
    mpo: npt.NDArray
) -> npt.NDArray:
    
    new = Tensordot(tensor, mps1, axes=[(0), (0)])
    
    ord_new = len(new.shape)
    lst = [i for i in range(ord_new)]
    
    new = new.transpose(
        rearrange_list_by_values(lst, [ord_new-2], [0])
    )
    
    new = Tensordot(new, mpo, axes=[(1, len(new.shape)-1), (0, 3)])
    
    ord_new = len(new.shape)
    lst = [i for i in range(ord_new)]
    
    new = new.transpose(
        rearrange_list_by_values(lst, [ord_new-2], [1])
    )
    
    return new

