import numpy as np
import numpy.typing as npt
from copy import deepcopy
from typing import List, Optional, Tuple, Union
from python.Backend import Backend

from python.utils import rearrange_list_by_values
from python.Decomposition import *
from python.Contract import *


def MPS_MPS_overlap(
    MPS1: list[npt.NDArray],
    MPS2: list[npt.NDArray],
    conj: bool = True,
    bk: Backend = Backend('auto')
) -> npt.NDArray:
    
    assert len(MPS1) == len(MPS2), f"{len(MPS1)=} != {len(MPS2)=}"
    ord_MPS = len(MPS1)
    
    # for it, _ in enumerate(MPS1):
    #     print(f"MPS1[{it}].shape={MPS1[it].shape} MPS2[{it}].shape={MPS2[it].shape}")
    
    overlap = bk.array([1]).reshape(1, 1)
    backward = True
    
    for i in range(ord_MPS):
        mps1 = bk.to_device(MPS1[i])
        mps2 = bk.to_device(MPS2[i]) if MPS2[i] is not None else None
    
        # print(f"{i}")
        if mps2 is None:
            # print(f"{mps1.shape=}, {mps2=}")
            overlap = mps_none(overlap, mps1, bk)
            # print(f"{overlap.shape=}")
            backward = False
        else:
            # print(f"{mps1.shape=}, {mps2.shape=}")
            if conj:
                overlap = mps_mps(overlap, mps1, bk.conj(mps2), backward, bk)
            else:
                overlap = mps_mps(overlap, mps1, mps2, backward, bk)
            # # print(f"{overlap.shape=}")
            backward = True
        i += 1
    
    if backward:
        overlap = bk.tensordot(overlap, bk.array([1]).reshape(1, 1), axes=[(0, 1), (0, 1)])
    
    else:
        overlap = bk.tensordot(overlap, bk.array([1]).reshape(1, 1), axes=[(0,), (0,)])
        
        ord_overlap = len(overlap.shape)
        lst = [i for i in range(ord_overlap)]
        
        overlap = bk.transpose(overlap, rearrange_list_by_values(lst, [ord_overlap-1], [1]))
    
    # print(f"{overlap.shape=}")
    
    return overlap


def mps_mps(
    tensor: npt.NDArray,
    mps1: npt.NDArray,
    mps2: npt.NDArray,
    backward: bool = True,
    bk: Backend = Backend('auto')
) -> npt.NDArray:
    
    new = mps_none(tensor, mps1, bk)
    
    if backward:
        new = bk.tensordot(
            new, mps2, axes=((1, len(new.shape)-1), (0, 2))
        )
        
        ord_new = len(new.shape)
        lst = [i for i in range(ord_new)]
        
        new = bk.transpose(
            new, rearrange_list_by_values(lst, [ord_new-1], [1])
        )
    
    else:
        new = bk.tensordot(
            new, mps2, axes=((len(new.shape)-1,), (2,))
        )
        
        ord_new = len(new.shape)
        lst = [i for i in range(ord_new)]
        
        new = bk.transpose(
            new, rearrange_list_by_values(lst, [ord_new-1, ord_new-2], [1, 3])
        )
    
    return new


def mps_none(
    tensor: npt.NDArray,
    mps1: npt.NDArray,
    bk: Backend = Backend('auto')
) -> npt.NDArray:
    
    new = bk.tensordot(tensor, mps1, axes=((0,), (0,)))
    
    ord_new = len(new.shape)
    lst = [i for i in range(ord_new)]
    
    new = bk.transpose(
        new, rearrange_list_by_values(lst, [ord_new-2], [0])
    )
    
    return new


def MPS_MPO_MPS_overlap(
    MPS1: list[npt.NDArray],
    MPO: list[npt.NDArray],
    MPS2: list[npt.NDArray],
    conj: bool = True,
    bk: Backend = Backend('auto')
):
    
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
    
    overlap = bk.array([1]).reshape(1, 1, 1)
    backward = True    
    
    for mps1, mpo, mps2 in zip(MPS1, MPO, MPS2):
        if mps2 is None:
            overlap = mps_mpo_none(overlap, mps1, mpo, bk)
            backward = False
        else:
            if conj:
                overlap = mps_mpo_mps(overlap, mps1, mpo, bk.conj(mps2), backward, bk)
            else:
                overlap = mps_mpo_mps(overlap, mps1, mpo, mps2, backward, bk)
            backward = True
    
    if backward:
        overlap = bk.tensordot(overlap, bk.array([1]).reshape(1, 1, 1), axes=((0, 1, 2), (0, 1, 2)))
    else:
        overlap = bk.tensordot(overlap, bk.array([1]).reshape(1, 1, 1), axes=((0, 1), (0, 1)))
        
        ord_overlap = len(overlap.shape)
        lst = [i for i in range(ord_overlap)]
        
        overlap = bk.transpose(
            overlap, rearrange_list_by_values(lst, [ord_overlap-1], [1])
        )
    
    return overlap


def MPS_MPO_multiplication(
    MPS: list[npt.NDArray],
    MPO: list[npt.NDArray],
    bk: Backend = Backend('auto')
) -> npt.NDArray:
    
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
    
    overlap = bk.array([1]).reshape(1, 1, 1)
    backward = True    
    
    for mps, mpo in zip(MPS, MPO):
        overlap = mps_mpo_none(overlap, mps, mpo, bk)
    
    overlap = bk.tensordot(overlap, bk.array([1]).reshape(1, 1, 1), axes=((0, 1), (0, 1)))   
    ord_overlap = len(overlap.shape)
    lst = [i for i in range(ord_overlap)]
    
    overlap = bk.transpose(
        overlap, rearrange_list_by_values(lst, [ord_overlap-1], [1])
    )
    overlap = bk.reshape(overlap, overlap.shape[2:])
    
    return bk.to_cpu(overlap)


def contract_MPS(MPS: list[npt.NDArray], bk: Backend = Backend('auto')) -> npt.NDArray:
    
    absolute = MPS[0]
    
    for it in range(1, len(MPS)):
        absolute = bk.tensordot(absolute, MPS[it], axes=((1,), (0,)))
        ord_absolute = len(absolute.shape)
        lst = [i for i in range(ord_absolute)]
        absolute = bk.transpose(
            absolute, rearrange_list_by_values(lst, [ord_absolute-2], [1])
        )
    
    absolute = bk.tensordot(absolute, bk.identity(absolute.shape[0]), axes=((0,1), (0,1)))

    assert len(absolute.shape) == len(MPS)

    return absolute


def Hamiltonian_to_MPO(
    Hamiltonian: npt.NDArray,
    bk: Backend = Backend('auto')
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
    
    tensor = bk.transpose(tensor, tuple(transpose))
    tensor = tensor.reshape((1,)+tensor.shape+(1,))
    
    MPO = []
    
    for it in range(n_sites - 1):
        
        matrix = tensor.reshape(
            np.prod(tensor.shape[0:3]), -1
        )
        
        Q, R = QR(matrix)
        
        MPO.append(bk.transpose(Q.reshape(tensor.shape[0:3]+(-1,)), (0, 3, 1, 2)))
        
        tensor = R.reshape((-1,)+tensor.shape[3:])
    
    MPO.append(bk.transpose(tensor, (0, 3, 1, 2)))
    
    return MPO
    

def MPO_to_Hamiltonian(
    MPO: list[npt.NDArray],
    bk: Backend = Backend('auto')
):
    
    """
       3|
    0 -- -- 1
       2|
    """

    n_sites = len(MPO)
    
    # Use the robust bk.transpose which handles permute/transpose internally
    rotate_MPO = [bk.transpose(mpo, (0, 2, 3, 1)) for mpo in MPO]
        
    physical_dim = rotate_MPO[0].shape[1]

    Hamiltonian = rotate_MPO[0]
    
    for it in range(1, n_sites):
        Hamiltonian = bk.tensordot(Hamiltonian, rotate_MPO[it], axes=((2*it+1,), (0,)))

    Hamiltonian = Hamiltonian.reshape(Hamiltonian.shape[1:2*n_sites+1])

    transpose_axes = []
    
    for it in range(n_sites):
        transpose_axes.append(2*it)
    for it in range(n_sites):
        transpose_axes.append(2*it+1)
        
    Hamiltonian = bk.transpose(Hamiltonian, tuple(transpose_axes))
    
    Hamiltonian = Hamiltonian.reshape(physical_dim**n_sites, physical_dim**n_sites)
    
    # If the backend is torch, convert to CPU numpy for compatibility with np.linalg.eigh
    # If the backend is numpy, it's already a numpy array
    # if bk.lib == "torch":
    #     Hamiltonian = bk.to_cpu(Hamiltonian)
    
    return Hamiltonian


def mps_mpo_mps(
    tensor: npt.NDArray,
    mps1: npt.NDArray,
    mpo: npt.NDArray,
    mps2: npt.NDArray,
    backward: bool = True,
    bk: Backend = Backend('auto')
) -> npt.NDArray:
    
    new = mps_mpo_none(tensor, mps1, mpo, bk)
    
    if backward:
        new = bk.tensordot(new, mps2, axes=((2, len(new.shape)-1), (0, 2)))
        
        ord_new = len(new.shape)
        lst = [i for i in range(ord_new)]
        
        new = bk.transpose(
            new, rearrange_list_by_values(lst, [ord_new-1], [2])
        )
    
    else:
        new = bk.tensordot(new, mps2, axes=((len(new.shape)-1,), (2,)))
        
        ord_new = len(new.shape)
        lst = [i for i in range(ord_new)]
        
        new = bk.transpose(
            new, rearrange_list_by_values(lst, [ord_new-1, ord_new-2], [2, 4])
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
    
    tensor = bk.array([[[1.0]]])
    
    for mps1, mpo, mps2 in zip(MPS1, MPO, MPS2):
        tensor = mps_mpo_mps(tensor, mps1, mpo, mps2)
    
    return tensor


def mps_mpo_none(
    tensor: npt.NDArray,
    mps1: npt.NDArray,
    mpo: npt.NDArray,
    bk: Backend = Backend('auto')
) -> npt.NDArray:
    
    new = bk.tensordot(tensor, mps1, axes=((0,), (0,)))
    
    ord_new = len(new.shape)
    lst = [i for i in range(ord_new)]
    
    new = bk.transpose(
        new, rearrange_list_by_values(lst, [ord_new-2], [0])
    )
    
    new = bk.tensordot(new, mpo, axes=((1, len(new.shape)-1), (0, 3)))
    
    ord_new = len(new.shape)
    lst = [i for i in range(ord_new)]
    
    new = bk.transpose(
        new, rearrange_list_by_values(lst, [ord_new-2], [1])
    )
    
    return new

