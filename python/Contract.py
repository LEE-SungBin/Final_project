import numpy as np
import numpy.typing as npt
# import dask.array as da
import opt_einsum as oe
import psutil
from typing import Union
# from pathlib import Path
import time
import pickle
import tempfile
import os
# import itertools
import multiprocessing as mp

from python.utils import round_sig, print_traceback
from python.memory import get_free_memory_windows, tensor_size_in_bytes, format_memory_size


def Contract(
    subscripts: str,
    *operands,
    out: npt.NDArray | None = None,
    dtype: str | None = None,
    order: str = 'K',
    casting: str = 'safe',
    use_blas: bool = True,
    optimize: str = 'dp',
    # space_limit: int = 17, # * In GB
    backend: str = 'numpy',
    gpu: bool = False,
) -> npt.NDArray:

    """
    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation.
    *operands : list of array_like
        These are the arrays for the operation.
    out : array_like
        A output array in which set the resulting output.
    dtype : str
        The dtype of the given contraction, see np.einsum.
    order : str
        The order of the resulting contraction, see np.einsum.
    casting : str
        The casting procedure for operations of different dtype, see np.einsum.
    use_blas : bool
        Do you use BLAS for valid operations, may use extra memory for more intermediates.
    optimize : str, list or bool, optional (default: ``auto``)
        Choose the type of path.

    Returns
    -------
    out : array_like
        The result of the einsum expression.
    """
    
    now = time.perf_counter()

    if gpu:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        operands = [torch.from_numpy(tensor).to(device).to(torch.complex64) if isinstance(tensor, np.ndarray) else tensor.to(device).to(torch.complex64) for tensor in operands]
        backend = 'torch'
        out = None
        for i, tensor in enumerate(operands):
            assert torch.isfinite(tensor).all(), (
                f"operand {i} is not finite\n"
                f"operands[{i}].shape={operands[i].shape}\n"
                f"operands[{i}]={operands[i]}"
            )

    else : 
        for i, tensor in enumerate(operands):
            assert np.isfinite(tensor).all(
            ), f"operand {i} is not finite\noperands[{i}].shape={operands[i].shape}\noperands[{i}]={operands[i]}"

    try:
        
        # estimated_memory, estimated_time = check_conditions_einsum(operands, subscripts)
        
        # memory_limit = get_free_memory_windows()
        
        # if estimated_memory > memory_limit:
        #     # Create Dask arrays with appropriate chunk sizes
        #     chunk_size = int((memory_limit / len(operands)) ** (1 / 3)) // operands[0].itemsize
        #     dask_operands = [da.from_array(tensor, chunks=chunk_size) for tensor in operands]

        #     # Perform the contraction using Dask
        #     result = da.einsum(subscripts, *dask_operands, optimize=optimize)

        #     if out is not None:
        #         result.store(out)
        #     else:
        #         result = result.compute()

        # else:
        result = oe.contract(
            subscripts, *operands, out=out, dtype=dtype, order=order, casting=casting, 
            use_blas=use_blas, optimize=optimize, backend=backend
        )

        if gpu : 
            result = result.cpu().numpy()
        
        return result

    # except MemoryError as e:
    #     print(f"Memory limit exceeded: {e}")
    #     print_traceback(e)
    #     raise

    except Exception as e:
        print(f"{subscripts=}")
        for i, tensor in enumerate(operands):
            print(f"operands[{i}].shape={operands[i].shape}")
        estimated_memory, estimated_time = check_conditions_einsum(operands, subscripts)
        memory_limit = get_free_memory_windows()
        print(f"Contract error {e}")
        print(f"estimated memory = {format_memory_size(estimated_memory)}\navailable memory = {format_memory_size(memory_limit)}")
        print_traceback(e)

        # raise Exception(f"Contract Error {e}")


def Tensordot(
    tensor1: npt.NDArray,
    tensor2: npt.NDArray,
    axes
) -> npt.NDArray:
    
    now = time.perf_counter()

    assert np.isfinite(tensor1).all(), f"tensor1 is not finite\n{tensor1=}"
    assert np.isfinite(tensor2).all(), f"tensor2 is not finite\n{tensor2=}"

    try:
        # return np.tensordot(
        #     tensor1, tensor2, axes=axes), round_sig(time.perf_counter()-now)
        
        
        # if estimated_memory > memory_limit:
        #     print(f"Chunked tensordot: estimated memory {format_memory_size(estimated_memory)} > available memory {format_memory_size(memory_limit)}")
        #     return chunked_tensordot(tensor1, tensor2, axes)

        # else:
        return np.tensordot(
            tensor1, tensor2, axes=axes
        )

    except Exception as e:
        print(f"Tensordot error\n{e}\n{tensor1.shape=}\n{tensor2.shape=}\n{axes=}")
        print_traceback(e)
        estimated_memory = check_conditions_tensordot(tensor1, tensor2, axes)
        memory_limit = get_free_memory_windows()
        print(f"estimated memory = {format_memory_size(estimated_memory)}\navailable memory = {format_memory_size(memory_limit)}")
        raise Exception(
            f"Tensordot Error\n{e}\n{tensor1.shape=}\n{tensor2.shape=}\n{axes=}")


def calculate_chunk_size(tensor1, tensor2, axes, available_memory):
    # Estimate the memory usage of one slice
    axis1, axis2 = axes
    if isinstance(axis1, int):
        axis1 = (axis1,)
    if isinstance(axis2, int):
        axis2 = (axis2,)
        
    # Calculate memory for one slice
    slice_size = np.prod([tensor1.shape[ax] for ax in axis1]) * tensor1.itemsize
    slice_size += np.prod([tensor2.shape[ax] for ax in axis2]) * tensor2.itemsize

    # Estimate how many slices can fit in the available memory
    num_slices = available_memory // slice_size
    return max(1, num_slices // 2)  # Use half of the available memory to be safe


def process_chunk(tensor1, tensor2, axes, chunk1, chunk2):
    axis1, axis2 = axes

    if isinstance(axis1, int):
        axis1 = (axis1,)
    if isinstance(axis2, int):
        axis2 = (axis2,)

    result_shape = [tensor1.shape[i] for i in range(tensor1.ndim) if i not in axis1]
    result_shape += [tensor2.shape[i] for i in range(tensor2.ndim) if i not in axis2]
    chunk_result = np.zeros(result_shape, dtype=np.result_type(tensor1, tensor2))
    
    for i in chunk1:
        index1 = tuple(slice(None) if dim not in axis1 else i[axis1.index(dim)] for dim in range(tensor1.ndim))
        for j in chunk2:
            index2 = tuple(slice(None) if dim not in axis2 else j[axis2.index(dim)] for dim in range(tensor2.ndim))
            chunk_result += np.tensordot(tensor1[index1], tensor2[index2], axes=0)
    
    return chunk_result


def chunked_tensordot(tensor1, tensor2, axes, num_processes=None):
    axis1, axis2 = axes

    # Ensure axes are tuples
    if isinstance(axis1, int):
        axis1 = (axis1,)
    if isinstance(axis2, int):
        axis2 = (axis2,)

    shape1 = [tensor1.shape[i] for i in range(tensor1.ndim) if i not in axis1]
    shape2 = [tensor2.shape[i] for i in range(tensor2.ndim) if i not in axis2]
    result_shape = shape1 + shape2

    # Create chunks for parallel processing
    indices1 = list(np.ndindex(*[tensor1.shape[ax] for ax in axis1]))
    indices2 = list(np.ndindex(*[tensor2.shape[ax] for ax in axis2]))
    
    chunk_size1 = len(indices1) // (num_processes or mp.cpu_count())
    chunk_size2 = len(indices2) // (num_processes or mp.cpu_count())
    
    chunks1 = [indices1[i:i + chunk_size1] for i in range(0, len(indices1), chunk_size1)]
    chunks2 = [indices2[i:i + chunk_size2] for i in range(0, len(indices2), chunk_size2)]
    
    # Multiprocessing pool
    with mp.Pool(processes=num_processes) as pool:
        results = []
        for chunk1 in chunks1:
            for chunk2 in chunks2:
                results.append(pool.apply_async(process_chunk, args=(tensor1, tensor2, axes, chunk1, chunk2)))
        
        # Aggregate results
        result = np.zeros(result_shape, dtype=np.result_type(tensor1, tensor2))
        for res in results:
            result += res.get()
    
    return result


def check_conditions_einsum(operands, subscripts):
    # Calculate memory usage of the inputs
    input_memory = sum(tensor_size_in_bytes(tensor) for tensor in operands)

    # Get the optimal contraction path
    path, path_info = oe.contract_path(subscripts, *operands, optimize='dp')
    max_intermediate_size = int(path_info.largest_intermediate * operands[0].itemsize)
    estimated_memory = max(input_memory, max_intermediate_size) * 4
    estimated_time = float(path_info.opt_cost) / (4*1.e9*8)
    memory_limit = get_free_memory_windows()

    if estimated_memory > memory_limit:
        for i, tensor in enumerate(operands):
            print(f"operands[{i}].shape={tensor.shape}")
        print(f"{subscripts=}")
        # raise MemoryError(f"Estimated memory usage {format_memory_size(estimated_memory)} exceeds {format_memory_size(memory_limit)} limit")

    if estimated_memory > 1024**3:
        print(f"Estimated memory usage {format_memory_size(estimated_memory)} exceeds 1GB, time: {round_sig(estimated_time)}s")
        for i, tensor in enumerate(operands):
            print(f"operands[{i}].shape={tensor.shape}")
        print(f"{subscripts=}")

    return estimated_memory, estimated_time  # This could be useful for logging or further checks


def check_conditions_tensordot(tensor1, tensor2, axes):
    # Calculate the memory usage of the input tensors
    input_memory = tensor_size_in_bytes(tensor1) + tensor_size_in_bytes(tensor2)
    shape1 = tensor1.shape
    shape2 = tensor2.shape
    
    if axes == 0:
        remaining_shape1 = [shape1[i] for i in range(len(shape1))]
        remaining_shape2 = [shape2[i] for i in range(len(shape2))]
    
    elif type(axes[0]) == int:
        shape1_axes = [shape1[axes[0]]]
        shape2_axes = [shape2[axes[1]]]
        
        if shape1_axes != shape2_axes:
            raise ValueError(f"Incompatible dimensions for tensordot: {shape1_axes} vs {shape2_axes}")
        
        remaining_shape1 = [shape1[i] for i in range(len(shape1)) if i != axes[0]]
        remaining_shape2 = [shape2[i] for i in range(len(shape2)) if i != axes[1]]
        
    else:
        # Ensure the axes indices are within the valid range for both tensors
        assert all(ax < len(shape1) for ax in axes[0]), f"axes[0] has indices out of range for tensor1.shape={shape1}"
        assert all(ax < len(shape2) for ax in axes[1]), f"axes[1] has indices out of range for tensor2.shape={shape2}"
        
        shape1_axes = [shape1[ax] for ax in axes[0]]
        shape2_axes = [shape2[ax] for ax in axes[1]]
        
        if shape1_axes != shape2_axes:
            raise ValueError(f"Incompatible dimensions for tensordot: {shape1_axes} vs {shape2_axes}")

        remaining_shape1 = [shape1[i] for i in range(len(shape1)) if i not in axes[0]]
        remaining_shape2 = [shape2[i] for i in range(len(shape2)) if i not in axes[1]]
        
    resulting_shape = remaining_shape1 + remaining_shape2
    output_size = np.prod(resulting_shape)
    
    max_intermediate_size = output_size * tensor1.itemsize

    estimated_memory = max(input_memory, max_intermediate_size) * 4

    memory_limit = get_free_memory_windows()
    if estimated_memory > memory_limit:
        print(f"Estimated memory usage {format_memory_size(estimated_memory)} exceeds {format_memory_size(memory_limit)} limit")
        print(f"tensor1.shape={tensor1.shape}, tensor2.shape={tensor2.shape}, axes={axes}")
        # raise MemoryError(f"Estimated memory usage {format_memory_size(estimated_memory)} exceeds {format_memory_size(memory_limit)} limit")
    
    if estimated_memory > 1024**3:
        print(f"Estimated memory usage {format_memory_size(estimated_memory)} exceeds 1GB")
        print(f"tensor1.shape={tensor1.shape}, tensor2.shape={tensor2.shape}, axes={axes}")

    return estimated_memory


def chunk_tensor(tensor, chunk_size):
    """Divide tensor into chunks of given size."""
    chunks = [tensor[i:i + chunk_size] for i in range(0, tensor.shape[0], chunk_size)]
    return chunks


def save_chunk_to_disk(chunk, filename):
    with open(filename, 'wb') as f:
        pickle.dump(chunk, f)


def load_chunk_from_disk(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def contract_with_file_io(subscripts: str, *operands, chunk_size: int, memory_limit: int):
    temp_dir = tempfile.mkdtemp()
    filenames = [os.path.join(temp_dir, f"tensor_{i}.pkl") for i in range(len(operands))]

    # Save chunks to disk
    for i, tensor in enumerate(operands):
        chunks = chunk_tensor(tensor, chunk_size)
        for j, chunk in enumerate(chunks):
            save_chunk_to_disk(chunk, filenames[i] + f"_{j}")

    # Load chunks and perform contraction in pieces
    result = None
    for i in range(len(operands)):
        for j in range(len(chunks)):
            chunk = load_chunk_from_disk(filenames[i] + f"_{j}")
            if result is None:
                result = chunk
            else:
                result = oe.contract(subscripts, result, chunk, optimize='dp')
    
    # Clean up temporary files
    for filename in filenames:
        os.remove(filename)
    os.rmdir(temp_dir)
    
    return result

