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
from python.Backend import Backend


def Contract(
    subscripts: str,
    *operands,
    bk: Backend = Backend('auto')
) -> npt.NDArray:
    
    now = time.perf_counter()

    try:
        # Convert operands to complex dtype if needed
        complex_operands = []
        for op in operands:
            if bk.lib == "torch":
                if not op.is_complex():
                    op = op.to(dtype=bk.complex)
            else:
                if not np.iscomplexobj(op):
                    op = op.astype(bk.complex)
            complex_operands.append(op)

        # Only pass the subscripts and operands to einsum, let the backend handle dtype
        result = bk.einsum(subscripts, *complex_operands)

        return result

    except Exception as e:
        print(f"{subscripts=}")
        for i, tensor in enumerate(operands):
            if bk.lib == "torch" and hasattr(tensor, 'device'):
                tensor = bk.to_cpu(tensor)
            print(f"operands[{i}].shape={tensor.shape}")
        
        # Move operands to CPU for memory estimation
        cpu_operands = [bk.to_cpu(op) if bk.lib == "torch" else op for op in operands]
        estimated_memory, estimated_time = check_conditions_einsum(cpu_operands, subscripts)
        memory_limit = get_free_memory_windows()
        print(f"Contract error {e}")
        print(f"estimated memory = {format_memory_size(estimated_memory)}\navailable memory = {format_memory_size(memory_limit)}")
        print_traceback(e)

        raise Exception(f"Contract Error {e}")


def Tensordot(
    tensor1: npt.NDArray,
    tensor2: npt.NDArray,
    axes,
    bk: Backend = Backend('auto')
) -> npt.NDArray:
    
    now = time.perf_counter()

    # Convert tensors to complex dtype if needed
    if bk.lib == "torch":
        if not tensor1.is_complex():
            tensor1 = tensor1.to(dtype=bk.complex)
        if not tensor2.is_complex():
            tensor2 = tensor2.to(dtype=bk.complex)
    else:
        if not np.iscomplexobj(tensor1):
            tensor1 = tensor1.astype(bk.complex)
        if not np.iscomplexobj(tensor2):
            tensor2 = tensor2.astype(bk.complex)

    assert bk.isfinite(tensor1).all(), f"tensor1 is not finite\n{tensor1=}"
    assert bk.isfinite(tensor2).all(), f"tensor2 is not finite\n{tensor2=}"

    try:
        # return np.tensordot(
        #     tensor1, tensor2, axes=axes), round_sig(time.perf_counter()-now)
        
        
        # if estimated_memory > memory_limit:
        #     print(f"Chunked tensordot: estimated memory {format_memory_size(estimated_memory)} > available memory {format_memory_size(memory_limit)}")
        #     return chunked_tensordot(tensor1, tensor2, axes)

        # else:
        return bk.tensordot(
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
