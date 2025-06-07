import numpy as np
import numpy.typing as npt
from scipy.linalg import expm
from scipy.stats import gmean
import sys
from pathlib import Path
# * Self only available after python 3.11
# from typing import Self
from typing import Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
import hashlib
import pickle
import pandas as pd
import argparse
import time
import itertools
from datetime import datetime
import opt_einsum as oe
import matplotlib.pyplot as plt
import math
from copy import deepcopy
from scipy.integrate import quad
import traceback
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import psutil
import os
import tracemalloc
from python.Backend import Backend


import numpy as np


def round_sig(arr, num_digits=3, bk: Backend = Backend('auto')):
    """
    Round an array or a single float/complex number to a specific number of significant digits.

    :param arr: Array or scalar (float or complex) compatible with the backend (NumPy or PyTorch)
    :param num_digits: Number of significant digits to round to
    :param bk: Backend instance for NumPy or PyTorch operations
    :return: Array or scalar rounded to the specified number of significant digits
    """

    def round_single_value(x):
        if bk.isfinite(x) == False:
            return x
        
        if bk.abs(x) == 0:
            return bk.array(0, dtype=x.dtype if hasattr(x, 'dtype') else None)
        
        # Check if x is complex (for arrays/tensors or scalars)
        is_complex = hasattr(x, 'dtype') and x.dtype.is_complex if hasattr(x, 'dtype') else isinstance(x, complex)
        if is_complex:
            real_part = round_real_or_imag(bk.real(x))
            imag_part = round_real_or_imag(bk.imag(x))
            return bk.complex(real_part, imag_part)  # Use the complex method
        else:
            return round_real_or_imag(x)

    def round_real_or_imag(x):
        if bk.isfinite(x) == False:
            return x
        
        if bk.abs(x) == 0:
            return bk.array(0, dtype=x.dtype if hasattr(x, 'dtype') else None)
        
        try:
            scale = -bk.floor(bk.log10(bk.abs(x)))
            scale = scale + (num_digits - 1)
            rounded_value = bk.round(x, decimals=scale)
            return bk.array(rounded_value, dtype=x.dtype if hasattr(x, 'dtype') else None)
        except Exception:
            return x

    arr = bk.to_device(arr)

    if bk.isscalar(arr):
        return round_single_value(arr)
    else:
        temp = bk.array(arr, dtype=arr.dtype if hasattr(arr, 'dtype') else None)
        is_complex = temp.dtype.is_complex if hasattr(temp, 'dtype') else False
        if is_complex:
            vectorized_round = bk.vectorize(round_single_value, otypes=[bk.complex])
        else:
            vectorized_round = bk.vectorize(round_single_value, otypes=[temp.dtype if hasattr(temp, 'dtype') else None])
        return vectorized_round(temp)


def get_shape(lst):
    """Get the shape of a nested list."""
    if not isinstance(lst, list) or not lst:  # Empty list or not a list
        return []
    return [len(lst)] + get_shape(lst[0])


def get_coord_from_num(M, N) -> npt.NDArray[np.int64]:

    coord = np.zeros(shape=(M*N, 2), dtype=int)

    for i in range(M*N):
        coord[i][0] = int((i - i % N) / N)
        coord[i][1] = i % N

    return coord


def get_num_from_coord(M, N) -> npt.NDArray[np.int64]:

    try:
        num = np.zeros(shape=(M, N), dtype=int)
    except Exception as e:
        print(f"{M=}, {N=}")

    for i in range(M):
        for j in range(N):
            num[i][j] = N * i + j

    return num


def get_coordinate(M: int, N: int, point: int):

    if point < M * N:
        return (int(point / N), point % N)

    else:
        raise ValueError(f"{point=} >= {M*N=}")


def permute_list(length: int, num1: int, num2: int) -> tuple[list[int], list[int]]:

    original_list = [i for i in range(length)]
    repermute_list = [i for i in range(length)]

    index_num1 = original_list.index(num1)
    index_num2 = original_list.index(num2)

    # * Remove the numbers from their original positions
    original_list.pop(index_num1)
    original_list.pop(
        index_num2 - 1) if index_num2 > index_num1 else original_list.pop(index_num2)

    # * Insert the numbers at the beginning of the list
    original_list.insert(0, num2)
    original_list.insert(0, num1)

    for i in range(length):
        repermute_list[i] = original_list.index(i)

    return original_list, repermute_list


def reshape_tensor(
    tensor: npt.NDArray,
    num1: int, num2: int, index_list: list[int],
    bk: Backend = Backend('auto')
):
    """
    tensor: shape[total up, total down, west, south, ...., single up, south, east, north, single down]
    return shape[total up * single up, total down * single down, ...]
    """
    
    # Ensure tensor is a numeric type
    if not np.issubdtype(tensor.dtype, np.number):
        tensor = tensor.astype(np.float64)

    assert np.isfinite(tensor).all(), f"Reshape Error, tensor is not finite\n{tensor=}"

    sizeup1, sizeup2, sizedown1, sizedown2 = (
        tensor.shape[0], tensor.shape[num1], tensor.shape[1], tensor.shape[num2])

    transpose_list = [i for i in range(sum(index_list))]

    index_num1 = transpose_list.index(num1)
    index_num2 = transpose_list.index(num2)

    transpose_list.pop(index_num1)
    transpose_list.pop(
        index_num2 - 1) if index_num2 > index_num1 else transpose_list.pop(index_num2)

    # * num 1 in second, num2 in the fourth position
    transpose_list.insert(2, num2)
    transpose_list.insert(1, num1)

    tensor = bk.transpose(tensor, tuple(transpose_list))

    reshape_list = [tensor.shape[0] * tensor.shape[1],
                    tensor.shape[2] * tensor.shape[3]]

    for i, shape in enumerate(tensor.shape):
        if i >= 4:
            reshape_list.append(shape)

    tensor = tensor.reshape(reshape_list)

    index_list[-1] -= 2

    return tensor, index_list


def transpose_into_original(tensor, num1: int, num2: int | None = None):
    """
    return [0, 1, 2, ..., M*N-2, ..., M*N-1, ...]
    """
    len_tensor = len(tensor.shape)

    if num2 == None:
        index = [i for i in range(len_tensor-1)]

        index.insert(num1, len_tensor-1)

    else:
        index = [i for i in range(len_tensor-2)]

        if num1 > num2:
            index.insert(num2, len_tensor-1)
            index.insert(num1, len_tensor-2)
        else:
            index.insert(num1, len_tensor-2)
            index.insert(num2, len_tensor-1)

    return index


def get_fidelity(exact, approx):

    # fidelity = np.trace(exact@approx.conj().T) / np.sqrt(np.trace(
    #     exact@exact.conj().T)*np.trace(approx@approx.conj().T)
    # )

    exact_fidelity = np.abs(
            exact @ approx.conj() / np.sqrt(
                (exact @ exact.conj())*(approx @ approx.conj())))
        
    return exact_fidelity**2


def get_fidelity_from_loss(full_losses, M: int, N: int):

    lowest, arithmetic_mean, geometric_mean = 1.0, 1.0, 1.0

    if full_losses == 0.0:
        return 1.0

    for i, losses in enumerate(full_losses):
        single_fidelity = np.empty((M*N))

        for j, loss in enumerate(losses):
            single_fidelity[j] = (
                (1-loss[0]) * (1-loss[1]) * (1-loss[2]) * (1-loss[3])
            )

    # * lowest case
        lowest *= min(single_fidelity)
        # if i == len(full_losses) - 1:
        #     print(f"{i+1} {min(single_fidelity)=}")

        # * geometric mean
        geometric_mean *= gmean(single_fidelity)

    return lowest


def geo_mean(arr: npt.NDArray, axis: int = 1):
    
    try:
        return gmean(arr, axis=axis)
    
    except Exception as e:
        return gmean(arr)


def geo_std(arr: np.ndarray, axis: int = 0):
    # Calculate geometric standard deviation along the specified axis
    try:
        log_arr = np.log(arr)
    except Exception as e:
        # If taking the log fails, return an array of zeros with the correct shape
        shape = list(arr.shape)
        shape.pop(axis)
        return np.zeros(shape)

    # Calculate the standard deviation along the specified axis
    std_log = np.std(log_arr, axis=axis)

    # Return the exponential of the standard deviation of logs
    geo_std_dev = np.exp(std_log)
    
    # Correct signs if needed (optional, can be removed if not relevant)
    mean_log = np.mean(log_arr, axis=axis)
    geo_std_dev[mean_log < 0] = np.exp(-std_log[mean_log < 0])

    return geo_std_dev


def rearrange_list_by_values(
    lst: list, values_to_move: list,
    new_positions: list[int]
) -> list[int]:
    """
    Rearrange a list by moving specified elements to new positions.
    
    :param lst: The original list.
    :param values_to_move: A list of values of the elements to move.
    :param new_positions: A list of new positions where the elements should be moved.
    """
    
    new_lst = deepcopy(lst)
    
    # Ensure the length of values_to_move and new_positions are the same
    if len(values_to_move) != len(new_positions):
        raise ValueError("values_to_move and new_positions must have the same length.")
    
    # Find the current indices of the elements to move
    indices_to_move = [new_lst.index(value) for value in values_to_move]
    
    # Sort the pairs of indices to move and new positions by the original indices in descending order
    # This avoids the issue of changing indices after popping elements
    moves = sorted(zip(indices_to_move, values_to_move, new_positions), reverse=True)

    # Pop elements from the list starting from the highest index
    for index, _, _ in moves:
        new_lst.pop(index)
    
    # Insert elements at the new positions in ascending order of their desired new position
    for _, value, new_position in sorted(moves, key=lambda x: x[2]):
        new_lst.insert(new_position, value)
    
    return new_lst


def get_repermutation(lst: list[int]):
    
    len = np.size(lst)
    repermute = []
    
    for loc in range(len):
        # print(f"{lst=} {loc=} {lst.index(loc)=}")
        repermute.append(lst.index(loc))
    
    return repermute


def get_entropy(Lambdas, bk: Backend = Backend('auto')):
    
    entropy = 0
    
    for Lambda in Lambdas:
        if Lambda < 1.e-8:
            continue
        entropy += -Lambda**2*bk.log2(Lambda**2)
        
    return entropy


def print_traceback(e: Exception):
    
    tb = e.__traceback__
    
    while tb is not None:
        print(f"File: {tb.tb_frame.f_code.co_filename}, Line: {tb.tb_lineno}, in {tb.tb_frame.f_code.co_name}")
        tb = tb.tb_next


def int_to_n_digit_binary(numbers, n=None):
    def convert_to_binary_array(number, n):
        binary_str = bin(number)[2:]  # Convert to binary and remove the "0b" prefix
        required_n = len(binary_str)
        if n is None or required_n > n:
            n = required_n  # Adjust n to fit the binary number
        padded_binary_str = binary_str.zfill(n)  # Pad with leading zeros to make it n digits
        return np.array([int(digit) for digit in padded_binary_str])

    if isinstance(numbers, (list, np.ndarray)):
        result = [convert_to_binary_array(int(number), n) for number in numbers]
        return result  # Return list of ndarrays if input was a list/ndarray of multiple numbers
    else:
        return [convert_to_binary_array(int(numbers), n)]

