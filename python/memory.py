import psutil
import os
import tracemalloc
import torch
import numpy as np
import numpy.typing as npt
from typing import List, Optional, Tuple, Union
from python.Backend import Backend

def get_free_memory_windows():
    mem = psutil.virtual_memory()
    free_memory = mem.available  # Available memory in bytes
    return free_memory


def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss  # in bytes


def tensor_size_in_bytes(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor.size * tensor.itemsize
    elif isinstance(tensor, torch.Tensor):
        return tensor.numel() * tensor.element_size()
    else:
        raise TypeError("Unsupported tensor type")


def format_memory_size(size_in_bytes):
    """
    Convert a size in bytes into a human-readable string with units.

    Parameters:
    - size_in_bytes: int - The size in bytes to be converted.

    Returns:
    - str - A string representation of the size in appropriate units (B, kB, MB, GB, TB).
    """
    units = ['B', 'kB', 'MB', 'GB', 'TB']
    size = size_in_bytes
    unit_index = 0

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1

    return f"{size:.2f} {units[unit_index]}"