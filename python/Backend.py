import numpy as np
import torch
from typing import Optional, Union, Any
from opt_einsum import contract

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Backend:
    """Facade enabling NumPy <-> Torch backends with identical calls."""

    def __init__(self, lib: str = "auto"):
        assert lib in ("numpy", "torch", "auto"), "Backend must be 'numpy' or 'torch' or 'auto'."
        if lib == "auto":
            lib = "numpy" if not torch.cuda.is_available() else "torch"

        self.lib = lib
        self.xp = np if lib == "numpy" else torch
        self._linalg = self.xp.linalg
        self.complex = self.xp.complex128
        self.device = device if lib == "torch" else None

        if lib == "torch":
            self.xp.set_default_dtype(self.xp.float64)

    def _map_dtype(self, dtype: Any) -> Any:
        """Map NumPy dtype to PyTorch dtype if necessary."""
        if self.lib == "torch" and dtype is not None:
            dtype_mapping = {
                np.float32: torch.float32,
                np.float64: torch.float64,
                np.complex64: torch.complex64,
                np.complex128: torch.complex128,
                np.int32: torch.int32,
                np.int64: torch.int64,
            }
            # Handle NumPy dtype objects
            if isinstance(dtype, np.dtype):
                dtype = dtype.type
            return dtype_mapping.get(dtype, dtype)
        return dtype

    def to_device(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Move tensor to appropriate device."""
        if self.lib == "torch" and isinstance(x, torch.Tensor):
            return x.to(self.device)
        return x

    def to_cpu(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert tensor to numpy array on CPU."""
        if self.lib == "torch" and isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x

    def array(self, x: Any, dtype=None) -> Union[np.ndarray, torch.Tensor]:
        current_dtype = self._map_dtype(dtype if dtype is not None else self.complex)
        if self.lib == "torch":
            if isinstance(x, list):
                processed_x = []
                for item in x:
                    if isinstance(item, torch.Tensor) and item.ndim == 0:
                        processed_x.append(item.item())
                    elif isinstance(item, np.ndarray) and item.ndim == 0:
                        processed_x.append(item.item())
                    else:
                        processed_x.append(item)
                x = processed_x
            if isinstance(x, torch.Tensor):
                return x.clone().detach().to(self.device, dtype=current_dtype)
            return self.xp.tensor(x, dtype=current_dtype, device=self.device)
        
        if isinstance(x, list):
            processed_x = []
            for item in x:
                if isinstance(item, torch.Tensor) and item.ndim == 0:
                    processed_x.append(item.item())
                elif isinstance(item, np.ndarray) and item.ndim == 0:
                    processed_x.append(item.item())
                else:
                    processed_x.append(item)
            x = processed_x
        return self.xp.array(x, dtype=current_dtype)

    def zeros(self, shape: Union[int, tuple], dtype=None) -> Union[np.ndarray, torch.Tensor]:
        current_dtype = self._map_dtype(dtype if dtype is not None else self.complex)
        if self.lib == "torch":
            return self.xp.zeros(shape, dtype=current_dtype, device=self.device)
        return self.xp.zeros(shape, dtype=current_dtype)

    def ones(self, shape: Union[int, tuple], dtype=None) -> Union[np.ndarray, torch.Tensor]:
        current_dtype = self._map_dtype(dtype if dtype is not None else self.complex)
        if self.lib == "torch":
            return self.xp.ones(shape, dtype=current_dtype, device=self.device)
        return self.xp.ones(shape, dtype=current_dtype)

    def eye(self, x: int, dtype=None) -> Union[np.ndarray, torch.Tensor]:
        current_dtype = self._map_dtype(dtype if dtype is not None else self.complex)
        if self.lib == "torch":
            return self.xp.eye(x, dtype=current_dtype, device=self.device)
        return self.xp.eye(x, dtype=current_dtype)

    def identity(self, x: int, dtype=None) -> Union[np.ndarray, torch.Tensor]:
        return self.eye(x, dtype)

    def diag(self, x: Union[np.ndarray, torch.Tensor], k: int = 0, dtype=None) -> Union[np.ndarray, torch.Tensor]:
        x = self.array(x)
        current_dtype = self._map_dtype(dtype if dtype is not None else self.complex)
        if self.lib == "torch":
            return self.xp.diag(x, k).type(current_dtype)
        return self.xp.diag(x, k)

    def randn(self, *shape: int, seed: Optional[int] = None, dtype=None) -> Union[np.ndarray, torch.Tensor]:
        current_dtype = self._map_dtype(dtype if dtype is not None else self.complex)
        if seed is not None:
            np.random.seed(seed) if self.lib == "numpy" else self.xp.manual_seed(seed)
        return self.xp.random.randn(*shape) if self.lib == "numpy" else self.xp.randn(*shape, dtype=current_dtype, device=self.device)

    def svd(self, a: Union[np.ndarray, torch.Tensor], full_matrices: bool = False):
        return self._linalg.svd(a, full_matrices=full_matrices)

    def eigh(self, a: Union[np.ndarray, torch.Tensor]):
        return self._linalg.eigh(a)

    def qr(self, a: Union[np.ndarray, torch.Tensor]):
        return self._linalg.qr(a)

    def norm(self, a: Union[np.ndarray, torch.Tensor]):
        return self._linalg.norm(a)

    def tensordot(self, a: Union[np.ndarray, torch.Tensor], b: Union[np.ndarray, torch.Tensor], axes):
        if self.lib == "torch":
            a = a.type(self.complex)
            b = b.type(self.complex)
            return self.xp.tensordot(a, b, dims=axes)
        else:
            dtype = self.complex
            a = np.array(a, dtype=dtype)
            b = np.array(b, dtype=dtype)
            return self.xp.tensordot(a, b, axes=axes)

    def conj(self, a: Union[np.ndarray, torch.Tensor]):
        return a.conj()

    def einsum(self, subscripts: str, *operands: Union[np.ndarray, torch.Tensor]):
        return contract(subscripts, *operands)

    def transpose(self, a: Union[np.ndarray, torch.Tensor], axes: tuple):
        a = self.array(a)
        if self.lib == "torch":
            return a.permute(axes)
        else:
            return a.transpose(axes)

    def reshape(self, a: Union[np.ndarray, torch.Tensor], shape: tuple):
        return a.reshape(shape)

    def matmul(self, a: Union[np.ndarray, torch.Tensor], b: Union[np.ndarray, torch.Tensor]):
        return self.xp.matmul(a, b)

    def trace(self, a: Union[np.ndarray, torch.Tensor]):
        return self.xp.trace(a)

    def isfinite(self, a: Union[np.ndarray, torch.Tensor]):
        return self.xp.isfinite(a)

    def all(self, a: Union[np.ndarray, torch.Tensor]):
        return self.xp.all(a)

    def isscalar(self, x: Union[np.ndarray, torch.Tensor, float, complex, int]) -> bool:
        """Check if input is a scalar."""
        if self.lib == "torch":
            return isinstance(x, torch.Tensor) and x.ndim == 0
        return self.xp.isscalar(x)

    def abs(self, x: Union[np.ndarray, torch.Tensor]):
        """Compute absolute value."""
        return self.xp.abs(x)

    def real(self, x: Union[np.ndarray, torch.Tensor]):
        """Extract real part."""
        if self.lib == "torch":
            return x.real
        return self.xp.real(x)

    def imag(self, x: Union[np.ndarray, torch.Tensor]):
        """Extract imaginary part."""
        if self.lib == "torch":
            return x.imag
        return self.xp.imag(x)

    def complex(self, real: Union[np.ndarray, torch.Tensor], imag: Union[np.ndarray, torch.Tensor]):
        """Construct complex number from real and imaginary parts."""
        if self.lib == "torch":
            return torch.complex(real, imag)
        return real + 1j * imag

    def floor(self, x: Union[np.ndarray, torch.Tensor]):
        """Compute floor of input."""
        return self.xp.floor(x)

    def log10(self, x: Union[np.ndarray, torch.Tensor]):
        """Compute base-10 logarithm."""
        return self.xp.log10(x)

    def log2(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Compute base-2 logarithm."""
        return self.xp.log2(x)

    def round(self, x: Union[np.ndarray, torch.Tensor], decimals: int = 0):
        """Round to specified decimals."""
        if self.lib == "torch":
            if decimals != 0:
                factor = self.xp.pow(10, decimals)
                return self.xp.round(x * factor) / factor
            return self.xp.round(x)
        return self.xp.round(x, decimals)

    def vectorize(self, func, otypes=None):
        """Vectorize a function."""
        if self.lib == "torch":
            def vectorized_func(x):
                if x.ndim == 0:
                    return func(x)
                return torch.stack([func(xi) for xi in x])
            return vectorized_func
        return self.xp.vectorize(func, otypes=otypes)
    
    def arange(self, start, stop=None, step=1, dtype=None):
        """
        Return evenly spaced values within a given interval.
        
        Parameters:
        - start: Number, start of interval (inclusive).
        - stop: Number, end of interval (exclusive). If None, start=0, stop=start.
        - step: Number, spacing between values (default=1).
        - dtype: Data type of output array (default=None, inferred from inputs).
        
        Returns:
        - Array of evenly spaced values.
        
        Examples:
        >>> bk_arange(3)
        array([0, 1, 2])
        >>> bk_arange(1, 5, 0.5)
        array([1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5])
        """
        if stop is None:
            stop = start
            start = 0
        
        # Determine number of elements
        num = int(np.ceil((stop - start) / step))
        if num < 0:
            num = 0
        
        # Generate array
        values = start + step * self.xp.arange(num)
        
        # Set dtype if specified
        if dtype is not None:
            values = values.astype(dtype)
        
        return values
    