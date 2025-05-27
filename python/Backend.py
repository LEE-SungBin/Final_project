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

        # Optional:  default float dtype for torch
        if lib == "torch":
            self.xp.set_default_dtype(self.xp.float64)

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
        current_dtype = dtype if dtype is not None else self.complex
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
        
        # Numpy backend
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
        current_dtype = dtype if dtype is not None else self.complex
        if self.lib == "torch":
            return self.xp.zeros(shape, dtype=current_dtype, device=self.device)
        return self.xp.zeros(shape, dtype=current_dtype)

    def ones(self, shape: Union[int, tuple], dtype=None) -> Union[np.ndarray, torch.Tensor]:
        current_dtype = dtype if dtype is not None else self.complex
        if self.lib == "torch":
            return self.xp.ones(shape, dtype=current_dtype, device=self.device)
        return self.xp.ones(shape, dtype=current_dtype)

    def eye(self, x: int, dtype=None) -> Union[np.ndarray, torch.Tensor]:
        current_dtype = dtype if dtype is not None else self.complex
        if self.lib == "torch":
            return self.xp.eye(x, dtype=current_dtype, device=self.device)
        return self.xp.eye(x, dtype=current_dtype)

    def identity(self, x: int, dtype=None) -> Union[np.ndarray, torch.Tensor]:
        return self.eye(x, dtype)

    def diag(self, x: Union[np.ndarray, torch.Tensor], k: int = 0, dtype=None) -> Union[np.ndarray, torch.Tensor]:

        if self.lib == "torch":
            return self.xp.diag(x, k).type(self.complex if dtype is None else dtype)
        return self.xp.diag(x, k)

    def randn(self, *shape: int, seed: Optional[int] = None, dtype=None) -> Union[np.ndarray, torch.Tensor]:
        if seed is not None:
            np.random.seed(seed) if self.lib == "numpy" else self.xp.manual_seed(seed)
        return self.xp.random.randn(*shape) if self.lib == "numpy" else self.xp.randn(*shape, dtype=dtype, device=self.device)

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
            a = a.type(self.complex) # Ensure complex type
            b = b.type(self.complex) # Ensure complex type
            return self.xp.tensordot(a, b, dims=axes)
        else: # numpy
            dtype = self.complex
            a = np.array(a, dtype=dtype) # Ensure 'a' is a NumPy array and then set dtype
            b = np.array(b, dtype=dtype) # Ensure 'b' is a NumPy array and then set dtype
            return self.xp.tensordot(a, b, axes=axes)

    def conj(self, a: Union[np.ndarray, torch.Tensor]):
        return a.conj()

    def einsum(self, subscripts: str, *operands: Union[np.ndarray, torch.Tensor]):
        return contract(subscripts, *operands)

    def transpose(self, a: Union[np.ndarray, torch.Tensor], axes: tuple):
        a = self.array(a)
        if self.lib == "torch":
            return a.permute(axes)
        else:  # self.lib == "numpy"
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
