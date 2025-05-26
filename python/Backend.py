import numpy as np
import torch
from typing import Optional
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

        # Optional:  default float dtype for torch
        if lib == "torch":
            self.xp.set_default_dtype(self.xp.float64)

    # --- thin wrappers on common ops ------------------------
    def array(self, x, dtype=None) -> np.ndarray | torch.Tensor:
        if self.lib == "torch":
            return self.xp.tensor(x, dtype=dtype, device=device)
        return self.xp.array(x, dtype=dtype)

    def zeros(self, x, dtype=None) -> np.ndarray | torch.Tensor:
        if self.lib == "torch":
            return self.xp.zeros(x, dtype=dtype, device=device)
        return self.xp.zeros(x, dtype=dtype)

    def ones(self, x, dtype=None) -> np.ndarray | torch.Tensor:
        if self.lib == "torch":
            return self.xp.ones(x, dtype=dtype, device=device)
        return self.xp.ones(x, dtype=dtype)

    def eye(self, x, dtype=None) -> np.ndarray | torch.Tensor:
        if self.lib == "torch":
            return self.xp.eye(x, dtype=dtype, device=device)
        return self.xp.eye(x, dtype=dtype)

    def diag(self, x, k=0, dtype=None) -> np.ndarray | torch.Tensor:
        if self.lib == "torch":
            return self.xp.diag(x, k).type(self.complex if dtype is None else dtype)
        return self.xp.diag(x, k)

    def randn(self, *shape, seed: Optional[int] = None, dtype=None) -> np.ndarray | torch.Tensor:
        if seed is not None:
            np.random.seed(seed) if self.lib == "numpy" else self.xp.manual_seed(seed)
        return self.xp.random.randn(*shape) if self.lib == "numpy" else self.xp.randn(*shape, dtype=dtype, device=device)

    def svd(self, a, full_matrices: bool = False):
        return self._linalg.svd(a, full_matrices=full_matrices)

    def eigh(self, a):
        return self._linalg.eigh(a)

    def qr(self, a):
        return self._linalg.qr(a)

    def norm(self, a):
        return self._linalg.norm(a)

    def tensordot(self, a, b, axes):
        if self.lib == "torch":
            return self.xp.tensordot(a, b, dims=axes)
        return self.xp.tensordot(a, b, axes=axes)

    def conj(self, a):
        return a.conj()

    def einsum(self, subscripts, *operands):
        return contract(subscripts, *operands)
