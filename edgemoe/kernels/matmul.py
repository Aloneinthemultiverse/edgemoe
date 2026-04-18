"""Python matmul kernel — pure-torch fallback when the C kernel isn't built.

Exposes a single entry point `matmul_4bit(input, packed_weights)` that
dispatches to the native AVX2/AMX kernel if available, otherwise falls
back to PyTorch.
"""

from __future__ import annotations

import ctypes
import os
import platform
from pathlib import Path

import torch


_NATIVE_LIB: ctypes.CDLL | None = None


def _try_load_native() -> ctypes.CDLL | None:
    global _NATIVE_LIB
    if _NATIVE_LIB is not None:
        return _NATIVE_LIB
    here = Path(__file__).parent
    if platform.system() == "Windows":
        candidate = here / "edgemoe_kernels.dll"
    elif platform.system() == "Darwin":
        candidate = here / "edgemoe_kernels.dylib"
    else:
        candidate = here / "edgemoe_kernels.so"
    if not candidate.exists():
        return None
    try:
        lib = ctypes.CDLL(str(candidate))
        lib.matmul_4bit_avx2.restype = None
        lib.matmul_4bit_avx2.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int,
        ]
        _NATIVE_LIB = lib
        return lib
    except OSError:
        return None


def has_native_kernel() -> bool:
    return _try_load_native() is not None


def matmul_4bit(
    input_fp: torch.Tensor, packed: dict
) -> torch.Tensor:
    """Dispatch to native kernel if on CPU; else PyTorch dequant+matmul."""
    if input_fp.is_cuda:
        return _matmul_torch(input_fp, packed)

    lib = _try_load_native()
    if lib is None or packed.get("mode") != "group_asym" or packed.get("bits") != 4:
        return _matmul_torch(input_fp, packed)

    # Native path: expects contiguous float32 / uint8 / float32 arrays.
    x = input_fp.contiguous().to(torch.float32)
    q = packed["q"].contiguous()
    scale = packed["scale"].contiguous().to(torch.float32)
    zp = packed["zp"].contiguous().to(torch.float32)
    rows, cols = q.shape
    out = torch.empty(x.shape[0], rows, dtype=torch.float32)
    lib.matmul_4bit_avx2(
        q.data_ptr(), x.data_ptr(), out.data_ptr(),
        scale.data_ptr(), zp.data_ptr(),
        rows, cols,
    )
    return out


def _matmul_torch(input_fp: torch.Tensor, packed: dict) -> torch.Tensor:
    from edgemoe.quantization.adaptive import AdaptiveQuantizer
    w = AdaptiveQuantizer().dequantize(packed).to(input_fp.device).to(input_fp.dtype)
    return input_fp @ w.T
