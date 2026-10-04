"""Custom Triton kernels for the frozen V2 structure workloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # pragma: no cover - exercised only in missing-runtime environments
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = exc
else:
    _TRITON_IMPORT_ERROR = None


@dataclass
class TritonPrepared:
    run: Callable[[], torch.Tensor]
    output: torch.Tensor


def _require_triton() -> None:
    if triton is None:
        raise RuntimeError(f"Triton is unavailable: {_TRITON_IMPORT_ERROR}")


if triton is not None:

    @triton.jit
    def _graph_sum_kernel(
        x_ptr,
        out_ptr,
        n,
        D: tl.constexpr,
        FANOUT: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for k in range(FANOUT):
            src = row - (k + 1)
            src = tl.where(src < 0, src + n, src)
            vals = tl.load(x_ptr + src * D + offs, mask=mask, other=0.0)
            acc += vals
        tl.store(out_ptr + row * D + offs, acc, mask=mask)


    @triton.jit
    def _hyper_mean_kernel(
        x_ptr,
        out_ptr,
        n,
        D: tl.constexpr,
        ARITY: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for k in range(ARITY):
            src = row + k
            src = tl.where(src >= n, src - n, src)
            vals = tl.load(x_ptr + src * D + offs, mask=mask, other=0.0)
            acc += vals
        acc = acc / ARITY
        tl.store(out_ptr + row * D + offs, acc, mask=mask)


    @triton.jit
    def _equivalence_mean_kernel(
        x_ptr,
        out_ptr,
        D: tl.constexpr,
        CLASS_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        cls = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D
        start = cls * CLASS_SIZE
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for k in range(CLASS_SIZE):
            vals = tl.load(x_ptr + (start + k) * D + offs, mask=mask, other=0.0)
            acc += vals
        acc = acc / CLASS_SIZE
        for k in range(CLASS_SIZE):
            tl.store(out_ptr + (start + k) * D + offs, acc, mask=mask)


def _features(size: int, feature_dim: int, device: torch.device) -> torch.Tensor:
    values = torch.arange(size * feature_dim, device=device, dtype=torch.float32)
    return (values.reshape(size, feature_dim).remainder(251.0)) / 251.0


def prepare_triton(
    family: str,
    *,
    size: int,
    feature_dim: int,
    fanout: int,
    arity: int,
    class_size: int,
    device: torch.device,
) -> TritonPrepared:
    _require_triton()
    if device.type != "cuda":
        raise RuntimeError("Triton V2 kernels require CUDA")
    if family == "equivalence" and size % class_size != 0:
        raise ValueError("V2 equivalence workload requires size divisible by class_size")

    x = _features(size, feature_dim, device)
    out = torch.empty_like(x)
    block_d = triton.next_power_of_2(feature_dim)

    if family == "graph":
        def run() -> torch.Tensor:
            _graph_sum_kernel[(size,)](
                x, out, size,
                D=feature_dim,
                FANOUT=fanout,
                BLOCK_D=block_d,
                num_warps=4,
            )
            return out
    elif family == "hypergraph":
        def run() -> torch.Tensor:
            _hyper_mean_kernel[(size,)](
                x, out, size,
                D=feature_dim,
                ARITY=arity,
                BLOCK_D=block_d,
                num_warps=4,
            )
            return out
    elif family == "equivalence":
        class_count = size // class_size
        def run() -> torch.Tensor:
            _equivalence_mean_kernel[(class_count,)](
                x, out,
                D=feature_dim,
                CLASS_SIZE=class_size,
                BLOCK_D=block_d,
                num_warps=4,
            )
            return out
    else:
        raise ValueError(f"unsupported family: {family}")

    return TritonPrepared(run=run, output=out)
