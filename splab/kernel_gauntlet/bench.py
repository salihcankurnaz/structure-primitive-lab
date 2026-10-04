"""Benchmark harness for Primitive Kernel Gauntlet V2."""

from __future__ import annotations

import gc
import math
from dataclasses import asdict, dataclass
from statistics import median

import torch

from splab.gauntlet.spec import WorkloadSpec
from splab.gauntlet.workloads import prepare_dense, prepare_native
from splab.kernel_gauntlet.spec import KernelWorkloadSpec
from splab.kernels.triton_reductions import prepare_triton


@dataclass(frozen=True)
class Stats:
    backend: str
    median_ms: float
    p95_ms: float
    peak_memory_mb: float
    samples_ms: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _p95(values: list[float]) -> float:
    values = sorted(values)
    return values[max(0, math.ceil(0.95 * len(values)) - 1)]


def _cleanup(device: torch.device) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


def _prepare(spec: KernelWorkloadSpec, backend: str, device: torch.device):
    common = dict(
        family=spec.family,
        size=spec.size,
        feature_dim=spec.feature_dim,
        fanout=spec.fanout,
        arity=spec.arity,
        class_size=spec.class_size,
    )
    if backend == "triton":
        return prepare_triton(**common, device=device)
    v1 = WorkloadSpec(**common)
    return prepare_native(v1, device) if backend == "torch_native" else prepare_dense(v1, device)


def _precompile_triton(spec: KernelWorkloadSpec, device: torch.device) -> None:
    prepared = _prepare(spec, "triton", device)
    with torch.inference_mode():
        prepared.run()
    torch.cuda.synchronize(device)
    del prepared
    _cleanup(device)


def _measure(
    spec: KernelWorkloadSpec,
    backend: str,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[Stats, torch.Tensor]:
    _cleanup(device)
    torch.cuda.reset_peak_memory_stats(device)
    prepared = _prepare(spec, backend, device)

    with torch.inference_mode():
        for _ in range(warmup):
            prepared.run()
        torch.cuda.synchronize(device)

        events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            prepared.run()
            end.record()
            events.append((start, end))
        torch.cuda.synchronize(device)
        samples = [float(start.elapsed_time(end)) for start, end in events]
        output = prepared.run().detach().cpu().clone()

    peak = float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
    stats = Stats(
        backend=backend,
        median_ms=round(median(samples), 6),
        p95_ms=round(_p95(samples), 6),
        peak_memory_mb=round(peak, 4),
        samples_ms=tuple(round(v, 6) for v in samples),
    )
    del prepared
    _cleanup(device)
    return stats, output


def _merge(a: Stats, b: Stats) -> Stats:
    samples = list(a.samples_ms) + list(b.samples_ms)
    return Stats(
        backend=a.backend,
        median_ms=round(median(samples), 6),
        p95_ms=round(_p95(samples), 6),
        peak_memory_mb=round(max(a.peak_memory_mb, b.peak_memory_mb), 4),
        samples_ms=tuple(samples),
    )


def benchmark_row(
    spec: KernelWorkloadSpec,
    device: torch.device,
    *,
    warmup: int,
    repeats: int,
    dense_limit: int = 16384,
) -> dict[str, object]:
    _precompile_triton(spec, device)

    torch_a, torch_out = _measure(spec, "torch_native", device, warmup, repeats)
    triton_a, triton_out = _measure(spec, "triton", device, warmup, repeats)
    triton_b, _ = _measure(spec, "triton", device, warmup, repeats)
    torch_b, _ = _measure(spec, "torch_native", device, warmup, repeats)

    torch_stats = _merge(torch_a, torch_b)
    triton_stats = _merge(triton_a, triton_b)

    delta = (torch_out - triton_out).abs()
    max_abs_error = float(delta.max().item()) if delta.numel() else 0.0
    correct = bool(torch.allclose(torch_out, triton_out, atol=1e-4, rtol=1e-4))

    speedup = torch_stats.median_ms / triton_stats.median_ms
    memory_ratio = triton_stats.peak_memory_mb / torch_stats.peak_memory_mb

    dense_payload: dict[str, object] | None = None
    if spec.size <= dense_limit:
        dense_stats, dense_out = _measure(spec, "dense", device, warmup, repeats)
        dense_delta = (dense_out - torch_out).abs()
        dense_payload = {
            "stats": dense_stats.to_dict(),
            "max_abs_error_vs_torch_native": float(dense_delta.max().item()) if dense_delta.numel() else 0.0,
            "speedup_of_triton_vs_dense": round(dense_stats.median_ms / triton_stats.median_ms, 6),
        }

    return {
        "family": spec.family,
        "size": spec.size,
        "feature_dim": spec.feature_dim,
        "fanout": spec.fanout,
        "arity": spec.arity,
        "class_size": spec.class_size,
        "correct": correct,
        "max_abs_error": max_abs_error,
        "torch_native": torch_stats.to_dict(),
        "triton": triton_stats.to_dict(),
        "triton_speedup_vs_torch_native": round(speedup, 6),
        "triton_to_torch_peak_memory_ratio": round(memory_ratio, 6),
        "dense_context": dense_payload,
    }
