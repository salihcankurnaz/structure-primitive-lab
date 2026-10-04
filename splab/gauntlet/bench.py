"""Timing and memory measurement for Primitive Gauntlet V1."""

from __future__ import annotations

import gc
import math
from dataclasses import asdict, dataclass
from statistics import median
from time import perf_counter

import torch

from splab.gauntlet.spec import WorkloadSpec
from splab.gauntlet.workloads import prepare_dense, prepare_native


@dataclass(frozen=True)
class BackendStats:
    backend: str
    median_ms: float
    p95_ms: float
    peak_memory_mb: float | None
    samples_ms: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def _cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _measure_block(
    spec: WorkloadSpec,
    backend: str,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[BackendStats, torch.Tensor]:
    _cleanup(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    prepared = prepare_dense(spec, device) if backend == "dense" else prepare_native(spec, device)

    with torch.inference_mode():
        for _ in range(warmup):
            prepared.run()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        samples: list[float] = []
        if device.type == "cuda":
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
            peak_mb: float | None = float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
        else:
            for _ in range(repeats):
                start = perf_counter()
                prepared.run()
                samples.append((perf_counter() - start) * 1000.0)
            peak_mb = None

        output = prepared.run().detach().cpu().clone()

    stats = BackendStats(
        backend=backend,
        median_ms=round(median(samples), 6),
        p95_ms=round(_p95(samples), 6),
        peak_memory_mb=round(peak_mb, 4) if peak_mb is not None else None,
        samples_ms=tuple(round(value, 6) for value in samples),
    )
    del prepared
    _cleanup(device)
    return stats, output


def _merge_stats(first: BackendStats, second: BackendStats) -> BackendStats:
    samples = list(first.samples_ms) + list(second.samples_ms)
    peaks = [value for value in (first.peak_memory_mb, second.peak_memory_mb) if value is not None]
    return BackendStats(
        backend=first.backend,
        median_ms=round(median(samples), 6),
        p95_ms=round(_p95(samples), 6),
        peak_memory_mb=round(max(peaks), 4) if peaks else None,
        samples_ms=tuple(samples),
    )


def benchmark_pair(
    spec: WorkloadSpec,
    device: torch.device,
    *,
    warmup: int = 5,
    repeats: int = 10,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> dict[str, object]:
    """Benchmark in ABBA order and compare semantically matched outputs."""

    dense_a, dense_output = _measure_block(spec, "dense", device, warmup, repeats)
    native_a, native_output = _measure_block(spec, "native", device, warmup, repeats)
    native_b, _ = _measure_block(spec, "native", device, warmup, repeats)
    dense_b, _ = _measure_block(spec, "dense", device, warmup, repeats)

    dense = _merge_stats(dense_a, dense_b)
    native = _merge_stats(native_a, native_b)

    if dense_output.shape != native_output.shape:
        max_abs_error = float("inf")
        correct = False
    else:
        delta = (dense_output - native_output).abs()
        max_abs_error = float(delta.max().item()) if delta.numel() else 0.0
        correct = bool(torch.allclose(dense_output, native_output, atol=atol, rtol=rtol))

    speedup = (dense.median_ms / native.median_ms) if native.median_ms > 0 else None
    memory_ratio = None
    if dense.peak_memory_mb and native.peak_memory_mb is not None:
        memory_ratio = native.peak_memory_mb / dense.peak_memory_mb

    return {
        "family": spec.family,
        "size": spec.size,
        "feature_dim": spec.feature_dim,
        "fanout": spec.fanout,
        "arity": spec.arity,
        "class_size": spec.class_size,
        "correct": correct,
        "max_abs_error": max_abs_error,
        "dense": dense.to_dict(),
        "native": native.to_dict(),
        "speedup_vs_dense": round(speedup, 6) if speedup is not None else None,
        "native_to_dense_peak_memory_ratio": round(memory_ratio, 6) if memory_ratio is not None else None,
    }
