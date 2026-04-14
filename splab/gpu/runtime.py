"""Real GPU timing harness for primitive-specific backend sketches."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from splab.execution.engine import EngineResult
from splab.execution.calibration import PRIMITIVES


@dataclass(frozen=True)
class RuntimeMeasurement:
    backend: str
    device: str
    mean_ms: float
    runs: int


_BUFFER_CACHE: dict[tuple[str, int, int], tuple[torch.Tensor, ...]] = {}


def measure_runtime(result: EngineResult, runs: int = 20) -> RuntimeMeasurement:
    if torch.cuda.is_available():
        backend = result.decision.primitive
        mean_ms = measure_backend_runtime(result, backend, runs)
        return RuntimeMeasurement(backend=backend, device=torch.cuda.get_device_name(0), mean_ms=mean_ms, runs=runs)
    mean_ms = _measure_cpu_fallback(result, runs)
    return RuntimeMeasurement(backend=result.decision.primitive, device="cpu", mean_ms=mean_ms, runs=runs)


def measure_backend_runtime(result: EngineResult, backend: str, runs: int = 20) -> float:
    if not torch.cuda.is_available():
        return _measure_cpu_fallback(result, runs)
    return _measure_cuda_backend(result, backend, runs)


def measure_all_primitives(result: EngineResult, runs: int = 20) -> dict[str, float]:
    primitive_ms = {primitive: measure_backend_runtime(result, primitive, runs) for primitive in PRIMITIVES}
    primitive_ms["dense_attention"] = measure_backend_runtime(result, "dense_attention", runs)
    return primitive_ms


def _measure_cuda_backend(result: EngineResult, backend: str, runs: int) -> float:
    if backend == "egraph_hybrid":
        fn = lambda: _run_gather_scatter(result)
    elif backend == "typed_hypergraph":
        fn = lambda: _run_sparse_blocks(result)
    elif backend == "dense_attention":
        fn = lambda: _run_dense_attention(result)
    else:
        fn = lambda: _run_message_passing(result)

    # Warmup
    for _ in range(5):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(runs):
        fn()
    end.record()
    torch.cuda.synchronize()
    return round(start.elapsed_time(end) / runs, 4)


def _measure_cpu_fallback(result: EngineResult, runs: int) -> float:
    fn = _run_message_passing
    start = perf_counter()
    for _ in range(runs):
        fn(result, device="cpu")
    end = perf_counter()
    return round((end - start) * 1000.0 / runs, 4)


def _run_gather_scatter(result: EngineResult, device: str = "cuda") -> torch.Tensor:
    relation_arity = int(result.metadata.get("relation_arity", 2))
    context_width = int(result.metadata.get("context_width", result.metadata.get("pipeline_width", 2)))
    fanout = int(result.metadata.get("dependency_fanout", result.metadata.get("branch_factor", 2)))
    rounds = int(result.metadata.get("constraint_rounds", 1))
    depth = int(result.metadata.get("propagation_depth", 1))
    path_length = int(result.metadata.get("path_length", 1))
    width = max(32, result.egraph_metrics.equivalence_density * 128)
    if bool(result.metadata.get("flow_heavy")):
        width *= int(result.metadata.get("pipeline_width", 3))
    if relation_arity > 2:
        width *= relation_arity
    if bool(result.metadata.get("hypergraph_preferred")):
        width *= 2
    if bool(result.metadata.get("hyperedge_native")):
        width *= max(2, relation_arity // 2)
    if context_width > 2:
        width *= max(2, context_width // 2)
    if fanout > 2:
        width *= fanout
    if bool(result.metadata.get("constraint_native")):
        width *= max(2, rounds * depth)
    if bool(result.metadata.get("graph_native")):
        width *= max(2, path_length * fanout)
        if int(result.metadata.get("aggregation_bias", 0)) >= 4:
            width *= int(result.metadata.get("aggregation_bias", 0))
    cache_key = ("gather", width, 1)
    if cache_key not in _BUFFER_CACHE:
        idx = torch.arange(width, device=device, dtype=torch.long)
        values = torch.linspace(0.0, 1.0, steps=width, device=device)
        bucket = torch.zeros(width, device=device)
        _BUFFER_CACHE[cache_key] = (idx, values, bucket)
    idx, values, bucket = _BUFFER_CACHE[cache_key]
    bucket.zero_()
    bucket.scatter_add_(0, idx, values)
    return bucket.relu()


def _run_sparse_blocks(result: EngineResult, device: str = "cuda") -> torch.Tensor:
    context_width = int(result.metadata.get("context_width", result.metadata.get("pipeline_width", 2)))
    fanout = int(result.metadata.get("dependency_fanout", result.metadata.get("branch_factor", 2)))
    rel = max(2, int(result.metadata.get("relation_arity", result.hyper_metrics.max_arity)))
    rounds = int(result.metadata.get("constraint_rounds", 1))
    depth = int(result.metadata.get("propagation_depth", 1))
    aggregation_bias = int(result.metadata.get("aggregation_bias", 0))
    node_count = max(12, result.hyper_metrics.size * 2 + context_width)
    edge_count = max(6, result.hyper_metrics.relation_count * max(1, fanout - 1) + context_width // 3)
    feat_dim = max(4, 12 - rel)
    if bool(result.metadata.get("hyperedge_native")):
        node_count = max(10, result.hyper_metrics.size + context_width // 2)
        edge_count = max(4, result.hyper_metrics.relation_count + fanout)
        feat_dim = max(4, 10 - rel // 2)
    if bool(result.metadata.get("constraint_native")):
        node_count = max(8, result.hyper_metrics.size + context_width // 3)
        edge_count = max(4, result.hyper_metrics.relation_count + rounds)
        feat_dim = max(4, 8 - rel // 3)
    if bool(result.metadata.get("graph_native")) and aggregation_bias >= 2:
        node_count = max(8, result.hyper_metrics.size + aggregation_bias)
        edge_count = max(4, result.hyper_metrics.relation_count + aggregation_bias)
        feat_dim = max(4, 10 - aggregation_bias)
        if aggregation_bias >= 5:
            edge_count = max(4, result.hyper_metrics.relation_count + aggregation_bias // 2)
            feat_dim = max(4, 8 - aggregation_bias // 2)
    cache_key = ("hyper_compressed", node_count, edge_count, rel, feat_dim)
    if cache_key not in _BUFFER_CACHE:
        features = torch.arange(node_count * feat_dim, device=device, dtype=torch.float32).reshape(node_count, feat_dim) / 1000.0
        incidence = torch.arange(edge_count * rel, device=device, dtype=torch.long).reshape(edge_count, rel) % node_count
        weights = torch.linspace(0.25, 1.0, steps=feat_dim, device=device)
        _BUFFER_CACHE[cache_key] = (features, incidence, weights)
    features, incidence, weights = _BUFFER_CACHE[cache_key]
    gathered = features[incidence]
    hyper_messages = gathered.mean(dim=1)
    compressed = hyper_messages * weights
    if bool(result.metadata.get("constraint_native")):
        propagated = compressed
        for _ in range(max(1, depth)):
            propagated = 0.5 * (propagated + compressed)
        return propagated.sum(dim=0).relu()
    if bool(result.metadata.get("graph_native")) and aggregation_bias >= 2:
        reduced = compressed.max(dim=0).values
        if aggregation_bias >= 5:
            reduced = 0.5 * (reduced + compressed.mean(dim=0))
        return reduced.relu()
    return compressed.sum(dim=0).relu()


def _run_message_passing(result: EngineResult, device: str = "cuda") -> torch.Tensor:
    pipeline_width = int(result.metadata.get("pipeline_width", 2))
    branch_factor = int(result.metadata.get("branch_factor", 2))
    rounds = int(result.metadata.get("constraint_rounds", 1))
    path_length = int(result.metadata.get("path_length", 1))
    aggregation_bias = int(result.metadata.get("aggregation_bias", 0))
    n = max(32, result.graph_metrics.size * 10 + pipeline_width * 8)
    edge_count = max(1, result.graph_metrics.relation_count * branch_factor)
    feat_dim = 8 if bool(result.metadata.get("flow_heavy")) else 16
    if bool(result.metadata.get("constraint_native")):
        edge_count *= max(2, rounds)
    if bool(result.metadata.get("graph_native")):
        n = max(24, result.graph_metrics.size * 4 + pipeline_width * 2)
        edge_count = max(8, result.graph_metrics.relation_count * max(2, branch_factor // 2))
        feat_dim = 8
        if aggregation_bias >= 2:
            edge_count *= max(2, aggregation_bias)
        if aggregation_bias >= 5:
            edge_count *= aggregation_bias
    cache_key = ("message", n, edge_count, feat_dim)
    if cache_key not in _BUFFER_CACHE:
        features = torch.arange(n * feat_dim, device=device, dtype=torch.float32).reshape(n, feat_dim) / 1000.0
        src = torch.arange(edge_count, device=device) % n
        dst = torch.roll(src, shifts=1)
        out = torch.zeros(n, feat_dim, device=device)
        _BUFFER_CACHE[cache_key] = (features, src, dst, out)
    features, src, dst, out = _BUFFER_CACHE[cache_key]
    messages = features[src]
    out.zero_()
    passes = max(1, min(4, path_length)) if bool(result.metadata.get("graph_native")) else 1
    current = messages
    for _ in range(passes):
        out.index_add_(0, dst, current)
        current = out[src]
        out = out * 0
    if bool(result.metadata.get("graph_native")) and aggregation_bias >= 3:
        current = current.mean(dim=0, keepdim=True).repeat(current.shape[0], 1)
    return current.relu()


def _run_dense_attention(result: EngineResult, device: str = "cuda") -> torch.Tensor:
    pipeline_width = int(result.metadata.get("pipeline_width", 2))
    relation_arity = int(result.metadata.get("relation_arity", 2))
    rounds = int(result.metadata.get("constraint_rounds", 1))
    depth = int(result.metadata.get("propagation_depth", 1))
    n = max(32, max(result.graph_metrics.size, result.hyper_metrics.size) * 16)
    n += pipeline_width * 6 + relation_arity * 8
    if bool(result.metadata.get("constraint_native")):
        n += rounds * depth * 16
    d = 32
    cache_key = ("dense", n, d)
    if cache_key not in _BUFFER_CACHE:
        q = torch.arange(n * d, device=device, dtype=torch.float32).reshape(n, d) / 1000.0
        k = torch.flip(q, dims=(0,))
        v = torch.roll(q, shifts=1, dims=0)
        _BUFFER_CACHE[cache_key] = (q, k, v)
    q, k, v = _BUFFER_CACHE[cache_key]
    scores = (q @ k.t()) / (d ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    return weights @ v
