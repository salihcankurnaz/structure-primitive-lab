"""Semantically matched dense and structure-native workloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from splab.gauntlet.spec import WorkloadSpec


@dataclass
class PreparedWorkload:
    run: Callable[[], torch.Tensor]
    output: torch.Tensor


def _features(spec: WorkloadSpec, device: torch.device) -> torch.Tensor:
    values = torch.arange(
        spec.size * spec.feature_dim,
        device=device,
        dtype=torch.float32,
    ).reshape(spec.size, spec.feature_dim)
    return (values.remainder(251.0)) / 251.0


def _graph_indices(spec: WorkloadSpec, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    src = torch.arange(spec.size, device=device, dtype=torch.long).repeat_interleave(spec.fanout)
    offsets = torch.arange(1, spec.fanout + 1, device=device, dtype=torch.long).repeat(spec.size)
    dst = (src + offsets).remainder(spec.size)
    return src, dst


def _hypergraph_incidence(spec: WorkloadSpec, device: torch.device) -> torch.Tensor:
    base = torch.arange(spec.size, device=device, dtype=torch.long).unsqueeze(1)
    offsets = torch.arange(spec.arity, device=device, dtype=torch.long).unsqueeze(0)
    return (base + offsets).remainder(spec.size)


def _class_index(spec: WorkloadSpec, device: torch.device) -> torch.Tensor:
    return torch.arange(spec.size, device=device, dtype=torch.long) // spec.class_size


def prepare_dense(spec: WorkloadSpec, device: torch.device) -> PreparedWorkload:
    spec.validate()
    if spec.family == "graph":
        return _prepare_graph_dense(spec, device)
    if spec.family == "hypergraph":
        return _prepare_hypergraph_dense(spec, device)
    if spec.family == "equivalence":
        return _prepare_equivalence_dense(spec, device)
    raise ValueError(spec.family)


def prepare_native(spec: WorkloadSpec, device: torch.device) -> PreparedWorkload:
    spec.validate()
    if spec.family == "graph":
        return _prepare_graph_native(spec, device)
    if spec.family == "hypergraph":
        return _prepare_hypergraph_native(spec, device)
    if spec.family == "equivalence":
        return _prepare_equivalence_native(spec, device)
    raise ValueError(spec.family)


def _prepare_graph_dense(spec: WorkloadSpec, device: torch.device) -> PreparedWorkload:
    x = _features(spec, device)
    src, dst = _graph_indices(spec, device)
    adjacency = torch.zeros(spec.size, spec.size, device=device, dtype=torch.float32)
    adjacency[dst, src] = 1.0
    out = torch.empty(spec.size, spec.feature_dim, device=device, dtype=torch.float32)

    def run() -> torch.Tensor:
        torch.mm(adjacency, x, out=out)
        return out

    return PreparedWorkload(run=run, output=out)


def _prepare_graph_native(spec: WorkloadSpec, device: torch.device) -> PreparedWorkload:
    x = _features(spec, device)
    src, dst = _graph_indices(spec, device)
    messages = torch.empty(
        src.numel(),
        spec.feature_dim,
        device=device,
        dtype=torch.float32,
    )
    out = torch.empty(spec.size, spec.feature_dim, device=device, dtype=torch.float32)

    def run() -> torch.Tensor:
        torch.index_select(x, 0, src, out=messages)
        out.zero_()
        out.index_add_(0, dst, messages)
        return out

    return PreparedWorkload(run=run, output=out)


def _prepare_hypergraph_dense(spec: WorkloadSpec, device: torch.device) -> PreparedWorkload:
    x = _features(spec, device)
    incidence_index = _hypergraph_incidence(spec, device)
    incidence = torch.zeros(spec.size, spec.size, device=device, dtype=torch.float32)
    rows = torch.arange(spec.size, device=device, dtype=torch.long).unsqueeze(1).expand_as(incidence_index)
    incidence[rows, incidence_index] = 1.0 / float(spec.arity)
    out = torch.empty(spec.size, spec.feature_dim, device=device, dtype=torch.float32)

    def run() -> torch.Tensor:
        torch.mm(incidence, x, out=out)
        return out

    return PreparedWorkload(run=run, output=out)


def _prepare_hypergraph_native(spec: WorkloadSpec, device: torch.device) -> PreparedWorkload:
    x = _features(spec, device)
    incidence_index = _hypergraph_incidence(spec, device)
    flat_index = incidence_index.reshape(-1)
    gathered = torch.empty(
        flat_index.numel(),
        spec.feature_dim,
        device=device,
        dtype=torch.float32,
    )
    out = torch.empty(spec.size, spec.feature_dim, device=device, dtype=torch.float32)

    def run() -> torch.Tensor:
        torch.index_select(x, 0, flat_index, out=gathered)
        torch.mean(gathered.view(spec.size, spec.arity, spec.feature_dim), dim=1, out=out)
        return out

    return PreparedWorkload(run=run, output=out)


def _prepare_equivalence_dense(spec: WorkloadSpec, device: torch.device) -> PreparedWorkload:
    x = _features(spec, device)
    class_of = _class_index(spec, device)
    class_count = int(class_of.max().item()) + 1
    membership = torch.zeros(class_count, spec.size, device=device, dtype=torch.float32)
    membership[class_of, torch.arange(spec.size, device=device, dtype=torch.long)] = 1.0
    counts = membership.sum(dim=1, keepdim=True).clamp_min_(1.0)
    class_means = torch.empty(class_count, spec.feature_dim, device=device, dtype=torch.float32)
    out = torch.empty(spec.size, spec.feature_dim, device=device, dtype=torch.float32)

    def run() -> torch.Tensor:
        torch.mm(membership, x, out=class_means)
        class_means.div_(counts)
        torch.mm(membership.t(), class_means, out=out)
        return out

    return PreparedWorkload(run=run, output=out)


def _prepare_equivalence_native(spec: WorkloadSpec, device: torch.device) -> PreparedWorkload:
    x = _features(spec, device)
    class_of = _class_index(spec, device)
    class_count = int(class_of.max().item()) + 1
    class_sums = torch.empty(class_count, spec.feature_dim, device=device, dtype=torch.float32)
    class_means = torch.empty_like(class_sums)
    counts = torch.bincount(class_of, minlength=class_count).to(torch.float32).unsqueeze(1).clamp_min_(1.0)
    out = torch.empty(spec.size, spec.feature_dim, device=device, dtype=torch.float32)

    def run() -> torch.Tensor:
        class_sums.zero_()
        class_sums.index_add_(0, class_of, x)
        torch.div(class_sums, counts, out=class_means)
        torch.index_select(class_means, 0, class_of, out=out)
        return out

    return PreparedWorkload(run=run, output=out)
