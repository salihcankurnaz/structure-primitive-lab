"""Minimal backend registry."""

from __future__ import annotations


def available_backends() -> tuple[str, ...]:
    return ("python", "torch", "triton", "custom_sparse", "dense_attention", "message_passing", "gather_scatter", "sparse_tensor")
