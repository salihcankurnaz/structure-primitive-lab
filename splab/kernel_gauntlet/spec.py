"""Frozen specifications for Primitive Kernel Gauntlet V2."""

from __future__ import annotations

from dataclasses import dataclass

FAMILIES: tuple[str, ...] = ("graph", "hypergraph", "equivalence")
G4_SIZES: tuple[int, ...] = (8192, 16384, 32768, 65536, 131072)


@dataclass(frozen=True)
class KernelWorkloadSpec:
    family: str
    size: int
    feature_dim: int = 64
    fanout: int = 8
    arity: int = 8
    class_size: int = 8

    def validate(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(self.family)
        if self.size < 2:
            raise ValueError("size must be >= 2")
        if self.feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if not 1 <= self.fanout < self.size:
            raise ValueError("invalid fanout")
        if not 1 <= self.arity <= self.size:
            raise ValueError("invalid arity")
        if not 1 <= self.class_size <= self.size:
            raise ValueError("invalid class_size")
        if self.family == "equivalence" and self.size % self.class_size != 0:
            raise ValueError("equivalence size must be divisible by class_size")


def g4_specs() -> tuple[KernelWorkloadSpec, ...]:
    out: list[KernelWorkloadSpec] = []
    for size in G4_SIZES:
        for family in FAMILIES:
            spec = KernelWorkloadSpec(family=family, size=size)
            spec.validate()
            out.append(spec)
    return tuple(out)
