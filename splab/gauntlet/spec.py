"""Frozen workload specifications for Primitive Gauntlet V1."""

from __future__ import annotations

from dataclasses import dataclass

FAMILIES: tuple[str, ...] = ("graph", "hypergraph", "equivalence")

_PROFILE_SIZES: dict[str, tuple[int, ...]] = {
    "smoke": (64, 128),
    "l4": (512, 2048, 4096),
    "a100": (1024, 4096, 8192),
    "g4": (1024, 4096, 8192, 16384),
}


@dataclass(frozen=True)
class WorkloadSpec:
    family: str
    size: int
    feature_dim: int = 64
    fanout: int = 8
    arity: int = 8
    class_size: int = 8

    def validate(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unsupported family: {self.family}")
        if self.size < 2:
            raise ValueError("size must be >= 2")
        if self.feature_dim < 1:
            raise ValueError("feature_dim must be >= 1")
        if not 1 <= self.fanout < self.size:
            raise ValueError("fanout must satisfy 1 <= fanout < size")
        if not 1 <= self.arity <= self.size:
            raise ValueError("arity must satisfy 1 <= arity <= size")
        if not 1 <= self.class_size <= self.size:
            raise ValueError("class_size must satisfy 1 <= class_size <= size")


def profile_specs(
    profile: str,
    *,
    feature_dim: int = 64,
    fanout: int = 8,
    arity: int = 8,
    class_size: int = 8,
) -> tuple[WorkloadSpec, ...]:
    try:
        sizes = _PROFILE_SIZES[profile]
    except KeyError as exc:
        raise ValueError(f"unknown profile: {profile}") from exc

    specs: list[WorkloadSpec] = []
    for size in sizes:
        for family in FAMILIES:
            spec = WorkloadSpec(
                family=family,
                size=size,
                feature_dim=feature_dim,
                fanout=min(fanout, size - 1),
                arity=min(arity, size),
                class_size=min(class_size, size),
            )
            spec.validate()
            specs.append(spec)
    return tuple(specs)


def available_profiles() -> tuple[str, ...]:
    return tuple(_PROFILE_SIZES)
