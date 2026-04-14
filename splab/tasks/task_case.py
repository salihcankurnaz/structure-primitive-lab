"""Shared task case model."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskCase:
    name: str
    family: str
    expr: object
    expected: object
    metadata: dict[str, float | int | bool | str] = field(default_factory=dict)
