"""Primitive frontier advisor derived from persisted benchmark reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from splab.tasks.task_case import TaskCase


@dataclass(frozen=True)
class FrontierAdvice:
    primitive: str
    confidence: float
    reason: str


def _load_report_summary() -> dict[str, object] | None:
    report_path = Path(__file__).resolve().parents[2] / "experiments" / "reports" / "structure_primitive_report.json"
    if not report_path.exists():
        return None
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = payload.get("summary")
    return summary if isinstance(summary, dict) else None


def advise_primitive(task: TaskCase) -> FrontierAdvice | None:
    summary = _load_report_summary()
    if not summary:
        return None
    family = task.family
    family_wins = summary.get("family_wins", {})
    family_frontier = summary.get("family_frontier", {})
    if not isinstance(family_wins, dict) or not isinstance(family_frontier, dict):
        return None
    wins = family_wins.get(family)
    frontier = family_frontier.get(family)
    if not isinstance(wins, dict) or not isinstance(frontier, dict) or not wins:
        return None
    winner, winner_count = max(wins.items(), key=lambda item: item[1])
    total = sum(int(count) for count in wins.values())
    confidence = round(winner_count / max(1, total), 4)
    frontier_sorted = sorted(
        ((primitive, float(value)) for primitive, value in frontier.items()),
        key=lambda item: item[1],
    )
    if not frontier_sorted:
        return None
    best_frontier_primitive, best_frontier_ms = frontier_sorted[0]
    if best_frontier_primitive == winner:
        return FrontierAdvice(
            primitive=winner,
            confidence=confidence,
            reason=f"family_win_and_frontier:{family}:{best_frontier_ms}",
        )
    return FrontierAdvice(
        primitive=best_frontier_primitive,
        confidence=confidence,
        reason=f"frontier_override:{family}:{best_frontier_ms}",
    )
