"""Lightweight runtime calibration from prior benchmark artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


PRIMITIVES: tuple[str, ...] = ("typed_hypergraph", "egraph_hybrid", "structure_graph")


@dataclass(frozen=True)
class CalibrationModel:
    family_means: dict[str, dict[str, float]]
    signature_means: dict[str, dict[str, float]]
    signature_oracles: dict[str, str]
    signature_oracle_confidence: dict[str, float]
    family_oracles: dict[str, str]
    family_oracle_confidence: dict[str, float]

    def adjusted_cost(
        self,
        primitive: str,
        family: str | None,
        metadata: dict[str, float | int | bool | str] | None,
        base_cost: float,
    ) -> float:
        signature = metadata_signature(family, metadata or {})
        if signature and primitive in self.signature_means.get(signature, {}):
            return self.signature_means[signature][primitive]
        if family and primitive in self.family_means.get(family, {}):
            family_cost = self.family_means[family][primitive]
            return round(0.6 * base_cost + 0.4 * family_cost, 4)
        return base_cost


def _artifact_path() -> Path:
    return Path(__file__).resolve().parents[2] / "experiments" / "reports" / "structure_primitive_report.json"


def metadata_signature(family: str | None, metadata: dict[str, float | int | bool | str]) -> str:
    if not family:
        return ""
    markers: list[str] = [family]
    if bool(metadata.get("equivalence_heavy")):
        markers.append("eq")
    if bool(metadata.get("graph_preferred")):
        markers.append("graph")
    if bool(metadata.get("hypergraph_preferred")):
        markers.append("hyper")
    if bool(metadata.get("hyperedge_native")):
        markers.append("hyperedge")
    if bool(metadata.get("constraint_native")):
        markers.append("constraint")
    if bool(metadata.get("flow_heavy")):
        markers.append("flow")
    if bool(metadata.get("graph_native")):
        markers.append("graphnative")
    aggregation_bias = float(metadata.get("aggregation_bias", 0))
    if aggregation_bias >= 4:
        markers.append("agg4p")
    elif aggregation_bias >= 2:
        markers.append("agg2p")
    relation_arity = float(metadata.get("relation_arity", 0))
    if relation_arity >= 4:
        markers.append("arity4p")
    elif relation_arity > 2:
        markers.append("arity3")
    return "|".join(markers)


def load_calibration() -> CalibrationModel | None:
    path = _artifact_path()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return None
    bucket: dict[str, dict[str, list[float]]] = {}
    signature_bucket: dict[str, dict[str, list[float]]] = {}
    oracle_votes: dict[str, dict[str, int]] = {}
    family_oracle_votes: dict[str, dict[str, int]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        family = row.get("family")
        all_ms = row.get("all_primitive_ms")
        metadata = row.get("metadata")
        oracle = row.get("oracle_primitive")
        if not isinstance(family, str) or not isinstance(all_ms, dict):
            continue
        signature = metadata_signature(family, metadata if isinstance(metadata, dict) else {})
        for primitive in PRIMITIVES:
            value = all_ms.get(primitive)
            if isinstance(value, (int, float)) and value > 0:
                bucket.setdefault(family, {}).setdefault(primitive, []).append(float(value))
                signature_bucket.setdefault(signature, {}).setdefault(primitive, []).append(float(value))
        if isinstance(oracle, str) and oracle in PRIMITIVES:
            oracle_votes.setdefault(signature, {}).setdefault(oracle, 0)
            oracle_votes[signature][oracle] += 1
            family_oracle_votes.setdefault(family, {}).setdefault(oracle, 0)
            family_oracle_votes[family][oracle] += 1
    family_means: dict[str, dict[str, float]] = {}
    for family, per_primitive in bucket.items():
        family_means[family] = {
            primitive: round(sum(values) / len(values), 4)
            for primitive, values in per_primitive.items()
            if values
        }
    signature_means: dict[str, dict[str, float]] = {}
    for signature, per_primitive in signature_bucket.items():
        signature_means[signature] = {
            primitive: round(sum(values) / len(values), 4)
            for primitive, values in per_primitive.items()
            if values
        }
    signature_oracles: dict[str, str] = {}
    signature_oracle_confidence: dict[str, float] = {}
    for signature, votes in oracle_votes.items():
        if not votes:
            continue
        winner, winner_count = max(votes.items(), key=lambda item: item[1])
        total = sum(votes.values())
        signature_oracles[signature] = winner
        signature_oracle_confidence[signature] = round(winner_count / total, 4)
    family_oracles: dict[str, str] = {}
    family_oracle_confidence: dict[str, float] = {}
    for family, votes in family_oracle_votes.items():
        if not votes:
            continue
        winner, winner_count = max(votes.items(), key=lambda item: item[1])
        total = sum(votes.values())
        family_oracles[family] = winner
        family_oracle_confidence[family] = round(winner_count / total, 4)
    if not family_means and not signature_means:
        return None
    return CalibrationModel(
        family_means=family_means,
        signature_means=signature_means,
        signature_oracles=signature_oracles,
        signature_oracle_confidence=signature_oracle_confidence,
        family_oracles=family_oracles,
        family_oracle_confidence=family_oracle_confidence,
    )
