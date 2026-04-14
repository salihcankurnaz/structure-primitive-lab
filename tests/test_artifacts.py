from pathlib import Path

from splab.eval.reports import summarize, write_report


def test_summary_contains_dense_baseline_fields() -> None:
    summary = summarize()
    assert summary["avg_dense_baseline_ms"] >= 0.0
    assert "symbolic_rewrite" in summary["family_speedups"]


def test_write_report_creates_json_artifact(tmp_path: Path) -> None:
    out_path = write_report(tmp_path)
    assert out_path.name == "structure_primitive_report.json"
    content = out_path.read_text(encoding="utf-8")
    assert "speedup_vs_dense" in content
    assert "family_speedups" in content
