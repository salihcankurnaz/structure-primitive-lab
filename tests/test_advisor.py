import splab.execution.advisor as advisor_module
from splab.tasks.constraint_native import constraint_native_cases
from splab.tasks.graph_native import graph_native_cases
from splab.tasks.symbolic_rewrite import symbolic_cases


def test_advisor_uses_family_frontier_from_deterministic_report(monkeypatch) -> None:
    summary = {
        "family_wins": {
            "symbolic_rewrite": {"egraph_hybrid": 5, "structure_graph": 1},
            "constraint_native": {"typed_hypergraph": 4, "structure_graph": 1},
            "graph_native": {"structure_graph": 4, "egraph_hybrid": 1},
        },
        "family_frontier": {
            "symbolic_rewrite": {
                "typed_hypergraph": 0.30,
                "egraph_hybrid": 0.10,
                "structure_graph": 0.20,
            },
            "constraint_native": {
                "typed_hypergraph": 0.20,
                "egraph_hybrid": 0.30,
                "structure_graph": 0.10,
            },
            "graph_native": {
                "typed_hypergraph": 0.25,
                "egraph_hybrid": 0.15,
                "structure_graph": 0.20,
            },
        },
    }
    monkeypatch.setattr(advisor_module, "_load_report_summary", lambda: summary)

    symbolic_advice = advisor_module.advise_primitive(symbolic_cases()[0])
    constraint_advice = advisor_module.advise_primitive(constraint_native_cases()[0])
    graph_advice = advisor_module.advise_primitive(graph_native_cases()[-1])

    assert symbolic_advice is not None
    assert symbolic_advice.primitive == "egraph_hybrid"

    assert constraint_advice is not None
    assert constraint_advice.primitive == "structure_graph"
    assert constraint_advice.reason.startswith("frontier_override:")

    assert graph_advice is not None
    assert graph_advice.primitive == "egraph_hybrid"
    assert graph_advice.reason.startswith("frontier_override:")
