from splab.eval.reports import write_report
from splab.execution.advisor import advise_primitive
from splab.tasks.constraint_native import constraint_native_cases
from splab.tasks.graph_native import graph_native_cases
from splab.tasks.symbolic_rewrite import symbolic_cases


def test_advisor_uses_family_frontier_after_report() -> None:
    write_report(bootstrap_rounds=2)
    symbolic_advice = advise_primitive(symbolic_cases()[0])
    constraint_advice = advise_primitive(constraint_native_cases()[0])
    graph_advice = advise_primitive(graph_native_cases()[-1])
    assert symbolic_advice is not None
    assert symbolic_advice.primitive == "egraph_hybrid"
    assert constraint_advice is not None
    assert constraint_advice.primitive == "structure_graph"
    assert graph_advice is not None
    assert graph_advice.primitive in {"typed_hypergraph", "egraph_hybrid"}
