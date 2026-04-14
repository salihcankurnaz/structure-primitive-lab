from splab.execution.engine import run_engine
from splab.tasks.program_transform import program_transform_cases
from splab.tasks.proof_state import proof_state_cases
from splab.tasks.symbolic_rewrite import Add, Const, Mul, Var


def test_equivalence_heavy_case_prefers_egraph() -> None:
    expr = Add(Var("x"), Const(0))
    result = run_engine(expr)
    assert result.decision.primitive == "egraph_hybrid"
    assert "egraph_hybrid" in result.decision.candidate_costs


def test_hypergraph_or_graph_route_is_valid_for_nested_case() -> None:
    expr = Mul(Add(Const(0), Var("x")), Add(Const(2), Const(3)))
    result = run_engine(expr)
    assert result.decision.primitive in {"typed_hypergraph", "egraph_hybrid", "structure_graph"}


def test_program_transform_cases_prefer_structure_graph() -> None:
    for case in program_transform_cases():
        result = run_engine(case)
        best = min(result.decision.candidate_costs, key=result.decision.candidate_costs.get)
        assert result.decision.primitive == best
        assert "graph_bias" in result.decision.reason


def test_proof_state_cases_prefer_typed_hypergraph() -> None:
    for case in proof_state_cases():
        result = run_engine(case)
        best = min(result.decision.candidate_costs, key=result.decision.candidate_costs.get)
        assert result.decision.primitive == best
        assert "higher_arity_bias" in result.decision.reason
        assert len(result.decision.candidate_costs) == 3
