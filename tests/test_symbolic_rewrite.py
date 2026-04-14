from splab.execution.rewrite_executor import normalize_expr
from splab.tasks.symbolic_rewrite import pretty_expr, symbolic_cases


def test_exact_rewrite_matches_expected_cases() -> None:
    for case in symbolic_cases():
        result = normalize_expr(case.expr)
        assert pretty_expr(result.expr) == pretty_expr(case.expected)
