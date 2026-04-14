"""Exact symbolic rewrite executor."""

from __future__ import annotations

from dataclasses import dataclass

from splab.tasks.symbolic_rewrite import Add, Const, Expr, Mul, pretty_expr


@dataclass(frozen=True)
class RewriteResult:
    expr: Expr
    steps: tuple[str, ...]


def normalize_expr(expr: Expr) -> RewriteResult:
    steps: list[str] = []
    current = expr
    while True:
        next_expr = _rewrite(current, steps)
        if next_expr == current:
            return RewriteResult(expr=current, steps=tuple(steps))
        current = next_expr


def _rewrite(expr: Expr, steps: list[str]) -> Expr:
    if isinstance(expr, Const):
        return expr
    if expr.__class__.__name__ == "Var":
        return expr
    if isinstance(expr, Add):
        left = _rewrite(expr.left, steps)
        right = _rewrite(expr.right, steps)
        if isinstance(left, Const) and left.value == 0:
            steps.append("add.zero_left")
            return right
        if isinstance(right, Const) and right.value == 0:
            steps.append("add.zero_right")
            return left
        if isinstance(left, Const) and isinstance(right, Const):
            steps.append("add.const_fold")
            return Const(left.value + right.value)
        terms = _flatten_add(left) + _flatten_add(right)
        rebuilt = _rebuild_add(sorted(terms, key=pretty_expr))
        if rebuilt != Add(left, right):
            steps.append("add.canonicalize")
        return rebuilt
    if isinstance(expr, Mul):
        left = _rewrite(expr.left, steps)
        right = _rewrite(expr.right, steps)
        if isinstance(left, Const) and left.value == 0:
            steps.append("mul.zero_left")
            return Const(0)
        if isinstance(right, Const) and right.value == 0:
            steps.append("mul.zero_right")
            return Const(0)
        if isinstance(left, Const) and left.value == 1:
            steps.append("mul.one_left")
            return right
        if isinstance(right, Const) and right.value == 1:
            steps.append("mul.one_right")
            return left
        if isinstance(left, Const) and isinstance(right, Const):
            steps.append("mul.const_fold")
            return Const(left.value * right.value)
        factors = _flatten_mul(left) + _flatten_mul(right)
        const_value = 1
        non_consts: list[Expr] = []
        for factor in factors:
            if isinstance(factor, Const):
                const_value *= factor.value
            else:
                non_consts.append(factor)
        if const_value == 0:
            steps.append("mul.zero_product")
            return Const(0)
        ordered = sorted(non_consts, key=pretty_expr)
        if const_value != 1:
            ordered.insert(0, Const(const_value))
        rebuilt = _rebuild_mul(ordered) if ordered else Const(1)
        if rebuilt != Mul(left, right):
            steps.append("mul.canonicalize")
        return rebuilt
    raise TypeError(f"Unsupported expr: {type(expr)!r}")


def _flatten_add(expr: Expr) -> list[Expr]:
    if isinstance(expr, Add):
        return _flatten_add(expr.left) + _flatten_add(expr.right)
    return [expr]


def _flatten_mul(expr: Expr) -> list[Expr]:
    if isinstance(expr, Mul):
        return _flatten_mul(expr.left) + _flatten_mul(expr.right)
    return [expr]


def _rebuild_add(terms: list[Expr]) -> Expr:
    if not terms:
        return Const(0)
    result = terms[0]
    for term in terms[1:]:
        result = Add(result, term)
    return result


def _rebuild_mul(terms: list[Expr]) -> Expr:
    if not terms:
        return Const(1)
    result = terms[0]
    for term in terms[1:]:
        result = Mul(result, term)
    return result
