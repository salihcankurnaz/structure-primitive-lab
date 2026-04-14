"""Tiny symbolic rewrite task family."""

from __future__ import annotations

from dataclasses import dataclass

from splab.tasks.task_case import TaskCase


class Expr:
    """Base expression class."""


@dataclass(frozen=True)
class Var(Expr):
    name: str


@dataclass(frozen=True)
class Const(Expr):
    value: int


@dataclass(frozen=True)
class Add(Expr):
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Mul(Expr):
    left: Expr
    right: Expr


def symbolic_cases() -> tuple[TaskCase, ...]:
    x = Var("x")
    y = Var("y")
    z = Var("z")
    return (
        TaskCase("add_zero_right", "symbolic_rewrite", Add(x, Const(0)), x, {"equivalence_heavy": True}),
        TaskCase("mul_one_left", "symbolic_rewrite", Mul(Const(1), x), x, {"equivalence_heavy": True}),
        TaskCase("mul_zero_nested", "symbolic_rewrite", Add(Mul(x, Const(0)), y), y, {"equivalence_heavy": True}),
        TaskCase("const_fold", "symbolic_rewrite", Add(Const(2), Add(Const(3), Const(0))), Const(5), {"equivalence_heavy": True}),
        TaskCase(
            "commutative_canonical",
            "symbolic_rewrite",
            Add(z, Add(y, x)),
            Add(Add(x, y), z),
            {"equivalence_heavy": True},
        ),
        TaskCase(
            "mixed",
            "symbolic_rewrite",
            Mul(Add(Const(0), x), Add(Const(2), Const(3))),
            Mul(Const(5), x),
            {"equivalence_heavy": True},
        ),
    )


def expr_size(expr: Expr) -> int:
    if isinstance(expr, (Var, Const)):
        return 1
    if isinstance(expr, (Add, Mul)):
        return 1 + expr_size(expr.left) + expr_size(expr.right)
    raise TypeError(f"Unsupported expr: {type(expr)!r}")


def pretty_expr(expr: Expr) -> str:
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, Const):
        return str(expr.value)
    if isinstance(expr, Add):
        return f"add({pretty_expr(expr.left)},{pretty_expr(expr.right)})"
    if isinstance(expr, Mul):
        return f"mul({pretty_expr(expr.left)},{pretty_expr(expr.right)})"
    raise TypeError(f"Unsupported expr: {type(expr)!r}")
