"""Typed hypergraph representation for symbolic expressions."""

from __future__ import annotations

from dataclasses import dataclass

from splab.tasks.symbolic_rewrite import Add, Const, Expr, Mul


@dataclass(frozen=True)
class HyperNode:
    node_id: int
    label: str
    node_type: str


@dataclass(frozen=True)
class HyperEdge:
    edge_id: int
    relation: str
    inputs: tuple[int, ...]
    output: int


@dataclass(frozen=True)
class TypedHypergraph:
    nodes: tuple[HyperNode, ...]
    edges: tuple[HyperEdge, ...]
    root: int

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def max_arity(self) -> int:
        return max((len(edge.inputs) for edge in self.edges), default=0)


def build_typed_hypergraph(expr: Expr) -> TypedHypergraph:
    nodes: list[HyperNode] = []
    edges: list[HyperEdge] = []
    next_node = 0
    next_edge = 0

    def walk(current: Expr) -> int:
        nonlocal next_node, next_edge
        if current.__class__.__name__ == "Var":
            node_id = next_node
            next_node += 1
            nodes.append(HyperNode(node_id, current.name, "var"))
            return node_id
        if isinstance(current, Const):
            node_id = next_node
            next_node += 1
            nodes.append(HyperNode(node_id, str(current.value), "const"))
            return node_id
        left_id = walk(current.left)
        right_id = walk(current.right)
        node_id = next_node
        next_node += 1
        label = "add" if isinstance(current, Add) else "mul"
        nodes.append(HyperNode(node_id, label, "op"))
        edges.append(HyperEdge(next_edge, label, (left_id, right_id), node_id))
        next_edge += 1
        return node_id

    root = walk(expr)
    return TypedHypergraph(nodes=tuple(nodes), edges=tuple(edges), root=root)
