"""Binary structure graph representation."""

from __future__ import annotations

from dataclasses import dataclass

from splab.tasks.symbolic_rewrite import Add, Const, Expr, Mul


@dataclass(frozen=True)
class GraphNode:
    node_id: int
    label: str


@dataclass(frozen=True)
class GraphEdge:
    source: int
    target: int
    label: str


@dataclass(frozen=True)
class StructureGraph:
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    root: int

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)


def build_structure_graph(expr: Expr) -> StructureGraph:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    next_node = 0

    def walk(current: Expr) -> int:
        nonlocal next_node
        if current.__class__.__name__ == "Var":
            node_id = next_node
            next_node += 1
            nodes.append(GraphNode(node_id, current.name))
            return node_id
        if isinstance(current, Const):
            node_id = next_node
            next_node += 1
            nodes.append(GraphNode(node_id, str(current.value)))
            return node_id
        label = "add" if isinstance(current, Add) else "mul"
        node_id = next_node
        next_node += 1
        nodes.append(GraphNode(node_id, label))
        left_id = walk(current.left)
        right_id = walk(current.right)
        edges.append(GraphEdge(node_id, left_id, "left"))
        edges.append(GraphEdge(node_id, right_id, "right"))
        return node_id

    root = walk(expr)
    return StructureGraph(nodes=tuple(nodes), edges=tuple(edges), root=root)
