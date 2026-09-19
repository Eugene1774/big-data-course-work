from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Set


class HypergraphIndex:
    """
    Lightweight in-memory hypergraph index:
    - edge -> nodes: read members by hyperedge
    - node -> edges: read incident hyperedges by node
    - edge -> role -> nodes: optional role-aware participants
    """

    def __init__(self) -> None:
        self._edge_to_nodes: Dict[str, Set[str]] = {}
        self._node_to_edges: Dict[str, Set[str]] = defaultdict(set)
        self._edge_to_roles: Dict[str, Dict[str, Set[str]]] = {}

    @staticmethod
    def _normalize_id(value: str) -> str:
        return str(value or "").strip()

    def add_node(self, node_id: str) -> None:
        """Register an isolated node explicitly."""
        node = self._normalize_id(node_id)
        if not node:
            raise ValueError("node_id cannot be empty")
        _ = self._node_to_edges[node]

    def _detach_edge_from_nodes(self, edge: str, nodes: Iterable[str]) -> None:
        for node in nodes:
            edge_bucket = self._node_to_edges.get(node)
            if edge_bucket is None:
                continue
            edge_bucket.discard(edge)
            # Prune empty buckets created by historical edge updates.
            if not edge_bucket:
                self._node_to_edges.pop(node, None)

    def add_hyperedge(self, edge_id: str, node_ids: Iterable[str]) -> None:
        """
        Add or overwrite a hyperedge.
        - Duplicate node IDs are deduplicated.
        - Overwriting the same edge cleans old reverse indexes first.
        """
        edge = self._normalize_id(edge_id)
        if not edge:
            raise ValueError("edge_id cannot be empty")

        normalized_nodes = {self._normalize_id(node_id) for node_id in node_ids}
        normalized_nodes.discard("")
        if not normalized_nodes:
            raise ValueError("hyperedge must contain at least one valid node")

        old_nodes = self._edge_to_nodes.get(edge, set())
        self._detach_edge_from_nodes(edge, old_nodes)

        self._edge_to_nodes[edge] = set(normalized_nodes)
        for node in normalized_nodes:
            self._node_to_edges[node].add(edge)

        # Plain add_hyperedge means role-less edge semantics.
        self._edge_to_roles.pop(edge, None)

    def add_role_hyperedge(self, edge_id: str, participants: Mapping[str, Iterable[str] | str]) -> None:
        """
        Add a role-aware hyperedge, preserving multi-entity semantics.
        Example:
            {"topic": ["融资", "估值"], "mentor": "mentor_001", "student": ["stu_a", "stu_b"]}
        """
        role_map: Dict[str, Set[str]] = {}
        for raw_role, raw_value in participants.items():
            role = self._normalize_id(raw_role)
            if not role:
                continue

            if isinstance(raw_value, str):
                values = [raw_value]
            else:
                values = list(raw_value)

            normalized_values = {self._normalize_id(value) for value in values}
            normalized_values.discard("")
            if normalized_values:
                role_map[role] = normalized_values

        member_nodes = {node for values in role_map.values() for node in values}
        if not member_nodes:
            raise ValueError("role hyperedge must contain at least one valid node")

        self.add_hyperedge(edge_id, member_nodes)
        self._edge_to_roles[self._normalize_id(edge_id)] = role_map

    def remove_hyperedge(self, edge_id: str) -> None:
        """Remove a hyperedge and clean reverse mappings."""
        edge = self._normalize_id(edge_id)
        if not edge:
            return

        old_nodes = self._edge_to_nodes.pop(edge, set())
        self._detach_edge_from_nodes(edge, old_nodes)
        self._edge_to_roles.pop(edge, None)

    def neighbors(self, node_id: str) -> Set[str]:
        """Return all nodes sharing at least one hyperedge with the target node."""
        node = self._normalize_id(node_id)
        if not node:
            return set()

        related_edges = self._node_to_edges.get(node, set())
        if not related_edges:
            return set()

        neighbor_nodes: Set[str] = set()
        for edge in related_edges:
            neighbor_nodes.update(self._edge_to_nodes.get(edge, set()))
        neighbor_nodes.discard(node)
        return neighbor_nodes

    def degree(self, node_id: str) -> int:
        node = self._normalize_id(node_id)
        return len(self._node_to_edges.get(node, set()))

    def edge_rank(self, edge_id: str) -> int:
        edge = self._normalize_id(edge_id)
        return len(self._edge_to_nodes.get(edge, set()))

    def edge_participants(self, edge_id: str) -> Dict[str, List[str]]:
        edge = self._normalize_id(edge_id)
        role_map = self._edge_to_roles.get(edge, {})
        return {role: sorted(nodes) for role, nodes in role_map.items()}

    def nodes(self) -> List[str]:
        return sorted(self._node_to_edges.keys())

    def edges(self) -> List[str]:
        return sorted(self._edge_to_nodes.keys())
