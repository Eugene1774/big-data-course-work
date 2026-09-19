from __future__ import annotations

import pytest

from core.hypergraph_index import HypergraphIndex


def test_add_hyperedge_rejects_empty_member_set() -> None:
    """
    Boundary case 1:
    adding a hyperedge without any valid member node should fail fast.
    """
    index = HypergraphIndex()
    with pytest.raises(ValueError, match="at least one valid node"):
        index.add_hyperedge("edge_empty", [])


def test_add_hyperedge_overwrite_cleans_reverse_index_and_neighbors() -> None:
    """
    Boundary case 2:
    overwriting an existing edge must clean stale reverse links and keep
    neighbor query consistent with the latest edge membership.
    """
    index = HypergraphIndex()
    index.add_hyperedge("edge_main", ["n1", "n1", "n2", "n2"])

    assert index.edge_rank("edge_main") == 2
    assert index.neighbors("n1") == {"n2"}

    index.add_hyperedge("edge_main", ["n1", "n3", "n3"])

    assert index.edge_rank("edge_main") == 2
    assert index.neighbors("n1") == {"n3"}
    assert index.degree("n2") == 0
    assert "n2" not in index.nodes()


def test_neighbors_of_unknown_or_isolated_node_return_empty_set() -> None:
    """
    Boundary case 3:
    querying an unknown or isolated node should always return an empty set.
    """
    index = HypergraphIndex()
    index.add_node("n_isolated")
    index.add_hyperedge("edge_main", ["n1", "n2", "n3"])

    assert index.neighbors("n_unknown") == set()
    assert index.neighbors("n_isolated") == set()
