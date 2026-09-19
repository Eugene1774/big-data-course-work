#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-neo4j}"
NEO4J_DATABASE="${NEO4J_DATABASE:-neo4j}"
CARDS_DIR="${CARDS_DIR:-$SCRIPT_DIR/knowledge_base/cards}"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/processed_json}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
KG_CLEAR_PIPELINE="${KG_CLEAR_PIPELINE:-0}"
KG_CLEAR_HYPERGRAPH="${KG_CLEAR_HYPERGRAPH:-0}"
KG_VERIFY_HYPERGRAPH="${KG_VERIFY_HYPERGRAPH:-1}"
KG_PIPELINE_LIMIT="${KG_PIPELINE_LIMIT:-}"

cd "$SCRIPT_DIR"

if [[ "$INSTALL_DEPS" == "1" ]]; then
  echo "[0/2] Installing dependencies from requirements.txt"
  "$PYTHON_BIN" -m pip install -r requirements.txt
fi

PIPELINE_ARGS=(
  --uri "$NEO4J_URI"
  --user "$NEO4J_USER"
  --password "$NEO4J_PASSWORD"
  --database "$NEO4J_DATABASE"
  --cards-dir "$CARDS_DIR"
)

HYPERGRAPH_ARGS=(
  --uri "$NEO4J_URI"
  --user "$NEO4J_USER"
  --password "$NEO4J_PASSWORD"
  --database "$NEO4J_DATABASE"
  --data-dir "$DATA_DIR"
)

if [[ "$KG_CLEAR_PIPELINE" == "1" ]]; then
  PIPELINE_ARGS+=(--clear)
fi

if [[ -n "$KG_PIPELINE_LIMIT" ]]; then
  PIPELINE_ARGS+=(--limit "$KG_PIPELINE_LIMIT")
fi

if [[ "$KG_CLEAR_HYPERGRAPH" == "1" ]]; then
  HYPERGRAPH_ARGS+=(--clear)
fi

if [[ "$KG_VERIFY_HYPERGRAPH" == "1" ]]; then
  HYPERGRAPH_ARGS+=(--verify)
fi

echo "[1/2] Running kg_pipeline.py"
"$PYTHON_BIN" kg_pipeline.py "${PIPELINE_ARGS[@]}"

echo "[2/2] Running kg_hypergraph.py"
"$PYTHON_BIN" kg_hypergraph.py "${HYPERGRAPH_ARGS[@]}"

echo "Knowledge graph build completed."
