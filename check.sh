#!/usr/bin/env bash
# Every gate CI runs, in CI's order, so a local pass cannot diverge from a CI pass.
set -euo pipefail
PY=.venv/bin/python
$PY -m ruff check .
$PY -m ruff format --check .
$PY -m mypy --strict src tools
$PY -m deptry src tools
$PY -m vulture src tools .vulture-whitelist.py --min-confidence 60
$PY -m pytest -q -m "not integration"
