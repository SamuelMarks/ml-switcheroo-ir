#!/bin/bash
set -euo pipefail
pytest tests/ --cov=src/ml_switcheroo_ir --cov=scripts --cov-branch --cov-report=term-missing --cov-fail-under=100
python3 scripts/check_function_coverage.py
