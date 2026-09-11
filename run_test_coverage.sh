#!/bin/bash
pytest tests/ --cov=src/ml_switcheroo_ir --cov=scripts --cov-branch --cov-report=xml --cov-report=term-missing
