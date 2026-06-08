#!/bin/bash
pytest tests/ --cov=src/ml_switcheroo_ir --cov-report=xml --cov-report=term-missing
