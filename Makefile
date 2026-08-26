VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest

.PHONY: install test

install:
	test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[test]"

# Requires Docker running locally — the test suite spins up a real
# pgvector/Postgres container via testcontainers.
test:
	$(PYTEST)
