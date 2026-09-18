# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# usersim developer and CI entry points.
#
# CI is a thin wrapper over these targets: every check
# runs identically on a laptop and on a runner. Keep the logic HERE rather than
# in pipeline YAML -- that is what keeps a move to a different CI provider a
# translation instead of a rewrite, and what stops "works locally" drift.

SHELL := /bin/bash

UV ?= uv

# A top-level `_`-prefixed directory holds material that is not part of the
# distribution and may carry its own targets. Absent one, this expands to
# nothing.
-include $(wildcard _*/targets.mk)

# Overridable so CI can scope a job without editing recipes.
#
# scripts/ and templates/ are in scope deliberately. The gate scripts run in
# CI, and templates/probe/ is example code a contributor copies, so both
# should meet the same bar as src/. Leaving them out meant a later scope
# change would trigger a second repo-wide reformat.
PY_SRC   ?= src scripts templates
PY_TESTS ?= tests conftest.py
PY_ALL   := $(PY_SRC) $(PY_TESTS)

# Worker count for the parallel suite. `auto` follows the local core count,
# which suits a CPU-bound suite with every LLM call mocked (~2x on a laptop).
#
# Override it in containers. `auto` resolves to the HOST's core count inside
# Docker, not the cgroup CPU limit, so on a shared many-core runner it spawns
# far more workers than there are usable CPUs and the run wedges during worker
# startup, so CI pins an explicit worker count.
#
# `test-fast` stays serial deliberately: interleaved worker output makes a
# single failure much harder to read, and that target exists for debugging.
PYTEST_WORKERS ?= auto

# Importable package names for coverage (not paths -- src/
# installs as `usersim.engine`).
COV_PKGS ?= usersim
COV_ARGS := $(foreach pkg,$(COV_PKGS),--cov=$(pkg))

# The coverage floor lives here, not in CI, so `make coverage` is the same gate
# everywhere. Measured at 85% (2,644 of 17,686 statements uncovered); 84 leaves
# a point of headroom for platform differences between a laptop and the Linux
# runner while still catching roughly 180 newly uncovered statements. Ratchet it
# upward as coverage improves; do not lower it to make a branch pass.
#
# Read it as a regression tripwire, not a quality score: reporting/ emits a
# React app from Python string literals, and those lines execute during tests
# while only being asserted on by substring match.
COV_FAIL_UNDER ?= 84

.DEFAULT_GOAL := help

.PHONY: help install-dev install-ci install-kernel lint lint-fix format format-check check-all \
        check-all-fix test test-fast coverage smoke check-wheel check-extensions check-doc-links clean-notebooks check-license-headers update-license-headers check-dependency-licenses clean

help:  ## Show the available targets
	@echo "usersim make targets:"
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-26s\033[0m %s\n", $$1, $$2}'

# --- setup -----------------------------------------------------------------

install-dev:  ## Sync the workspace with dev tooling and install pre-commit hooks
	$(UV) sync
	@if [ ! -f .git/hooks/pre-commit ]; then \
		echo "Installing pre-commit hooks..."; \
		$(UV) run pre-commit install; \
	else \
		echo "Pre-commit hooks already installed"; \
	fi

# --locked installs strictly from uv.lock and refuses to re-resolve, so
# dependency drift becomes a build failure rather than a silent
# re-resolution. No pre-commit hooks -- CI has no working tree to guard.
install-ci:  ## Sync exactly what CI needs, enforcing the lockfile
	$(UV) sync --locked

# --- lint / format ---------------------------------------------------------

lint:  ## Check lint rules (no changes written)
	$(UV) run ruff check --output-format=full $(PY_ALL)
	@echo "Lint clean. Run 'make lint-fix' to auto-fix issues."

lint-fix:  ## Fix what ruff can fix automatically
	$(UV) run ruff check --fix $(PY_ALL)

format-check:  ## Check formatting (no changes written)
	$(UV) run ruff format --check $(PY_ALL)
	@echo "Formatting clean. Run 'make format' to auto-fix issues."

format:  ## Reformat in place
	$(UV) run ruff format $(PY_ALL)

check-all: lint format-check  ## Run every static check CI enforces

check-all-fix: lint-fix format  ## Fix everything fixable, then you re-review

# --- test ------------------------------------------------------------------

test:  ## Run the full test suite (parallel)
	$(UV) run pytest -n $(PYTEST_WORKERS)

test-fast:  ## Stop at the first failure, serially (tight debugging loop)
	$(UV) run pytest -x -q

coverage:  ## Run tests with coverage and enforce the floor
	$(UV) run pytest -n $(PYTEST_WORKERS) $(COV_ARGS) \
		--cov-report=term-missing \
		--cov-report=xml \
		--cov-fail-under=$(COV_FAIL_UNDER)

smoke:  ## Offline self-check of config, registries, and report builders
	$(UV) run usersim smoke

# The editable install resolves imports from the source tree and finds assets
# next to them, so `make test` and `make smoke` stay green even when the built
# artifact is unusable. This target is the only check that exercises what a
# user actually installs: it builds both wheels, installs them into a throwaway
# environment with no access to this checkout, and runs from an empty directory.
#
# Run it before any release, and after any change to packaging metadata, asset
# locations, or resource loading.
check-wheel:  ## Build the wheel, clean-install it, and smoke from an empty dir
	@set -eu; \
	WORK=$$(mktemp -d); \
	EMPTY=$$(mktemp -d); \
	trap 'rm -rf "$$WORK" "$$EMPTY"' EXIT; \
	echo "→ building wheels into $$WORK"; \
	$(UV) build --wheel -o "$$WORK" >/dev/null; \
	echo "→ asset files in the wheel:"; \
	BANKS=$$(unzip -l "$$WORK"/*.whl \
		| grep -cE '\.(yaml|yml|json|parquet)$$' || true); \
	echo "   $$BANKS"; \
	if [ "$$BANKS" -eq 0 ]; then \
		echo "   ✗ the banks did not ship; every probe would fail at asset resolution"; \
		exit 1; \
	fi; \
	echo "→ installing into a throwaway venv"; \
	$(UV) venv "$$WORK/venv" --python 3.13 >/dev/null 2>&1; \
	VIRTUAL_ENV="$$WORK/venv" $(UV) pip install --quiet "$$WORK"/*.whl; \
	echo "→ running from $$EMPTY (no repo in sight)"; \
	cd "$$EMPTY" && "$$WORK/venv/bin/usersim" smoke; \
	cd "$$EMPTY" && "$$WORK/venv/bin/usersim" simulate --dry-run \
		--locale en_US --num-rows 1 --out ./out >/dev/null; \
	echo "✓ installed package runs from a clean environment"

check-dependency-licenses:  ## Check runtime dependency licenses against the policy
	@$(UV) run --isolated --no-dev --group license-check --locked \
		python scripts/check_dependency_licenses.py

check-license-headers:  ## Verify every source file carries the SPDX header
	@$(UV) run python scripts/update_license_headers.py --check

update-license-headers:  ## Add or refresh the SPDX header on every source file
	@$(UV) run python scripts/update_license_headers.py

clean-notebooks:  ## Strip outputs from every notebook (source only)
	@$(UV) run --with nbstripout nbstripout $$(git ls-files '*.ipynb' | grep -v '^_')
	@echo "✓ notebook outputs stripped"

# `uv run jupyter lab` already sees the kernel inside .venv. This registers a
# NAMED one so editors that pick kernels system-wide can find this project's
# environment rather than offering a bare "python3" per checkout.
install-kernel:  ## Register a Jupyter kernel named for this project's venv
	@$(UV) run python -m ipykernel install --user \
		--name usersim --display-name "UserSim (.venv)"

check-doc-links:  ## Fail on broken relative links in tracked markdown
	@$(UV) run python scripts/check_doc_links.py

check-extensions:  ## Install an out-of-tree package and prove all six seams work
	@set -eu; \
	WORK=$$(mktemp -d); \
	EMPTY=$$(mktemp -d); \
	trap 'rm -rf "$$WORK" "$$EMPTY"' EXIT; \
	echo "→ building wheels into $$WORK"; \
	$(UV) build --wheel -o "$$WORK" >/dev/null; \
	echo "→ installing them plus the fixture extension into a throwaway venv"; \
	$(UV) venv "$$WORK/venv" --python 3.13 >/dev/null 2>&1; \
	VIRTUAL_ENV="$$WORK/venv" $(UV) pip install --quiet "$$WORK"/*.whl; \
	cp -R tests/fixtures/extension_package "$$WORK/ext"; \
	VIRTUAL_ENV="$$WORK/venv" $(UV) pip install --quiet "$$WORK/ext"; \
	echo "→ asking the installed CLI what it can see, from $$EMPTY"; \
	cp tests/fixtures/extension_package/verify_seams.py "$$EMPTY/"; \
	cd "$$EMPTY" && "$$WORK/venv/bin/python" verify_seams.py

# --- housekeeping ----------------------------------------------------------

clean:  ## Remove caches, coverage output, and compiled artifacts
	find . -type d -name __pycache__ -not -path './.venv*' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.py[co]' -not -path './.venv*' -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml htmlcov
	@echo "Cleaned."
