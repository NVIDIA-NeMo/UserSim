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
UV_VERSION := $(shell sed -n 's/^required-version = "==\([^"]*\)"/\1/p' pyproject.toml)

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
PY_E2E   ?= tests_e2e
PY_ALL   := $(PY_SRC) $(PY_TESTS) $(PY_E2E)

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
COV_XML ?= coverage.xml
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

.PHONY: help bootstrap check-tools check-lock install-dev install-ci install-kernel health-check check-notebooks-clean lint lint-fix format format-check check-all \
        check-hooks check-all-fix validate-native validate-ci test test-fast test-e2e coverage smoke dist check-dist check-release-tag check-extensions check-doc-links clean-notebooks check-license-headers update-license-headers check-dependency-licenses clean


help:  ## Show the available targets
	@echo "usersim make targets:"
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-26s\033[0m %s\n", $$1, $$2}'

# --- setup -----------------------------------------------------------------

# Requires an existing uv installation; installs the declared release on PATH.
bootstrap:  ## Install the repository's required uv version as a standalone tool
	$(UV) tool install --force "uv==$(UV_VERSION)"

check-tools:  ## Reject an unsupported uv version
	@actual=$$($(UV) --version 2>/dev/null | awk '{print $$2}'); \
	if [ -z "$(UV_VERSION)" ] || [ "$$actual" != "$(UV_VERSION)" ]; then \
		echo "Expected uv $(UV_VERSION), found $${actual:-none}. Install uv, then run make bootstrap and ensure its bin directory is on PATH." >&2; \
		exit 1; \
	fi

check-lock: check-tools  ## Check dependency metadata against uv.lock without rewriting it
	$(UV) lock --check

# Reinstalling the project restores any file that removing another distribution
# took with it, such as the `usersim` command an earlier name also installed.
install-dev: check-tools  ## Sync the workspace with dev tooling and install pre-commit hooks
	$(UV) sync --locked --reinstall-package usersim
	@if [ ! -f "$$(git rev-parse --git-path hooks/pre-commit)" ]; then \
		echo "Installing pre-commit hooks..."; \
		$(UV) run pre-commit install; \
	else \
		echo "Pre-commit hooks already installed"; \
	fi

# --locked installs strictly from uv.lock and refuses to re-resolve, so
# dependency drift becomes a build failure rather than a silent
# re-resolution. No pre-commit hooks -- CI has no working tree to guard.
install-ci: check-tools  ## Sync exactly what CI needs, enforcing the lockfile
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

check-all: check-tools check-lock lint format-check check-license-headers check-doc-links check-notebooks-clean check-hooks  ## Run ordinary static gates (hooks may rewrite files)

check-hooks:  ## Run hooks on every tracked file; review rewrites and rerun after failure
	$(UV) run pre-commit run --all-files --show-diff-on-failure

check-all-fix: lint-fix format  ## Fix everything fixable, then you re-review

# --- test ------------------------------------------------------------------

# PYTEST_ARGS is a passthrough, so a one-off run does not need a new target:
#   make test PYTEST_ARGS="--durations=25 -k financial"
E2E_ARGS ?=
WHEEL_DIR ?=

PYTEST_ARGS ?=

test:  ## Run the full test suite (parallel)
	$(UV) run pytest -n $(PYTEST_WORKERS) $(PYTEST_ARGS)

test-fast:  ## Stop at the first failure, serially (tight debugging loop)
	$(UV) run pytest -x -q

coverage:  ## Run tests with coverage and enforce the floor
	$(UV) run pytest -n $(PYTEST_WORKERS) $(COV_ARGS) \
		--cov-report=term-missing \
		--cov-report="xml:$(COV_XML)" \
		--cov-fail-under=$(COV_FAIL_UNDER) $(PYTEST_ARGS)

# Keep the installed suite separate: ordinary pytest's root conftest blocks
# loopback, and the e2e runner must never import the editable source tree.
test-e2e: check-tools  ## Test an installed wheel through CLI subprocesses and a local HTTP provider
	$(UV) run --frozen python scripts/run_e2e.py --uv "$(UV)" $(if $(WHEEL_DIR),--wheel-dir "$(WHEEL_DIR)") $(E2E_ARGS)

validate-native: check-all check-dependency-licenses coverage smoke check-extensions  ## Check source and installed extension behavior

validate-ci: validate-native check-dist  ## Validate source, then test the same clean distribution artifacts
	$(MAKE) test-e2e WHEEL_DIR=dist

smoke:  ## Offline self-check of config, registries, and report builders
	$(UV) run usersim smoke

dist:  ## Build the sdist, then the wheel from it, into a fresh dist/
	rm -rf dist
	$(UV) build -q -o dist
	@ls -1 dist

# The editable install resolves imports from the source tree and finds assets
# next to them, so `make test` and `make smoke` stay green even when the built
# artifact is unusable. This target checks what a user actually installs: it
# builds both archives, checks them (scripts/check_dist.py), installs the wheel
# into a throwaway environment with no access to this checkout, and runs it
# from an empty directory.
#
# Run it before any release, and after any change to packaging metadata, asset
# locations, or resource loading.
check-dist: dist  ## Build dist/, check both archives, clean-install the wheel and smoke it from an empty dir
	@set -eu; \
	$(UV) run python scripts/check_dist.py dist; \
	WORK=$$(mktemp -d); \
	EMPTY=$$(mktemp -d); \
	trap 'rm -rf "$$WORK" "$$EMPTY"' EXIT; \
	echo "→ installing the wheel into a throwaway venv"; \
	$(UV) venv "$$WORK/venv" >/dev/null; \
	test -x "$$WORK/venv/bin/python" || { echo "venv was not created"; exit 1; }; \
	$(UV) pip install --quiet --python "$$WORK/venv/bin/python" dist/*.whl; \
	test -x "$$WORK/venv/bin/usersim" \
		|| { echo "   ✗ the wheel installed but shipped no usersim entry point"; exit 1; }; \
	echo "→ running from $$EMPTY (no repo in sight)"; \
	cd "$$EMPTY" && "$$WORK/venv/bin/usersim" smoke; \
	cd "$$EMPTY" && "$$WORK/venv/bin/usersim" simulate --dry-run \
		--locale en_US --num-rows 1 --out ./out >/dev/null; \
	echo "✓ installed package runs from a clean environment"

# A tag ruleset limits who can push a `v*` tag, not which commit it names or
# which version it claims, so the release workflow runs this on the tagged
# commit before publishing. Run it from the commit you are about to tag.
# A tag name may contain `$(shell ...)`, backticks or quotes. The recipe reads
# it as $$TAG, which no shell parses, and CI passes TAG in the environment,
# because make expands a value given on its command line.
check-release-tag:  ## Fail unless TAG is v + the release version in dist/ and HEAD is on origin/main (TAG=v<version>)
	@test -n "$$TAG" || { echo "usage: make check-release-tag TAG=v<version>"; exit 1; }
	@$(UV) run python scripts/check_release_tag.py "$$TAG" dist

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

# The pre-commit hook that strips outputs only exists if someone ran
# `make install-dev`; a fresh clone has none. This is what actually keeps
# outputs, and the absolute paths and endpoint URLs inside them, out of the
# repository.
check-notebooks-clean:  ## Fail if any committed notebook carries outputs
	@$(UV) run python -c "$$CHECK_NOTEBOOKS_PY"

define CHECK_NOTEBOOKS_PY
import json, subprocess, sys
files = subprocess.run(
    ["git", "ls-files", "*.ipynb"], capture_output=True, text=True, check=True
).stdout.split()
dirty = []
for path in files:
    nb = json.load(open(path))
    n = sum(len(c.get("outputs", [])) for c in nb["cells"])
    n += sum(1 for c in nb["cells"] if c.get("execution_count"))
    if n:
        dirty.append(f"{path} ({n} output/execution entries)")
if dirty:
    print("Committed notebooks carry outputs:", *dirty, sep="\n  ")
    print("\nRun `make clean-notebooks` and commit the result.")
    sys.exit(1)
print(f"OK {len(files)} notebook(s) carry no outputs")
endef
export CHECK_NOTEBOOKS_PY

# `uv run jupyter lab` already sees the kernel inside .venv. This registers a
# NAMED one so editors that pick kernels system-wide can find this project's
# environment rather than offering a bare "python3" per checkout.
install-kernel:  ## Register a Jupyter kernel named for this project's venv
	@$(UV) run python -m ipykernel install --user \
		--name usersim --display-name "User Sim (.venv)"

check-doc-links:  ## Fail on broken relative links in tracked markdown
	@$(UV) run python scripts/check_doc_links.py

# --- live checks (need credentials; never run in PR CI) --------------------

# The offline suite proves the code is self-consistent. This proves a real
# provider accepts what we send it, which is a different question: the
# parameters, the provider routing and the engine's token budget are only
# exercised by an actual run. One row, two turns, so the cost is negligible.
PROVIDER_CONFIG ?= src/usersim/cli/models_default.toml
HEALTH_LOCALE   ?= ja_JP
HEALTH_PROBE    ?= sov_ai_dynamic

# Only missing credentials skip; configuration and execution errors fail.
HEALTH_TIMEOUT ?= 900
HEALTH_SUMMARY_FILE ?=

health-check:  ## Live one-row simulate + eval + report against a provider
	@$(UV) run python scripts/check_health.py \
		--models "$(PROVIDER_CONFIG)" --locale "$(HEALTH_LOCALE)" \
		--probe "$(HEALTH_PROBE)" --timeout "$(HEALTH_TIMEOUT)" \
		$(if $(HEALTH_SUMMARY_FILE),--summary-file "$(HEALTH_SUMMARY_FILE)")

check-extensions:  ## Install an out-of-tree package and prove all six seams work
	@set -eu; \
	WORK=$$(mktemp -d); \
	EMPTY=$$(mktemp -d); \
	trap 'rm -rf "$$WORK" "$$EMPTY"' EXIT; \
	echo "→ building wheels into $$WORK"; \
	$(UV) build --wheel -o "$$WORK" >/dev/null; \
	echo "→ installing them plus the fixture extension into a throwaway venv"; \
	$(UV) venv "$$WORK/venv" >/dev/null; \
	test -x "$$WORK/venv/bin/python" || { echo "venv was not created"; exit 1; }; \
	$(UV) pip install --quiet --python "$$WORK/venv/bin/python" "$$WORK"/*.whl; \
	cp -R tests/fixtures/extension_package "$$WORK/ext"; \
	$(UV) pip install --quiet --python "$$WORK/venv/bin/python" "$$WORK/ext"; \
	echo "→ asking the installed CLI what it can see, from $$EMPTY"; \
	cp tests/fixtures/extension_package/verify_seams.py "$$EMPTY/"; \
	cd "$$EMPTY" && "$$WORK/venv/bin/python" verify_seams.py

# --- housekeeping ----------------------------------------------------------

clean:  ## Remove caches, coverage output, compiled artifacts, build output and package metadata
	find . -type d -name __pycache__ -not -path './.venv*' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.py[co]' -not -path './.venv*' -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml htmlcov build dist src/*.egg-info
	@echo "Cleaned."
