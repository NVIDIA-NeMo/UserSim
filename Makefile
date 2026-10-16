# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Compatibility only. Mise owns implementations, dependencies and defaults.
SHELL := /bin/bash
MISE ?= scripts/mise
.DEFAULT_GOAL := help

# GNU Make forwards environment and command-line variable overrides to tasks.
# Defaults live only in the canonical Mise task implementation.

-include $(wildcard _*/targets.mk)

.PHONY: help bootstrap
help:
	@$(MISE) tasks

bootstrap:
	@bash scripts/install_mise.sh
	@$(MISE) run setup:tools

.PHONY: check-tools
check-tools:
	@$(MISE) run check:tools

.PHONY: check-lock
check-lock:
	@$(MISE) run check:lock

.PHONY: install-dev
install-dev:
	@$(MISE) run setup

.PHONY: install-ci
install-ci:
	@$(MISE) run setup:ci

.PHONY: install-kernel
install-kernel:
	@$(MISE) run setup:kernel

.PHONY: lint
lint:
	@$(MISE) run check:lint

.PHONY: lint-fix
lint-fix:
	@$(MISE) run lint:fix

.PHONY: format-check
format-check:
	@$(MISE) run check:format

.PHONY: format
format:
	@$(MISE) run format

.PHONY: check-all
check-all:
	@$(MISE) run check

.PHONY: check-hooks
check-hooks:
	@$(MISE) run check:hooks

.PHONY: check-all-fix
check-all-fix:
	@$(MISE) run format:all

.PHONY: test
test:
	@$(MISE) run test

.PHONY: test-fast
test-fast:
	@$(MISE) run test:fast

.PHONY: coverage
coverage:
	@$(MISE) run test:ci

.PHONY: test-e2e
test-e2e:
	@$(MISE) run test:e2e

.PHONY: validate-native
validate-native:
	@$(MISE) run validate:native

.PHONY: validate-ci
validate-ci:
	@$(MISE) run validate:ci

.PHONY: smoke
smoke:
	@$(MISE) run test:smoke

.PHONY: dist
dist:
	@$(MISE) run build:dist

.PHONY: check-dist
check-dist:
	@$(MISE) run check:dist

.PHONY: check-release-tag
check-release-tag:
	@$(MISE) run check:release-tag

.PHONY: check-dependency-licenses
check-dependency-licenses:
	@$(MISE) run check:license:dependencies

.PHONY: check-license-headers
check-license-headers:
	@$(MISE) run check:license:headers

.PHONY: update-license-headers
update-license-headers:
	@$(MISE) run license:headers:fix

.PHONY: clean-notebooks
clean-notebooks:
	@$(MISE) run notebooks:clean

.PHONY: check-notebooks-clean
check-notebooks-clean:
	@$(MISE) run check:notebooks

.PHONY: check-doc-links
check-doc-links:
	@$(MISE) run check:docs

.PHONY: health-check
health-check:
	@$(MISE) run test:health

.PHONY: check-extensions
check-extensions:
	@$(MISE) run check:extensions

.PHONY: clean
clean:
	@$(MISE) run clean

.PHONY: container-audit
container-audit:
	@$(MISE) run container:audit

.PHONY: container-build-test
container-build-test:
	@$(MISE) run container:build:test

.PHONY: test-container
test-container:
	@$(MISE) run test:ci-container

.PHONY: typecheck check-ci-status
typecheck:
	@$(MISE) run check:type

check-ci-status:
	@$(MISE) run check:ci-status

.PHONY: release-digests
release-digests:
	@$(MISE) run release:digests
