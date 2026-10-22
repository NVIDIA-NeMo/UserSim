# Mise tools and tasks

Mise owns the developer task graph and the exact uv release in `.mise.toml`.
`mise.lock` records the downloads for Linux x64/ARM64 and macOS x64/ARM64.
Ruff and ty belong to the locked Python development environment rather than a
second tool registry.

```bash
make bootstrap
scripts/mise run setup
scripts/mise run check
```

Bootstrap requires Bash, Python 3, curl, and GPG. It reads the supported Mise release
from `.mise.toml`, verifies the versioned upstream installer against the pinned
release-key fingerprint, and installs under `.cache/tools/`. Verification
failure stops installation. It does not update shell profiles or global Mise
configuration. `scripts/mise` provides repository-scoped writable cache, data,
configuration, and state directories; existing `MISE_*_DIR` overrides are honored.
Use `scripts/mise tasks` to list task descriptions.

Make targets remain compatibility commands. Their implementations and defaults
live in the Mise tasks, so `make coverage PYTEST_WORKERS=2` and
`PYTEST_WORKERS=2 scripts/mise run test:ci` select the same operation. Environment
and Make command-line overrides keep their existing names. `UV_PYTHON` selects
the interpreter explicitly in matrix CI; `UV_PROJECT_ENVIRONMENT` selects a
separate Python environment without forcing checkout activation into e2e.

Quality commands execute Ruff through `uv run`, using the same locked version
as hooks. `scripts/mise run check:tasks` validates task definitions and Bash
syntax, including extensionless task scripts. SPDX checks include those scripts
and the bootstrap helper. Task execution does not silently install missing
tools: run bootstrap/setup first. Live health checks remain a separate operation
with their established CLI credential-loading rules.

## Common commands

```bash
# Focused ordinary tests; PYTEST_ARGS is parsed without shell evaluation.
PYTEST_WORKERS=0 PYTEST_ARGS="tests/test_mise_tasks.py" scripts/mise run test
# Native gates, coverage, archives and e2e against the original wheel.
PYTEST_WORKERS=2 scripts/mise run validate:ci
# Fresh CPU image, resource bounds and network-disabled test execution.
scripts/mise run test:ci-container
# Check a pending commit message, or the configured PR commit range.
scripts/mise run check:dco "$(git rev-parse --git-path COMMIT_EDITMSG)"
# Replace main with your PR's target ref for a stacked branch.
BASE_SHA="$(git rev-parse --verify 'main^{commit}')" \
HEAD_SHA="$(git rev-parse --verify 'HEAD^{commit}')" scripts/mise run check:dco:range
```

`PYTEST_ARGS` and the quality scope variables support quoted arguments and
explicit empty scope overrides. Hook tasks forward filenames as separate
arguments. Full validation pins the final e2e `--wheel-dir dist` selector so an
ambient `WHEEL_DIR` or `E2E_ARGS` cannot redirect it to another archive. Failed
archive validation prevents e2e execution.
