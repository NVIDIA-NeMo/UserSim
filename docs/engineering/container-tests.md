# CPU container tests

The test image prepares a source unit-test environment and a separate installed-wheel
e2e environment during build. Execution uses the same `test:ci` and `test:e2e`
Mise tasks as native validation, with external networking disabled. Make targets
remain compatibility wrappers.

Install Docker with a running daemon. Build dependencies require internet access;
test execution needs no provider credentials. The image uses Python 3.12 pinned by
manifest digest and installs uv from `.mise.toml` and `mise.lock` through the
same verified Mise bootstrap used locally and in CI.

```bash
make container-build-test
make test-container
```

Artifacts appear in fresh `test-results/container/run-*/` directories: build and container logs,
`unit.log`, `unit.xml`, `coverage.xml`, `.coverage`, `cleanup.log`, and the e2e
artifact directory.
A failing suite or timeout returns nonzero. Both suites run so one failure does
not hide the other suite's diagnostics. The host runner allows one hour for image build and another hour for execution;
each suite allows 25 minutes. Runtime limits are two CPUs, two pytest workers,
8 GiB memory with no additional swap, and 512 processes. Temporary storage is
limited to 1 GiB and permits executable test fixtures. Tests run with the caller's UID/GID so exported files remain
owned by the caller.

Only the artifact directory is mounted from the host. The image includes source,
test fixtures, documentation, templates, and workflow files that unit tests read.
`.dockerignore` excludes Git metadata, host environments, private endpoint
configurations, credentials, editor/agent settings, caches and generated output.
Git is installed for tests that create their own temporary repositories.
The e2e runner operates outside `/workspace` against its installed wheel and
prepares its tokenizer cache during build. Docker's disabled external network
still permits the fake provider to use container loopback.

Every image build first runs a required context audit. It exports a synthetic
scratch context with Docker's native ignore semantics and checks every shared
example in `tests/fixtures/exclusion-policy.json`. Generated financial-services
`embeddings.parquet` sidecars stay outside the context; source asset banks remain.
Run this bounded check separately with:

```bash
make container-audit
# Or: scripts/mise run container:audit
```

The audit, image build, execution, and cleanup require `/var/run/docker.sock`
to be a Unix socket.
They reject nonempty inherited `DOCKER_*`, `BUILDX_*`, and `BUILDKIT_*` selectors
before invoking Docker, including host, context, client config, TLS, and builder
settings. Unset these variables explicitly when using this local validation
workflow. Each operation uses a temporary empty client config, explicit
`--host unix:///var/run/docker.sock`; image builds use the `default` builder
with the `docker` driver. Execution and cleanup share the same admitted client
configuration and environment, keeping the temporary config alive until cleanup
finishes. Stored default or remote contexts cannot select a different daemon
for running the image. A failed builder check, missing daemon, failed export, or
incorrect canary result returns nonzero. No missing prerequisite becomes a successful skip.
The scratch audit pulls no image and supplies `--network=none`.

Artifacts include `audit-settings.json`, `audit-builder.log`, `audit-export.log`,
and `audit-result.json`; the receipt records every example and its result.
Temporary client config and synthetic context files are removed afterward.
Ordinary pytest mocks Docker orchestration and never invokes a Docker executable.
Actual daemon access belongs to this explicit audit and container validation.

Each `audit`, `build`, or `run` invocation exclusively creates a `run-*` directory
beneath `--artifacts ROOT` and prints its exact path before starting. A `build`
attempt contains its audit and build output; an execution attempt contains the
container, cleanup, unit, and e2e logs. Earlier attempt directories and unrelated
root files are untouched. There are no ownership markers or history migrations;
remove result directories when you no longer need them.

The host mounts only the new execution directory at `/artifacts`. The `inside`
action writes directly into that already-owned directory, without another
container-level directory or cleanup step. Installed e2e creates its own fresh
attempt below `/artifacts/e2e`. Preparation remains in `/opt/e2e`, separate from
results. Failed attempts retain their partial diagnostics without exposing a
previous attempt's passing audit, coverage, JUnit, or scores as current. CI can
continue uploading the selected output root recursively.

`cleanup.log` records the container name, cleanup status, exit code, and captured
output, including partial timeout output. Successful removal and the exact
already-removed response for that container preserve the primary result.
Other removal errors require recovery and return nonzero if the primary run
succeeded. A primary test failure or timeout remains the returned status even
when cleanup fails. The diagnostic identifies the container and local endpoint
for inspection and manual recovery; cleanup never falls back to another daemon.

For direct runner overrides:

```bash
python scripts/run_container_tests.py build --image usersim-tests:local
python scripts/run_container_tests.py run --image usersim-tests:local --artifacts /tmp/usersim-test-artifacts --timeout 3600
```

After changing the base image, resolve and review its public manifest digest,
update the Dockerfile, and rebuild the image before running validation. Changes
to source or fixtures also require rebuilding; execution never mounts the host
checkout over the image.
