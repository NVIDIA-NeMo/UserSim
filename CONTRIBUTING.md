# Contributing to NeMo UserSim

Thanks for your interest. This guide covers **developing** UserSim. To *use*
it, start with the [README](README.md).

## Developer Certificate of Origin

Every commit must be signed off, certifying that you wrote the patch or
otherwise have the right to contribute it under Apache-2.0. See [DCO](DCO)
for the full text.

```bash
git commit -s -m "your message"
```

That appends a `Signed-off-by: Your Name <you@example.com>` line, which must
match your commit author identity. To sign off a branch you already wrote:

```bash
git rebase --signoff main
```

## Getting set up

Requires Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras
uv run usersim smoke        # offline self-check; no API key, no network
```

`smoke` is the fastest way to confirm an install is sound. It resolves the
model catalog, every probe's assets, and the plugin entry points without
making a single model call.

## Before you open a pull request

```bash
make check-all               # ruff lint
make test                    # the full suite
make check-license-headers   # SPDX header on every source file
make check-doc-links         # every relative link in markdown resolves
```

Licensing has its own gate, worth running if you added a dependency:

```bash
make check-dependency-licenses   # runtime licences against the policy allowlist
```

Two heavier gates run in CI and are worth running locally if you touched
packaging or the extension seams:

```bash
make check-wheel        # builds a wheel, installs it clean, runs from an empty dir
make check-extensions   # installs an out-of-tree package, asserts all six seams work
```

`make update-license-headers` adds the SPDX header to new files.

## What the tests expect

The suite is hermetic: every model call is mocked, and a network guard fails
any test that tries to reach out. It runs in well under a minute on a laptop.

Two conventions matter more than they look:

- **No vacuous skips.** A `pytest.skip` that fires because a path moved is a
  silent regression, not a skipped test. Prefer a hard assertion; if
  something is genuinely conditional, assert on the condition itself.
- **Extension discovery is off under test.** `USERSIM_DISABLE_EXTENSIONS` is
  set via `pytest-env`, because a package installed in your virtualenv would
  otherwise register itself into the probe and scorer registries and make
  results depend on your machine.

## Adding a probe or a scorer

You do not need to modify this repository. Probes, scorers, asset domains,
execution backends, capabilities, and CLI subcommands are all discovered
through entry points; see the seven groups in
`src/usersim/engine/core/_extensions.py`.

To check your extension satisfies the contract:

```python
from usersim.testing import assert_probe_conforms, assert_scorer_conforms

def test_my_probe_conforms():
    assert_probe_conforms("my_probe")
```

Those same checks run against every built-in probe and scorer, so they cannot
drift into describing a contract this project does not itself keep.

If you are adding a probe *in tree*, `templates/probe/` has a copy-and-edit
gallery and [`docs/engine/AUTHORING_A_PROBE.md`](docs/engine/AUTHORING_A_PROBE.md)
is the end-to-end walkthrough.

## Code style

`ruff` owns formatting and linting; there is no separate style config to
learn. Beyond that, one house rule about comments: write them for the reader
of the current code. A comment explaining what the code *used to* do, or why
a past change was made, goes stale the moment it merges.

## Reporting bugs and security issues

Functional bugs belong in GitHub Issues. **Security vulnerabilities do
not**. See [SECURITY.md](SECURITY.md) for the private disclosure channels.

## Code of conduct

Participation is governed by the [Contributor Covenant](CODE_OF_CONDUCT.md).
