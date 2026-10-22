# Contributing to NeMo User Sim

Thanks for your interest. This guide covers **developing** User Sim. To *use*
it, start with the [README](README.md).

## Developer Certificate of Origin

### Signing Off Your Work

* We require that all contributors "sign-off" on their commits. This certifies
  that the contribution is your original work, or you have rights to submit it
  under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be
    accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when
  committing your changes:

  ```bash
  $ git commit -s -m "Add cool feature."
  ```

  This will append the following to your commit message:

  ```
  Signed-off-by: Your Name <your@email.com>
  ```

  The sign-off must match your commit author identity. To sign off a branch you
  already wrote:

  ```bash
  git rebase --signoff main
  ```

### Recording your signature, once

The first pull request you open will get a comment from the DCO Assistant bot
asking you to sign. Reply with exactly:

```
I have read the DCO document and I hereby sign the DCO.
```

This is separate from signing off your commits, and both are needed. The
sign-off travels with each commit and is what certifies that contribution; the
comment records your agreement in one place so the bot does not have to ask
again. You sign once, not once per pull request, and every later contribution
passes the check automatically.

* Full text of the DCO (https://developercertificate.org/):

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

    Everyone is permitted to copy and distribute verbatim copies of this
    license document, but changing it is not allowed.


    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I
        have the right to submit it under the open source license
        indicated in the file; or

    (b) The contribution is based upon previous work that, to the best
        of my knowledge, is covered under an appropriate open source
        license and I have the right under that license to submit that
        work with modifications, whether created in whole or in part
        by me, under the same open source license (unless I am
        permitted to submit under a different license), as indicated
        in the file; or

    (c) The contribution was provided directly to me by some other
        person who certified (a), (b) or (c) and I have not modified
        it.

    (d) I understand and agree that this project and the contribution
        are public and that a record of the contribution (including all
        personal information I submit with it, including my sign-off) is
        maintained indefinitely and may be redistributed consistent with
        this project or the open source license(s) involved.
  ```

## Getting set up

Supports Python 3.11 to 3.14. The default development interpreter is Python 3.12
from `.python-version`; CI tests every supported version. Start with an existing
[uv](https://docs.astral.sh/uv/) installation, then install the exact
`required-version` from `pyproject.toml`:

```bash
make bootstrap              # installs the declared uv; ensure its bin directory is on PATH
make install-dev            # locked dev dependencies and pre-commit hook
uv run usersim smoke        # offline self-check; no API key, no network
```

`make check-tools` rejects another uv version. `make check-lock` checks that
`uv.lock` agrees with dependency metadata without rewriting it. Both setup
targets use `--locked`, so stale metadata fails rather than updating the lock.
After a deliberate dependency change, run `uv lock`, review `uv.lock`, and
commit it with the metadata change. CI reads the same uv version declaration;
update it deliberately and validate the lock before committing a tool upgrade.

`make install-ci` performs locked setup without installing hooks. In a Git
worktree, `make install-dev` finds the hook directory through Git rather than
assuming `.git` is a directory.

`smoke` is the fastest way to confirm an install is sound. It resolves the
model catalog, every probe's assets, and the plugin entry points without
making a single model call.

### Reproducing CI and coverage

Run `make validate-ci` for native checks, clean distribution validation, and
installed CLI e2e tests against that same validated wheel.
`make validate-native` runs static checks, dependency license policy, coverage,
the offline self-check, and installed extension checks without building the
distribution archives. CI runs the same test suite across supported Python
versions and enforces the 84% coverage floor on Python 3.12.

Use `make coverage PYTEST_WORKERS=2` locally. The XML report is `coverage.xml`;
change its path with `COV_XML`, or add pytest options through `PYTEST_ARGS`, for
example `make coverage PYTEST_ARGS="--junitxml=junit.xml"`. See
[Local and CI validation](docs/engineering/ci.md) for bundle membership, report
artifacts, and failure diagnosis.

Run `make test-e2e` for the isolated CLI workflow alone, or
`make test-e2e WHEEL_DIR=dist` to use the single existing wheel in `dist/`
without rebuilding. See [Installed CLI e2e tests](docs/engineering/e2e.md) for
setup, offline execution, scenarios, and diagnostics.

### Choosing models and providers

Three providers are built in, each needing only a credential. Pick the shipped
config that matches:

| Provider | Credential | Config |
|---|---|---|
| `nvidia` (build.nvidia.com) | `NVIDIA_API_KEY` | `models_default.toml` |
| `openai` | `OPENAI_API_KEY` | `models_openai.toml` |
| `openrouter` | `OPENROUTER_API_KEY` | `models_openrouter.toml` |

All three live in [`src/usersim/cli/`](src/usersim/cli/). Pass one with
`--models`, or leave it off to get the default.

To use any other OpenAI-compatible endpoint, copy one of those files, declare a
provider, and pass it with `--models`. The `api_key` field is the *name* of an
environment variable, never the key itself:

```toml
[[providers]]
name = "my-endpoint"
endpoint = "https://my-inference-endpoint.example/v1"
provider_type = "openai"
api_key = "MY_ENDPOINT_API_KEY"
```

**Keep your own endpoints out of the repository.** Name such a file
`models.local.toml`: `.gitignore` covers it, along with anything matching
`*.local.toml`, so it cannot be committed by accident.

A `models.local.toml` is picked up automatically. When no config is named,
resolution is:

1. `$USERSIM_MODELS_CONFIG`, if set, which redirects every command at once.
   A path that does not exist is an error rather than a silent fallback.
2. The nearest `models.local.toml` in the working directory or any parent.
3. The bundled `models_default.toml`.

An explicit `--models` beats all three. Whichever wins is printed before any
model call, so a run always says which config and which provider it used:

```
Models config: /path/to/models.local.toml (local override; the bundled default is unused)
Loaded 9 models (providers: ['my-endpoint'])
```

If you would rather configure an endpoint once for every project rather than
per checkout, add it to `~/.data-designer/model_providers.yaml`: Data Designer
reads that file and User Sim merges it, and it sits outside any repository.

### Credentials

Every provider reads its key from an environment variable. Export them, or put
them in a `.env.local` at the project root, which is gitignored:

```bash
NVIDIA_API_KEY=...
OPENAI_API_KEY=...
```

`.env.local` then `.env` are read at startup, searching upward from the working
directory. **Variables already set in your environment always win**, so the
file fills gaps rather than overriding a deliberate export. Nothing is required:
a missing file is normal, and the notebooks prompt for any key still missing.

**Reasoning models reject sampling parameters.** Most return an HTTP 400 for
any non-default `temperature`, and some reject `top_p` outright. Cataloged
models handle this already. For one the catalog does not know, such as a model
behind your own provider, name the parameters to leave out of the request.
TOML has no null literal, so this is how a row says "send nothing here":

```toml
{ alias = "evaluator_model", model = "my-reasoning-model", provider = "my-endpoint", drop_params = ["temperature", "top_p"] }
```

Without it the project defaults (`temperature = 0.7`, `top_p = 0.95`) apply and
the call fails. `max_tokens` is droppable the same way.

Adding a model means adding it to
[`model_catalog.py`](src/usersim/cli/model_catalog.py) as well, so its sampling
parameters resolve. `usersim smoke` fails if a shipped config names a model the
catalog does not define. A model absent from `VLLM_DEFAULTS` also needs an
explicitly declared provider: overriding an alias to one without a provider to
route it to raises `ConfigError` rather than guessing an endpoint.

## Before you open a pull request

```bash
make check-all               # lock, lint, format, SPDX, docs, notebooks, and all tracked-file hooks
make test                    # the full suite
```

Licensing has its own gate, worth running if you added a dependency:

```bash
make check-dependency-licenses   # runtime licenses against the policy allowlist
```

Two heavier gates run in CI and are worth running locally if you touched
packaging or the extension seams:

```bash
make check-dist         # builds and checks the sdist and wheel, installs the wheel clean, runs it from an empty dir
make check-extensions   # installs an out-of-tree package, asserts all six seams work
```

`make check-all` includes `make check-hooks`, which runs pre-commit against all
tracked files. Hooks may rewrite files and fail: review those changes and rerun
`make check-all`. Use `make install-dev` to install local pre-commit hooks;
CI runs the same checks even when local hooks are bypassed.

The SPDX hook fixes only the staged paths passed to it. `make update-license-headers`
adds or refreshes headers throughout `src/`, `tests/`, `scripts/`, `templates/`,
`Makefile`, and root `conftest.py`; `make check-license-headers` checks those
roots without writing. Python, shell, and `.template` files are covered; Markdown,
YAML, and TOML do not require SPDX headers under this policy.

Keep credentials and generated output local; keep required source and fixtures
trackable. Git ignores do not control archive or Docker inclusion.
Maintainers own gate scripts, tool declarations, hook configuration, and exclusion
policy through `.github/CODEOWNERS`.

Your PR title must follow [Conventional Commits](https://www.conventionalcommits.org/)
(`feat:`, `fix:`, `docs:`, `ci:`, `chore:`, `refactor:`, `test:`, `style:`,
`perf:`), because it becomes the squashed commit subject on `main`.

## What CI runs

`CI Status` aggregates static checks, all supported Python versions, packaging,
installed CLI e2e tests, and CPU container tests. Missing, skipped, cancelled, or
failed mandatory jobs fail that status. Run the same Make targets locally; see
[local and CI validation](docs/engineering/ci.md) for commands, diagnostics, and
the maintainer checklist for changing required checks. DCO and PR-title checks
remain separate required policies.

Every job in `.github/workflows/ci.yml` calls a `make` target, so anything CI
rejects reproduces locally with the same command. Tests and the dependency
license check run on Python 3.11 to 3.14. Pull-request checks run for every
base branch so stacked PRs receive CI; push checks remain scoped to `main`.

No job triggered by a pull request has access to a credential, so a PR from a
fork runs the complete suite. The consequence is that CI cannot tell you
whether a provider still accepts what we send it. That check lives in
`.github/workflows/health-checks.yml`, which runs weekly and on demand, and
you can run it yourself:

```bash
make health-check                                             # build.nvidia.com
make health-check PROVIDER_CONFIG=src/usersim/cli/models_openai.toml
```

A provider whose API key you have not set is reported as skipped rather than
failed, so you can run this with one key and check one provider.

It runs a real one-row `simulate`, `eval` and `report`. Please run it if you
touched models, providers, prompts, scorers or the trajectory store. The
offline suite mocks every model call, and `usersim smoke` builds its own
request rather than the one a run sends, so neither can catch a payload a
provider rejects or a file we write but cannot read back.

## Releases

Maintainers release from `main`:

1. Merge a pull request that sets `version` in `pyproject.toml` to the release
   version: `X.Y.Z`, or `X.Y.ZrcN` for a release candidate.
2. From an up-to-date `main`, run the checks the release runs, then tag the
   merge commit and push the tag by name:

   ```bash
   git switch main && git pull --ff-only
   make check-dist && make check-release-tag TAG=v0.2.0rc1
   git tag -a v0.2.0rc1 -m "usersim 0.2.0rc1"
   git push origin v0.2.0rc1
   ```

3. `.github/workflows/release.yml` runs those checks and the test suite on the
   tagged commit. After a maintainer approves the `release` environment, it
   creates the GitHub release with the sdist and wheel attached, marked as a
   pre-release for a release candidate, and writes their SHA-256 digests to
   the run summary.
4. Merge a pull request that moves `version` to the next development version,
   for example `0.2.0rc2.dev0`.

If the workflow fails before the release is published, delete the tag
(`git tag -d v0.2.0rc1 && git push origin --delete v0.2.0rc1`) and any draft
release it left, fix the cause and tag again. Once a release is published,
its tag and files stay as they are, and a fix ships as a new version. The
files attached to a release are the ones published to PyPI;
`gh release verify-asset v0.2.0rc1 <file>` checks a downloaded file against
them.

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
`src/usersim/engine/core/_extensions.py`. Give a probe or scorer a name no
built-in uses, since registering one under a taken name replaces the earlier
one.

To check your extension satisfies the contract:

```python
from usersim.testing import assert_probe_conforms, assert_scorer_conforms

def test_my_probe_conforms():
    assert_probe_conforms("my_probe")

async def test_my_scorer_conforms():
    await assert_scorer_conforms("my_scorer")
```

The scorer check is a coroutine, because it calls your scorer and the
evaluator awaits every scorer. Calling it without `await` returns a
coroutine that is never run, and the test passes having checked nothing.
Pass `sample_row=` a row shaped like your probe's trajectories to also check
empty cells, which the evaluator hands a scorer as `None`; see
[`architecture/plugins.md`](architecture/plugins.md).

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

## Live provider checks

See [Live provider health checks](docs/engineering/health-checks.md) for credential
precedence, local commands, and the pass, skip, and failure outcomes. These checks
make real model requests and are separate from offline contributor validation.

## Container validation

Run `make test-container` to build a fresh CPU test image and run coverage plus
installed-wheel e2e without external networking. See
[CPU container tests](docs/engineering/container-tests.md) for Docker prerequisites,
resource limits, context policy, and exported diagnostics.
