# Developing UserSim

This file is for coding agents developing UserSim. To use UserSim, start with
[README.md](README.md). For contribution policy and setup, read
[CONTRIBUTING.md](CONTRIBUTING.md); for the system map, read
[architecture/overview.md](architecture/overview.md).

UserSim generates persona-grounded conversations, evaluates trajectories, and
produces reports and selected datasets. Its simulator and evaluator are Data
Designer column generators; reuse Data Designer's orchestration, parallelism,
and provider handling rather than implementing a second execution framework.

## Module map

| Path under `src/usersim/` | Owns |
| --- | --- |
| `cli/` | Command arguments, configuration loading, and pipeline orchestration. |
| `engine/config.py`, `engine/core/` | Simulation configuration, conversation runtime, model calls, and storage. |
| `engine/probes/`, `engine/assets/` | Probe behavior, prompts, and shipped asset banks. |
| `engine/evaluator/` | Evaluation configuration, rubric axes, and scorers. |
| `reporting/`, `selection/` | Reports, selection profiles, quality gates, and exports. |
| `taxonomy/` | Shared capability vocabulary and evaluation-cell decoding. |
| `asset_gen/`, `testing/` | Asset generation and conformance checks for extension authors. |

## Preserve the contracts

- Keep behavior with its owning module and follow neighboring code. Search for
  existing helpers before adding configuration, credential, or registry logic.
- Preserve the dependency boundaries in the architecture guide. Shared
  capability definitions belong in `taxonomy/`, where configuration and reporting
  can both use them without reversing their dependency direction.
- Changes to prompts, probes, assets, or scorers can change experimental results
  even when tests pass. Explain that impact in the PR and update relevant docs.
- UserSim's extension entry points are separate from its two Data Designer
  plugins. For extension work, read [architecture/plugins.md](architecture/plugins.md);
  for probes, read [the authoring guide](docs/engine/AUTHORING_A_PROBE.md).

## Develop and validate

Use Make and uv; `make help` lists the available targets. Reuse an existing
suitable environment for investigation; sync when setup or dependencies require it.

```bash
make install-dev                         # sync dependencies and install hooks
make check-all                           # read-only lint and format checks
make test PYTEST_ARGS='tests/test_cli.py -q' PYTEST_WORKERS=2
make check-doc-links                     # check repository documentation links
```

Test changed behavior and integration boundaries, using focused tests first.
Run packaging checks when installed-package behavior changes; see CONTRIBUTING
for the existing targets. Keep ordinary tests free of live provider calls.
`make health-check` is a separate credentialed operation; report whether it ran.

Follow the PR template and use `git commit --signoff` for DCO compliance.
Report exact checks and outcomes, including checks not run; do not claim human
review or live-provider acceptance without evidence. Keep durable contracts in
source docstrings and relevant docs. See [.agents/README.md](.agents/README.md)
for shared guidance and private local files.
