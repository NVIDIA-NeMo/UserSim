# CLI

`usersim` is an argparse driver in `usersim.cli`. Every subcommand module
exposes `register(subparsers)` and binds a `run(args)` through
`set_defaults(func=...)`, which is also the contract a contributed command
implements; see [plugins.md](plugins.md).

## Subcommands

| Command | LLM calls | What it does |
|---|:---:|---|
| `panel` | no | Sample personas into a parquet that fixes a run's locales and per-locale row counts |
| `simulate` | yes | Run the conversation simulator, write a partitioned trajectory dataset |
| `eval` | yes | Score trajectories with the judge ensemble and deterministic scorers |
| `rescore` | no | Recompute deterministic scorers on an existing evaluation parquet |
| `select` | no | Curate trajectories into an exportable dataset |
| `report` | no | Render the capability dashboard from trajectory + evaluation parquets |
| `gen-assets` | yes | Generate an asset bank for a domain |
| `validate-assets` | no | Vet a generated bank offline |
| `evaluate-assets` | yes | Judge-scored quality scorecard for a generated bank |
| `smoke` | no | Offline end-to-end verification |
| `setup-ngc` | no | Install the NGC CLI into the project venv |

The commands that make no LLM calls are free and work offline. `gen-assets`
and `evaluate-assets` are live, paid steps and are not needed to use the
shipped banks.

`simulate`, `eval` and `gen-assets` accept `--dry-run`, which resolves the
run's plan (for `simulate`, the models, probe mix, asset paths and toolset
seed) and exits before the first network call.

## Model configuration

Models are declared in TOML rather than code, so a deployment can repoint
providers without touching Python. `src/usersim/cli/models_default.toml` is
the shipped catalogue, with `models_openai.toml` and `models_openrouter.toml`
alongside it for those providers; `--models` points at your own file.

Three aliases are required for a simulation run: `user_model`,
`assistant_model` and `judge_model`. Two more are used when a run reaches for
them: `api_response_model` for tool-calling probes and `summary_model` for
context compression. `eval` requires `evaluator_model`, kept distinct from
`judge_model` so the in-sim gate and the post-hoc rubric can be sized
independently. A missing required alias fails at config construction, not
mid-run.

`usersim.cli.model_catalog` holds per-model inference defaults: temperature,
top_p, max_tokens, and any model-family-specific `extra_body`. A TOML row
overrides the catalogue per field, and the row is self-describing for
anything it sets.

## Failure handling

`usersim.cli._errors.ConfigError` is a `ValueError` subclass, never a
`SystemExit`, so library callers can catch it. `main()` catches it at the
boundary and prints a message rather than a traceback, exiting 2.

## Execution backends

`simulate` routes execution through an `ExecutionBackend`, with `local`
running in-process. `--backend` takes its choices from the registry, so an
installed backend appears in `--help` without editing the CLI. Validation
(model config, probe mix, asset pre-flight) happens before submission, so a
bad configuration fails on your machine rather than after a job reaches a
scheduler.
