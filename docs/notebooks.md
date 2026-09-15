# Notebooks

The `notebooks/` directory mirrors the CLI flow for interactive iteration.
They need the Jupyter kernel, which the runtime dependencies deliberately do
not carry:

```bash
uv sync --all-extras
```

Outputs are stripped from the committed notebooks, so you will see source
only until you run them. That keeps machine-specific paths and endpoint URLs
out of the repository, and keeps the repository small.

## The three notebooks

**[`01_simulate.ipynb`](../notebooks/01_simulate.ipynb)** drives the simulator
and writes the trajectory dataset. It is resumable across re-runs, skipping
trajectories whose `trajectory_id` is already present, and surfaces the
registered probe set with example `PROBE_MIX` presets plus a Rich-formatted
conversation preview.

**[`02_evaluate_simulation.ipynb`](../notebooks/02_evaluate_simulation.ipynb)**
runs the trajectory evaluator and builds the capability dashboard
(`build_capability_report` + `write_capability_dashboard_artifacts`), emitting
`report_manifest.json`, `capability_matrix.json`, `coverage_summary.json` and
`triage_queue.json` alongside a static `index.html`.

**[`03_extract_training_data.ipynb`](../notebooks/03_extract_training_data.ipynb)**
is a deliberate placeholder. Training-data extraction is a separate
workstream; the notebook publishes the planned five-record-type interface
contract and the questions still open before it can be implemented. The
schemas it describes are pinned by `tests/test_extraction_example.py`.

## Notebook or CLI?

The two paths call the same factories, so a notebook run and a CLI run of the
same configuration produce the same artifacts. Prefer the notebook for
interactive iteration: small row counts, locale-by-locale inspection,
conversation previews. Prefer the CLI for unattended and reproducible runs:

```bash
usersim simulate --locale en_US --num-rows 20 --out output/trajectories
```

Every LLM-driven subcommand accepts `--dry-run`, which prints the resolved
plan and exits without touching the network.
