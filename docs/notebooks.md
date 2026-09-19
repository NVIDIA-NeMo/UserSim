# Notebooks

The `notebooks/` directory mirrors the CLI flow for interactive iteration.
They need the Jupyter kernel, which the runtime dependencies deliberately do
not carry. It comes with the dev group, which `uv sync` installs by default:

```bash
uv sync
uv run jupyter lab      # sees the kernel inside .venv
```

If you open the notebooks in an editor that picks kernels system-wide rather
than launching Jupyter yourself, register a named kernel once so this
project's environment is selectable:

```bash
make install-kernel     # appears as "UserSim (.venv)"
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
joins a run's trajectories and evaluations, applies a quality profile, and
writes the passing subset to `output/curated/run=<id>/profile=<name>/`. It
drives the same `selection` package as `usersim select`, so the two produce
identical results. The curated output carries the full trajectory schema plus
the `selection_*` quality columns, which is what a downstream job filters on
to shape training records. `tests/test_extraction_example.py` pins the schema contract for the SFT and
pairwise shapes, confirming the trajectory and evaluation columns are
sufficient to drive them.

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
