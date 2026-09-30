# What a run produces

`usersim panel → simulate → eval → report` leaves you with three things: a
trajectory dataset, an evaluation dataset, and a dashboard. This page is the
reference for all three: the parquet columns, the evaluation cell, and the
report files.

**Trajectories**: a directory-partitioned parquet dataset under
`output/trajectories/run=<id>/locale=<X>/probe_family=<Y>/`. One row per
simulated conversation, each carrying:

- `conversation_messages` (JSON): the full multi-turn message list. Each entry is `{role, content}`, plus `tool_calls` / `tool_call_id` on tool-using turns and `reasoning_content` on an assistant turn whose model emitted a thinking trace, unless `--no-store-reasoning` was passed. The trace is stored for analysis only; it is never replayed into a later turn (see [`docs/engine/README.md`](engine/README.md)).
- `simulation_outcome` (JSON) holds a row-level structured record: status, failure-class taxonomy, per-actor attribution, per-model token counts, wall-clock, provenance (code SHA + asset bank versions + prompt versions).
- `simulation_traces` (JSON): per-turn / per-call telemetry (judge-gate ratings, fourth-wall pre-filter triggers, capitulation checks, tool-call verification, etc.).
- A content-hashed `persona_uuid` (stable across runs) and a deterministic `trajectory_id` (hash of `persona + probe + resolved model identities + prompt version`). These are the simulator's idempotency keys.

> **Reasoning-trace export:** `usersim select` and its Hugging Face upload path
> retain `conversation_messages`, so captured `reasoning_content` travels with
> curated exports. Provider-emitted reasoning may be sensitive and is not
> guaranteed to be faithful or suitable as PRM supervision. Use
> `--no-store-reasoning` for runs whose traces should not leave the simulation
> environment, and apply the same review/governance as other model internals
> before sharing a dataset.

**Evaluations**: a wide JSON cell per trajectory containing per-axis × per-judge
scores plus deterministic scorer outputs. The `trajectory-evaluator` plugin
configures a judge ensemble (a single judge by default; any ensemble of two or
more must span at least two architecturally diverse model families, validated at
config-construct time, so inter-judge agreement means something) and a set of
scorers, of which twelve ship today (`tool_use`, `sov_ai_facts`,
`sov_ai_dynamic`, `sov_ai_multilingual_parity`, `safety_chat_pressure`,
`safety_agentic`, `financial_services`, `health_disclosure_concealment`,
`identity_disclosure`, plus the deterministic `language_compliance` / `response_shape` / `refusal_basics`). Cells carry an
envelope describing exactly which judges / axes / scorers / prompt version
produced them, and the evaluator skips rows whose envelope already matches
on re-run, so iterating on a judge or rubric is cheap.

**Capability dashboard**: a static React/Vega report keyed off the
trajectory + evaluator parquets. Rather than a per-bundle scorecard, the
dashboard organizes signal as a **capability × locale matrix** answering
"where is the model struggling, how strong is the evidence, and what should
we inspect or probe next?" Each cell carries an evidence state, a score, a
threshold, and links to the lowest-scoring trajectories so a reviewer can
drill in. A single cell looks like:

```json
{
  "capability": "assistant_quality",
  "locale": "en_US",
  "state": "blocked",
  "score": 0.6067,
  "threshold": 0.8,
  "n": 10,
  "source_axes": ["helpfulness", "accuracy", "coherence"],
  "blocker": "Below threshold by 0.193.",
  "evidence_policy": "show_lowest_scoring_trajectories",
  "trajectory_ids": ["baf81705dc9f7924", "0ec3a883410e860f", "..."]
}
```

The dashboard surfaces a **coverage matrix** (which probe / locale combinations
are populated, with how many rows), a **capability heatmap** (one row per
capability, one column per locale), and a **triage queue** (failed cells
sorted by impact × evidence strength). Build it from
[`notebooks/02_evaluate_simulation.ipynb`](../notebooks/02_evaluate_simulation.ipynb)
or directly from the CLI:

```bash
usersim report \
  --trajectories output/trajectories \
  --evaluations output/evaluations \
  --out output/report/
```

Both paths call `reporting.build_capability_report` +
`reporting.write_capability_dashboard_artifacts` and emit the same five
artifacts under `output/report/run=<id>/`: `index.html`,
`report_manifest.json`, `capability_matrix.json`,
`coverage_summary.json`, `triage_queue.json`.

**Multi-model comparison dashboard**: a side-by-side bake-off of every
per-run report under `output/report/`. Seven-layer narrative: simulator
health · model leaderboard · quality-vs-verbosity Pareto scatter ·
verbosity profile · per-locale grouped bars · per-capability flipped
facet · coverage matrix. Locked color + shape per model across every
chart, hard-fail on `eval_column` mismatch, soft warnings for sample-mode
/ persona-panel drift, deep-link from any cell tooltip back to that
run's per-run dashboard. Built by `reporting.build_comparison_report` +
`reporting.write_comparison_dashboard_artifacts`; emits three artifacts
under `output/comparison/run=cmp_<epoch>/`: `index.html`,
`comparison_manifest.json`, `comparison_matrix.json`. This surface is a
library API rather than a subcommand: call
`usersim.reporting.build_comparison_report` over several per-model reports
and render with `write_comparison_dashboard_artifacts`.
