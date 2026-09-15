# Reporting

Turns trajectory and evaluation parquets into a capability dashboard. No LLM
calls: pandas, JSON, and an HTML writer, so `usersim report` is free and
offline.

## The capability matrix

The organising idea is a **capability × locale matrix** rather than a
per-bundle scorecard. It answers "where is the model struggling, how strong
is the evidence, and what should I inspect next?" Each cell carries an
evidence state, a score, a threshold, and links to the lowest-scoring
trajectories behind it.

A capability is defined in [`usersim.taxonomy`](overview.md#taxonomy), not
here: reporting reads definitions, it does not own them.

## Artifacts

`usersim report` writes five files under `output/report/run=<id>/`:

| File | Contents |
|---|---|
| `index.html` | the dashboard; the one file a person opens |
| `report_manifest.json` | run provenance and configuration |
| `capability_matrix.json` | the capability × locale cells |
| `coverage_summary.json` | which probe and locale combinations are populated |
| `triage_queue.json` | failed cells ranked by impact × evidence strength |

`usersim report --open` launches a browser; it is opt-in so CI and headless
runs never spawn one. `--list-runs` answers "which runs do I have" without
rendering anything.

## Rendering

`dashboard.py` writes a self-contained `index.html`. React, Vega and `marked`
load from a CDN at view time, pinned to exact versions, under a
Content-Security-Policy. Untrusted text is sanitised before it reaches the
DOM.

Opening the file offline is a supported path: the static fallback view is
rendered *inline* with the real data, and the React app replaces it only if
the CDN loads. You get a usable report either way, and the page says which
view you are looking at.

## Multi-model comparison

`comparison.py` and `comparison_dashboard.py` build a side-by-side bake-off
across per-run reports: simulator health, a leaderboard, a
quality-versus-verbosity Pareto scatter, verbosity profile, per-locale bars,
per-capability facets, and a coverage matrix. Colour and shape are locked per
model across every chart, and an `eval_column` mismatch is a hard failure
rather than a silently mixed comparison.

This surface is a library API rather than a subcommand, so call
`build_comparison_report` over several reports and render with
`write_comparison_dashboard_artifacts`.

## Sim health

`sim_health.py` builds the rollup embedded in every dashboard: the
behavioural-realism diagnostics (front-loading ratio, politeness fraction,
verbosity CV, frustration markers, lexical diversity) that say whether the
*simulation* behaved, separately from whether the assistant scored well. A
run with degenerate user behaviour produces scores that are not worth
reading, and this is where that shows up.
