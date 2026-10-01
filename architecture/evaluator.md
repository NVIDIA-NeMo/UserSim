# Evaluator

Scoring runs *out of sim*. The simulator writes a trajectory parquet; the
evaluator reads it. That separation is the reason iterating on a judge or a
rubric costs nothing: no conversation is regenerated.

## Two kinds of signal

**Judges** are LLM calls that score subjective axes against a rubric. An
ensemble of two or more must span at least two architecturally diverse model
families, validated at config construction rather than at run time, because a
single-family ensemble measures agreement with itself. A single judge is
allowed and opts out of inter-judge agreement reporting.

**Scorers** are per-probe functions over stored columns, each returning a
structured verdict rather than a free-form opinion. Three are fully
deterministic (`language_compliance`, `refusal_basics`, `response_shape`); the
other nine call a model to reach a judgement the stored columns cannot settle on
their own, such as whether a retrieved fact is actually supported or what a
free-form turn actually disclosed (`health_disclosure_concealment`, which falls
back to deterministic arithmetic over committed intent when no auditor is wired).
Twelve ship:

| Scorer | Reads |
|---|---|
| `tool_use` | tool calls against the offered schema |
| `sov_ai_facts` | answers against a per-locale fact bank |
| `sov_ai_dynamic` | self-consistency where no ground truth exists |
| `sov_ai_multilingual_parity` | matched-pair joins on `query_id` |
| `safety_chat_pressure` | capitulation, turn-of-flip, refusal consistency |
| `safety_agentic` | boundary violations, scope creep, consequence disclosure |
| `financial_services` | verifiable task state plus reference grounding |
| `health_disclosure_concealment` | disclosure handling and red-flag capture |
| `identity_disclosure` | who the model says built it, against the expected identity |
| `language_compliance` | language and script of each turn |
| `response_shape` | structural expectations |
| `refusal_basics` | refusal presence and form |

Scorers are opt-in per run. A capability whose scorer was not dispatched
renders as *untested* rather than as a failure, and
`capabilities_missing_scorer()` exists so a run can warn about that instead
of silently producing an empty column.

## The evaluation cell

One wide JSON cell per trajectory, carrying per-axis and per-judge scores
plus deterministic outputs. Each cell embeds an envelope naming the judges,
axes, scorers, and prompt version that produced it. On re-run, a row whose
envelope already matches is skipped.

A parquet may key the assistant score as `eval_v1` or `assistant_eval_v2`
rather than the canonical `assistant_eval`; all three are accepted, so an
existing parquet renders without re-evaluation.

## Extending

Scorers register through `register_scorer` at import, and are discovered
out-of-tree through the `usersim.scorers` entry point. `usersim.testing`
publishes a conformance check, including that a scorer tolerates a row
missing any given field, since the evaluator calls it on incomplete
trajectories. See [plugins.md](plugins.md).
