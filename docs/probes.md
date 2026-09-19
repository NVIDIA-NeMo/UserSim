# Probe catalog

A probe decides *what* a simulated user does: the interaction pattern, the
turn-one content, and what counts as the assistant failing. This page is the
catalogue: which probe catches which failure mode. For how the probe
substrate works, see [`architecture/probes.md`](../architecture/probes.md);
for writing one, [`docs/engine/AUTHORING_A_PROBE.md`](engine/AUTHORING_A_PROBE.md).

Thirteen probes ship today, each aimed at a distinct failure mode. Select them
with `--probe-mix` on `usersim simulate`, or `PROBE_MIX = {...}` in
[`notebooks/01_simulate.ipynb`](../notebooks/01_simulate.ipynb); the registry
validates the mix at parse time so an unknown probe fails immediately.

| Family | Probe | Catches |
|--------|-------|---------|
| General | `tool_calling` | Tool hallucination, parameter fabrication, schema violations, missed-clarification cases (`call/clarify/refuse` precision) |
| General | `general_open_ended` | Information-seeking degradation across personality types, register mismatch, unhelpful hedging |
| General | `general_educational` | Pedagogical failures, level mismatch, condescension, lack of scaffolding |
| Sovereign-AI | `sov_ai_facts` | Locale-specific factual errors, false-premise traps, fabricated institutional knowledge, graded against a curated per-locale fact bank |
| Sovereign-AI | `sov_ai_dynamic` | Hallucinations and self-inconsistency on locale topics with no ground truth, where judges flag suspected fabrication |
| Sovereign-AI | `sov_ai_multilingual_parity` | Respect-parity gaps across demographic strata for the same conceptual question across locales (matched-pair joins on `query_id`) |
| Safety | `safety_chat_pressure` | Multi-turn capitulation under reframing pressure, scored by `Turn_of_Flip`, `pressure_resistance_half_life`, refusal consistency |
| Safety | `safety_agentic` | Tool-use boundary violations, scope creep, prompt-injection susceptibility, missing consequence disclosure under realistic mock-environment load |
| Financial-services | `financial_services` | Multi-institution financial support/advisory: wrong/unauthorized tool use and poor retrieval on verifiable tasks, plus ungrounded/fabricated/numerically-wrong answers on open-ended advisory (two tiers: deterministic verifier + reference-grounded judge) |
| Health-disclosure | `health_therapy_disclosure` | Mishandled self-harm / crisis content when a therapy patient discloses gradually, where the user side paces disclosure via a deterministic move-space + Guard |
| Health-disclosure | `health_triage_disclosure` | Missed urgent red-flag symptoms that a patient only surfaces once a care-navigation / triage assistant earns them |
| Health-disclosure | `health_decision_support_disclosure` | Missed clinical danger signs when a *clinician* presents a case to a decision-support assistant |
| Health-disclosure | `health_general_disclosure` | Downplayed red-flag symptoms and quiet medication lapses when a patient with an ongoing condition consults a general health assistant |

Content review status travels with the results. The eleven non-`general`
probes draw on curated banks whose entries carry a review flag, set to
`placeholder: true` until a native reviewer or domain expert signs off. Every
trajectory that drew on a marked entry records a `used_placeholder_*` warning
on its outcome, written to the trajectory parquet alongside the scores, so you
can always see which findings rest on expert-reviewed content. Status is per
entry rather than per bank: the `sov_ai_dynamic` taxonomies are reviewed
throughout, the safety and multilingual-parity banks are part-way, and the
finance, sovereign-AI fact and health banks are awaiting review.

## Choosing a mix

`--probe-mix` takes weights that need not sum to 1; they are normalised. The
registry validates every name at parse time, so a typo fails before any model
is called:

```bash
usersim simulate --locale en_US --num-rows 30 \
  --probe-mix 'sov_ai_facts=1/3,sov_ai_dynamic=1/3,sov_ai_multilingual_parity=1/3' \
  --out output/trajectories
```

Probes are discovered through the `usersim.probes` entry point, so an
installed package can add one without modifying this project. See
[`architecture/plugins.md`](../architecture/plugins.md).
