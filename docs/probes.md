# Probe catalog

A probe decides *what* a simulated user does: the interaction pattern, the
turn-one content, and what counts as the assistant failing. This page is the
catalog: which probe catches which failure mode. For how the probe
substrate works, see [`architecture/probes.md`](../architecture/probes.md);
for writing one, [`docs/engine/AUTHORING_A_PROBE.md`](engine/AUTHORING_A_PROBE.md).

Fourteen probes ship today, each aimed at a distinct failure mode. Select them
with `--probe-mix` on `usersim simulate`, or `PROBE_MIX = {...}` in
[`notebooks/01_simulate.ipynb`](../notebooks/01_simulate.ipynb); the registry
validates the mix at parse time so an unknown probe fails immediately.

| Family | Probe | Catches |
|--------|-------|---------|
| General | `tool_calling` | Wrong tool choice, fabricated or malformed arguments, unnecessary tool calls, missed information gathering and architecture leaks, scored on the `tool_use` scorer's eight axes |
| General | `general_open_ended` | Information-seeking degradation across personality types, register mismatch, unhelpful hedging |
| General | `general_educational` | Pedagogical failures, level mismatch, condescension, lack of scaffolding |
| Sovereign-AI | `sov_ai_facts` | Locale-specific factual errors, false-premise traps, fabricated institutional knowledge, graded against a curated per-locale fact bank |
| Sovereign-AI | `sov_ai_dynamic` | Hallucinations and self-inconsistency on locale topics with no ground truth, where judges flag suspected fabrication |
| Sovereign-AI | `sov_ai_multilingual_parity` | Respect-parity gaps (condescension, miscalibrated complexity, help pitched below the user's proficiency) on the same conceptual question asked in every locale; each row records its `query_id`, so a question's answers line up locale by locale |
| Safety | `safety_chat_pressure` | Multi-turn capitulation under reframing pressure, scored on whether the refusal holds (`pressure.refusal_held`) and stays contextually safe (`pressure.contextual_safety_persistence`), with the turn of the first flip (`turn_of_flip`) on each row |
| Safety | `safety_agentic` | Tool-use boundary violations, scope creep, prompt-injection susceptibility, missing consequence disclosure under realistic mock-environment load |
| Financial-services | `financial_services` | Multi-institution financial support/advisory: wrong/unauthorized tool use and poor retrieval on verifiable tasks, plus ungrounded/fabricated/numerically-wrong answers on open-ended advisory (two tiers: deterministic verifier + reference-grounded judge) |
| Health-disclosure | `health_therapy_disclosure` | Mishandled self-harm / crisis content when a therapy patient discloses gradually, where the user side paces disclosure via a deterministic move-space + Guard |
| Health-disclosure | `health_triage_disclosure` | Missed urgent red-flag symptoms that a patient only surfaces once a care-navigation / triage assistant earns them |
| Health-disclosure | `health_decision_support_disclosure` | Missed clinical danger signs when a *clinician* presents a case to a decision-support assistant |
| Health-disclosure | `health_general_disclosure` | Downplayed red-flag symptoms and quiet medication lapses when a patient with an ongoing condition consults a general health assistant |
| Identity | `identity_disclosure` | Self-attribution errors: the model names a developer other than its own when asked who it is or who made it, unprompted, when a question suggests one, or after several turns of a user pressing a competitor. The probe infers who made the model from its id; [declare](../src/usersim/engine/probes/identity_disclosure/README.md#what-a-row-is-graded-against) any model it cannot infer |

## Scope and coverage

Every probe runs in all eleven shipped locales except `financial_services`,
which runs where a locale has a bank of its own. The 22 India language
variants run every probe; probes with per-locale content use the `en_IN`
bank, and an opening taken from a bank is machine-translated and marked
`USED_MACHINE_TRANSLATION`.

| Probe | Shipped locales | India variants | Content |
|---|---|---|---|
| `tool_calling` | all 11 | yes | 143 toolsets shared by every locale |
| `general_open_ended`, `general_educational` | all 11 | yes | shared prompts; topic and subject seeds with native extensions for `en_GB`, `de_DE`, `fr_FR`, `pt_BR`, `ja_JP`, `ko_KR`, `hi_Deva_IN` and `hi_Latn_IN` |
| `sov_ai_facts` | all 11 | through `en_IN` | 60 facts per locale, with region-specific facts for `pt_BR` (Brazilian macro-regions), `en_GB` (UK nations) and `de_DE` (Länder) |
| `sov_ai_dynamic` | all 11 | through `en_IN` | 10 topic categories per locale |
| `sov_ai_multilingual_parity` | all 11 | through `en_IN` | 48 questions, each written in every shipped locale |
| `safety_chat_pressure` | all 11 | through `en_IN` | 64 requests across 8 harm categories, with 6 pressure strategies; every request is written in every shipped locale |
| `safety_agentic` | all 11 | through `en_IN` | 48 action requests across 4 sub-protocols, each written in every shipped locale |
| `financial_services` | `en_US`, `en_IN`, `en_GB`, `fr_FR`, `de_DE` | through `en_IN` | 4 to 6 institutions and 60 to 109 verifiable tasks per locale |
| `health_*_disclosure` (four clients) | all 11 | yes | one synthetic clinical profile per client; bring your own profile bank |
| `identity_disclosure` | all 11 | through `en_IN` | 14 worded openings in every locale (the English locales share one wording), plus 7 the simulated user writes |

**Review flags.** Eleven probes draw on curated banks whose entries carry a
review flag: every probe except `tool_calling` and the two `general_*` probes.
`placeholder: true` marks an entry no native speaker or domain expert has
verified. Every trajectory that draws on one records a `used_placeholder_*`
warning on its outcome, stored in the trajectory parquet with the scores, so
you can always tell which results rest on flagged content; for those, SME
verification is recommended. The `sov_ai_dynamic` taxonomies are reviewed for
nine locales, with `en_GB` and `de_DE` flagged; the safety and
multilingual-parity banks mix reviewed and flagged entries; the finance, fact
and health banks and the identity spec are flagged.

**Bring your own clinical profiles.** The health probes run out of the box:
each client ships one synthetic profile that every persona falls back to, and
rows that use it carry `used_placeholder_clinical_profile`. For concealment
ground truth from your own population, point `USERSIM_THERAPY_PROFILES`,
`USERSIM_TRIAGE_PROFILES`, `USERSIM_CDS_PROFILES` or
`USERSIM_GENERIC_HEALTH_PROFILES` at a profile bank keyed by persona uuid; see
the [health_disclosure README](../src/usersim/engine/probes/health_disclosure/README.md#clinical-profile-bank-concealment-ground-truth).

## Choosing a mix

`--probe-mix` takes weights that need not sum to 1; they are normalized. The
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
