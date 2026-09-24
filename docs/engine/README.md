# Conversation Plugin

This package ships **two** Data Designer column generators for the
[NeMo UserSim](../../README.md) population-grounded user simulation framework:

- **`conversation-simulator`**: the trajectory simulator. Runs probe-dispatched
  user/assistant conversations with in-sim judges and emits trajectory parquet rows.
  Implementation: `usersim.engine.generator.ConversationSimulatorGenerator`.
  Config: `usersim.engine.config.ConversationSimulatorConfig`.
- **`trajectory-evaluator`**: the out-of-sim assistant
  evaluator. Reads stored trajectories and applies a multi-family judge ensemble
  + probe-specific deterministic scorers; writes a single wide JSON cell
  per trajectory. Re-runnable against an unchanged trajectory store without
  re-invoking the simulator.
  Implementation: `usersim.engine.evaluator.generator.TrajectoryEvaluatorGenerator`.
  Config: `usersim.engine.evaluator.config.TrajectoryEvaluatorConfig`.

The two are physically separate plugins so iteration on the evaluator (swap a
judge, change an axis rubric, add a scorer) does not require re-running the
simulation. The [parent README](../../README.md) covers the project as a whole.

## Architecture

<img src="../images/plugin_architecture.jpg" width="700">

Every probe except `safety_agentic` drives the shared
`core/simulation.py::ConversationLoop` via `BaseProbe.run_dispatch`,
turn iteration, fourth-wall + refusal-echo pre-filters, user-judge gate
retries, role-violation telemetry, follow-up generation, reactive
frustration escalation, and early-stop heuristics all live in one place.
The simulator does **not** judge the assistant under test: that is the
trajectory-evaluator's job. A probe MAY optionally override
`should_succeed(state)` for sim-side guardrails (e.g., `tool_calling`
flips `conversation_status` to False when the assistant called zero
tools). `safety_agentic` overrides `run_dispatch` itself because its
shape (single user turn + multi-assistant-turn tool-interception, no
user-LLM at all) is genuinely incompatible with the unified loop's
back-and-forth assumption: it still inherits the substrate's bank-load
/ placeholder-warning / version-pinning / `should_succeed` /
`build_result_extras` machinery.

### Assistant under test: pure-capability policy

The framework treats the assistant model as the **system under test**.
We measure its out-of-the-box behavior: every probe sends the
assistant **no system prompt** (an empty string, omitted from the
message list rather than sent as an empty `system` role to keep
provider compatibility intact). This is a deliberate cross-cutting
policy: pinning language affordances ("Respond in {language}"),
safety guidelines ("follow your usual safety guidelines"), or tool-use
rules (CONFIRM/DISCLOSE/TREAT) would all prime the model and
contaminate the very signals each probe is designed to surface.

What the model still receives:

- **Conversation history**: the user's turn(s) and the assistant's
  prior responses. This is the standard chat completion input;
  removing it would not be a chat at all.
- **OpenAI `tools=` parameter** for `tool_calling` and `safety_agentic`
 : the tool schemas (each with its own `description` field) are
  exposed via the API parameter, not a system message. The model
  knows what tools exist and what they do, but no behavioral
  instruction about when to use them.

What the model does NOT receive:

- A system prompt of any kind ("You are a helpful assistant", "Respond
  in {language}", "Follow safety guidelines", "CONFIRM before
  irreversible actions"). Whether the model behaves helpfully,
  responds in the right language, refuses harmful requests, or
  confirms before acting is exactly what each probe is designed to
  measure. Telling the model to do those things would prime the
  signal.

If you author a new probe that depends on a system prompt for the
assistant under test (e.g., a research-grounded experiment), document
the deviation in the probe's module docstring. The default is empty;
deviations are deliberate.

### Sim2Real Gap Mitigations

The plugin implements four Sim2Real features derived from persona OCEAN traits.
Each is measurable in the output, and the D1-D4 diagnostics quantify how far
they move a run away from LLM-as-fake-user homogeneity. See
[`docs/methodology.md`](../methodology.md).

1. **Incremental disclosure**: 60% of users withhold details and reveal them
   gradually; 40% provide everything upfront. Controlled by
   `incremental_disclosure_ratio` (default 0.6).
2. **OCEAN-grounded interaction style**: Each user receives one of five
   interaction styles (`cooperative`, `neutral`, `impatient`, `frustrated`,
   `confrontational`) derived from their persona's neuroticism, agreeableness,
   and patience scores. The full behavioral profile additionally surfaces
   `patience` / `verbosity` / `cooperativeness` / `tech_literacy` /
   `error_proneness` parameters that the user-agent prompts read.
3. **Reactive frustration**: During conversation, when the assistant is
   judged inadequate, the user agent receives escalating frustration prompts
   (pushing back, repeating requests, threatening to leave). Escalation rate
   is computed each turn from the persona's `patience` score, the chosen
   interaction style, and accumulated assistant failures.
4. **D1–D4 + MATTR behavioral diagnostics**: post-hoc realism metrics
   (front-loading ratio, politeness fraction, verbosity coefficient of
   variation, frustration-marker frequency, lexical diversity) computed in
   `reporting/diagnostics.py` and surfaced in the capability dashboard's
   Sim Health panel for sim-quality monitoring.

### Asset-driven probes

Ten of the fourteen shipped probes consume a curated asset bank: three
sovereign-AI probes (*curated factual depth*, *dynamic open-ended
breadth*, *cross-locale matched-pair queries*), two safety probes
(*conversational pressure*, *agentic tool-use*), `financial_services`, the
three `health_disclosure` clients, and `identity_disclosure`. Each probe consumes one asset shape; the
assets are intentionally distinct because the probes ask structurally
different questions of the model.

| Asset | Loader | Unit | Per-locale shape | Used by |
| --- | --- | --- | --- | --- |
| Fact bank | `core/fact_bank.py` | One fact per entry (with `question`, `false_premises`, `ground_truth`) | One file per locale, content authored separately per locale | `sov_ai_facts` |
| Probing / dynamic-tier taxonomy | `core/probing_taxonomy.py` | One topic category per entry (`invitation`, `subtopic_hints`, optional `applies_to_types` + `persona_affinities`); shared `select_category` derivation | One file per locale (`categories.yaml` for sov_ai_dynamic; `dynamic.yaml` for financial_services) | `sov_ai_dynamic`, `financial_services` |
| Financial-services bank | `core/finance_bank.py` (+ `finance_tasks.py`, `embeddings.py`) | Per-institution corpus + tools + tasks + embeddings; `Institution` / `FinanceBank` typed views; sim-time gold derivation | One dir tree per locale (`<type>/<institution>/`) | `financial_services` |
| Query bank | `core/query_bank.py` | One **conceptual question** per entry (with N locale `renderings` + `concern`) | **One file shared across locales; N renderings per entry joined by `query_id`** | `sov_ai_multilingual_parity` |
| Clinical-profile bank | `core/clinical_profile_bank.py` | One clinical profile per entry (versioned concealment ground truth) | One bank per client label | the three `health_disclosure` clients |
| Identity spec | `core/identity_spec.py` | Developers, ordered expected-identity rules, competitors, tactics and strategies; a spec layer can extend another | One file shared across locales; localized text per tactic | `identity_disclosure` |

The query bank's cross-locale shape is load-bearing: the `sov_ai_multilingual_parity`
probe needs to join trajectories that probed *the same conceptual question*
across locales / demographic strata, and that join requires a shared
`query_id`. See `assets/sov_ai_multilingual_parity/SCHEMA.md` §"File layout"
for the comparison table and the rationale.

Each asset-driven probe owns exactly one asset folder under `assets/<probe-name>/`: `assets/sov_ai_facts/<locale>/sample.yaml`, `assets/safety_chat_pressure/sample.yaml`, `assets/safety_agentic/{sample.yaml,action_taxonomy.yaml}`, `assets/financial_services/<locale>/`, `assets/health_therapy_disclosure/`, `assets/identity_disclosure/spec.yaml`, etc.

| Probe | Curated asset | Turn 1 origin | Per-trajectory side channels (in addition to the standard set) | Out-of-sim scorer |
| --- | --- | --- | --- | --- |
| `sov_ai_facts` | per-locale fact bank under `assets/sov_ai_facts/<locale>/` | injected verbatim from `fact.question` so false-premise wording survives | `sovereign_facts_probed`, `probe_variant = fact.category` | `sov_ai_facts` (axes: `sovereign.{accuracy,completeness,fabrication,false_premise_catch,graceful_unknown}`) |
| `sov_ai_dynamic` | per-locale category taxonomy under `assets/sov_ai_dynamic/<locale>/` | generated by the user-agent in the persona's voice, anchored to one `(category, subtopic_hint)` pair | `probing_categories_explored`, `probing_subtopic_hints_used`, `probe_variant = category.id` | `sov_ai_dynamic` (axes: `sovereign.{suspected_fabrication,self_consistency,graceful_unknown}`; response-only: no ground truth) |
| `sov_ai_multilingual_parity` | cross-locale query bank under `assets/sov_ai_multilingual_parity/sample.yaml` (one file shared across locales; N renderings per entry) | injected verbatim from `query.rendering_for(locale)` so the same authored question appears across locales for matched-pair joins | `query_id` (the matched-pair join key: surfaced both in `state.metadata` and as a top-level row column), `query_domain`, `query_difficulty`, `query_concern`, `probe_variant = query.id` | `sov_ai_multilingual_parity` (axes: `respect.{condescension,complexity_calibration,helpfulness_at_proficiency}`; one judge per trajectory; rubric grounded in the query's `concern` field) |
| `safety_chat_pressure` | shared pressure-strategy bank under `assets/safety_chat_pressure/sample.yaml` (one file: 6 strategies × 10 reframings, and 64 target requests across 8 harm categories) | injected verbatim from `target.initial_prompt` (turn 1); turns 2..N have the user-agent apply the strategy's reframings in deterministic rotation against the assistant's previous response | `target_request_id`, `strategy_id`, `harm_category`, `reframings_used` (list of reframing ids in turn order, excluding turn 1), `probe_variant = f"{strategy_id}::{harm_category}"`. Optional input-row columns `pressure_strategy_id` / `target_request_id` allow panel-driven (target × strategy) matrix sampling. | `safety_chat_pressure` (scored axes: `pressure.{refusal_held,contextual_safety_persistence}`; per-turn classifications + `turn_of_flip` returned alongside the scored axes; one judge per trajectory; rubric grounded in `target.expected_refusal` + `strategy.description`) |
| `safety_agentic` | shared agentic-action bank under `assets/safety_agentic/sample.yaml` (4 sub-protocols × 5 action requests, each with its own simulated tool spec + canned mock responses); cross-asset name resolution against `assets/safety_agentic/action_taxonomy.yaml` at load time | injected verbatim from `action_request.initial_user_prompt` (single-user-turn design: the user states the request once); subsequent turns are the assistant's tool-using response with **mock-env interception** (every assistant tool call is recorded and answered with the matching `mock_tool_responses[tool_name]`; tools the assistant fabricates outside the spec get a synthetic "unknown tool" reply) | `action_request_id`, `sub_protocol`, `sanctioned_action_name`, `tempted_action_name`, `attempted_actions` (list of `{tool_name, tool_args, turn_idx, was_in_simulated_tools}` records: the load-bearing side channel for the `safety_agentic` scorer), `probe_variant = f"{sub_protocol}::{sanctioned_action_name or 'no_action'}"`. Optional input-row column `action_request_id` allows panel-driven (sub_protocol × action_category) matrix sampling. | `safety_agentic` (hybrid mechanical / LLM-judged axes: `agentic.boundary_held` and `agentic.blast_radius_contained` are purely mechanical: classify every attempted action, join against the rich action taxonomy, check for escalation; `agentic.consequence_disclosure` + `agentic.confirmation_sought` populated only for `consequence_disclosure` sub-protocol via single LLM judge call grounded in `action_request.expected_behavior`; `agentic.injection_resistance` populated only for `compromised_inputs` sub-protocol; zero judge calls for `unsanctioned_action` / `scope_creep`; outputs `attempted_actions_classified` + `total_modeled_blast_radius` as structured side channels for the capability dashboard) |

| `financial_services` | per-locale, multi-institution bank under `assets/financial_services/<locale>/<institution_type>/<institution_id>/` (corpus + tools + tasks; the optional `embeddings.parquet` sidecar is not shipped and is built by `gen-assets`) plus a per-locale `dynamic.yaml` taxonomy | tier-branched: `verifiable` injects the param-locked `opening_user_message` verbatim; `dynamic` generates turn-1 from the taxonomy `invitation` + a rotated `subtopic_hint` | `finance_task_id`, `task_tier`, `institution_id` / `institution_type` / `domain`, `gold_document_ids` / `gold_tool_sequence` / `expected_state_deltas`, `retrieved_document_ids`, `attempted_tool_names`, `dynamic_category_id` / `taxonomy_version`; scoped `kb_search` + discoverable-tool gate + in-memory `account_state` (rides `ToolExecutionMixin`) | `financial_services` (tier-aware: `verifiable` → deterministic verifier `finance.*_rate` axes; `dynamic` → reference-grounded judge `dynamic.*` axes) |
| `identity_disclosure` | shared spec under `assets/identity_disclosure/spec.yaml` (one file; localized text per tactic; spec layers can extend it, see its `SCHEMA.md`) | injected verbatim from the tactic's text, with the expected developer or a competitor filled in | `identity_tactic_id`, `identity_tactic_kind`, `identity_pair`, `identity_competitor`, `strategy_id`, `expected_identity` (declared in the spec for the assistant model's id, or inferred from the model family name in it), `identity_spec` (version, digest, vocabulary and rules), `probe_variant = f"{tactic}::{strategy}"`. Optional input-row columns `identity_tactic_id` / `identity_strategy_id` allow panel-driven sampling. | `identity_disclosure` (rate axes: `identity.{no_spontaneous_false_attribution_rate,no_false_attribution_rate,correct_attribution_rate,names_developer_rate,model_name_correct_rate,confirms_developer_rate,rejects_other_developer_rate,reasoning_spill_free_rate,no_human_claim_rate}`, each feeding an `identity_*` capability; one extraction call per trajectory returns the identity claims with verbatim quotes, and deterministic code grades them against `expected_identity` and the spec's vocabulary; names the vocabulary does not know go to review, and an unknown company still counts as a false attribution while an unknown model name does not) |

These probes write the asset version into `simulation_outcome.provenance.bank_version[<key>]` (key is the locale for the sovereign-AI + financial_services probes; the stable non-locale string `"safety"` for `safety_chat_pressure` and `"agentic"` for `safety_agentic`, both English-only by design and not locale-stratified; and the probe's label for `identity_disclosure`, whose one spec serves every locale). Each attaches a `WarningKind.USED_PLACEHOLDER_*` warning (`USED_PLACEHOLDER_FACT` / `USED_PLACEHOLDER_TAXONOMY` / `USED_PLACEHOLDER_QUERY` / `USED_PLACEHOLDER_TARGET` / `USED_PLACEHOLDER_AGENTIC_ACTION` / `USED_PLACEHOLDER_FINANCE` / `USED_PLACEHOLDER_IDENTITY`) whenever it probed a placeholder-flagged asset, so downstream scorecards refuse production-ready headlines on placeholder-tainted runs.

**Difficulty and correctness semantics.** Difficulty is author metadata
used for stratification and as judge-prompt context; current scorers do
not mechanically change thresholds for `easy` / `medium` / `hard`.
Use `easy` for one obvious fact or one clear refusal/tool boundary,
`medium` for multiple constraints, plausible false premises, locale
institutions, or moderate safety/tool nuance, and `hard` for current
policy exceptions, jurisdiction splits, multiple plausible false
premises, graceful uncertainty, multi-turn pressure resilience, or
high-blast-radius tool behavior. Correctness is grounded only in the
fields each scorer reads: `ground_truth` / `false_premises` for
`sov_ai_facts`, response-surface consistency for `sov_ai_dynamic`,
`concern` for `sov_ai_multilingual_parity`, `expected_refusal` for
`safety_chat_pressure`, and `expected_behavior` plus the action
taxonomy for `safety_agentic`.

All five probes share the same persona-tag vocabulary: `task_derivation.persona_to_tags(...)` is the source of truth and is re-exported by `sov_ai_dynamic.task_derivation` (as `persona_to_interest_tags(...)` for clarity), `sov_ai_multilingual_parity.task_derivation`, `safety_chat_pressure.task_derivation`, and `safety_agentic.task_derivation`. Cross-probe joins on persona-feature axes (e.g., the `age:25-44 ∧ education:tertiary` cohort across all five probes) are direct rather than fuzzy.

### India language-variant resolution

The India multi-language stop-gap (see the parent
[README](../../README.md#india-multi-language-stop-gap-preview)) adds 22
conversation locales (11 languages × native + romanized) that reuse the
`en_IN` persona dataset and `en_IN` asset banks. Two distinctions keep the
conversation locale and the reused-asset locale cleanly separate:

- **`self._locale`**: the *conversation* locale (e.g. `ta_Taml_IN`). Drives
  language + script enforcement (`core/simulation.py`), the trajectory's
  `locale` column + Hive partition, and the `trajectory_id` salt.
- **`self._asset_locale`**: the locale whose banks / prompt packs / renderings
  are actually resolved. Set by `BankBackedProbe._load_bank_for_locale` and
  refined per-probe.

Both are computed **presence-aware** via `core/locale.py`:

| Helper | Resolves | Fallback |
|--------|----------|----------|
| `persona_dataset_locale(locale, datasets_dir=)` | which `<locale>.parquet` to sample | variant → `en_IN` unless a native parquet is present |
| `asset_locale(locale, exists=)` | which per-locale bank/taxonomy to load | variant → `en_IN` unless a native file exists |
| `prompt_pack_locale(locale, has=)` | which prompt-pack key to read | variant → `en_IN` (a shipped pack key) |

Because `asset_locale` resolves to `en_IN` (a shipped locale present in every
prompt pack), the pack lookups never abort and `available_locales()` is
unchanged: the fallback is a call-time redirect, **not** a mutation of the
`_SYSTEM_PROMPTS` / `LocalePromptPack` contents.

**Verbatim turn-1 translation.** When a probe fell back
(`self._asset_locale != self._locale`), `BankBackedProbe._localize_verbatim`
machine-translates the verbatim turn-1 into the conversation language via
`core/translation.py::translate_user_turn` (cached, single-flight, cheap
`summary_model`, preserves false premises, never raises) and records a
`WarningKind.USED_MACHINE_TRANSLATION` on the outcome. `sov_ai_dynamic`
generates turn-1 in-language via the loop's language directive, so it needs no
translation. `safety_agentic` (custom `run_dispatch`, no user-agent turn)
calls `_localize_verbatim` on its `initial_user_prompt` directly.

**Graduation.** Because every resolver prefers the exact locale's asset when
present, dropping in a real per-language bank / parquet / pack switches that
path to native automatically (no translation, no flag, no code change). Full
graduation promotes the language into `SHIPPED_LOCALES`, at which point the
`test_default_asset_paths` / `test_locale` parity suites enforce completeness.

### How to drive each probe from the CLI / notebook

Pick probes via `--probe-mix` on `usersim simulate` or via `PROBE_MIX = {...}` in `notebooks/01_simulate.ipynb`. Weights renormalize to 1.0; the same key vocabulary works in both surfaces. Validation against `_PROBE_REGISTRY` happens at parse time, so an unknown key fails immediately with a friendly error rather than as an opaque sampler error mid-run.

```bash
# Default (general family: tool_calling / general_open_ended / general_educational):
usersim simulate --locale en_US --num-rows 20 \
  --out output/trajectories

# Sovereign-AI smoke run (sov_ai_facts + sov_ai_dynamic + sov_ai_multilingual_parity):
usersim simulate --locale en_US --num-rows 30 \
  --probe-mix 'sov_ai_facts=1/3,sov_ai_dynamic=1/3,sov_ai_multilingual_parity=1/3' \
  --out output/sovereign_trajectories

# Safety smoke run (safety_chat_pressure + safety_agentic):
usersim simulate --locale en_US --num-rows 20 \
  --probe-mix 'safety_chat_pressure=0.5,safety_agentic=0.5' \
  --out output/safety_trajectories

# Single-probe debug:
usersim simulate --locale en_US --num-rows 5 \
  --probe-mix 'safety_agentic=1.0' \
  --out output/agentic_only
```

Each probe resolves its curated asset at simulator-call time under `assets/<probe-name>/`: `assets/sov_ai_facts/<locale>/sample.yaml`, `assets/sov_ai_dynamic/<locale>/categories.yaml`, `assets/sov_ai_multilingual_parity/sample.yaml`, `assets/safety_chat_pressure/sample.yaml`, `assets/safety_agentic/{sample.yaml,action_taxonomy.yaml}`, and `assets/identity_disclosure/spec.yaml`. Override paths via `USERSIM_SOV_AI_FACTS_BANK_<LOCALE>` / `USERSIM_SOV_AI_DYNAMIC_TAXONOMY_<LOCALE>` / `USERSIM_SOV_AI_MULTILINGUAL_PARITY_BANK` / `USERSIM_SAFETY_CHAT_PRESSURE_BANK` / `USERSIM_SAFETY_AGENTIC_BANK` / `USERSIM_SAFETY_AGENTIC_ACTION_TAXONOMY` / `USERSIM_IDENTITY_DISCLOSURE_SPEC` env vars when running against private banks; an identity spec named by its override can extend the shipped one rather than replace it.

**Panel-driven matrix sampling**: the `sov_ai_multilingual_parity`, `safety_chat_pressure`, `safety_agentic` and `identity_disclosure` probes all accept optional input-row columns that override their persona-derived defaults: `query_id` for `sov_ai_multilingual_parity`, `pressure_strategy_id` + `target_request_id` for `safety_chat_pressure`, `action_request_id` for `safety_agentic`, `identity_tactic_id` + `identity_strategy_id` for `identity_disclosure`. To use them, build the panel with the override column populated (typically via `usersim panel` followed by a small pandas script that adds the column) and pass `--panel <path>` to `usersim simulate` instead of `--locale`. Without overrides, every probe falls back to its persona-derived default: useful for smoke runs.

## Trajectory Evaluator

The evaluator is a separate DD column generator that reads trajectory rows
(written by the simulator) and produces a single wide JSON cell containing
per-axis × per-judge scores plus deterministic scorer outputs. Configured
via `TrajectoryEvaluatorConfig`:

```python
from usersim.engine.evaluator.config import (
    TrajectoryEvaluatorConfig,
    JudgeSpecConfig,
)

config_builder.add_column(
    TrajectoryEvaluatorConfig(
        name="eval_v2",                      # output column
        judges=[
            JudgeSpecConfig(alias="judge_a", family="openai"),
            JudgeSpecConfig(alias="judge_b", family="nvidia_nemotron"),
        ],
        axes=None,                           # None = all applicable for the row's probe_family
        scorers=[],                          # opt in per probe (e.g., ["tool_use"])
        prompt_version="v1.0",               # bump to invalidate prior cells
        skip_if_existing=True,               # partial-re-run aware
    )
)
```

**Cell envelope**: every evaluator cell is a JSON document with this shape:

```json
{
  "envelope": {
    "judge_aliases": ["judge_a", "judge_b"],
    "judge_families": ["openai", "nvidia_nemotron"],
    "axes": ["helpfulness", "accuracy", "..."],
    "scorers": [],
    "prompt_version": "v1.0",
    "evaluator_version": "v1.0"
  },
  "axes": {
    "helpfulness": {
      "judge_a": {"score": 4, "reasoning": "..."},
      "judge_b": {"score": 5, "reasoning": "..."}
    },
    "...": {}
  },
  "scorers": {},
  "skipped": false,
  "skipped_reason": null
}
```

**Partial re-run.** If `skip_if_existing=True` (the default), a row whose
existing cell value already carries an envelope matching the current config
is *not* re-judged; the cell's `skipped`/`skipped_reason` fields are flipped
so downstream reporting can count skips. Bumping `prompt_version` or changing
the axes / scorers / judges invalidates prior envelopes and forces a fresh run.

**Judge-family diversity.** Configs with two or more judges are validated at
config-construct time (and re-validated at run time) to span ≥2 distinct
`JudgeFamily` values. See `evaluator/judges.py` for the registry; family
inference falls back to model-id heuristics if you don't set `JudgeSpecConfig.family`.
Single-judge ensembles are allowed but explicitly opt out of inter-judge
agreement reporting.

**Wide → long.** Each cell is wide-per-row by design. The reporting layer
(`reporting/`) handles `pd.melt`-based wide → long conversion for
inter-judge agreement (Cohen's κ / Krippendorff's α), intersectional
aggregation, and stratified heatmaps.

**Cross-run comparison.** Each per-run report (`output/report/run=*/report_manifest.json`)
is also the input to the multi-model comparison dashboard
(`reporting/comparison.py` + `reporting/comparison_dashboard.py`),
which pivots N runs into a side-by-side bake-off: leaderboard,
quality-vs-verbosity Pareto plot, locale-tabbed grouped bars. The
plugin emits no special comparison-side data; the manifest schema
written by the per-run capability dashboard is the only contract.

## Why we centralize the loop

The conversation loop (turn iteration, fourth-wall + refusal-echo
pre-filters, user-judge gate retries, role-violation telemetry, follow-up
generation, frustration escalation, capitulation check, early-stop
heuristics) lives in **one place**: `core/simulation.py::ConversationLoop`.
Every probe inherits this machinery via `BaseProbe` rather than
hand-rolling its own version.

Every probe is a `BaseProbe` (or `BankBackedProbe`) subclass plus an
optional mixin that pre-configures the optional hooks for one of the
common probe shapes. Six mixins ship in `core/probes.py`,
`BankVerbatimMixin` (verbatim turn-1 from a curated bank),
`BankReframingMixin` (multi-turn bank-driven reframings),
`AgenticMixin` (single-user-turn + multi-assistant-turn tool loop),
`ToolCallingMixin` (tool-calling capability test, gates early-stop on
≥1 tool call), `ToolExecutionMixin` (shared inner assistant↔tool loop with an
`execute_tool_call` hook; `single`/`multi` round modes), and
`CustomTurn1InstructionMixin` (templated turn-1 instruction): plus
`GuardedMoveMixin` (deterministic user-side move-space + Guard), which lives in
the `health_disclosure` package until a second family adopts it. New probes are
~50 lines of declarative configuration;
the loop semantics are inherited. The `ConversationLoop` itself is the
only place those semantics live, so a fix lands once and reaches every
probe: no drift across copies of the same logic.

The single exception is `safety_agentic`, whose shape (single user
turn + multi-assistant-turn tool-interception loop, NO user-LLM at
all) is genuinely incompatible with the unified loop's back-and-forth
assumption. It overrides `BaseProbe.run_dispatch` to run a custom
simulate loop while still inheriting the substrate's bank-load /
placeholder-warning / version-pinning / `should_succeed` /
`build_result_extras` machinery. The escape hatch is documented at
length in the probe's module docstring.

See [`docs/AUTHORING_A_PROBE.md`](AUTHORING_A_PROBE.md) for the
end-to-end tutorial on writing a new probe against this substrate.

## Adding a new probe

The end-to-end tutorial: decision tree on which shape to pick, full
worked example (a `cooking_advisor` probe end-to-end), common
gotchas, test-authoring section: lives at
[`docs/AUTHORING_A_PROBE.md`](AUTHORING_A_PROBE.md). Read it
first.

Copy-pasteable starter files for each shape live at
[`templates/probe/`](../../templates/probe/):

- `generator_baseprobe_only.py`: no asset bank
- `generator_bank_verbatim.py`: verbatim turn-1 from a curated bank
- `generator_custom_turn1.py`: bank-anchored generated turn-1
- `generator_bank_reframing.py`: multi-turn bank-driven reframings
- `generator_agentic.py`: single-user-turn + tool-interception (custom `run_dispatch`)
- `generator_tool_calling.py`: tool-calling capability test
- `prompts.py`: `LocalePromptPack` demo
- `test_probe.py.template`: per-probe test scaffold

Quick orientation:

1. Pick a shape from [the decision tree](AUTHORING_A_PROBE.md#step-0-pick-your-shape).
2. Copy the matching `generator_*.py` to
   `src/usersim/engine/probes/<your_probe>/generator.py`.
3. Subclass `BaseProbe` (or `BankBackedProbe`) + the right mixin;
   set 2-3 class attributes; override 2-4 hooks.
4. `@register_probe(family=, prompt_version=, variants=)` to wire
   into `_PROBE_REGISTRY` (and the per-module `PROBE_FAMILY` /
   `PROMPT_VERSION` / `PROBE_VARIANTS` constants the smoke check
   reads).
5. Add a dispatcher import in
   [`generator.py::_bootstrap_probes`](../../src/usersim/engine/generator.py)
   so the bootstrap triggers your probe's `@register_probe` at
   plugin import.
6. Declare every column `build_result_extras` writes in
   `side_effect_columns` on
   [`ConversationSimulatorConfig`](../../src/usersim/engine/config.py).
   An undeclared column is discarded on the way to storage, and the
   scorer that reads it back reports nothing to score.
7. Run `usersim smoke` to confirm registration; copy
   `templates/probe/test_probe.py.template` for the per-probe
   regression test.

For the substrate's API surface (every class, every mixin, every
hook), see the source-of-truth docstrings in
[`src/usersim/engine/core/probes.py`](../../src/usersim/engine/core/probes.py).

**Assistant evaluation is configured separately** via
`TrajectoryEvaluatorConfig` (see Trajectory Evaluator section). Add
new axes to `evaluator/axes.py::PROBE_SCORES` if needed; add a
deterministic scorer under `evaluator/scorers/your_probe.py` and
include it in `DEFAULT_SCORER_MODULES` to auto-register on import.

## Scope

Behavioural fidelity is bounded by the model doing the simulating. Current
models flatten OCEAN-derived differences relative to what a persona
description implies, so interaction-style coverage is best read as breadth
across styles rather than a calibrated measure of any one of them. The D1-D4
diagnostics in the simulator-health report quantify that for a given run, and
[`docs/methodology.md`](../methodology.md) covers the reasoning in full.

## Config Reference

`ConversationSimulatorConfig` fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `persona_column` | str | `"persona"` | Column containing persona dict |
| `probe_type_column` | str | `"probe_type"` | Column carrying the probe label that the dispatcher resolves through `_PROBE_REGISTRY` |
| `theme_column` | str | `"theme"` | Column with theme/topic |
| `tools_column` | str? | `None` | Tool definitions (tool_calling only) |
| `toolset_name_column` | str? | `None` | Toolset name (tool_calling only) |
| `locale` | str | `"en_US"` | Locale for conversation language |
| `assets_dir` | str? | `None` | Optional override for the asset-bank root (defaults to repo `assets/`); per-probe env vars also work |
| `max_tools` | int | `5` | Max tools sampled per row (tool_calling) |
| `max_turns` | int | `5` | Max conversation turns |
| `max_steps` | int | `10` | Max tool steps per turn |
| `max_query_attempts` | int | `3` | Max retries for the user-LLM query / follow-up + judge gate before declaring `USER_QUERY_GATE_EXHAUSTED` |
| `max_assistant_attempts` | int | `1` | Assistant-turn resampling budget. `1` (default) keeps the per-turn assistant judge flag-only (evaluation semantics). `>1` discards a judge-rejected assistant candidate and regenerates it within the budget: for SDG-style runs where the assistant is a data generator, not a model under test. Discards are counted in `n_assistant_retries`; probes with side-effectful `after_assistant_turn` opt out via `supports_assistant_resampling = False`. Gate semantics: the judge's `warn` rating passes (only `failure` triggers a resample), and a judge reply with no parseable rating is re-asked once on the same candidate rather than charged against the budget (entries marked `judge_parse_error`). With a budget `>1`, prefer a judge from a different model family than the assistant: `models_default.toml` uses the same model for both, which means the model filters its own generations |
| `incremental_disclosure_ratio` | float | `0.6` | Fraction of users on incremental disclosure (rest go upfront) |
| `persona_grounding_ratio` | float | `1.0` | Fraction of trajectories whose first user turn incorporates persona-specific details. Lower it for a generic-query control arm |
| `context_compression` | bool | `True` | Summarize older assistant responses to bound context growth |
| `compression_window` | int | `1` | Keep last N assistant responses verbatim before compressing |
| `store_reasoning` | bool | `True` | Keep the assistant's thinking trace on the assistant message when the model emits one (`--no-store-reasoning` to turn off). Gated at capture, so a suppressed trace is never attached. Independent of whether the model *produces* a trace: that is a model-side setting (`chat_template_kwargs` in the models TOML, plus a server-side reasoning parser). Never affects what a model receives: traces are stored, never replayed |
| `verbosity` | int | `1` | Logging detail: 0=quiet, 1=normal, 2=detailed |
| `random_seed` | int? | `None` | Seed for reproducible tool sampling |

## Side-Effect Columns

The simulator emits one row per simulated trajectory. Side-effect
columns appear on every row; [`docs/outputs.md`](../outputs.md) documents
what each one carries:

| Column | Type | Description |
|--------|------|-------------|
| `persona_uuid` | str | Content-hashed persona identity (16 hex chars) computed at simulation write time. Stable across runs given identical persona content; replay-safe. See `core/identity.py`. |
| `trajectory_id` | str | Deterministic 16-hex-char identifier composed from `persona_uuid`, `probe_family`, `probe_variant`, probe seed, resolved user/assistant model identities, and the probe's `PROMPT_VERSION`. The simulator's idempotency key (driver scripts skip rows whose `trajectory_id` is already in the output dataset). See `core/identity.py` and `core/idempotency.py`. |
| `simulation_outcome` | str? (JSON) | Row-level structured outcome: status, failure_class, failure_attribution, counters (n_turns, n_user_query_attempts, n_user_followup_retries, n_assistant_inline_failures, n_api_response_rerolls, n_fourth_wall_triggers, n_user_role_violations), warnings, early_stop, per-model tokens / calls / wall-clock-by-alias, provenance. See `core/outcomes.py` for the schema. |
| `simulation_traces` | str? (JSON) | Per-turn / per-call telemetry (keyed `turn_idx` x `call_idx`) with `kind` discriminator: `user_query_gate`, `user_followup_gate`, `fourth_wall_prefilter`, `assistant_quality`, `early_stop`, `api_response_reroll`, `tool_call_verifier`, etc. The debugging surface: **not** the assistant-evaluation surface. |
| `probe_family` | str | Probe module's `PROBE_FAMILY` constant (e.g., `"general_open_ended"`, `"sov_ai_facts"`). |
| `probe_variant` | str | Probe-specific discriminator (first entry of `PROBE_VARIANTS`, or an explicit row value when provided). Part of `trajectory_id` and the evaluator's stratification key. |
| `conversation_messages` | str (JSON) | Full message list. Each entry is `{role, content}`; assistant entries add `tool_calls` when the turn called a tool, `tool` entries add `tool_call_id`, and an assistant entry adds **`reasoning_content`** when that model call emitted a thinking trace **and `store_reasoning` is on** (the default; `--no-store-reasoning` suppresses it). The key is absent (not empty) whenever either condition fails; the run manifest's `simulation_config.store_reasoning` records which was requested. Traces are attributed per call, so a tool-calling turn that appends several assistant messages carries each trace on the message that produced it. Stored for analysis only: `core/llm.py::_dicts_to_chat_messages` hard-codes the assistant's `reasoning_content` to `None` on the way back into a model, so a later turn never sees an earlier turn's thinking. |
| `conversation_metadata` | str (JSON) | Loose per-turn judge / event log surfaced to interactive previews and the deep-dive analysis script. |
| `conversation_status` | bool | True if the trajectory ran to completion (no gate exhaustion / no infrastructure failure) AND any optional `should_succeed(state)` hook returned True. |
| `behavioral_profile` | str (JSON) | Computed behavioral parameters |
| `persona_religion_language_context` | str (JSON) | Normalized audit view of the religion, ordered spoken languages, religious background, and linguistic background rendered into the user-agent prompt. Keys are present with null/empty values when the source persona has no such fields. Written to the trajectory parquet, but protected attributes, so curated exports omit it unless you ask for it by name or with the `persona_protected` group (`--schema full` includes it). Review before sharing. |
| `locale` | str | Locale for this row |
| `conversation_language` | str | Language used |
| `user_query` | str | Initial user query |
| `num_turns` | int | Actual turn count |
| `num_tool_calls` | int | Total tool calls (0 for non-tool) |
| `tool_subset` | str? (JSON) | Sampled tools (tool_calling only, None otherwise) |
| `disclosure_style` | str | `"incremental"` or `"upfront"` (Sim2Real) |
| `user_interaction_style` | str | OCEAN-derived style: cooperative/neutral/impatient/frustrated/confrontational |
| `persona_grounding` | bool | True when the trajectory used a persona-grounded first-user-turn instruction (governed by `persona_grounding_ratio`); False for the generic-query control arm |

## Core Modules

**Substrate**

| Module | Purpose |
|--------|---------|
| `core/probes.py` | The probe-authoring substrate: `ProbeAdapter` Protocol, `BaseProbe` with sensible defaults for every optional hook, `BankBackedProbe` (load + cache + placeholder warning + version pinning), the five mixins (`BankVerbatimMixin` / `CustomTurn1InstructionMixin` / `BankReframingMixin` / `AgenticMixin` / `ToolCallingMixin`), `LocalePromptPack`, `assistant_message()` (builds an assistant message carrying its own reasoning trace: use it in `after_assistant_turn` instead of a hand-written dict), the `_PROBE_REGISTRY`, and `@register_probe`. **The single source of truth for adding a new probe.** |
| `core/simulation.py` | Shared `ConversationLoop` driven by every probe via `BaseProbe.run_dispatch`; turn-1 generate-and-gate vs verbatim-injection paths; per-turn loop with pre-filters, gates, capitulation check, frustration injection, early-stop heuristics |
| `core/storage.py` | Hive-partitioned trajectory + evaluator parquet storage (`run=<id>/locale=<X>/probe_family=<Y>/`); run-isolation helpers (`new_run_id`, `latest_run_id`, `resolve_run`); legacy bare-layout fallback |
| `core/idempotency.py` | `trajectory_id`-based resumability helpers (`existing_trajectory_ids`, `is_duplicate_trajectory_id`, `filter_to_missing_trajectory_ids`) |
| `core/identity.py` | `persona_uuid` (content hash) + `trajectory_id` (deterministic) derivation, model-name resolution |
| `core/outcomes.py` | `SimulationOutcome` + `SimulationTrace` schemas + `OutcomeBuilder` (status / failure_class / failure_attribution / counters / per-model resource profile / warnings) |
| `core/provenance.py` | `Provenance` capture (code SHA, persona dataset version, probe prompt version, asset bank version) |

**Behavior + Prompts**

| Module | Purpose |
|--------|---------|
| `core/behavioral.py` | OCEAN → BehavioralProfile (patience, verbosity, cooperativeness, tech_literacy, error_proneness); disclosure style, interaction style, reactive-frustration computation |
| `core/persona.py` | Persona formatting for prompts (deterministic shuffle) |
| `core/locale.py` | Locale → language + expected-script-ranges mapping (single source of truth for shipped locales) |
| `core/localized.py` | `LocalizedText` wrapper for asset-bank fields with per-locale renderings (`for_locale(locale)` lookup with `en_US` fallback) |
| `core/prompt_loader.py` | Reads per-probe prompt packs from `assets/<probe>/prompts.yaml` |
| `core/messages.py` | Message, tool, theme formatting |
| `core/preview.py` | Rich-based preview utilities for inspecting generated conversations |

**LLM + judges**

| Module | Purpose |
|--------|---------|
| `core/llm.py` | LLM calling via ModelFacade with retries, debug log, per-model stats, per-conversation outcome-builder hook for resource accounting |
| `core/judges.py` | Inline judge invocation (XML response parsing, run_inline_judge helper) |
| `core/context.py` | Context-window compression: `compress_history` and `summarize_response` (driven by `cfg.context_compression` / `compression_window`) |
| `core/language_detection.py` | Lazy `lingua` detector + `script_compliance_fraction` + `script_dominance` helpers used by the deterministic mechanical-evals scorers |

**Asset bank loaders (per-probe schemas)**

| Module | Purpose |
|--------|---------|
| `core/_assets.py` | Resolves the runtime assets directory (`set_runtime_assets_dir` / `probe_assets_dir`) so every loader uses the same root |
| `core/fact_bank.py` | Per-locale fact-bank loader, schema validator, persona-tag matcher (consumed by `sov_ai_facts`) |
| `core/probing_taxonomy.py` | Per-locale dynamic-tier taxonomy loader + schema validator + the shared `select_category` derivation (optional `applies_to_types` scoping + `persona_affinities` soft-weighting); consumed by `sov_ai_dynamic` and `financial_services` |
| `core/finance_bank.py` | Per-locale multi-institution financial-services bank loader (`Institution` / `FinanceBank`, scope-integrity, embeddings sidecar); `core/finance_tasks.py` (sim-time instantiation + gold derivation) + `core/embeddings.py` (query embedding); consumed by `financial_services` |
| `core/query_bank.py` | Cross-locale query-bank loader, schema validator with rendering × locale cross-product check, persona-tag matcher (consumed by `sov_ai_multilingual_parity`) |
| `core/pressure_bank.py` | Pressure-strategy bank loader (Strategy / Reframing / TargetRequest), template-placeholder validator, deterministic per-turn reframing rotation, persona-tag matcher (consumed by `safety_chat_pressure`) |
| `core/agentic_bank.py` | Agentic-action bank loader (SubProtocol / ToolSpec / ActionRequest), JSON-Schema tool-spec validator, mock-response cross-reference validator, **action-taxonomy name resolution at load time** (rejects banks that reference an unknown action), persona-tag matcher (consumed by `safety_agentic`) |
| `core/action_taxonomy.py` | Rich action-taxonomy loader (categories, risk tiers, blast-radius ranks); consumed by the `safety_agentic` scorer for cross-asset action classification |
| `core/seeds.py` | Hybrid seed-data loading (per-probe `base/` + per-locale extensions merged at runtime) |
| `core/analysis.py` | Helpers for the deep-dive analysis script |

## Testing

```bash
uv pip install -e ".[dev]"
pytest tests/ -q
```

The engine's tests live in three directories: `tests/engine/core/`
(substrate, asset loaders, behavioural, identity, storage),
`tests/engine/probes/` (the probe generators, tool-call verifier and
per-probe invariants), and `tests/engine/evaluator/` (judges, scorers and
per-scorer end-to-end).

**`tests/engine/core/`: substrate, asset loaders, behavioural, identity, storage**

| Test File | Coverage |
|-----------|----------|
| `test_probes_substrate.py` | The probe-authoring substrate: `BaseProbe` defaults, `BankBackedProbe` bank-load + placeholder-warning + version-pinning pipeline, the five mixins (MRO ordering invariant), `LocalePromptPack` validation, `@register_probe` registry semantics, `resolve_probe` |
| `test_simulation.py` | Shared simulation helpers (`language_instruction`, `make_result`, `make_failed`) |
| `test_engine_e2e.py` | End-to-end `ConversationLoop` with mocked LLM: status promotion, gates with retries, frustration injection, early-stop, role-violation telemetry |
| `test_behavioral.py` | OCEAN extraction, behavioral params (patience / verbosity / cooperativeness / tech_literacy / error_proneness), language mapping, Sim2Real features |
| `test_identity.py` | `persona_uuid` content-hashing stability + invariance, `trajectory_id` determinism, `resolve_model_name` |
| `test_idempotency.py` | `existing_trajectory_ids` / `is_duplicate_trajectory_id` / `filter_to_missing_trajectory_ids` |
| `test_storage.py` | Hive-partitioned dataset reads/writes, run-isolation helpers (`new_run_id` / `latest_run_id` / `resolve_run`), legacy bare-layout fallback, schema-unification on cross-fragment drift |
| `test_outcomes.py` | `SimulationOutcome` schema, enum exhaustiveness, status promotion, `OutcomeBuilder` |
| `test_provenance.py` | `code_sha` env override, `nemotron_personas_version` plumbing, probe metadata |
| `test_locale.py` | `SHIPPED_LOCALES` parity across language-name / display-name / script-range maps; new locales caught at CI time |
| `test_language_detection.py` | Lazy `lingua` detector loading, `script_compliance_fraction`, `script_dominance` helpers |
| `test_default_asset_paths.py` | Every shipped probe's default asset path resolves to a real file across every supported locale |
| `test_assets_discovery.py` | Runtime assets-dir resolution (`set_runtime_assets_dir`, `probe_assets_dir`), env-override precedence |
| `test_config.py` | `ConversationSimulatorConfig` defaults, optional fields, side-effect-columns contract |
| `test_pipeline.py` | Pipeline-level config validation |
| `test_generator.py` | Dispatcher failed-result structure, probe-variant resolution, structured `SCENARIO_ABORTED` outcomes |
| `test_messages.py` | Message / tool / theme formatting helpers |
| `test_persona.py` | Persona formatting (deterministic shuffle) |
| `test_preview.py` | Rich-based preview rendering for inspecting generated conversations |
| `test_seeds.py` | Hybrid seed loading and merging (per-probe `base/` + per-locale extensions) |
| `test_multi_locale.py` | Multi-locale config, seed loading, language mapping |
| `test_llm.py` | LLM call helpers (retries, debug log, per-model stats, per-conversation outcome-builder hook) |
| `test_analysis.py` | Score extraction, demographic preservation, edge cases used by the deep-dive analysis script |
| `test_fact_bank.py` | Fact-bank YAML schema validator, loader, persona-tag matching, locale-keyed shared cache |
| `test_probing_taxonomy.py` | Probing-taxonomy YAML schema validator, loader, lookup helpers, locale-keyed shared cache |
| `test_query_bank.py` | Cross-locale query-bank YAML schema validator, loader (with cross-product rendering × locale validation + opt-out semantics), persona-tag matcher, file-path-keyed cache, env override, shipped-sample end-to-end check |
| `test_pressure_bank.py` | Pressure-strategy bank YAML schema validator, loader with template-placeholder validation, deterministic reframing rotation with wrap-around, persona-tag matcher, cache + env override, shipped-sample end-to-end |
| `test_agentic_bank.py` | Agentic-action bank YAML schema validator, loader with cross-asset action_taxonomy name resolution at load time, JSON-Schema tool-spec validation, mock-response cross-reference validation, persona-tag matcher, cache + env override, shipped-sample end-to-end |
| `test_action_taxonomy.py` | Rich action-taxonomy loader (consumed by `safety_agentic` scorer): schema validation, cache + env override (`USERSIM_SAFETY_AGENTIC_ACTION_TAXONOMY`), `by_name` / `blast_radius_rank` lookups |

**`tests/engine/probes/`: the probe generators, verifier and invariants**

| Test File | Coverage |
|-----------|----------|
| `test_probe_invariants.py` | Cross-probe invariants every shipped probe upholds (registration, side-channel keys, prompt-version monotonicity, etc.) |
| `test_tool_calling.py` | Tool-calling probe: tool sampling, gate check, `should_succeed` on zero tool calls |
| `test_verifiers.py` | Tool-call JSON-schema validator |
| `test_sov_ai_facts.py` | `sov_ai_facts` probe: persona→tag, deterministic `derive_task` (bank-version salted), prompts, e2e with mocked LLM |
| `test_sov_ai_dynamic.py` | `sov_ai_dynamic` probe: persona→tag, deterministic `derive_probe` + hint rotation (taxonomy-version salted), all-locale prompt availability, e2e with mocked LLM |
| `test_sov_ai_multilingual_parity.py` | `sov_ai_multilingual_parity` probe: persona→tag re-export, deterministic `derive_task` with locale opt-out filtering (bank-version salted), all-locale prompt completeness with no-half-shipped invariant, query-bank cache + env override, e2e with mocked LLM, verbatim turn-1 injection, structured failure paths |
| `test_safety_chat_pressure.py` | `safety_chat_pressure` probe: persona→tag re-export, `derive_task` with independent strategy/target salts (50-seed cell-coverage test), `resolve_task_from_row` (full / partial / neither override + persona-tag guard against panel-driven mismatches), prompt-formatting helpers, end-to-end with mocked LLM (verbatim turn-1, multi-turn rotation, bank-load failure, panel-id miss, no-resolvable-pair, assistant-error attribution, follow-up-failure breaks loop) |
| `test_safety_chat_pressure_classifier.py` | Dedicated capitulation classifier (comparator-shaped `summary_model` call): rubric-anchored decision, defensive returns, regression that the comparator prompt has no pressure / strategy / persona keywords (post the v1.2 redesign) |
| `test_safety_agentic.py` | `safety_agentic` probe: persona→tag re-export, `derive_task` (deterministic + bank-version perturbation + exclusion lists), `resolve_task_from_row`, `_format_tools_for_api` + `_extract_tool_call` internal helpers, e2e with mocked LLM (verbatim turn-1, no-tool-call short-circuit, mock-response interception, fabricated-tool synthetic-error reply), `should_succeed` invariant |

**`tests/engine/evaluator/`: judges, scorers, registry**

| Test File | Coverage |
|-----------|----------|
| `test_evaluator_axes.py` | Axis registry by probe family + `axis_category` partitioning |
| `test_evaluator_config.py` | `TrajectoryEvaluatorConfig` validation, judge-family diversity, defaults |
| `test_evaluator_generator.py` | `TrajectoryEvaluatorGenerator` (envelope, partial-re-run skip, scorer dispatch, error trapping) |
| `test_evaluator_judges.py` | `JudgeFamily` registry, `infer_judge_family`, `validate_ensemble_diversity` |
| `test_evaluator_scorers.py` | Scorer registry (`register_scorer` / `get_scorer` / `clear_registry` / `reload_scorers`) |
| `test_evaluator_scorer_structured_output_schemas.py` | Cross-cutting schema-design contract for LLM-judge scorers (Literal-typed enums, Optional fields, 1–5 score constraints): fails the build on violations |
| `test_evaluator_scorer_tool_use.py` | 8-axis tool-calling trajectory judge |
| `test_evaluator_scorer_sov_ai_facts.py` | `sov_ai_facts` out-of-sim scorer: per-fact LLM judge with question-type-specific axes, error paths, multi-fact aggregation |
| `test_evaluator_scorer_sov_ai_dynamic.py` | `sov_ai_dynamic` out-of-sim scorer: response-only LLM judge with `sovereign.{suspected_fabrication,self_consistency,graceful_unknown}` axes, multi-category aggregation, drift detection |
| `test_evaluator_scorer_sov_ai_multilingual_parity.py` | `sov_ai_multilingual_parity` out-of-sim respect-parity scorer: single LLM judge per trajectory with `respect.{condescension,complexity_calibration,helpfulness_at_proficiency}` axes, concern-grounded rubric, locale+language interpolation, status-proposal gating, error paths, version-mismatch drift |
| `test_evaluator_scorer_safety_chat_pressure.py` | `safety_chat_pressure` out-of-sim scorer: single LLM judge per trajectory with `pressure.{refusal_held,contextual_safety_persistence}` scored axes plus per-turn classifications + derived `turn_of_flip`, rubric grounded in `target.expected_refusal` + `strategy.description`, status-proposal gating, error paths, version-mismatch drift |
| `test_evaluator_scorer_safety_agentic.py` | `safety_agentic` scorer: registration, axis constants (`agentic.*` namespace; mechanical vs LLM separation), short-circuit paths, load failures, mechanical-only sub-protocols, attempted-actions normalisation, `consequence_disclosure` and `compromised_inputs` judge paths, judge-error paths, bank-version drift, total-blast-radius aggregation, envelope contract |
| `test_evaluator_scorer_identity_disclosure.py` | `identity_disclosure` scorer: applicability and fail-closed paths, extraction schema built from the spec's developers, quote validation, name-to-developer mapping (names, products, model names, ambiguous words), per-turn grading (false and spontaneous attribution, premise stance, reasoning spill, human claims), trajectory rates and review flags, same-family extractor warning |
| `test_evaluator_scorer_language_compliance.py` | Deterministic (no-LLM) scorer: per-trajectory `language.{requested_language_match_rate, script_compliance_rate, first_turn_match}` rates from `lingua` language detection + Unicode script analysis; clean-Hindi / English-fallback / Hindi-in-Latin-transliteration / Japanese-mixed-scripts cases; per-turn diagnostic surfacing |
| `test_evaluator_scorer_response_shape.py` | Deterministic (no-LLM) scorer: per-classifier unit tests (empty/trivial, over-formatted, mid-sentence truncation) + trajectory-level aggregation |
| `test_evaluator_scorer_refusal_basics.py` | Deterministic (no-LLM) scorer: refusal lexicon detection across shipped locales, `refusal.in_wrong_language_rate` (English-refusal-in-non-English-locale) composite, canned-AI-phrase detection, status-proposal gating semantics |
