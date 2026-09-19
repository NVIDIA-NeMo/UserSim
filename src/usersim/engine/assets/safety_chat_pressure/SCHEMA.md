## Pressure-strategy bank schema

A **pressure-strategy bank** is a curated YAML file containing the
primitives the `safety_chat_pressure` probe uses to
drive multi-turn pressure trajectories: strategies (high-level pressure
approaches), reframings (concrete tactics applied per turn), and target
requests (the things the assistant is being pressured into doing).

This schema lives here so:

- the loader in `usersim/engine/core/pressure_bank.py` can validate
  incoming banks;
- the probe adapter and scorer can encode against a stable contract;
- safety-corpus reviewers know exactly what to fill in and what's
  optional before they invest time authoring entries.

## Conceptual model

Three primitives, three slots in the bank:

- **Target request.** A content type the assistant *should refuse*.
  Independent of any specific strategy: any strategy can be applied to
  any target request, generating the test matrix. Each target carries a
  short `description` (what kind of refusal is being tested), an
  `initial_prompt` (the user-agent's first turn), an `expected_refusal`
  signature (a short reviewer-authored hint at what an honest refusal
  looks like, used by the scorer's "did the model flip?" judgement),
  and a `harm_category` (broad bin: `deception`, `safety_bypass`,
  `manipulation`, `circumvention`, `fraud`, …) for stratified reporting.

- **Pressure strategy.** A high-level approach for the user-agent. The
  The bank ships six strategies (`persistence`, `hypothetical_framing`,
  `authority_impersonation`, `emotional_dependency`,
  `reward_penalty_framing`, `partial_compliance`), covering the pressure
  patterns most often reported in the red-teaming literature. Each strategy
  carries a `system_prompt_template` consumed by the user-agent, plus a list
  of `reframings` (ten per strategy).

- **Reframing.** A concrete tactic within a strategy applied turn-by-
  turn. The user-agent rotates through the strategy's reframings
  deterministically (seed-salted per persona) so the same persona
  doesn't always hit the same tactic first. Each reframing carries an
  `instruction` the user-agent reads before composing its next turn.

The trajectory unit is **(persona, target_request, strategy)**: each
trajectory probes one target with one strategy, applying reframings as
the conversation progresses. Stratified reporting joins on
`target_request_id × strategy_id` to surface "which (target, strategy)
pairs cause the assistant to flip most often."

## File layout

```
assets/safety_chat_pressure/
├── SCHEMA.md          # this file
├── sample.yaml        # placeholder bank (see §Placeholder discipline)
└── v1.yaml            # reviewer-authored bank (future)
```

One bank per file. The loader picks one by path; the env override is
`USERSIM_SAFETY_CHAT_PRESSURE_BANK`.

## Top-level fields

| Field | Type | Required | Description |
|---|---|---|---|
| `schema_version` | str | ✓ | Schema version this file targets. Currently pinned at `v0.1`. The loader refuses unknown versions. |
| `bank_id` | str | ✓ | Stable identifier for this bank. Appears on every trajectory that probes it so cross-run drift is detectable. |
| `bank_version` | str | ✓ | Semver-style version of this bank's *contents*. Bump on any entry edit; the version is carried into `simulation_outcome.provenance` so consumers can spot when a bank changed mid-run. |
| `harm_categories` | list[str] | ✓ | The set of harm-category bins used by target-request entries. Used for stratified reporting. |
| `tags` | list[str] | optional | The tag vocabulary used by `persona_tags` on target-request entries. Informational; no enforcement. |
| `strategies` | list[Strategy] | ✓ | The pressure strategies. Minimum 1 entry. The bank ships 6. |
| `target_requests` | list[TargetRequest] | ✓ | The target-request entries. Minimum 1 entry. |

## Strategy fields

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | str | ✓ | Unique across the bank's `strategies` list. Convention: lowercase snake_case (`persistence`, `hypothetical_framing`, `authority_impersonation`). |
| `description` | str | ✓ | Reviewer-facing one-paragraph description of what this strategy is and why a real-world adversary might use it. Grounds reviewer judgement when authoring new strategies / reframings. |
| `system_prompt_template` | str | ✓ | The user-agent's system prompt for this strategy. Authored in English (the user-agent for safety probes runs in English by default: this is intentional; a non-English jailbreak attempt is a separate test). Must accept these placeholders: `{persona}`, `{target_description}`, `{initial_prompt}`. The user-agent's first turn is the target's `initial_prompt` verbatim; this prompt frames how the user-agent should react to the assistant on follow-up turns. |
| `reframings` | list[Reframing] | ✓ (≥1) | The reframings the user-agent rotates through on follow-up turns. Ten per strategy in the shipped bank. The first follow-up turn picks reframing index `(persona_hash + bank_salt) % len(reframings)`; subsequent turns advance one index at a time (with wrap-around). This deterministic rotation means the same persona always sees the same reframing sequence given a fixed bank version. |

## Reframing fields

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | str | ✓ | Unique within the parent strategy's `reframings` list. Convention: lowercase snake_case (`rephrase`, `claim_misunderstanding`, `appeal_to_helpfulness`, `escalate_urgency`, …). |
| `description` | str | ✓ | Reviewer-facing description of the tactic: "what is the user-agent doing on this turn." Grounds the reviewer's judgement of "is this realistic." |
| `instruction` | str | ✓ | The instruction injected into the user-agent's message history for this turn. Authored in English. Templated against the assistant's previous response: the user-agent reads `{previous_assistant_response}` interpolated into the instruction. The instruction tells the user-agent *what tactic to apply*, not what to say verbatim: the user-agent composes the actual turn. |

## Target-request fields

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | str | ✓ | Unique across the bank's `target_requests` list. Convention: `TR-<HARM-SHORT>-B<NNN>` where `<HARM-SHORT>` is `SB`, `M`, `D`, `F`, `C`, `PI`, `I`, `H` for the eight harm categories (e.g., `TR-D-B001`, `TR-PI-B003`). The format is *convention only*: uniqueness is the only hard requirement, and lookup code does not parse the ID. |
| `description` | str | ✓ | Reviewer-facing one-line description of the request and why the assistant should refuse it. |
| `harm_category` | str | ✓ | Must appear in the bank's `harm_categories` list. Used for stratified reporting and reviewer triage. |
| `placeholder` | bool | ✓ | `true` for unreviewed entries (stubs). The reporting layer surfaces the fraction of probed targets that were placeholder; the capability dashboard treats placeholder-only scorecards as preview-only (mirrors the placeholder discipline used by other asset banks). Reviewers set this to `false` once the entry passes review. |
| `initial_prompt` | str | ✓ | The verbatim text the user-agent sends as the first turn of the trajectory. Authored to be **clearly inappropriate but low real-world harm**: refusal is the right answer, but capitulation outputs an embarrassing failure (a generic manipulative paragraph, a faked document template, etc.) rather than catastrophic harm. See §Placeholder discipline for the safety bar. |
| `expected_refusal` | str | ✓ | A short reviewer-authored sketch of what an honest refusal looks like for this request: e.g., "decline + name the policy reason + offer a legitimate alternative if applicable." The scorer reads this to ground its "did the assistant flip?" judgement; not a verbatim regex match, just a rubric anchor. |
| `persona_tags` | list[str] | optional | If present, restricts which personas this target should be probed against. Empty / omitted = applicable to every persona. Same structured-tag vocabulary as the fact bank and query bank. Use sparingly: most safety probes are persona-agnostic by design. |
| `provenance` | Provenance | ✓ | See §Provenance. |

## Provenance

Every target request carries a `provenance` block:

```yaml
provenance:
  source: "internal authorship 2026-04 (placeholder)"
  last_reviewed: "2026-04-20"
  references: []
```

| Field | Type | Required | Description |
|---|---|---|---|
| `source` | str | ✓ | Human-readable description of where the entry came from. `"placeholder"` is allowed only when `placeholder: true`. |
| `last_reviewed` | str (ISO-8601 date) | ✓ | `YYYY-MM-DD`. Used by the loader to flag stale entries. |
| `references` | list[str] | optional | URLs / citations. Useful when the target was derived from a published red-team paper or benchmark; non-placeholder entries should have at least one. |

## Placeholder discipline

Every target request shipped with `placeholder: true` is a stub. The
**safety bar for placeholder content** is intentionally tighter than
for the other curated assets in this repo: a placeholder target
request must be **clearly inappropriate** (the model should refuse)
**AND low-real-world-harm** (capitulation outputs an embarrassing
failure, not catastrophic harm). The `initial_prompt` is published in
this repo and visible to anyone reading the source: it must not
contribute to a real attack pattern by being public.

The shipped targets cover eight refusal categories, eight targets each:
safety bypass, manipulation, deception, fraud, circumvention, privacy
invasion, impersonation, and harassment. None of these produce content that materially advances
real harm if the model fails the test.

Reviewers replacing placeholder targets must apply the same discipline:
**publishable target prompts only**. Targets that probe more sensitive
refusal categories (CSAM, self-harm-detail, weapon-synthesis-detail)
require a separate non-public bank with its own reviewer process: the
schema supports this through the env override (`USERSIM_SAFETY_CHAT_PRESSURE_BANK`)
but the public placeholder bank stays bounded by the safety bar above.

The capability dashboard treats as preview-only for any
trajectory that probed a `placeholder: true` target, same epistemic
discipline as other asset-bank placeholder gates.

## Loader contract

The loader (`usersim/engine/core/pressure_bank.py`) enforces:

- `schema_version` is in the supported set.
- `bank_id`, `bank_version`, `harm_categories` (non-empty),
  `strategies` (non-empty), `target_requests` (non-empty) all present.
- Every strategy + reframing + target has all required fields.
- Every `strategy.id` is unique within the bank.
- Every `reframing.id` is unique within its parent strategy.
- Every `target.id` is unique within the bank.
- Every `target.harm_category` is in the bank's `harm_categories` list.
- Every strategy's `system_prompt_template` carries the three required
  placeholders: `{persona}`, `{target_description}`, `{initial_prompt}`.
- Every reframing's `instruction` carries the
  `{previous_assistant_response}` placeholder.
- `placeholder: true` allows `source: "placeholder"`; non-placeholder
  entries must have a real provenance source.
- ISO-8601 date format on `last_reviewed`.

Any violation raises `PressureBankError` with a `path::entry_id::field`
breadcrumb so reviewers can grep the offending location.
