# Population-probing taxonomy schema

A **probing taxonomy** is a curated, per-locale YAML file that drives
the `sov_ai_dynamic` probe. It declares the
categories the user-agent is allowed to ask about, plus per-category
prompt fragments used to seed the conversation.

This file is **not** a fact bank. It contains no ground truth, no
canonical answers, no embedded false premises. The whole point of
sov_ai_dynamic is to surface the model's behavior on the
**long tail** of questions personas naturally ask in those
categories, which a fact bank can never enumerate exhaustively.
Scoring happens via heuristic LLM judges (suspected fabrication,
self-inconsistency, graceful unknown) that read the assistant's
response on its own terms; ground truth is intentionally absent.

The category vocabulary today is identical to the fact bank's
(`brazil-geography`, `brazil-history`, `brazil-culture`,
`brazil-poetry-and-music`, `brazil-public-services`) so per-category
comparison between the two probes is direct. The two taxonomies
are deliberately stored separately (`assets/sov_ai_facts/<locale>/*.yaml`
vs. `assets/sov_ai_dynamic/<locale>/categories.yaml`) so they can
diverge as runs reveal what's worth curating vs. what's worth probing
dynamically. The documented discovery loop
("Why sovereign-AI readiness ships as two complementary probes")
explicitly relies on that separation.

## File layout

```
assets/sov_ai_dynamic/
├── SCHEMA.md             # this file
├── pt_BR/
│   └── categories.yaml   # the pt_BR taxonomy
├── ja_JP/
│   └── categories.yaml   # (future)
└── …
```

One taxonomy per locale per file. Override the default path with the
env var `USERSIM_SOV_AI_DYNAMIC_TAXONOMY_<LOCALE>` (e.g.
`USERSIM_SOV_AI_DYNAMIC_TAXONOMY_PT_BR=/abs/path/categories.yaml`).

## Top-level fields

| Field | Type | Required | Description |
|---|---|---|---|
| `schema_version` | str | ✓ | Schema version this file targets. Currently pinned at `v0.1`. The loader refuses unknown versions. |
| `locale` | str | ✓ | Nemotron-Personas locale id (`pt_BR`, `ja_JP`, …). Should match the parent directory. |
| `taxonomy_id` | str | ✓ | Stable identifier for this taxonomy. Carried on every trajectory in `sov_ai_dynamic` trajectories so cross-run drift is detectable. |
| `taxonomy_version` | str | ✓ | Semver-style version of this file's *contents*. Bump on any edit (added category, reworded invitation, edited sub-topic hints). The version is folded into trajectory provenance the same way `bank_version` is for fact banks. |
| `placeholder` | bool | optional, default `false` | `true` while the taxonomy contains invitations + hints that have **not** been audited by a native-speaker reviewer (typical for placeholder taxonomies that ship to exercise the engineering pipeline). The loader emits a one-line INFO log on placeholder load, and the downstream sovereign-AI scorecard refuses its "production-ready" headline whenever any trajectory probed against a placeholder taxonomy. Reviewers flip to `false` (or delete the field: both are accepted; `placeholder: true` is the opt-in mark) once a native-speaker review has occurred. Bump `taxonomy_version` in the same edit. |
| `categories` | list[Category] | ✓ | Minimum 1 entry. |

## Category fields

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | str | ✓ | Unique within the taxonomy. Should match the corresponding fact-bank category string when one exists (e.g., `brazil-geography`) so the scorecard can stratify across both probes on the same key. |
| `invitation` | str | ✓ | Locale-language prose used in the user-agent system prompt to instruct the persona to ask about something in this category. Should refer to the persona's voice ("pergunte como você falaria com…") rather than dictating a question. |
| `subtopic_hints` | list[str] | ✓ (≥ 2) | Locale-language phrases that nudge the user-agent toward different sub-areas of the category. The probe rotates through these per trajectory to prevent topic collapse: without rotation, the user-agent often falls back on the same one or two stock topics across hundreds of trajectories. **The loader rejects taxonomies with fewer than 2 hints per category**: rotation requires variety. |

## Authoring guidance

- **Invitations should not specify a question.** They should set the topic boundary and let the persona's voice + the sub-topic hint do the work. "Pergunte sobre algo de geografia brasileira que tenha a ver com seu cotidiano" is right; "Pergunte qual é o rio mais longo do Brasil" is wrong (that's a fact-bank entry).
- **Sub-topic hints should be *neighborhoods*, not questions.** "sobre um rio, serra, ou bioma" is right; "qual é o maior rio do Brasil?" is wrong.
- **Sub-topic count.** A useful starting point is 5-8 hints per category. Two is the absolute minimum (so rotation isn't a no-op); fewer than three risks repetitive runs at moderate sample sizes.
- **Authoring language.** Every reviewer-facing string lives in the locale's native language. The judge prompts (English, infrastructure) are the only exception: they reason about responses in the locale language.
- **Discovery loop ergonomics.** As real sov_ai_dynamic runs accumulate, mining "high-flag-rate sub-topics" can suggest new sub-topic hints to add (broadening coverage) AND new fact-bank entries to author (turning dynamic-only knowledge into curated ground truth). That mining step is out-of-band; the column plumbing on the trajectory parquet (`probe_variant = category`, plus per-trajectory judge flags) is what makes it possible.

## Versioning and drift

`taxonomy_version` is consumed exactly like `bank_version`: it lands
on `simulation_outcome.provenance.bank_version[locale]` (we re-use
the same provenance slot: the field name is historical, the value
is generic), and the `sov_ai_dynamic` scorer compare against the
currently-loaded taxonomy version to surface a drift flag per
trajectory when a taxonomy revision lands mid-run.

`schema_version` gates the loader. Bumping it requires lock-step
loader work + an explicit migration of existing taxonomy files.

## Authoring checklist (for reviewers)

- [ ] `schema_version: "v0.1"` pinned at the top.
- [ ] `locale` matches the parent directory.
- [ ] `taxonomy_id` is stable (don't rename mid-revision; bump
      `taxonomy_version` instead).
- [ ] Every category has `id`, `invitation`, `subtopic_hints` (≥ 2).
- [ ] Category ids match the fact-bank's category vocabulary when
      one exists (e.g., `brazil-geography`, `india-public-services`).
- [ ] Invitations describe a topic boundary in the locale's voice;
      they do **not** specify a question.
- [ ] Sub-topic hints are sub-area neighborhoods, not questions.
- [ ] `taxonomy_version` bumped when any reviewer-facing string
      changes.
- [ ] `placeholder: true` is set **iff** the file has not yet been
      audited by a native-speaker reviewer. Promote to `false` (or
      delete the field) only after the audit; bump `taxonomy_version`
      in the same edit so downstream drift detection picks it up.
