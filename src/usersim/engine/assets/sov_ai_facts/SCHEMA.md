# Fact bank schema

A **fact bank** is a curated, per-locale YAML file that pins the ground
truth used by the `sov_ai_facts` probe. Every entry is meant to
be reviewed by a subject-matter expert (SME) for the locale. Every entry
in the shipped banks carries `placeholder: true`: that review hasn't
happened, and SME verification is recommended before relying on results
that rest on them (see §Placeholder discipline).

This schema lives here so:

- the loader in `usersim/engine/core/fact_bank.py` can validate
  incoming banks;
- probe scorers can encode against a stable contract;
- native-language reviewers know exactly what to fill in (and what's
  optional) before they invest time authoring entries.

## File layout

```
assets/sov_ai_facts/
├── SCHEMA.md          # this file
├── pt_BR/
│   └── sample.yaml    # placeholder bank (see §Placeholder discipline)
├── ja_JP/
├── fr_FR/
└── …
```

One bank per file. Each locale directory can hold multiple banks
(e.g. a public bank + a safety-sensitive bank); the loader picks one
by path.

## Top-level fields

| Field | Type | Required | Description |
|---|---|---|---|
| `schema_version` | str | ✓ | Schema version this file targets. Currently pinned at `v0.1`. The loader refuses unknown versions. |
| `locale` | str | ✓ | Nemotron-Personas locale id (`pt_BR`, `ja_JP`, `fr_FR`, …). |
| `bank_id` | str | ✓ | Stable identifier for this bank. Appears on every trajectory that probes it so cross-run drift is detectable. |
| `bank_version` | str | ✓ | Semver-style version of this bank's *contents*. Bump on any entry edit; the `bank_version` is carried into `simulation_outcome.provenance` so consumers can spot when a bank changed mid-run. |
| `categories` | list[str] | ✓ | The set of categories used by entries. Used for stratified scoring. Category ids are English, kebab-case, and locale-prefixed (e.g. `brazil-geography`, `japan-history`) so grep parity across locales is preserved. |
| `tags` | list[str] | optional | The tag vocabulary used by `persona_tags` on entries. Informational; no enforcement. |
| `entries` | list[Entry] | ✓ | The facts themselves. Minimum 1 entry. |

## Entry fields

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | str | ✓ | Unique across the bank. Convention: `<LOCALE>-<CATEGORY-SHORT>-<NNN>` (e.g., `BR-GEO-001`). |
| `category` | str | ✓ | Must appear in the bank's `categories` list. |
| `question_type` | str | ✓ | One of `factual_recall` or `completion`. Drives the scorer rubric (see §Question types). |
| `placeholder` | bool | optional, default `false` | `true` for unreviewed entries (stubs). Every trajectory that probes a placeholder fact records a `used_placeholder_fact` warning on its `simulation_outcome`, and the scorer copies the flag into its per-fact output (`fact_placeholder`). Reviewers delete the field (or set it to `false`) once the entry passes review. |
| `question` | str | ✓ | The probe, written in the locale's language. For `factual_recall` this is the question asked (often with embedded false premises). For `completion` this is the partial verse/passage + the prompt asking the model to continue. |
| `false_premises` | list[str] | required for `factual_recall`; empty allowed | Each entry is one deliberate false assertion embedded in the question. Write each premise **in the locale's native language** (matching `question` and `ground_truth`): the judge prompt operates in the same language as the conversation, and language consistency within an entry is part of the sovereign-AI design. A short parenthetical correction within the same language is recommended (e.g., `"O Amazonas desagua no Pacífico (na verdade no Atlântico, em Belém)"`). Leave empty `[]` if the question carries no false premises. |
| `ground_truth` | str | ✓ | The authoritative answer, written in the locale's language. For `completion`, this is the canonical continuation plus (optionally) a note on what the model should do when it doesn't know the source verbatim. |
| `graceful_unknown` | str | required for `completion`; optional for `factual_recall` | The preferred response when the model doesn't know. For `completion` entries this is what the model *should* say instead of fabricating fluent-but-wrong continuations: e.g., "I don't recall this verse precisely; I'd risk misquoting it." Used by the scorer to reward graceful refusal over hallucinated content. |
| `persona_tags` | list[str] | ✓ (≥1) | Tags used by `derive_task(persona, fact_bank)` to match facts to personas. Follow the structured form below. At least one tag required so every fact is reachable via persona-targeted selection. |
| `difficulty` | str | ✓ | One of `easy`, `medium`, `hard`. Used for stratified scoring. |
| `provenance` | Provenance | ✓ | See §Provenance. |

## Persona tags

Persona tags are **structured** so the matcher can intersect them
deterministically. Allowed prefixes:

| Prefix | Example | Source persona attribute |
|---|---|---|
| `region:` | `region:northeast` / `region:scotland` / `region:any` | derived for pt_BR (macro-region, from the persona's state), en_GB (nation) and de_DE (Land); personas in other locales carry only `region:any` |
| `age:` | `age:18-24` / `age:65+` / `age:any` | the persona's age, binned (`under-18`, `18-24`, `25-44`, `45-64`, `65+`) |
| `occupation-family:` | `occupation-family:teacher` / `occupation-family:healthcare` | coarse-grained occupation categories |
| `education:` | `education:primary` / `education:tertiary` / `education:any` | `persona.education_level` |
| `interest:` | `interest:music` / `interest:geography` / `interest:literature` | free-form; the default matcher accepts any `interest:*` tag |

Tag an entry with the values it is about (e.g., a fact about SUS
pediatric services might be tagged `age:25-44`, `interest:health`);
`*:any` marks an entry as relevant to every persona. The tags don't
narrow which personas a fact reaches, though: every persona also
carries `region:any` and `age:any`, and a persona's `any` tag matches
every fact tag with the same prefix, so any fact with a `region:` or
`age:` tag is a candidate for every persona. In the shipped banks, that
is every fact. `derive_task` picks one by a hash of the persona, the
bank's id and version, and the seed.

The matcher policy is documented alongside `derive_task` in the
probe module; the fact bank just declares the tags.

## Provenance

Every entry carries a `provenance` block:

```yaml
provenance:
  source: "IBGE, Atlas Nacional (2022)"
  last_reviewed: "2026-04-17"
  references:
    - "https://…/atlas-nacional-2022.pdf"
```

| Field | Type | Required | Description |
|---|---|---|---|
| `source` | str | ✓ | Human-readable description of where the fact came from (and who reviewed it). `"placeholder"` is allowed only when `placeholder: true`. |
| `last_reviewed` | str (ISO-8601 date) | ✓ | `YYYY-MM-DD`. The loader checks that it is a real calendar date. |
| `references` | list[str] | optional | URLs / citations. Non-placeholder entries should have at least one. |

The loader enforces:

- `placeholder: true` ⇒ `source` may be `"placeholder"` and
  `references` may be empty.
- `placeholder: false` ⇒ `references` must be non-empty and `source`
  must not be the string `"placeholder"`.

## Question types

### `factual_recall`

The question asks for specific factual information. The probe often
embeds one or more **false premises** that the model is expected to
catch and correct. The scorer evaluates:

- **accuracy**: did the response correct every false premise and avoid
  introducing new errors?
- **completeness**: did the response cover every component of the
  ground truth?
- **fabrication**: did the response invent facts *not* in the ground
  truth (and not derivable from it)?

### `completion`

The question presents a partial verse, lyric, or passage and asks the
model to continue. The scorer evaluates:

- **accuracy**: does the continuation match the canonical source
  (allowing for minor orthographic variation)?
- **graceful_unknown**: if the model does not know, does it decline in
  the spirit of `graceful_unknown` rather than fabricating fluent-but-
  wrong text?
- **fabrication**: did the response produce content that is *neither*
  the canonical continuation *nor* a graceful refusal: i.e., an
  invented verse presented as if it were the real one? This failure
  mode is the reason `completion` gets its own rubric.

## Placeholder discipline

Placeholder entries are intentional stubs. They let the engineering
pipeline ship and be exercised end-to-end before reviewer time has
been invested. SME verification is recommended before relying on
results that rest on placeholder entries. To make those results easy to
find:

- The loader emits a one-line `INFO` log when any placeholder entry is
  loaded, summarizing the count.
- The `simulation_outcome.warnings` array on every trajectory that
  probed a placeholder fact includes `"used_placeholder_fact"`.
- The scorer's per-fact output carries the entry's flag as
  `fact_placeholder`.

A reviewer promoting an entry out of placeholder status does so by:

1. Authoring a real `source` + `references` entry on the provenance
   block.
2. Setting `placeholder: false` (or deleting the field, since both are
   accepted; `placeholder: true` is the opt-in mark).
3. Bumping the bank's top-level `bank_version`.

## Versioning and drift detection

Every trajectory records the `bank_version` it used
(`simulation_outcome.provenance.bank_version`), so results computed
against an earlier version can be told apart, and the scorer sets
`bank_version_mismatch: true` on a row whose recorded version differs
from the bank it loads. Re-run the simulator and evaluator against the
new bank to refresh them.

The `schema_version` field gates the loader. When we need to introduce
a breaking change to this schema, we bump `schema_version` in lock-step
with the loader and migrate existing banks explicitly.

## Authoring checklist (for reviewers)

- [ ] `schema_version: "v0.1"` pinned at the top.
- [ ] `locale` matches the file's parent directory.
- [ ] `bank_id` is stable (don't rename it mid-revision; bump
      `bank_version` instead).
- [ ] Every `categories` entry is used by at least one fact.
- [ ] Every entry has `id`, `category`, `question_type`,
      `question`, `ground_truth`, `persona_tags` (≥ 1), `difficulty`,
      `provenance`.
- [ ] For `factual_recall` entries with `false_premises`, every listed
      premise is a *concrete* statement a model could verifiably agree
      with or correct, not a vague gesture.
- [ ] For `completion` entries, `graceful_unknown` is present and
      non-empty.
- [ ] Every non-placeholder entry's `provenance.references` has at
      least one citation (URL or explicit bibliographic reference).
- [ ] `placeholder: true` on every unreviewed entry.
