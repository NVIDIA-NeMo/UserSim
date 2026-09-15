## Query bank schema

A **query bank** is a curated YAML file containing **cross-locale query
templates** used by the `sov_ai_multilingual_parity` probe
to test whether the assistant treats different demographic strata,
both *within a locale* and *across locales*, with comparable quality
and respect.

The unit is the **conceptual question**, not the locale-specific
phrasing. Each entry holds N locale renderings of the same underlying
query, plus a single `query_id` so trajectories that probed the same
conceptual question (in different locales, by different personas) can
be joined for matched-pair / similarity-based comparison downstream.

This schema is structurally different from the fact bank
(`assets/sov_ai_facts/SCHEMA.md`) and the probing taxonomy
(`assets/sov_ai_dynamic/SCHEMA.md`):

| Asset | Unit | Per-locale shape | Ground truth | Adversarial framing |
|---|---|---|---|---|
| Fact bank | One fact per entry | One file per locale, content authored separately per locale | ✓ | ✓ (false premises) |
| Probing taxonomy | One topic category per entry | One file per locale, content authored separately per locale | ✗ | ✗ |
| **Query bank** | **One conceptual query per entry** | **One file shared across locales; N renderings per entry** | ✗ (scoring is respect-parity, not correctness) | ✗ (queries are neutral; the bias is in the model's response) |

The cross-locale design is the load-bearing feature: matched-pair
comparison ("how does the model respond to *the same question* in
Hindi vs. English by *demographically matched* personas?") needs the
same conceptual content to appear in N locales tied by a shared
identifier. Per-locale separate authoring would not give us this.

## File layout

```
assets/sov_ai_multilingual_parity/
├── SCHEMA.md          # this file
├── sample.yaml        # placeholder bank (see §Placeholder discipline)
└── v1.yaml            # reviewer-authored bank (future)
```

One bank per file. Multiple banks (e.g., a public bank + a
safety-sensitive bank) can coexist; the loader picks one by path.

## Top-level fields

| Field | Type | Required | Description |
|---|---|---|---|
| `schema_version` | str | ✓ | Schema version this file targets. Currently pinned at `v0.1`. The loader refuses unknown versions. |
| `bank_id` | str | ✓ | Stable identifier for this bank. Appears on every trajectory that probes it so cross-run drift is detectable. |
| `bank_version` | str | ✓ | Semver-style version of this bank's *contents*. Bump on any entry edit; carried into `simulation_outcome.provenance` so consumers can spot when a bank changed mid-run. |
| `locales` | list[str] | ✓ | The locales this bank supports renderings for. Every entry must provide a rendering for every locale listed here, OR explicitly opt out via the per-entry `renderings` map. The loader validates the cross-product. |
| `domains` | list[str] | ✓ | The set of domain tags used by entries (e.g., `health`, `public-services`, `civic`, `tech`, `finance`, `education`). Used for stratified reporting. |
| `tags` | list[str] | optional | The tag vocabulary used by `persona_tags` on entries. Informational; no enforcement. |
| `entries` | list[Entry] | ✓ | The query templates themselves. Minimum 1 entry. |

## Entry fields

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | str | ✓ | Unique across the bank. Convention: `Q-<DOMAIN-SHORT>-<NNN>` (e.g., `Q-MEDS-001`, `Q-TAX-002`, `Q-VOTE-001`). The id is **locale-independent** and joins all renderings of the same query. |
| `domain` | str | ✓ | Must appear in the bank's `domains` list. Used for stratified reporting. |
| `placeholder` | bool | ✓ | `true` for unreviewed entries (stubs). The reporting layer surfaces the fraction of probed queries that were placeholder; the scorer flags placeholder-only scorecards as preview-only. Reviewers set this to `false` once the entry passes review. |
| `difficulty` | str | ✓ | One of `easy`, `medium`, `hard`. Refers to the **conceptual difficulty of the question**, not the language difficulty (which varies by locale and reviewer). Used for stratified scoring. |
| `persona_tags` | list[str] | ✓ (≥1) | Tags used by the probe to match queries to personas. Follow the structured form below. At least one tag required so every query is reachable via persona-targeted selection. |
| `concern` | str | ✓ | A reviewer-facing prose description of *what failure mode this query is designed to surface*. Examples: "tests whether the model gives the same level of clinical detail to a low-education user as to a high-education user asking the same medication question"; "tests whether the model invents an English-language US-centric process when asked in Hindi about Indian voter registration". The concern is what makes the query useful: it tells the reviewer what they should look for in the response. Required and non-empty for every entry. |
| `renderings` | dict[str, str] | ✓ | The locale-specific renderings. Keys are locale ids matching the bank's `locales` list. Values are the actual question text in that locale's native language, written in the persona's voice (informal where appropriate, locale-appropriate idiom). The loader validates that every locale in the bank's `locales` list has a rendering, OR that the entry explicitly opts out by setting `renderings_opted_out: [locale, ...]` (rare; for queries that genuinely don't translate, e.g., a US-tax-specific question in non-US locales). |
| `renderings_opted_out` | list[str] | optional | Locales for which this query has no meaningful rendering and should NOT be probed. Use only when the question genuinely has no cross-locale equivalent (the reviewer can't honestly translate it). The loader skips these locales when matching personas to queries. Default: empty (the entry must have a rendering for every locale in the bank). |
| `provenance` | Provenance | ✓ | See §Provenance. |

## Persona tags

Persona tags are **structured** so the matcher can intersect them
deterministically. Same vocabulary as the fact bank and probing
taxonomy (`region:`, `age:`, `occupation-family:`, `education:`,
`interest:`); see `assets/sov_ai_facts/SCHEMA.md` §Persona tags for the
full list. Use `*:any` to opt an entry in for every persona along an
axis; use specific values to target a demographic.

The matcher policy is documented alongside the probe; the
query bank just declares the tags.

## Provenance

Every entry carries a `provenance` block:

```yaml
provenance:
  source: "internal authorship 2026-04 (placeholder)"
  last_reviewed: "2026-04-20"
  references: []
```

| Field | Type | Required | Description |
|---|---|---|---|
| `source` | str | ✓ | Human-readable description of where the query came from (and who reviewed it). `"placeholder"` is allowed only when `placeholder: true`. |
| `last_reviewed` | str (ISO-8601 date) | ✓ | `YYYY-MM-DD`. Used by the loader to flag stale entries. |
| `references` | list[str] | optional | URLs / citations. Useful when the query is derived from a real-world report or benchmark; non-placeholder entries SHOULD have at least one. |

## Placeholder discipline

Every entry in a bank shipped with `placeholder: true` is a stub.
Reviewers must replace placeholder content with locally-vetted
queries before any production claim is made off the resulting
scorecard. The capability dashboard treats as preview-only
production-ready claims for any locale that probed against
placeholder queries, same epistemic discipline as fact-bank
placeholders and probing-taxonomy placeholders.

The placeholder-flagged shipped queries serve three concrete purposes:

1. exercise the engineering pipeline end-to-end before reviewer-time
   is committed;
2. give reviewers a structural template they can extend rather than
   author from scratch;
3. document the cross-locale-rendering contract via working examples.

Placeholder queries should **not** be used to draw conclusions about
specific models or specific locales' equity gaps. They're pipeline
plumbing, not signal.

## Loader contract

The loader (`usersim/engine/core/query_bank.py`) enforces:

- `schema_version` is in the supported set.
- `bank_id`, `bank_version`, `locales` (non-empty), `domains`
  (non-empty), `entries` (non-empty) all present.
- Every entry has all required fields.
- Every `entry.id` is unique within the bank.
- Every `entry.domain` is in the bank's `domains` list.
- Every locale in the bank's `locales` list either has a rendering on
  every entry OR is in that entry's `renderings_opted_out` list.
- Every `renderings_opted_out` locale must be in the bank's `locales`
  list (catches typos).
- Every `entry.persona_tags` is non-empty.
- Every `entry.difficulty` is one of `easy` / `medium` / `hard`.
- `placeholder: true` allows `source: "placeholder"`; non-placeholder
  entries must have a real provenance source.
- ISO-8601 date format on `last_reviewed`.

Any violation raises `QueryBankError` with a `path::entry_id::field`
breadcrumb so reviewers can grep the offending location.
