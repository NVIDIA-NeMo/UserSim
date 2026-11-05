# sov_ai_multilingual_parity

Tests respect and calibration on the same question asked in every locale: does
the assistant engage each user as a peer and pitch a helpful answer to them,
rather than condescend or deflect? Each row records its `query_id`, so the
answers to one question can be compared across locales.
[`docs/probes.md`](../../../../../docs/probes.md) places it among the other
probes.

## How a row runs

- The probe picks a query with a rendering in the row's locale, the same query
  for the same persona, bank version and seed.
- The first turn is that rendering, sent as written. The simulated user writes
  the follow-ups from the locale's prompt, with the query's domain and
  difficulty.
- The assistant under test gets no tools and no system prompt.
- A row succeeds when it records its query (`query_id`).

## Locales

One bank serves every locale: each query holds a rendering per shipped locale.
An India language variant runs on the `en_IN` rendering, machine-translated
into the row's language, and the row is marked `USED_MACHINE_TRANSLATION`.

## Assets

| File | What it holds |
|---|---|
| `assets/sov_ai_multilingual_parity/sample.yaml` | The queries, each with its domain, difficulty, persona tags, the `concern` it is designed to surface, a rendering per locale and its provenance. |
| `assets/sov_ai_multilingual_parity/SCHEMA.md` | The bank's fields and its rules for placeholder entries. |

Each query records its `source`. A query a reviewer has checked is
`placeholder: false`; its references note that native speakers have not
verified the per-locale renderings. The others were authored for this bank and
are `placeholder: true`, and a row that uses one records a
`used_placeholder_*` warning.

## Adding a locale

1. In `sample.yaml`, add the locale to `locales` and a rendering of every query
   in it, or list the locale in a query's `renderings_opted_out`. The loader
   refuses a bank with a query that has neither.
2. Add the locale's prompt to `_SYSTEM_PROMPTS` and its follow-up instruction
   to `_FOLLOWUP_INSTRUCTIONS` in `prompts.py`.

The probe's tests check that the bank and the prompts cover every shipped
locale.

## Scoring

The `sov_ai_multilingual_parity` scorer, in `default_scorers`, has a judge
grade each conversation against the query's `concern` on
`respect.condescension`, `respect.complexity_calibration` and
`respect.helpfulness_at_proficiency`. Its evidence feeds the `respect_parity`
capability in `capabilities.py`.

## Tests

`tests/probes/sov_ai_multilingual_parity/` holds the probe's, the bank's and
the scorer's tests, with `contract.py` for the shared tests.
