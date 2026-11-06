# sov_ai_facts

Tests recall of locale-specific facts, drawn from a curated bank and graded
against the bank's ground truth: what a model serving users in that locale is
expected to know about its history, geography, culture, public services and
more. Some questions embed a false premise the assistant should correct; others
give part of a verse for it to continue.
[`docs/probes.md`](../../../../../docs/probes.md) places it among the other
probes, and `sov_ai_dynamic` covers the same ground without a curated answer.

## How a row runs

- The probe picks a fact whose persona tags match the row's persona, the same
  fact for the same persona, bank version and seed. A persona no fact matches
  aborts the row.
- The first turn is the fact's question, sent as written, so a false premise or
  a partial verse reaches the assistant unchanged. The simulated user writes the
  follow-ups from the locale's prompt for the question type.
- The assistant under test gets no tools and no system prompt.
- A row succeeds when it records the fact it asked about (`facts_probed`).

## Locales

Every shipped locale has a bank of its own,
`assets/sov_ai_facts/<locale>/sample.yaml`. An India language variant runs on
the `en_IN` bank: its first turn is machine-translated into the row's language
and the row is marked `USED_MACHINE_TRANSLATION`.

## Assets

| File | What it holds |
|---|---|
| `assets/sov_ai_facts/<locale>/sample.yaml` | The locale's facts: question, question type, ground truth, persona tags and provenance. |
| `assets/sov_ai_facts/SCHEMA.md` | The bank's fields and the checklist for authoring one. |

Each fact records its `source`, and the facts that cite sources list them in
`references`. The `en_GB` and `de_DE` banks were drafted from the public
sources each entry cites. In the other banks, a fact is either one of the
original sample entries or an AI-authored entry with references; the `hi_Latn_IN`
bank is a transliteration of `hi_Deva_IN`. Every fact is marked
`placeholder: true`, since no subject-matter expert has reviewed it, and a row
that uses one records a `used_placeholder_*` warning.

## Adding a locale

1. Add `assets/sov_ai_facts/<locale>/sample.yaml`, following
   [`SCHEMA.md`](assets/sov_ai_facts/SCHEMA.md) and its authoring checklist.
2. Add the locale's prompts to `prompts.py`: a system prompt for each question
   type in `_SYSTEM_PROMPTS`, and a follow-up instruction in
   `_FOLLOWUP_INSTRUCTIONS`.

A run loads each requested locale's bank before any model call, and
`asset_problems` reports a shipped locale with no bank. The probe's tests
require prompts for every shipped locale.

## Scoring

The `sov_ai_facts` scorer, in `default_scorers`, has a judge grade each answer
against the fact's ground truth: `sovereign.accuracy`,
`sovereign.completeness` and `sovereign.fabrication`, plus
`sovereign.false_premise_catch` for factual-recall questions and
`sovereign.graceful_unknown` for completions. Its evidence feeds the
`sovereign_local_knowledge` capability in `capabilities.py`, which
`sov_ai_dynamic` feeds too.

## Tests

`tests/probes/sov_ai_facts/` holds the probe's, the bank's and the scorer's
tests, with `contract.py` for the shared tests.
