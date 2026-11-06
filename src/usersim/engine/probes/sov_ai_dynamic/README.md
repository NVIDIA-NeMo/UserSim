# sov_ai_dynamic

Tests locale-specific knowledge where no curated answer exists. A persona-voiced
user explores a topic from the locale's taxonomy, such as its history,
geography, culture or public services, and a judge flags suspected fabrication,
self-contradiction across turns, and confident guesses where declining was
better. It is the breadth counterpart of `sov_ai_facts`, which grades curated
facts against ground truth. [`docs/probes.md`](../../../../../docs/probes.md)
places it among the other probes.

## How a row runs

- The probe picks a category from the locale's taxonomy and one of its
  sub-topic hints, the same pick for the same persona, taxonomy version and
  seed.
- The simulated user writes the first turn in the persona's voice from the
  category's invitation and the hint, and the user judge checks it. Each
  follow-up carries the locale's follow-up instruction.
- The assistant under test gets no tools and no system prompt.
- A row succeeds when it records the category it explored
  (`probing_categories_explored`), and `probe_variant` holds the category.

## Locales

Every shipped locale has a taxonomy of its own,
`assets/sov_ai_dynamic/<locale>/categories.yaml`. An India language variant
runs on the `en_IN` taxonomy, and the simulated user writes its turns in the
row's language: the first turn is written, not translated.

## Assets

| File | What it holds |
|---|---|
| `assets/sov_ai_dynamic/<locale>/categories.yaml` | The locale's taxonomy: its id and version, a `placeholder` flag, and its categories, each with an invitation and sub-topic hints. |
| `assets/sov_ai_dynamic/SCHEMA.md` | The taxonomy's fields, authoring guidance and the reviewer checklist. |

A taxonomy is `placeholder: true` until a native-speaker reviewer has audited
its invitations and hints, and `false` after; the review flags in
[`docs/probes.md`](../../../../../docs/probes.md#scope-and-coverage) say which
locales are reviewed.

## Adding a locale

1. Add `assets/sov_ai_dynamic/<locale>/categories.yaml`, following
   [`SCHEMA.md`](assets/sov_ai_dynamic/SCHEMA.md): every category needs at
   least two sub-topic hints. `USERSIM_SOV_AI_DYNAMIC_TAXONOMY_<LOCALE>` points
   a run at a file elsewhere.
2. Add the locale's opening prompt to `_OPENING_PROMPTS` and its follow-up
   instruction to `_FOLLOWUP_INSTRUCTIONS` in `prompts.py`.

A run loads each requested locale's taxonomy before any model call, and
`asset_problems` reports a shipped locale with none. The probe's tests check
the prompts and the taxonomy of every shipped locale.

## Scoring

The `sov_ai_dynamic` scorer, in `default_scorers`, has a judge read the
assistant's replies alone, with no ground truth, for
`sovereign.suspected_fabrication`, `sovereign.self_consistency` and
`sovereign.graceful_unknown`. Its evidence feeds the
`sovereign_local_knowledge` capability, declared with `sov_ai_facts` in
[`sov_ai_facts/capabilities.py`](../sov_ai_facts/capabilities.py).

## Tests

`tests/probes/sov_ai_dynamic/` holds the probe's, the taxonomy's and the
scorer's tests, with `contract.py` for the shared tests.
