# general_educational

A persona-grounded learner asks about a subject, and the probe looks at whether
the explanations are accurate, pitched at the learner's level and scaffolded
with examples, without condescension. The conversations use no tools and no
curated bank; the trajectory evaluator judges their quality.
[`docs/probes.md`](../../../../../docs/probes.md) places it among the other
probes.

## How a row runs

- The run draws one subject per row from the locale's subject list and stores
  it in the row's `theme`.
- The simulated user plays a learner and writes the first turn and every
  follow-up from the subject and its persona, in the row's language. The user
  judge checks each learner turn before the assistant sees it.
- The assistant under test gets no system prompt (see the
  [pure-capability policy](../../../../../docs/engine/README.md#assistant-under-test-pure-capability-policy)).
- A learner who says they are satisfied ends the conversation before its last
  turn.
- Every conversation that completes counts as a success of the probe's
  protocol.

## Locales

A locale with a subject list of its own reads
`assets/general_educational/<locale>/subjects.json`; any other locale, the
India language variants included, reads the `en_US` list. The prompts are
shared by every locale, and the simulated user writes in the row's language
whichever list it reads. The locale table in
[`docs/probes.md`](../../../../../docs/probes.md#scope-and-coverage) names the
locales with their own list.

## Assets

| File | What it holds |
|---|---|
| `assets/general_educational/<locale>/subjects.json` | The locale's subjects, each a `type` and a `description` written in the locale's language. Each file is a complete list, not additions to the `en_US` one. |
| `assets/general_educational/prompts.yaml` | The simulated learner's prompt, the user judge's rubric and the assistant judge's prompt. |

The subject lists were written for this project.

## Adding a locale

Add `assets/general_educational/<locale>/subjects.json` with the subjects
written in that locale's language. Nothing else is needed: the prompts serve
every locale. `asset_problems`, which `usersim smoke` and
`assert_probe_conforms` run, reports a shipped locale with no list to read, and
`tests/engine/core/test_default_asset_paths.py` checks that every shipped
locale's list loads and gives each subject a `type` and a `description`.

## Scoring

The probe has no scorer of its own. `usersim eval --scorers auto` runs the
generic `language_compliance`, `refusal_basics` and `response_shape` scorers on
its rows, and the trajectory judge scores them on the shared axes plus
`pedagogical_quality`: whether the response taught at the right level.

## Tests

`tests/probes/general_educational/test_probe.py` runs the probe through
`run_episode` with a learner scripted in Brazilian Portuguese, and
`contract.py` in the same folder holds the probe's entries in the shared tests.
