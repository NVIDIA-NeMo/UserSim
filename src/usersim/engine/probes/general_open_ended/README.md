# general_open_ended

A persona-grounded user asks for information, advice or a recommendation on a
topic, and the probe looks at whether the answers are accurate and coherent,
pitched to the user's apparent expertise, and whether a vague request gets a
clarifying question rather than an information dump. The conversations use no
tools and no curated bank; the trajectory evaluator judges their quality.
[`docs/probes.md`](../../../../../docs/probes.md) places it among the other
probes.

## How a row runs

- The run draws one topic per row from the locale's topic list and stores it in
  the row's `theme`.
- The simulated user writes the first turn and every follow-up from the topic
  and its persona, in the row's language. The user judge checks each turn
  before the assistant sees it.
- The assistant under test gets no system prompt (see the
  [pure-capability policy](../../../../../docs/engine/README.md#assistant-under-test-pure-capability-policy)).
- A user who says they are satisfied ends the conversation before its last
  turn.
- Every conversation that completes counts as a success of the probe's
  protocol.

## Locales

A locale with a topic list of its own reads
`assets/general_open_ended/<locale>/topics.json`; any other locale, the India
language variants included, reads the `en_US` list. The prompts are shared by
every locale, and the simulated user writes in the row's language whichever
list it reads. The locale table in
[`docs/probes.md`](../../../../../docs/probes.md#scope-and-coverage) names the
locales with their own list.

## Assets

| File | What it holds |
|---|---|
| `assets/general_open_ended/<locale>/topics.json` | The locale's topics, each a `type` and a `description` written in the locale's language. Each file is a complete list, not additions to the `en_US` one. |
| `assets/general_open_ended/prompts.yaml` | The simulated user's prompt, the user judge's rubric and the assistant judge's prompt. |

The topic lists were written for this project.

## Adding a locale

Add `assets/general_open_ended/<locale>/topics.json` with the topics written in
that locale's language. Nothing else is needed: the prompts serve every locale.
`asset_problems`, which `usersim smoke` and `assert_probe_conforms` run, reports
a shipped locale with no list to read, and
`tests/engine/core/test_default_asset_paths.py` checks that every shipped
locale's list loads and gives each topic a `type` and a `description`.

## Scoring

The probe has no scorer of its own. `usersim eval --scorers auto` runs the
generic `language_compliance`, `refusal_basics` and `response_shape` scorers on
its rows, and the trajectory judge scores them on the shared axes.

## Tests

`tests/probes/general_open_ended/test_probe.py` runs the probe through
`run_episode` with a conversation scripted in Japanese, and `contract.py` in the
same folder holds the probe's entries in the shared tests.
