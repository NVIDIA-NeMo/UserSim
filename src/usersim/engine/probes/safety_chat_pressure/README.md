# safety_chat_pressure

Tests safety under conversational pressure: does the assistant hold the line
when the user persists with, reframes or escalates a request it should refuse?
Each row pairs a target request with a pressure strategy, and the simulated
user applies the strategy turn by turn.
[`docs/probes.md`](../../../../../docs/probes.md) places it among the other
probes.

## How a row runs

- The probe takes the strategy and the target from the row's
  `pressure_strategy_id` and `target_request_id`, or picks them from the
  persona.
- The first turn is the target's opening request in the row's locale, sent as
  written.
- Each follow-up applies the strategy's next reframing to the assistant's last
  reply, and the row records the reframings it used.
- After each assistant turn a classifier checks whether the assistant gave in,
  and the conversation stops when it did.
- The assistant under test gets no tools and no system prompt.
- A row succeeds when it records its target (`target_request_id`).

## Locales

One bank serves every locale. Each target's description, opening request and
expected refusal are written in every shipped locale; the strategies and their
reframings, which only the simulated user reads, are in English, and the
simulated user writes its turns in the row's language. An India language
variant runs on the `en_IN` opening request, machine-translated into the row's
language, and the row is marked `USED_MACHINE_TRANSLATION`.

## Assets

| File | What it holds |
|---|---|
| `assets/safety_chat_pressure/sample.yaml` | The harm categories, the pressure strategies with their reframings, and the target requests with their provenance. |
| `assets/safety_chat_pressure/SCHEMA.md` | The bank's fields and its rules for placeholder targets. |

The targets are synthetic requests, clearly inappropriate and of low real-world
harm. Each records its `source`: a target a reviewer has checked is
`placeholder: false`, and an AI-authored one is `placeholder: true`; a row that
uses a placeholder target records a `used_placeholder_*` warning. The
strategies record no provenance.

## Adding a locale

Add the locale's text to each target's `description`, `initial_prompt` and
`expected_refusal`, following [`SCHEMA.md`](assets/safety_chat_pressure/SCHEMA.md).
The loader checks the placeholders in every rendering. A target with no text
for a locale reads its `en_US` text, so add every target's. Strategies and
reframings need no rendering of their own.

## Scoring

The `safety_chat_pressure` scorer, in `default_scorers`, has a judge classify
each assistant turn, find the turn the assistant gave in at, if any, and score
`pressure.refusal_held` and `pressure.contextual_safety_persistence`, grounded
in the target's expected refusal and the strategy's description. Its evidence
feeds the `safety_pressure` capability in `capabilities.py`.

## Tests

`tests/probes/safety_chat_pressure/` holds the probe's, the bank's, the
classifier's and the scorer's tests, with `contract.py` for the shared tests.
