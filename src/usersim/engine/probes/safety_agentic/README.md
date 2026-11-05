# safety_agentic

Tests safety when the assistant acts through tools: whether it takes an action
it was not asked for, reaches beyond the task, discloses an action's
consequences, and resists instructions planted in what its tools return. Each
row holds one action request, and the assistant works through it against mock
tools. A conversation that ends with the assistant asking what to do next is
expected, not a truncated transcript.
[`docs/probes.md`](../../../../../docs/probes.md) places it among the other
probes.

## How a row runs

- The probe takes the action request from the row's `action_request_id`, or
  picks one from the persona.
- The single user turn is the request's opening prompt in the row's locale,
  sent as written. There is no simulated user after it.
- The assistant is offered the request's tools. Each call is recorded and
  answered with the bank's mock response for that tool, and a call to a tool
  the request does not offer gets an error.
- The conversation ends when the assistant replies without a tool call, or at
  the run's turn limit.
- A row succeeds when it records its request (`action_request_id`).

The probe runs its own loop in `run_dispatch`, since there is no simulated user
to alternate with.

## Locales

One bank serves every locale. Each request's description, opening prompt and
expected behavior are written in every shipped locale; the tool specs and mock
responses are in English. An India language variant runs on the `en_IN`
opening prompt, machine-translated into the row's language, and the row is
marked `USED_MACHINE_TRANSLATION`.

## Assets

| File | What it holds |
|---|---|
| `assets/safety_agentic/sample.yaml` | The sub-protocols and the action requests, each with its tools, mock responses, expected behavior and provenance. |
| `assets/safety_agentic/action_taxonomy.yaml` | The actions the scorer classifies calls against: risk tier, reversibility, whether confirmation or disclosure is required, and blast radius. |
| `assets/safety_agentic/SCHEMA.md` | The bank's fields and its rules for placeholder entries. |

Each request records its `source`: a request a reviewer has checked is
`placeholder: false`, and an AI-authored one is `placeholder: true`; a row that
uses a placeholder request records a `used_placeholder_*` warning. Each action
in the taxonomy records its `source` and when it was last reviewed.

## Adding a locale

Add the locale's text to each request's `description`, `initial_user_prompt`
and `expected_behavior`, following
[`SCHEMA.md`](assets/safety_agentic/SCHEMA.md). A request with no text for a
locale reads its `en_US` text, so add every request's.

## Scoring

The `safety_agentic` scorer, in `default_scorers`, classifies the actions the
assistant attempted against the action taxonomy for `agentic.boundary_held`
and `agentic.blast_radius_contained`, and has a judge score
`agentic.consequence_disclosure`, `agentic.confirmation_sought` and
`agentic.injection_resistance` for the sub-protocols that need them. Its
evidence feeds the `agentic_boundary`, `agentic_disclosure` and
`prompt_injection_resistance` capabilities in `capabilities.py`.

## Tests

`tests/probes/safety_agentic/` holds the probe's, the bank's, the action
taxonomy's and the scorer's tests, with `contract.py` for the shared tests.
