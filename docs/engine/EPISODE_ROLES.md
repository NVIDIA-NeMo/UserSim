# Design: separating the three roles inside a UserSim episode

**Status:** proposal. No code changes in this PR.

## The observation

An episode does three jobs at once: it simulates the user, it runs the probe's
tools, and it judges what comes back. All three mutate the same
`ConversationState`:

```python
@dataclass
class ConversationState:
    messages: list[dict[str, Any]]        # the shared transcript
    user_history: list[dict[str, Any]]    # the user simulator's own view
    metadata: dict[str, Any]              # everything else
    conv_summaries: dict[int, str]
    uh_echo_positions: dict[int, str]
    assistant_failures: int
    outcome: OutcomeBuilder
```

`metadata` is where the three overlap most visibly. It currently holds, side
by side: `attempted_actions`, `tools_called`, `account_state`,
`kb_search_queries`, `verifier_results`, `tool_call_count_per_turn` —
and `user_judge_ratings`, `assistant_judge_ratings`, `capitulation_checks`,
`early_stop`, `frustration_events` — and a dozen probe-specific side channels.

Three jobs, one bag.

| Job | What it owns |
|---|---|
| **User simulation** | `user_history`, the persona-derived system prompt, frustration |
| **Tool environment** | `attempted_actions`, `tools_called`, `account_state`, `kb_search_queries`, `verifier_results` |
| **Evaluation** | judge ratings, capitulation checks, early stop, `outcome` and its traces |

## Why this matters

The user simulator is the *least* state-hungry of the three. It is called with
its own role-flipped history and never reads the shared transcript:

```python
state.user_history = [
    {"role": "system", "content": probe.get_user_system_prompt()},
    {"role": "user", "content": query_instruction},
]
...
user_resp = await acall_llm(models, MODEL_USER, compressed_uh)
```

So a persona turn needs: persona prompt, probe instruction, its own history,
pending frustration. Nothing else.

Everything that makes an episode hard to run from outside UserSim —
tool-call indices, the tool simulator's context, the assistant's per-call
document view — belongs to the *tool environment*, not to the user simulator.
They are only entangled because they share `state`.

This is what collides with a host like NeMo Gym, whose Environment Server
alternates between participant Agents and exposes only `/run`, with tools
served by a Resources Server from inside an Agent's own loop. A host cannot
take the user simulator without also taking the tool environment, because
there is no seam between them.

## Proposed seam

Split `ConversationState` along the three jobs, so each can be held by
whoever is responsible for it.

**User simulation** becomes self-contained: `(persona prompt, instruction,
user_history, frustration) -> next user turn`, plus the gate that accepts or
rejects it (`max_query_attempts`, language and script compliance, fourth-wall
and echo pre-filters). It needs no transcript and no tool state. A host can
run this as an ordinary participant Agent.

**The tool environment** keeps the probe's tool state and owns everything
derived from it: execution with its own `turn_idx`/`call_idx`, the simulated
payload, and the assistant's per-call document view. `financial_services`'
`transform_assistant_history` bounds retrieved bodies, which is a statement
about retrieved documents — a tool concern that currently lives in harness
code. A host can run this as a Resources Server.

**Evaluation** keeps judge verdicts, capitulation, early stop, and the
outcome and trace sink, consumed at `verify()` and at the per-turn decisions.

The shared transcript is then exactly what it looks like — the conversation —
and belongs to whoever alternates the participants.

## What does not decompose cleanly

Named rather than glossed, because these are the design's real questions.

1. **The trace sink.** `outcome` is written by tool execution, by judges, and
   by every model call for token accounting. It is genuinely cross-cutting.
   Either one side owns it and the others report into it, or traces are
   merged at finalize.
2. **The assistant's per-call view.** `prepare_assistant_history` composes
   compression (harness) with `transform_assistant_history` (probe, and in
   the one implementing case a tool concern). Splitting it means deciding
   whether compression is a harness or an environment policy.
3. **Blast radius.** 14 files touch `state.metadata` directly, including 8 of
   the probe generators. A seam that changes its shape is a change to the
   probe contract, not an additive surface.

## Fallback: an episode step API

If the decomposition is too invasive, the same host problem can be solved
without it, by exposing the existing single-state episode as addressable
steps: `next_user_prompt`, `submit_user_turn`, `assistant_turn_inputs`,
`simulate_tool_call`, `submit_assistant_turn`, `finalize`.

`simulate_tool_call` takes the assistant message that requested the call and
returns a refreshed `assistant_turn_inputs` alongside the payload. The Agent
already makes that call, so this costs no extra round trip, and it carries
everything the probe owns about the inner loop: the assistant message is
appended before simulating so the tool simulator sees the tool-call line;
indices come from the loop's counters; the returned view has the probe's
projection applied; and an empty returned tool set expresses a tools-disabled
synthesis or an exhausted per-turn cap, so no static loop policy is needed.

This routes around the overlap rather than resolving it. It keeps one
`ConversationState` and adds a surface over it, which means the host's call
pattern has to match the episode's internal shape — the coupling is still
there, just expressed in HTTP.

## Why the assistant turn can be the Agent's, either way

`after_assistant_turn` sits inside a retryable unit: a judge-rejected
candidate is rolled back and regenerated. That would mean asking an Agent to
un-execute tools it has already reported.

It never arises. `AgenticMixin`, `ToolCallingMixin` and `ToolExecutionMixin`
all declare `supports_assistant_resampling = False`, with the reason stated on
`BaseProbe`: *"replaying those side effects is not safe, so they are capped to
one attempt."* Every tool-executing probe is already a single, non-retried
assistant turn.

## Alternatives already tried

**Host executes tools, then replays its transcript into the runtime.** The
first revision of the Gym integration. The host ran ahead of the loop, so the
tool simulator saw a context without the assistant's tool-call line and
indices came from a host counter: `tool_calling` recorded `turn_idx` 3 and 5
standalone, 2 and 3 hosted.

**Static loop policy.** A config describing round mode, caps and a final
synthesis step, handed to the Agent at seed time. Carries limits but not
content: the per-call assistant view is a transformation of live state.

**Host rebuilds the user side.** Also tried, in `_run_external_probe`. It
skipped the gates rather than calling them; across 40 test personas the
`tool_calling` user prompt carried gradual-disclosure instructions 0/40 times
and an interaction style 0/40 times, against 25/40 and 23/40 standalone.

## Open questions

1. Is the seam worth cutting, or should the step API ship as an additive
   surface over the current single state?
2. If the seam is cut, who owns the trace sink and the compression policy?
3. Is the user simulator worth extracting on its own merits — independent of
   any host — given it already runs on a separate `user_history`?
