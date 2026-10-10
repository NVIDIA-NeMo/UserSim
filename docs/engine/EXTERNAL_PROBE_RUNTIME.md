# External Probe Runtime

`ProbeEpisodeRuntime` lets an external host run a User Sim episode without
reimplementing any part of it. User Sim runs the unmodified probe dispatch as a
background task and pauses at every model boundary; the host answers each pause
with a model response. Per-episode settings, user-turn gates, judges, early
stop, tool execution and finalize stay User Sim's.

The host never assembles a prompt, picks a tool index, or decides when a turn
ends. Each paused request states what comes next.

## Materializing inputs

Materialize resolved rows without model calls, then construct runtimes from
them:

```python
from usersim.engine.external import ProbeEpisodeRuntime, materialize_episode_inputs

rows = materialize_episode_inputs(
    locale="en_US",
    num_rows=1,
    probe_mix={"tool_calling": 1.0},
    random_seed=42,
)

runtime = ProbeEpisodeRuntime.from_resolved_row(rows[0], models=runtime_models)
```

This uses the same Data Designer persona, probe, theme and toolset sampler
graph as `usersim simulate`; the host does not select panel indices or author
themes and toolsets. Rows carry the persona, probe inputs, locale-aware
behavioral settings, probe family and variant, trajectory ID, provenance, and
the `usersim_config` snapshot `from_resolved_row()` restores. `models_path` and
`assets_dir` are optional overrides. Model configuration supplies trajectory
identities; no configured model is called.

Rows are the dataset: store them, and replay from the stored rows. Regenerating
the same rows from the same seed is explicitly *not* guaranteed, because
nothing depends on it.

The equivalent CLI is:

```bash
usersim simulate --locale en_US --num-rows 1 \
  --probe-mix tool_calling=1 --random-seed 42 \
  --materialize-inputs --out episode-inputs.jsonl
```

## Declaring the host's models

Trajectory identity is keyed on the resolved model id, and `identity_disclosure`
cannot grade an assistant it cannot name, so a host that proxies model calls
still declares what is behind each role:

```python
from usersim.engine.external import HostRoleModel

runtime_models = {
    "user_model": HostRoleModel(model_name="my-provider/user-model", max_tokens=2048),
    "assistant_model": HostRoleModel(model_name="my-provider/model-under-test", max_tokens=4096),
    "judge_model": HostRoleModel(model_name="my-provider/judge-model"),
    "summary_model": HostRoleModel(model_name="my-provider/summary-model"),
}
```

`max_tokens` is the role's budget; User Sim scales it for non-Latin-script
locales exactly as it does for a configured model, and the scaled value arrives
in the request's `parameters`.

`tool_calling` also simulates each tool's response with `api_response_model`, a
model User Sim calls itself rather than sending to the host. A host running
`tool_calling` therefore passes one: anything exposing the same `acompletion()`
as a Data Designer model facade.

```python
runtime_models["api_response_model"] = my_tool_response_model
```

`financial_services` optionally embeds its `kb_search` queries with the model
named by `finance_embedding_model_alias` (`embedding_model` by default, anything
exposing `agenerate_text_embeddings()`); without one, dense retrieval falls back
to lexical search with a warning, so results can differ from a standalone run
that has one configured.

### Replaying the assistant's reasoning

When the row's `usersim_config` sets `replay_assistant_reasoning` (from
`replay_reasoning = true` on `assistant_model` in the models file), each
assistant request's earlier assistant messages carry their
`reasoning_content`. Forward them to the model as they are. No other role's
messages ever carry the assistant's reasoning. Replay only helps where the endpoint passes `reasoning_content` on earlier assistant messages through to the model; a server or chat template that drops earlier thinking makes it a no-op.

## Driving the episode

`advance()` is the whole loop. The first call takes no result; every later call
records one model response:

```python
event = await runtime.advance()
while isinstance(event, ActivationRequest):
    reply = await call_your_model(event.role, event.messages, event.tools, event.parameters)
    event = await runtime.advance({
        "activation_id": event.activation_id,
        "response": {
            "role": "assistant",
            "content": reply.content,
            "reasoning_content": reply.reasoning,
            "tool_calls": reply.tool_calls,
        },
        "usage": {"input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens},
    })

row = await runtime.finalize()
```

Each `ActivationRequest` carries:

- `role`: `user`, `assistant`, `judge` or `summary`;
- `messages`: the exact input User Sim would send, so the host never builds one;
- `tools` and `tools_enabled`: whether User Sim is offering tools for *this*
  call, usually to the assistant, and to the user when the probe's simulated
  user acts through a tool. It disables them when the probe's loop asks the
  assistant for a final answer;
- `continues_turn`: whether this request continues its side's turn already in
  progress rather than opening a new one. An assistant request continues the
  turn when no user message has arrived since the assistant's previous request
  (a tool loop's answer step). A user request continues it when no assistant
  message has arrived since the user's previous request: a guarded user's move
  proposal, a re-ask after a veto and the utterance that follows are one turn.
  A continuing request can also be a retry whose output replaces the previous
  attempt rather than adding to it, such as a user message regenerated after
  the judge rejects it, or an assistant reply resampled;
- `parameters`: the sampling options User Sim would have passed.

Recording the same `activation_id` with an identical payload is idempotent and
replays the same transition, including after the episode completes. Reusing an
id with a different payload is rejected.

`finalize()` returns the row User Sim's own dispatch produced: the resolved input
row updated with the result, which is what `ConversationSimulatorGenerator`
writes for the same inputs. Loop-written columns such as `user_query` are on it.
`close()` cancels a running episode.

## Tool calls

The host records the assistant's response, tool calls included, **before**
issuing those calls. User Sim's own loop then executes them, in its own context
and with its own `turn_idx` and `call_idx`, and the payloads become available:

```python
event = await runtime.advance({"activation_id": ..., "response": response_with_tool_calls})
for call in response_with_tool_calls["tool_calls"]:
    payload = await runtime.tool_result(call["id"])
```

Payloads are returned verbatim, as plain strings, and are never parsed or
re-encoded; several probes' simulated responses are not JSON.

A probe caps how many calls it executes per turn. A recorded call beyond that
cap never runs, and `tool_result()` raises for it; `executed_tool_calls()` is
the authoritative list of what ran.

A host that cannot record tool calls simply returns one model response per
request with no `tool_calls`; User Sim runs the tools either way.

A user activation can offer tools too, when a probe's simulated user acts
through one: the guarded health-disclosure variants commit each move with
`commit_move`. Record the user's tool calls in the response as usual. The
probe's own loop consumes them, as in a local run: they are not executed as
tools, so do not call `tool_result()` for them, and they appear in neither
`executed_tool_calls()` nor `num_tool_calls`. If the probe rejects a move, it
answers the call with a `tool` message and asks again, so that answer arrives in
the next user activation's `messages`.

## Contract errors

Recording a response that breaks the probe's loop rules (tool calls when the
request offers no tools or comes from a judge or summary activation, an
unoffered tool name, a missing or repeated call id)
raises `EpisodeContractError` from `advance()`. These are host bugs, so they are
raised at the call that made them: they never consume a model retry and are
never attributed to the model under test.

## Writing a hostable probe

Probes on the shared `ConversationLoop` are hostable with no extra work. A probe
that does its own work has two obligations:

- A custom `run_dispatch` must accept `state` and `seed_state`, reuse the
  supplied state, and skip seeding when asked. See `templates/probe/generator_agentic.py`.
- A probe that executes its own tool calls must call `on_tool_call_executed()`
  immediately after appending each `role: "tool"` message, or the payloads it
  produced never reach its host. See `templates/probe/generator_tool_calling.py`.

Check both, and overall parity, from your own test suite:

```python
from usersim.testing import assert_hosted_parity

async def test_my_probe_is_hostable():
    await assert_hosted_parity("my_probe", config=cfg, persona=persona, data=data)
```

It runs one episode standalone and one hosted against scripted models and
reports every difference, including prompts. `usersim.testing.hosted_parity_report`
returns the findings as a list instead of raising.

## Supported probes

Every registered probe is hostable, and the check above runs against all of
them so this document cannot drift from what the probes actually do.
