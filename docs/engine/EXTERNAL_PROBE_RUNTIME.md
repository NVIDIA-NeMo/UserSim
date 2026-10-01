# External Probe Runtime

UserSim exposes two independently hostable episode components:

- `ConversationRuntime` belongs with the environment server. It owns native
  probe dispatch, outer participant activation, conversation state,
  completion, idempotency and finalization. It delegates each Assistant turn
  once and never steps that turn model-call by model-call or tool by tool.
- `ProbeToolSession` belongs with the resources server. Constructed separately
  from the same immutable resolved row, it owns native probe tool behavior and
  mutable tool-effect state. The Agent calls it directly while it owns the turn.

Environment sends the Agent one typed `AssistantTurnRequest`. The Agent runs
the complete model → Resources tools → model loop and returns one
`CompletedAssistantTurn`, including the ordered transcript, per-model-call
usage/reasoning, and typed Resources evidence. There is no record-before-tool
callback to Environment or UserSim. `ProbeEpisodeRuntime` remains the
compatibility facade and adapts the existing per-model `advance()` API locally.

The Agent receives generic declarative loop constraints (caps, whether tools
remain enabled, and whether a final synthesis call is required). Those are
constraints, not callbacks: once delegated, UserSim does not choose the next
step between model calls. Resources assigns semantic `turn_idx`/`call_idx` and
decides which calls are executable under the cap.

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

`max_tokens` is the role's budget; UserSim scales it for non-Latin-script
locales exactly as it does for a configured model, and the scaled value arrives
in the request's `parameters`.

`tool_calling` also simulates each tool's response with `api_response_model`.
With the compatibility facade, include that support model in the same mapping:

```python
runtime_models["api_response_model"] = my_tool_response_model
```

In a split deployment, `ConversationRuntime` receives the role declarations
above. `ProbeToolSession` receives only models used while executing effects:
`api_response_model` for `tool_calling`; no models for the shipped
`safety_agentic` tools; and the optional finance embedding/translation support
models for `financial_services`. Both constructors still reconstruct the
native probe from the same resolved row, whose explicit trajectory identity
keeps the two halves aligned.

`financial_services` optionally embeds its `kb_search` queries with the model
named by `finance_embedding_model_alias` (`embedding_model` by default, anything
exposing `agenerate_text_embeddings()`); without one, dense retrieval falls back
to lexical search with a warning, so results can differ from a standalone run
that has one configured.

## Driving the episode

For the compatibility facade, `advance()` remains unchanged. The first call
takes no result; every later call records one model response:

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

Non-Assistant participants still use `ActivationRequest`. The compatibility
facade also adapts Assistant turns back into this shape. Each request carries:

- `role`: `user`, `assistant`, `judge` or `summary`;
- `messages`: the exact input UserSim would send, so the host never builds one;
- `tools` and `tools_enabled`: whether UserSim is offering tools for *this*
  call. It disables them when the probe's loop asks for a final answer;
- `continues_turn`: whether this request continues the turn already in
  progress rather than opening a new one;
- `parameters`: the sampling options UserSim would have passed.

Recording the same `activation_id` with an identical payload is idempotent and
replays the same transition, including after the episode completes. Reusing an
id with a different payload is rejected.

`finalize()` returns the row UserSim's own dispatch produced: the resolved input
row updated with the result, which is what `ConversationSimulatorGenerator`
writes for the same inputs. Loop-written columns such as `user_query` are on it.
`close()` cancels a running episode.

## Split Assistant turns

In a split deployment, Environment sees one event for the whole Assistant
turn. This example shows the ownership boundary; `agent_model` and
`resources_client` may be remote services:

```python
from usersim.engine.external import (
    ActivationRequest,
    ActivationResult,
    AssistantModelCall,
    AssistantToolRound,
    AssistantTurnRequest,
    CompletedAssistantTurn,
    ConversationRuntime,
    EpisodeLifecycleComplete,
)

conversation = ConversationRuntime.from_resolved_row(row, models=role_models)

event = await conversation.advance()
while not isinstance(event, EpisodeLifecycleComplete):
    if not isinstance(event, AssistantTurnRequest):
        reply = await activate_outer_participant(event)
        event = await conversation.advance(
            ActivationResult(event.activation_id, reply.message, reply.usage)
        )
        continue

    # One Environment -> Agent delegation. Everything below runs in Agent.
    await resources_client.begin_turn(event.tool_context)
    transcript = []
    model_calls = []
    tool_count = 0
    limit_reached = False

    while True:
        tools = (
            event.tools
            if event.loop_policy.tools_enabled(
                model_calls=len(model_calls),
                tool_calls=tool_count,
                limit_reached=limit_reached,
            )
            else ()
        )
        reply = await agent_model(
            event.model_messages(transcript),
            tools=tools,
            **event.parameters,
        )
        call = AssistantModelCall(response=reply.message, usage=reply.usage)
        model_calls.append(call)
        transcript.append(call.response)
        if not call.response.get("tool_calls"):
            break

        batch = await resources_client.execute_round(
            AssistantToolRound(
                turn_id=event.turn_id,
                round_id=f"round-{len(model_calls)}",
                assistant_response=call.response,
            )
        )
        transcript.extend(batch.tool_messages)
        tool_count += len(batch.receipts)
        limit_reached = batch.limit_reached
        if not event.loop_policy.should_continue(
            model_calls=len(model_calls),
            tool_calls=tool_count,
        ):
            break

    completed = CompletedAssistantTurn(
        turn_id=event.turn_id,
        transcript=tuple(transcript),
        model_calls=tuple(model_calls),
        evidence=await resources_client.complete_turn(event.turn_id),
    )
    # One Agent -> Environment result for the entire turn.
    event = await conversation.advance(completed)
```

`begin_turn()` initializes only Resources-owned state. `execute_round()` takes
an ordered batch and returns opaque string payloads plus typed receipts;
Resources—not Agent—assigns `turn_idx` and `call_idx`, owns mutable tool state,
and marks `limit_reached`. Parallel calls retain response order. A call beyond
the native cap receives no receipt and is not executed.

When Environment accepts the completed turn, `ConversationRuntime` validates
the transcript against Resources evidence, then replays the recorded assistant
subresponses and receipts through the native probe loop. Replay runs native
bookkeeping and custom `run_dispatch` logic but never executes a tool effect a
second time. Internal model/tool boundaries do not surface as lifecycle events.

`AssistantTurnRequest.model_messages()` applies the declared generic replay
policy for later calls (for example, reasoning is retained as evidence but not
fed back to the model, and bounded document-result projection is applied where
declared). The Agent remains autonomous: no UserSim callback chooses its next
model or tool step.

## Contract errors

An unoffered tool, reused id, cap violation, or any disagreement among the
completed transcript, model-call records, and Resources receipts raises
`EpisodeContractError`. These are host bugs: they never consume a model retry
or become assistant failures.

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
