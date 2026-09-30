# External Probe Runtime

`ProbeEpisodeRuntime` lets an external host run a tool-using probe without
moving probe policy or simulated state into the host.

Materialize inputs without model calls before constructing runtimes:

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

This uses the same Data Designer persona, probe, theme, and toolset sampler
graph as `usersim simulate`; the host does not select panel indices or author
themes/toolsets. Rows include the persona, probe inputs, locale-aware
behavioral settings, probe family/variant, trajectory ID, config snapshot, and
UserSim provenance. `models_path` and `assets_dir` are optional keyword
overrides; their packaged UserSim defaults are used otherwise. Model
configuration supplies trajectory identities but no configured model is
called. The equivalent CLI is:

```bash
usersim simulate --locale en_US --num-rows 1 \
  --probe-mix tool_calling=1 --random-seed 42 \
  --materialize-inputs --out episode-inputs.jsonl
```

`from_resolved_row()` restores the embedded UserSim config and preserves the
resolved trajectory identity and provenance. One runtime represents one such
resolved episode. It owns:

- the resolved probe and scenario state;
- Assistant tool schemas and the allowlist;
- probe-specific tool simulation and evidence;
- completion checks and result extras; and
- final native UserSim output.

The host owns only transport and mechanical orchestration. In a NeMo Gym
deployment, the Environment chooses the next participant, the Assistant Agent
runs the model-to-tool-to-model loop, and the Resources Server retains the
runtime.

## Host contract

Await `runtime.descriptor()` once after constructing the runtime. The
descriptor is immutable and JSON serializable. It includes participant
prompts, the initial verbatim user message when applicable, tool schemas, user
turn policy, and `AssistantToolLoopPolicy`.

The Assistant loop policy declares:

- `tool_round_mode`;
- `max_assistant_activations`;
- `max_tool_calls_per_turn`;
- whether a final tools-disabled synthesis activation is required;
- whether the probe has only one user turn; and
- retry and resampling behavior.

Hosts must not derive these rules from `probe_type`.

For every selected tool call, invoke `simulate_tool_call()` with only the
original non-empty `tool_call_id`, name, and arguments. For parallel calls
from one Assistant response, use `simulate_tool_calls()` with the ordered
batch. UserSim assigns native `turn_idx` and `call_idx`; the host must not
derive them. An identical retry is idempotent, while reusing an ID with a
different name or arguments is rejected. Simulated payloads are returned as
plain strings.

Use `synchronize_transcript()` to install the complete observed transcript as
it grows. Synchronization is monotonic. Every tool result must match runtime
execution evidence, and every executed call must match the Assistant function
call's ID, name, and arguments. JSON tool payloads are compared semantically
so transport whitespace does not change evidence identity.

The runtime exposes delegated follow-up, capitulation, success, and result
hooks over that synchronized state. Call `finalize()` once verification is
ready. Repeating `finalize()` without changing the transcript returns the same
result.

## Supported probes

The resumable lifecycle covers every registered probe. The hosted-parity suite
runs one standalone `ConversationSimulatorGenerator` execution and one
`ProbeEpisodeRuntime` execution for each registry entry and compares native
results after removing only wall-clock fields.
