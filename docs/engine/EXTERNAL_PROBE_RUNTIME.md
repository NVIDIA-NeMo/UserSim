# External Probe Runtime

`ProbeEpisodeRuntime` lets an external host run a tool-using probe without
moving probe policy or simulated state into the host.

One runtime represents one resolved episode. It owns:

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

For every selected tool call, invoke `simulate_tool_call()` with the original
non-empty `tool_call_id`, semantic `turn_idx`, and `call_idx`. The runtime
rejects unavailable tools, duplicate identities, and mutation after
finalization.

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

The external runtime currently supports:

- `tool_calling`;
- `safety_agentic`; and
- `financial_services`.

Other probes continue to use the native Conversation Loop. Adding another
probe requires an explicit external-runtime opt-in and host-contract parity
tests; it is not inferred from the presence of tool schemas.
