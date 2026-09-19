# Probe substrate

A probe owns *what* a simulated user does. This page is the substrate: the
base class, the hooks, the mixins. For the catalogue of what ships, see
[`docs/probes.md`](../docs/probes.md); for a worked example,
[`docs/engine/AUTHORING_A_PROBE.md`](../docs/engine/AUTHORING_A_PROBE.md).

## Registration

`@register_probe` on a `BaseProbe` subclass does everything: it sets the
module constants the dispatcher reads (`PROBE_FAMILY`, `PROMPT_VERSION`,
`PROBE_VARIANTS`) and adds the class to the registry under its `label`.

```python
@register_probe(family="general", prompt_version="v1.0", variants=("default",))
class OpenEndedProbe(BaseProbe):
    label = "general_open_ended"
```

The class *is* the registration; there is no separate table to edit. A
duplicate label is rejected rather than silently overwriting.

## Hooks the loop calls

`core/simulation.py` calls these on the probe. Only the first two are
required; the rest have working defaults on `BaseProbe`.

| Hook | When |
|---|---|
| `get_user_system_prompt` | every user turn |
| `get_assistant_system_prompt` | every assistant turn |
| `get_verbatim_first_user_turn` | turn 1, when a bank supplies it literally |
| `format_gate_prompt` | judging a generated user turn before accepting it |
| `format_followup_user_instructions` | turns after the first |
| `on_followup_failure` | a follow-up turn fails its gate |
| `get_tools_for_assistant` | tool-using probes |
| `after_assistant_turn` | after each assistant reply; may execute tools |
| `format_assistant_judge_prompt` | in-sim assessment of the assistant turn |
| `is_capitulation_detected` | pressure probes checking for a flip |
| `allow_early_stop_at_turn` | deciding whether the conversation may end |
| `seed_state_metadata` | recording bank/seed provenance on the row |
| `build_result_extras` | adding probe-specific columns to the trajectory |

Anything a probe does not override inherits a default, which is what keeps a
new probe to roughly fifty lines.

## Mixins

Six mixins cover the shapes that recur, all in `core/probes.py`:

- **`BankVerbatimMixin`**: turn 1 comes from a bank, used literally.
- **`BankReframingMixin`**: turn 1 from a bank, then reframed under pressure.
- **`CustomTurn1InstructionMixin`**: turn 1 generated from an instruction
  rather than a bank.
- **`ToolCallingMixin`**: the assistant is offered tools; the probe checks
  call/clarify/refuse behaviour.
- **`ToolExecutionMixin`**: tool calls are actually executed against a mock
  environment.
- **`AgenticMixin`**: bank-seeded, tool-executing, multi-step.

Mixins that offer or execute tools (`AgenticMixin`, `ToolCallingMixin`,
`ToolExecutionMixin`) set `supports_assistant_resampling = False`. A resample
rolls the transcript back and re-calls the assistant, which is only safe when
the turn had no side effects.

## Assets

A probe's content lives at `engine/assets/<label>/`, resolved through
`probe_assets_dir()`. A run fails pre-flight if a selected probe's bank is
missing, naming the probe and the path rather than failing from inside a
loader mid-run. See [assets.md](assets.md).
