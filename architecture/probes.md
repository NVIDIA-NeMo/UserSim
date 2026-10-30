# Probe substrate

A probe owns *what* a simulated user does. This page is the substrate: the
base class, the hooks, the mixins. For the catalog of what ships, see
[`docs/probes.md`](../docs/probes.md); for a worked example,
[`docs/engine/AUTHORING_A_PROBE.md`](../docs/engine/AUTHORING_A_PROBE.md).

## Registration

`@register_probe` on a `BaseProbe` subclass does everything: it sets the
class attributes the dispatcher reads (`family`, `prompt_version`,
`variants`) and adds the class to the registry under its `label`. Each
probe carries its own, so two probes defined in one module record their own
prompt versions. The decorator also sets the same values on the module as
`PROBE_FAMILY`, `PROMPT_VERSION` and `PROBE_VARIANTS`; with two probes in one
module those hold whichever was registered last.

```python
@register_probe(family="general", prompt_version="v1.0", variants=("default",))
class OpenEndedProbe(BaseProbe):
    label = "general_open_ended"
```

The class *is* the registration; there is no separate label table to edit. A
second class registered under a taken label raises `ProbeRegistrationError`,
naming both classes, so give each probe a label of its own.

Shipped probes are listed by module in `BUILTIN_PROBE_MODULES`
(`core/probes.py`); the registry imports them the first time a probe is
looked up, which is when their decorators run. Out-of-tree probes are
imported through the `usersim.probes` entry-point group instead.

## Hooks the loop calls

`core/simulation.py` calls these on the probe. Only the first two are
required; the rest have working defaults on `BaseProbe`.

| Hook | When |
|---|---|
| `get_user_system_prompt` | every user turn |
| `get_assistant_system_prompt` | every assistant turn |
| `get_verbatim_first_user_turn` | turn 1, when a bank supplies it literally |
| `get_user_query_instruction` | turn 1, when the user-LLM writes it from the probe's own instruction |
| `format_gate_prompt` | judging a generated user turn before accepting it |
| `format_followup_user_instructions` | turns after the first |
| `on_followup_failure` | a follow-up turn fails its gate |
| `get_tools_for_assistant` | tool-using probes |
| `after_assistant_turn` | after each assistant reply; may execute tools |
| `format_assistant_judge_prompt` | in-sim assessment of the assistant turn |
| `is_capitulation_detected` | pressure probes checking for a flip |
| `allow_early_stop_at_turn` | deciding whether the conversation may end |
| `user_turn_policy` | once per trajectory: whether earlier replies may be summarized for context management, whether the last follow-ups may wrap up, which filter phrases the user may write, which names the script check skips, and a check each generated opening must pass before the gate |
| `seed_state_metadata` | recording bank/seed provenance on the row |
| `build_result_extras` | adding probe-specific columns to the trajectory, each declared in the class's `output_columns` |

Anything a probe does not override inherits a default, so a new probe declares
only what is specific to it. Its `description`, a paragraph on what it tests, is
what the per-row preview and the capability report show with its rows.

A probe's own columns are the one part of this that is declared twice. Whatever
`build_result_extras` returns is written only if the name is a core column or
appears in some registered probe's `output_columns`; anything else is
discarded, and the scorer that reads it back finds nothing to score. The
generator logs a warning naming the probe and the column the first time it
happens. A run declares the core columns plus the union of every registered
probe's `output_columns` as `ConversationSimulatorConfig.side_effect_columns`,
because Data Designer reads the declaration once, before any row says which
probe produced it.

Per-row data that is not a column of its own goes in `state.metadata`, which is
stored with the row as `conversation_metadata` and read from there by the
probe's scorer. Write it during the run, for example in `after_assistant_turn`:
the metadata is serialized before `build_result_extras` runs, so anything added
there is lost.

The conversation loop writes eight keys of its own there, listed with what each
holds in `RESERVED_METADATA_KEYS` (`core/simulation.py`): `user_judge_ratings`,
`assistant_judge_ratings`, `frustration_events`, `capitulation_checks`,
`stopped_on_capitulation`, `capitulated_at_turn`, `user_followup_retries` and
`early_stop`. A probe reads them and writes keys of its own;
`assert_probe_conforms` reads a probe's classes and reports any that write a
reserved key.

Six hooks can reach a model, and the loop awaits them: `get_verbatim_first_user_turn`,
`format_followup_user_instructions`, `after_assistant_turn`, `is_capitulation_detected`,
`execute_tool_call` and `run_dispatch`. Declare an override of any of these with
`async def`. `usersim.testing.assert_probe_conforms` reports a plain `def` override,
which would otherwise raise partway through a conversation.

## Mixins

Six mixins cover the shapes that recur, all in `core/probes.py`:

- **`BankVerbatimMixin`**: turn 1 comes from a bank, used literally.
- **`BankReframingMixin`**: turn 1 from a bank, then reframed under pressure.
- **`CustomTurn1InstructionMixin`**: turn 1 generated from an instruction
  rather than a bank.
- **`ToolCallingMixin`**: the assistant is offered tools, and the conversation
  cannot stop early until it has used one.
- **`ToolExecutionMixin`**: tool calls are actually executed against a mock
  environment.
- **`AgenticMixin`**: bank-seeded, tool-executing, multi-step.

Mixins that offer or execute tools (`AgenticMixin`, `ToolCallingMixin`,
`ToolExecutionMixin`) set `supports_assistant_resampling = False`. A resample
rolls the transcript back and re-calls the assistant, which is only safe when
the turn had no side effects.

## Assets

A probe's content lives at `engine/assets/<label>/`, resolved through
`probe_assets_dir()`. A run fails pre-flight, before any model call, if a
selected probe's bank is missing or, for the probes whose banks are split by
locale, doesn't load for a run locale (a finance bank without tasks, for
instance). Those probes declare a `locale_bank_check` on the class, which the
pre-flight calls on the bank the class's `bank_locale` names. The refusal names
the probe, the locale and the locales that can run, rather than failing from
inside a loader mid-run. `simulate` and its `--dry-run`, `--materialize-inputs`,
`--inputs` and notebook 01 all run the check. See [assets.md](assets.md).

A probe whose conversations vary by topic lists its themes in a `themes`
classmethod, read under its asset folder. A run draws one per row into the
`theme` column, and the probe reads it back with `theme()`. A probe that lists
none gets a placeholder it never reads. `tool_calling` takes its themes and
toolsets from the run's settings instead.
