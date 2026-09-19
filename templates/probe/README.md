# Probe template gallery

A copy-and-edit reference for adding a new probe to NeMo UserSim via the
`BaseProbe` substrate.

For the **end-to-end tutorial** (decision tree on which shape to
pick, full worked example, common gotchas), read
[`docs/engine/AUTHORING_A_PROBE.md`](../../docs/engine/AUTHORING_A_PROBE.md).
This README is a quick orientation to the files in this directory.

## When to add a probe

A probe answers the question *what kind of conversation is the user
having with the assistant under test, and what about that conversation
do we need to surface?* Each probe owns:

- The user-agent system prompt (how the simulated user is told to act).
- The per-turn user-judge gate prompt (the in-sim guardrail that
  keeps the user-agent from slipping into "AI helpful assistant" mode).
- An optional asset bank (curated facts / queries / strategies /
  action requests) that grounds the probe in reviewer-vetted content.
- Probe-specific side-channel columns (`facts_probed`, `query_id`,
  `target_request_id`, `attempted_actions`, …) that the matching
  out-of-sim scorer reads.

If the new "probe" is just a different prompt for an existing probe,
prefer a `probe_variant` over a new module.

## Files in this gallery

```
templates/probe/
├── README.md                       # this file
├── __init__.py                     # marker
├── generator_baseprobe_only.py     # BaseProbe-only: no asset bank
├── generator_bank_verbatim.py      # BankVerbatimMixin: verbatim turn-1 from a bank
├── generator_custom_turn1.py       # CustomTurn1InstructionMixin: bank-anchored turn-1, generated
├── generator_bank_reframing.py     # BankReframingMixin: multi-turn pressure with reframings
├── generator_agentic.py            # AgenticMixin: single user turn + tool-interception
├── generator_tool_calling.py       # ToolCallingMixin: early-stop deferred until tool used
├── prompts.py                      # LocalePromptPack demo
└── test_probe.py.template          # copy-and-edit per-probe test scaffold
```

When you port the template, copy the `generator_<shape>.py` whose
shape matches your probe into
`src/usersim/engine/probes/<your_probe>/generator.py`,
copy `prompts.py` into the same directory, and copy
`test_probe.py.template` to
`tests/engine/probes/test_<your_probe>.py`. Then follow the
authoring tutorial.

## The contract a probe class must satisfy

A probe class is a `BaseProbe` subclass (plus an optional mixin)
decorated with `@register_probe(...)`. Three module-level constants
land automatically via the decorator (`PROBE_FAMILY` /
`PROMPT_VERSION` / `PROBE_VARIANTS`); the smoke check verifies
they're present.

The runtime `ProbeAdapter` contract has 8 required hooks: class-level
`label`, two system-prompt accessors, two judge-prompt formatters, the
tool accessor, `after_assistant_turn`, and `build_result_extras`.
`BaseProbe` supplies defaults for five of them, so a normal subclass only
sets `label` and implements the two system-prompt accessors. It also
provides up to 9 optional hooks. The mixins pre-configure the optional
hooks for the four common probe shapes; what you override differs by shape.

| Shape | Recommended base | Mixin | Override these | Skip these |
|---|---|---|---|---|
| No asset bank | `BaseProbe` | none | `get_user_system_prompt`, `get_assistant_system_prompt` | everything else (use defaults) |
| Verbatim turn-1 from a bank | `BankBackedProbe` | `BankVerbatimMixin` | `derive_task`, `get_verbatim_first_user_turn`, `should_succeed`, `build_result_extras` | turn-1 generation (the mixin handles the verbatim path) |
| Bank-anchored generated turn-1 | `BankBackedProbe` | `CustomTurn1InstructionMixin` | `derive_task`, `seed_state_metadata`, `get_user_query_instruction`, `should_succeed`, `build_result_extras` | reframings (no follow-up template logic) |
| Multi-turn bank-driven reframings | `BankBackedProbe` | `BankReframingMixin` | `derive_task`, `get_verbatim_first_user_turn`, `format_followup_user_instructions`, `should_succeed`, `build_result_extras` | the follow-up failure mode (mixin sets `"abort"`) |
| Agentic single-user-turn + tool loop | `BankBackedProbe` | `AgenticMixin` | `derive_task`, `run_dispatch` (custom loop, not `ConversationLoop`), `build_result_extras` | follow-up generation (no user-LLM at all) |
| Tool-calling capability test | `BaseProbe` | `ToolCallingMixin` | tool-sampling logic in `__init__`, `after_assistant_turn` (tool execution), `should_succeed`, `build_result_extras` | early-stop heuristic (mixin gates on tool-use first) |

See the per-shape generator file in this directory for a copy-pasteable starter.

## Where probe judging lives

- **In-sim judges** (kept in this module + `core/simulation.py`): the
  user-query gate, the fourth-wall pre-filter, the per-turn user-judge
  rubric, the optional assistant-quality flag, the early-stop signal.
  Their job is to keep the conversation coherent so the trajectory is
  worth scoring at all.
- **Out-of-sim judges** (live in
  `evaluator/scorers/<your_probe>.py`): any LLM- or rule-based
  scoring that judges the *assistant under test*. The simulator never
  runs these. Register your probe's scorer in
  `evaluator/scorers/__init__.py::DEFAULT_SCORER_MODULES` so the
  auto-load path picks it up.

See the "in-sim / out-of-sim boundary" discussion
for the rationale.

## Wiring checklist

When porting one of the templates into a new module:

- [ ] Pick a `label` for `@register_probe`, used by `--probe-mix`
      and the `_PROBE_REGISTRY` lookup. Underscores, lowercase.
- [ ] Pick a `family` (the `PROBE_FAMILY` constant the decorator
      sets). The reporting layer stratifies on this.
- [ ] Bump `prompt_version` on **any** semantic change to the
      user-agent prompt or the per-turn judge rubric. The
      evaluator's partial-re-run dedup key includes it.
- [ ] Wire the asset bank (if applicable): set `bank_loader`,
      `placeholder_warning_kind`, `bank_version_key`. See
      `core/probes.py::BankBackedProbe` for the contract.
- [ ] Author the user-agent system prompt + the per-turn judge
      rubric in `prompts.py`. Use `LocalePromptPack` for any prompt
      that ships across multiple locales.
- [ ] Add a deterministic scorer under
      `evaluator/scorers/<your_probe>.py` and include the module
      path in `DEFAULT_SCORER_MODULES`.
- [ ] Write tests using the patterns in `test_probe.py.template`:
      use `_patched_call_llm` to patch BOTH `core.simulation.call_llm`
      AND `core.judges.call_llm`; use the alias-dispatching mock with
      XML-shaped judge payloads. See
      [`tests/engine/probes/test_sov_ai_facts.py`](../../tests/engine/probes/test_sov_ai_facts.py)
      for a real reference.
- [ ] Update `docs/engine/README.md`'s probe table.
- [ ] Record the design choices in the probe module docstring
      (variant set, prompt version, what the scorer measures).
