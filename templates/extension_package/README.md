# Extension package template

A package to copy when you start a probe of your own: the probe, its scorer, the
capability report rows they feed, the assets they read and their tests, wired
together. Every name in it is a
placeholder to replace. [`docs/extensions.md`](../../docs/extensions.md) builds a
complete package step by step and explains each piece.

```
extension_package/
├── pyproject.toml
├── src/my_probe/
│   ├── __init__.py
│   ├── probe.py
│   ├── scorer.py
│   ├── capabilities.py
│   └── assets/my_probe/
│       ├── prompts.yaml
│       ├── en_US/themes.yaml
│       └── pt_BR/themes.yaml
└── tests/
    ├── conftest.py
    └── test_my_probe.py
```

## The probe sets the area, and the persona explores it

Each locale's `themes.yaml` lists the areas a run draws from, one per row: a
topic and the bounds the conversation stays within. The simulated user gets the
row's area through `prompts.yaml` and explores it in the persona's own words, so
the probe never writes the user's messages. Keep it that way as you fill it in:
write areas and their bounds, not messages to send.

## Make it yours

1. Copy this folder out of the repository, and rename `my-probe`, the
   distribution, and `my_probe`, the package, the probe's label and the
   scorer's name. Every name the package adds starts with your prefix, as
   `my_tool_calls` and `my_probe.tool_used_rate` start with `my_`; see
   [Names](../../docs/extensions.md#names).
2. Write your areas in `themes.yaml`, one file per locale in its language, in
   a folder named for the locale.
3. Write the user's instructions in `prompts.yaml`, once, in English.
   `render_prompt` fills `{persona}`, `{area}` and any column of the row, and
   `language_instruction` has the user write in the row's language. Give the
   area and its bounds, not lines to say: the persona writes every turn.
4. Replace `DO_TASK` with the tool your probe offers the assistant, or drop it,
   `ToolExecutionMixin` and `execute_tool_call` if it offers none.
5. Replace the scorer with what your probe measures, and `capabilities.py`
   with the report rows it feeds: without a capability the probe's rows are
   absent from the capability report.
6. Set the `usersim` range in `pyproject.toml` to the releases the package
   works with: from the first release with every hook it uses, up to the next
   minor release.

## Test it

```bash
uv sync
uv run pytest
uv run usersim smoke
```

The tests check the probe and the scorer against the contract, run an episode
with scripted models and no provider, and check that a host driving the probe
reproduces a standalone run. `make check-extensions` installs this template next
to a freshly built User Sim and runs them, so the template keeps working as User
Sim changes.

## In this repository instead

A probe that belongs in this repository keeps the same shape: `probe.py` with
its assets in `assets/<label>/` beside it, under
`src/usersim/engine/probes/<label>/`.
[Where should a probe live?](../../CONTRIBUTING.md#where-should-a-probe-live)
says when a probe belongs here, and
[`AUTHORING_A_PROBE.md`](../../docs/engine/AUTHORING_A_PROBE.md) what else an
in-tree probe needs.
