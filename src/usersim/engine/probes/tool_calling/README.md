# tool_calling

Tests whether the assistant calls tools correctly when a persona-grounded user
asks for something a tool can do: choosing the tool, shaping its arguments to
the tool's schema, and recovering when a call fails. An invalid call gets the
error an API would return, not a correction. A conversation in which the
assistant calls no tool is a completed trajectory, and the judge scores it like
any other. [`docs/probes.md`](../../../../../docs/probes.md) places it among
the other probes.

## How a row runs

- Each row gets one toolset from the catalog, the probe's `seed_dataset`, and
  up to `max_tools` of its tools, drawn the same way for the same row every
  run. The toolset is the row's theme.
- The simulated user writes requests the tools can serve, in the row's
  language, and holds any credential a tool declares.
- The assistant is offered the tools. A call that breaks the tool's schema gets
  a 400, a call without the user's credential gets a 401, and any other call
  gets a response the `api_response_model` writes for that tool.
- Within a turn, the tools stay on offer after each round of calls, so the
  assistant can chain calls, until it replies without calling one or the turn
  has made 8 calls; a call past that is not run. The conversation does not end
  early before the assistant has called a tool.
- Each row records `tool_subset`, the tools it offered, and `num_tool_calls`,
  the calls that passed the schema and credential checks. `verifier_results` in
  `conversation_metadata` lists every call and whether it passed.

Two settings, each set with `--probe-option tool_calling.<name>=<value>`:
`max_tools`, the most tools a conversation offers (default `5`), and
`tools_column`, the row column holding the toolset's tools (default `tools`).

## Locales

One catalog and one `prompts.yaml` serve every locale, the India language
variants included; the simulated user writes in the row's language.

## Assets

| File | What it holds |
|---|---|
| `assets/tool_calling/toolsets_seed.parquet` | The tool catalog, one toolset per row with its tools as OpenAI-style function specs. |
| `assets/tool_calling/README.md` | The catalog's schema, how its tools are simulated, how to use a catalog of your own, and its sources and licenses. |
| `assets/tool_calling/prompts.yaml` | The simulated user's prompt and credentials block, the tool simulator's prompt, and the judge prompts. |

The catalog's specs come from public tool-calling benchmarks, each pinned to an
upstream commit with its license, plus toolsets written for this project; every
row names its source in `toolset_source`. The
[catalog README](assets/tool_calling/README.md#sources-and-licenses) records
each source and every change made to its specs.

## Adding a locale

Nothing is needed per locale. To run on other tools, see
[using your own catalog](assets/tool_calling/README.md#using-your-own-catalog).

## Scoring

The `tool_use` scorer, in `default_scorers`, judges each conversation on task
completion, tool selection, argument quality, unnecessary tool use, information
gathering, role adherence, architecture leaking and an overall score. Its
evidence feeds the `tool_use_correctness` and `tool_use_restraint`
capabilities in `capabilities.py`.

## Tests

`tests/probes/tool_calling/` holds the probe's tests (`test_probe.py`), the
catalog's (`test_catalog.py`), the credentials' (`test_credentials.py`) and the
call verifier's (`test_verifiers.py`), with `contract.py` for the shared tests.
