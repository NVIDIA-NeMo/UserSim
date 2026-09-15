# `tool_calling`: Simulated Tool Catalog

This folder holds the seed assets for the **`tool_calling`** probe, which tests
whether the assistant-under-test invokes tools correctly: right tool selection,
well-formed arguments, schema compliance, and graceful handling of the
`call` / `clarify` / `refuse` meta-decision.

The tools the assistant is offered are **not hand-authored here**: they are a
curated catalog of real, public tool/function specs in
[`toolsets_seed.parquet`](toolsets_seed.parquet). This README documents that
catalog (what it contains, where it came from, and how it is used); the
authoritative per-tool descriptions live inside each tool's JSON spec.

## Catalog at a glance

**143 toolsets · 3,407 tools · 79 categories.** A "toolset" is a named bundle
of related tools (e.g. `openweathermap` → `current_weather`); `num_tools`
counts the functions in that bundle.

## Tools by source

Specs are drawn from public tool-calling benchmarks plus a small curated /
auto-generated set. Counts:

| Source | Toolsets | Tools |
|---|---:|---:|
| `UltraTools` | 22 | 2,328 |
| `ToolEyes` | 95 | 568 |
| `API-Bank` | 10 | 225 |
| `Auto-generated` | 6 | 177 |
| `Custom` | 10 | 109 |
| **Total** | **143** | **3,407** |

> `UltraTools` dominates the raw tool count; `ToolEyes` contributes the most
> distinct toolsets (95). Per-toolset size ranges from 1 to 354 functions
> (mean ≈ 24).

### Provenance

`toolset_source` records where each row came from, and the label is worth
reading precisely:

- **`Custom`**: authored for this project. Generic domains (Inventory
  Warehouse, Hospital Ward Manager, Recipe Manager, and so on) with no
  external derivation.
- **`Auto-generated`**: despite the name, these describe publicly
  documented third-party APIs: Spoonacular, OpenCritic and Genius via
  RapidAPI, plus Spotify, TMDB and Meteosource. The descriptions retain the
  vendors' own wording and endpoint URLs.
- **`ToolEyes` / `UltraTools` / `API-Bank`**: drawn from the public
  tool-calling benchmarks of those names.

Only the schemas are bundled; nothing here calls a live API, and the
catalog exists to exercise a model's tool-selection behaviour rather than to
reach any of these services.

## Tools by category (top 15 of 79)

| Category | Tools |
|---|---:|
| Finance | 444 |
| Travel | 290 |
| Meeting | 227 |
| Document | 195 |
| Repair | 171 |
| Agenda | 171 |
| Flight | 171 |
| Other | 129 |
| Hotel | 120 |
| All | 100 |
| Competition | 95 |
| Account | 87 |
| Train | 86 |
| Food | 85 |
| Alarm | 70 |

The remaining ~64 categories (Calendar, Database, Smart Home & IoT, Health &
Medical, Music, News, Translation, Stock, Weather, Hospital Ward Manager,
Library Management System, …) make up the long tail.

## Histograms

**Tools per source** (bars scaled to the largest):

```
UltraTools      ████████████████████████████████████████ 2328
ToolEyes        ██████████ 568
API-Bank        ████ 225
Auto-generated  ███ 177
Custom          ██ 109
```

**Toolsets by size** (how many functions each toolset bundles):

```
   1 tool   ███████ 8
  2-5       ██████████████████████████████████ 39
  6-10      ████████████████████████████████████████ 46
 11-25      ████████████████████████ 28
 26-50      ██████ 7
 51-100     █████ 6
101-500     ████████ 9
```

Most toolsets are small (≈85% have ≤25 functions); the `UltraTools` bundles
form the long tail that dominates the raw tool count.

## How the tools are simulated

```
toolsets_seed.parquet ── sample a toolset ──▶ assistant_model (model under test)
                                                   │ emits tool call(s)
                                                   ▼
                                     api_response_model synthesizes a
                                     deterministic, schema-shaped response
                                                   │
                                                   ▼
                              tool_use scorer grades the trajectory
                              (task_completion, tool_selection,
                               argument_quality, unnecessary_tool_use,
                               architecture_leaking, overall, ...)
```

- One toolset is sampled per conversation (`toolset_name` / `tools` columns are
  threaded into the simulator by [`cli/_pipeline.py`](../../../cli/_pipeline.py)).
- The assistant sees the toolset's tools as OpenAI-style function specs and
  decides whether/which to call.
- There is **no real execution**: the `api_response_model` alias (see
  [`cli/models_default.toml`](../../../cli/models_default.toml)) synthesizes the
  tool's response from the schema, so behavior is reproducible.
- Conversation themes (some tool-expected, some not) come from
  `DEFAULT_TOOL_CALLING_THEMES` in `cli/_pipeline.py`, exercising the
  call/clarify/refuse decision.

## Schema

One row per toolset. The same columns apply whether the file is
`toolsets_seed.parquet` or `toolsets_seed.jsonl`: jsonl being one JSON object
per line, with these keys:

| Column | Type | Description |
|---|---|---|
| `toolset_source` | str | Provenance (`ToolEyes`, `API-Bank`, `UltraTools`, `Custom`, `Auto-generated`). |
| `toolset_category` | str | Coarse domain label (e.g. `Weather`, `Finance`). |
| `toolset_name` | str | Bundle name (e.g. `openweathermap`). |
| `num_tools` | int | Number of functions in the bundle. |
| `tools` | str (JSON) or list | List of OpenAI-style function specs (`{"type": "function", "function": {"name", "description", "parameters"}}`). |

`tools` is the one column whose form differs by container. Parquet stores it
as a JSON-encoded string; in jsonl you can write it either way: a nested
array, which is the natural shape there, or the same encoded string. The probe
accepts both (`json.loads` when it is a string, normalized to a Python list
when Data Designer loads it as a nested array), so a hand-written jsonl catalog
does not need its tool specs
double-encoded.

### Using your own catalog

A run reads `<assets-dir>/tool_calling/toolsets_seed.parquet` by default, or
`toolsets_seed.jsonl` if that is what is present. Keep exactly one: both
suffixes in the same directory is an error rather than a precedence rule,
so that editing the one the run ignores cannot look like an edit that had no effect.

To point a run at a file somewhere else, without touching this tree:

```bash
usersim simulate --locale en_US --num-rows 10 \
    --probe-mix 'tool_calling=1.0' \
    --toolset-seed-path /path/to/my_toolsets.jsonl \
    --out output/trajectories
```

The file must carry the columns above and use a `.parquet` or `.jsonl`
suffix. A missing or unsupported file is an error: a run that silently fell
back to the shipped catalog would report results for tools you did not ask
about. Supplying this flag without `tool_calling` in `--probe-mix` is also an
error because the override would otherwise be silently unused.

### Example tool spec (`ToolEyes / Trend / Google Trends`)

```json
{
  "type": "function",
  "function": {
    "name": "google_trends_search",
    "description": "Scrape results from the Google Trends search page.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "The query or queries to search (up to 5 for some data types)."
        },
        "geo": {
          "type": "string",
          "description": "Two-letter country code to scope the trend (e.g. US)."
        }
      }
    }
  }
}
```

## Inspecting the catalog

```python
import pandas as pd
df = pd.read_parquet("assets/tool_calling/toolsets_seed.parquet")

# tools per source
df.groupby("toolset_source")["num_tools"].agg(["count", "sum"])

# peek at a toolset's function specs
import json
json.loads(df.iloc[0]["tools"])
```

## Related

- Probe implementation: [`src/usersim/engine/probes/tool_calling/generator.py`](../../probes/tool_calling/generator.py)
- Scorer: [`src/usersim/engine/evaluator/scorers/tool_use.py`](../../evaluator/scorers/tool_use.py)
- Turn-1 / probing prompts: [`prompts.yaml`](prompts.yaml)
- The **agentic** safety surface uses a *different*, hand-authored mock-tool
  mechanism (per-action tool specs + canned responses) documented in
  [`assets/safety_agentic/SCHEMA.md`](../safety_agentic/SCHEMA.md).
