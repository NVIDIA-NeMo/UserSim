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

**143 toolsets · 3,406 tools · 79 categories.** A "toolset" is a named bundle
of related tools (e.g. `openweathermap` → `current_weather`); `num_tools`
counts the functions in that bundle.

## Tools by source

Specs come from five public tool-calling benchmarks, plus a set written for
this project. Counts:

| Source | Toolsets | Tools |
|---|---:|---:|
| `UltraTools` | 22 | 2,328 |
| `ToolEyes` | 95 | 567 |
| `API-Bank` | 10 | 225 |
| `Custom` | 10 | 109 |
| `RestGPT` | 2 | 94 |
| `AutoTools` | 4 | 83 |
| **Total** | **143** | **3,406** |

> `UltraTools` dominates the raw tool count; `ToolEyes` contributes the most
> distinct toolsets (95). Per-toolset size ranges from 1 to 354 functions
> (mean ≈ 24).

`toolset_source` records where each row came from. [Sources and
licenses](#sources-and-licenses) lists each upstream project, the revision
the specs match, what changed and the license. Only the schemas are bundled;
nothing here calls a live API, and the catalog exists to exercise a model's
tool-selection behaviour rather than to reach any of these services.

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
| Food | 84 |
| Alarm | 70 |

The remaining 64 categories (Calendar, Database, Smart Home & IoT, Health &
Medical, Music, News, Translation, Stock, Weather, Hospital Ward Manager,
Library Management System, …) make up the long tail.

## Histograms

**Tools per source** (bars scaled to the largest):

```
UltraTools      ████████████████████████████████████████ 2328
ToolEyes        ██████████ 567
API-Bank        ████ 225
Custom          ██ 109
RestGPT         ██ 94
AutoTools       █ 83
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
                                     schema-shaped response
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
  tool's response from the schema, so a run needs no live service.
- Conversation themes (some tool-expected, some not) come from
  `DEFAULT_TOOL_CALLING_THEMES` in `cli/_pipeline.py`, exercising the
  call/clarify/refuse decision.

## Schema

One row per toolset. The same columns apply whether the file is
`toolsets_seed.parquet` or `toolsets_seed.jsonl`: jsonl being one JSON object
per line, with these keys:

| Column | Type | Description |
|---|---|---|
| `toolset_source` | str | Provenance (`UltraTools`, `ToolEyes`, `API-Bank`, `Custom`, `RestGPT`, `AutoTools`). |
| `toolset_category` | str | Coarse domain label (e.g. `Weather`, `Finance`). |
| `toolset_name` | str | Bundle name (e.g. `openweathermap`). The shipped catalog sets it on the `ToolEyes` rows and leaves it empty on the others. |
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

Abridged to two of its 11 parameters; the wording is the upstream text.

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
          "description": "Parameter defines the query or queries you want to search. You can use anything that you would use in a regular Google Trends search. The maximum number of queries per search is 5 (this only applies to \"Interest over time\" and \"Compared breakdown by region\" data_type, other types of data will only accept 1 query per search)."
        },
        "geo": {
          "type": "string",
          "description": "Parameter defines the location from where you want the search to originate. It defaults to Worldwide (activated when the value of geo parameter is not set or empty). Head to the Google Trends Locations for a full list of supported Google Trends locations."
        }
      },
      "required": ["query"]
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
  [`probes/safety_agentic/assets/safety_agentic/SCHEMA.md`](../../probes/safety_agentic/assets/safety_agentic/SCHEMA.md).

## Sources and licenses

The catalog combines tool specs from five public research benchmarks with
toolsets written for NeMo User Sim. Only tool schemas (name, description and
parameters) are included: no code, dialogues, test cases or recorded API
responses from these projects.

### Research benchmarks

| `toolset_source` | Toolsets | Tools | Upstream | License |
|---|---:|---:|---|---|
| `UltraTools` | 22 | 2,328 | [UltraTool](https://github.com/JoeYing1019/UltraTool) | [Apache-2.0](https://github.com/JoeYing1019/UltraTool/blob/main/LICENSE) |
| `ToolEyes` | 95 | 567 | [ToolEyes](https://github.com/Junjie-Ye/ToolEyes) | [Apache-2.0](https://github.com/Junjie-Ye/ToolEyes/blob/main/LICENSE) |
| `API-Bank` | 10 | 225 | [API-Bank](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank), part of DAMO-ConvAI | [Apache-2.0](https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/api-bank/LICENSE) for the `api-bank` folder, [MIT](https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/LICENSE) for the repository |
| `RestGPT` (categories `Spotify`, `TMDB`) | 2 | 94 | [RestGPT](https://github.com/Yifan-Song793/RestGPT) (RestBench) | [MIT](https://github.com/Yifan-Song793/RestGPT/blob/main/LICENSE) |
| `AutoTools` (categories `Food`, `Game`, `Music`, `Weather`) | 4 | 83 | [AutoTools](https://github.com/mangopy/AutoTools) | [Apache-2.0](https://github.com/mangopy/AutoTools/blob/main/LICENSE) |

Every spec is wrapped in the OpenAI function-calling envelope
(`{"type": "function", "function": {...}}`) and placed in a toolset. The
other changes are listed per source:

- **UltraTool** ([Huang et al., 2024](https://aclanthology.org/2024.findings-acl.259/)):
  every distinct tool spec in the English `dev.json` split of
  [`data/English-dataset`](https://github.com/JoeYing1019/UltraTool/tree/5d340adc26d5881725308398dd5dfc6441eb0156/data/English-dataset)
  (commit `5d340ad`), with one toolset per upstream `domain` label. Tool names
  and descriptions are unchanged. The `arguments` field is renamed
  `parameters`, and the `results` (output) schema is removed, including where
  upstream nests it in the arguments of `seat_preference_query` and
  `select_and_change_ticket`. Argument schemas are otherwise unchanged, except
  that `population_growth_rate` in `analyze_impact_on_economy` has type
  `number`, where upstream writes `float`, which is not a JSON Schema type.
- **ToolEyes** ([Ye et al., 2025](https://aclanthology.org/2025.coling-main.12/)):
  all 95 toolsets in
  [`Tool_Library`](https://github.com/Junjie-Ye/ToolEyes/tree/c818fcebe017951570a25e552a7984e40093fd52/Tool_Library)
  (commit `c818fce`), read from each toolset's `config_gpt4.json` and named
  after its folders. Specs are unchanged and keep their upstream order. The two
  agent-control functions that every upstream toolset includes, `ask_to_user`
  and `finish`, are removed. Three toolsets repeat a name upstream: the second
  `drivers` in `API F1` is `teams` and the second `blacklist_ip` in
  `Anti Malware` is `blacklist_url`, the functions their descriptions match in
  the toolset's `tool.py`, and `spoonacular products` keeps one of its two
  identical `classify_grocery_product` specs. In `Execute`, two parameter names
  are the ones the functions' signatures in `tool.py` use: `code` in
  `execute_pycode` (upstream `code:`) and `return_err_output` in
  `execute_bash_code` (upstream `return_err_outputpersistent=False`).
- **API-Bank** ([Li et al., 2023](https://aclanthology.org/2023.emnlp-main.187/)):
  100 of the 101 APIs listed in
  [`api-bank/data/all_apis.csv`](https://github.com/AlibabaResearch/DAMO-ConvAI/blob/faf754671f5b850e1b4d1ff870db37a6a2480df6/api-bank/data/all_apis.csv)
  (commit `faf7546`), all except `ImageCaption`. Descriptions and parameter
  descriptions are unchanged. Tool names are the upstream class names in lower
  case (`AddAlarm` becomes `addalarm`); parameter types are written as JSON
  Schema types (`str` as `string`, `int` as `integer`, `datetime` as `string`,
  `List[str]` as `array`); 32 tools gain a `required` list; and output
  parameters are removed. The `All` toolset holds every API, and nine topical
  toolsets repeat subsets of them.
- **RestGPT** ([Song et al., 2023](https://arxiv.org/abs/2306.06624)):
  all 40 Spotify and 54 TMDB operations in RestBench's OpenAPI specs under
  [`specs`](https://github.com/Yifan-Song793/RestGPT/tree/d99012860cb45323975bf5780152d7aa44f390f5/specs)
  (commit `d990128`), one tool per operation, named after its HTTP method and
  path. A tool's parameters are the operation's own, those declared on its
  path, and the fields of its JSON request body; a body field that repeats a
  query parameter appears once, as the query parameter. A parameter is
  required when upstream marks it so, and the Spotify spec writes that mark as
  the string `"true"` or `"false"`. Parameter schemas keep their type, `enum`,
  `default` and `items`. Descriptions are joined onto one line; each tool
  description ends with the operation's method and URL, and each parameter
  description ends with its location (for example `Location: query` or
  `Location: body`). One operation and 16 parameters that have no upstream
  description were given a short one: four query parameters and the 12 TMDB
  path parameters. In TMDB's `get_discover_tv`, the values that `with_status`
  and `with_type` list for a string parameter are written as strings.
- **AutoTools** ([Shi et al., 2025](https://arxiv.org/abs/2405.16533)):
  all 83 endpoints in the `food`, `game`, `music` and `weather` files under
  [`tools`](https://github.com/mangopy/AutoTools/tree/09c88df0959b2baf03e570aeff4365ea9f4d8b64/tools)
  (commit `09c88df`), one toolset per file and one tool per endpoint. They
  document APIs listed on RapidAPI: Spoonacular, OpenCritic, Genius and
  Meteosource. Tool names are the upstream names in lower case
  (`GET_menu_item_information` becomes `get_menu_item_information`).
  Parameters come from each endpoint's `parameters` list, which the `game`
  file names `params`. Descriptions are trimmed; each tool description ends
  with the endpoint's method and URL, and each parameter description ends with
  its location (for example `Location: query`). Parameter types are written in
  lower case, `float` as `number` and an empty type as `string`; a parameter
  named in the endpoint's URL is required; and the sample responses are
  removed. One `game` endpoint, `get_game_reviewed_today`, has its description
  in the upstream `url` field, so its tool description ends with that text and
  the API's host.

### Toolsets written for this project

`Custom` (10 toolsets, 109 tools) covers generic domains such as Inventory
Warehouse, Hospital Ward Manager and Recipe Manager. These toolsets were
written for NeMo User Sim and are covered by this project's Apache-2.0
license.

### License notices

UltraTool, ToolEyes, AutoTools and the `api-bank` folder of DAMO-ConvAI are
licensed under the Apache License, Version 2.0. None of them includes a NOTICE file.
The full license text is in this project's
[LICENSE](https://github.com/NVIDIA-NeMo/UserSim/blob/main/LICENSE) file.

RestGPT and the DAMO-ConvAI repository are licensed under the MIT License.
Their notices follow.

```text
MIT License

Copyright (c) 2023 Yifan Song

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

```text
MIT License

Copyright (c) 2022 Alibaba Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Trademarks and third-party services

Product and service names in the catalog, such as Spotify, TMDB, Spoonacular,
OpenWeatherMap and Google Trends, are trademarks of their owners and identify
the APIs that the specs describe. The catalog holds schemas only: nothing in
it calls a live API, and using any of these services is subject to its
provider's terms.

### Citation

If you publish results obtained with this catalog, please also cite the
benchmarks it draws on:

- Shijue Huang et al. 2024. [Planning, Creation, Usage: Benchmarking LLMs for Comprehensive Tool Utilization in Real-World Complex Scenarios](https://aclanthology.org/2024.findings-acl.259/). In *Findings of the Association for Computational Linguistics: ACL 2024*.
- Junjie Ye et al. 2025. [ToolEyes: Fine-Grained Evaluation for Tool Learning Capabilities of Large Language Models in Real-world Scenarios](https://aclanthology.org/2025.coling-main.12/). In *Proceedings of the 31st International Conference on Computational Linguistics (COLING 2025)*, pages 156-187.
- Minghao Li et al. 2023. [API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs](https://aclanthology.org/2023.emnlp-main.187/). In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023)*.
- Yifan Song et al. 2023. [RestGPT: Connecting Large Language Models with Real-World RESTful APIs](https://arxiv.org/abs/2306.06624). arXiv:2306.06624.
- Zhengliang Shi et al. 2025. [Tool Learning in the Wild: Empowering Language Models as Automatic Tool Agents](https://arxiv.org/abs/2405.16533). WWW 2025. arXiv:2405.16533.
