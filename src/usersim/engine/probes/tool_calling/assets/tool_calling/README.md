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
  threaded into the simulator by [`cli/_pipeline.py`](../../../../../cli/_pipeline.py)).
- The assistant sees the toolset's tools as OpenAI-style function specs and
  decides whether/which to call.
- There is **no real execution**: the `api_response_model` alias (see
  [`cli/models_default.toml`](../../../../../cli/models_default.toml)) synthesizes the
  tool's response from the schema, so a run needs no live service.
- A call that breaks its tool's schema gets a `400` response instead of a
  simulated one, and a call without the user's key for a tool that takes one
  gets a `401` (see [Credentials](#credentials)).
- A conversation's theme is its toolset: the user is asked for a task in the
  toolset's `toolset_category` that involves its `toolset_name`, or its
  category where the name is empty. A catalog without categories gives each
  row its toolset's name. The probe lists no themes, so a row's `theme`
  column holds the run's `__not_used__` placeholder, as it does for every
  probe that draws its area from its own data.

## Schema

One row per toolset. The same columns apply whether the file is
`toolsets_seed.parquet` or `toolsets_seed.jsonl`: jsonl being one JSON object
per line, with these keys. A run stores them on its `tool_calling` rows, and
leaves them empty on the rows of the other probes in its mix.

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

### Credentials

A tool entry may declare `credentials` beside its function spec: a mapping
from a parameter of the function to the credential that parameter takes,
named the way its user knows it.

```json
{"type": "function", "function": {"name": "google_trends_search", "...": "..."}, "credentials": {"api_key": "SerpApi key"}}
```

Each conversation draws one random value for every credential its tools
declare, so tools that share a credential take the same value and no two
conversations share one. The simulated user holds those values and gives one
when the assistant asks for it. A call whose argument is not the value the
user holds, because it is empty or made up, gets a `401` response, and nothing
is simulated for it; a call that leaves out a required parameter gets a `400`,
as any call that breaks its schema does. The declaration stays out of the
request the assistant's API receives, and the tool list each judge reads names
the credentials a tool needs, so asking the user for one counts as part of the
call.

A tool that sets a new value for a credential, such as a password change or a
registration, declares `sets_credentials` the same way:

```json
{"type": "function", "function": {"name": "modifypassword", "...": "..."}, "credentials": {"old_password": "account password"}, "sets_credentials": {"new_password": "account password"}}
```

Once such a call gets past the schema and credential checks, the value it
passed is the one the user holds: later calls must carry it, the old value
gets a `401`, and the user's list of credentials shows the new one. This
declaration stays out of the request too.

### Using your own catalog

The probe declares the catalog as its seed table (`seed_dataset =
"toolsets_seed"`), so a run reads `toolsets_seed.parquet` from the probe's
asset folder, or `toolsets_seed.jsonl` if that is what is present, and gives
each row the columns of one catalog row. Under `--assets-dir`, the folder is
`<assets-dir>/tool_calling/`. Keep exactly one: both suffixes in the same
directory is an error rather than a precedence rule, so that editing the one
the run ignores cannot look like an edit that had no effect. A run holds one
seed table, so `tool_calling` cannot share a run with another probe that
reads one.

To point a run at a file somewhere else, without touching this tree:

```bash
usersim simulate --locale en_US --num-rows 10 \
    --probe-mix 'tool_calling=1.0' \
    --seed-dataset-path /path/to/my_toolsets.jsonl \
    --out output/trajectories
```

The file must carry the columns above and use a `.parquet` or `.jsonl`
suffix. A missing or unsupported file is an error: a run that silently fell
back to the shipped catalog would report results for tools you did not ask
about. Supplying this flag without `tool_calling` in `--probe-mix` is also an
error because the override would otherwise be silently unused.

### Example tool spec (`ToolEyes / Trend / Google Trends`)

Abridged to two of its 11 parameters: `query`, in the upstream wording, and
`api_key`, the key the tool declares under `credentials`.

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
        "api_key": {
          "type": "string",
          "description": "Your SerpApi key."
        }
      },
      "required": ["query", "api_key"]
    }
  },
  "credentials": {"api_key": "SerpApi key"}
}
```

## Inspecting the catalog

```python
import pandas as pd
df = pd.read_parquet("src/usersim/engine/probes/tool_calling/assets/tool_calling/toolsets_seed.parquet")

# tools per source
df.groupby("toolset_source")["num_tools"].agg(["count", "sum"])

# peek at a toolset's function specs
import json
json.loads(df.iloc[0]["tools"])
```

## Related

- Probe implementation: [`src/usersim/engine/probes/tool_calling/probe.py`](../../probe.py)
- Scorer: [`src/usersim/engine/evaluator/scorers/tool_use.py`](../../../../evaluator/scorers/tool_use.py)
- Turn-1 / probing prompts: [`prompts.yaml`](prompts.yaml)
- The **agentic** safety surface uses a *different*, hand-authored mock-tool
  mechanism (per-action tool specs + canned responses) documented in
  [`probes/safety_agentic/assets/safety_agentic/SCHEMA.md`](../../../safety_agentic/assets/safety_agentic/SCHEMA.md).

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
  are unchanged. The `arguments` field is renamed `parameters`, and the
  `results` (output) schema is removed, including where upstream nests it in
  the arguments of `seat_preference_query` and `select_and_change_ticket`.
  UltraTool's tools have no implementation, so each spec is checked against its
  own description, the `dev.json` plan steps that call it and the `results` of
  the tools that feed it:
  - Every array declares its item type (179 parameters). A parameter whose
    description, examples or plan calls show a number, a boolean or a list has
    that type (98), among them `population_growth_rate` in
    `analyze_impact_on_economy`, where upstream writes `float`.
  - 103 parameters that a call needs and its schema lacks, such as the origin
    and destination of `book_best_flight`, are added, named and typed after the
    matching parameter of a related tool, and 33 tools require the inputs they
    cannot run without.
  - `example` is written `examples`, `min` is `minimum`, `dependencies` is
    `dependentRequired`, `nullable` is removed, an enum lists each value once
    and the values the tool's plans use, and a `format` outside JSON Schema becomes a `pattern`, a JSON Schema
    format or text in the description (32 formats).
  - Every nested field has a description (29 added), and 72 descriptions that
    contradict their tool, or that leave a list's layout unstated, are
    rewritten.
  - The 40 parameters that take a password the user already holds, such as
    `password` in `account_login` and `old_password` in `change_password`, are
    required, described by the account they open, such as `Your online banking
    password.`, and declared under their tool's `credentials`. The 9
    `new_password` parameters of the change, reset and update tools in
    `Account`, `Finance` and `Meeting` are declared under `sets_credentials`
    (see [Credentials](#credentials)).
- **ToolEyes** ([Ye et al., 2025](https://aclanthology.org/2025.coling-main.12/)):
  all 95 toolsets in
  [`Tool_Library`](https://github.com/Junjie-Ye/ToolEyes/tree/c818fcebe017951570a25e552a7984e40093fd52/Tool_Library)
  (commit `c818fce`), read from each toolset's `config_gpt4.json` and named
  after its folders. Specs keep their upstream order. The two
  agent-control functions that every upstream toolset includes, `ask_to_user`
  and `finish`, are removed. Three toolsets repeat a name upstream: the second
  `drivers` in `API F1` is `teams` and the second `blacklist_ip` in
  `Anti Malware` is `blacklist_url`, the functions their descriptions match in
  the toolset's `tool.py`, and `spoonacular products` keeps one of its two
  identical `classify_grocery_product` specs. In `Execute`, two parameter names
  are the ones the functions' signatures in `tool.py` use: `code` in
  `execute_pycode` (upstream `code:`) and `return_err_output` in
  `execute_bash_code` (upstream `return_err_outputpersistent=False`). Each
  spec is checked against the function of the same name in its toolset's
  `tool.py`, the code that calls the API:
  - A parameter that upstream types `string` has the type the function's
    signature gives it, or that its use shows where the signature gives none
    (669 parameters, most of them integers and booleans), and each array
    declares its item type (80).
  - A tool or parameter description that contradicts the function, such as the
    nine CoinMarketCap tools that share one tool's text or Ergast's ids
    described as names, or that is empty, such as 13 search tools, is written
    from the endpoint the function calls (228 descriptions).
  - `required` follows the signature where upstream's list differs from it
    (14 tools), five parameters the functions take are added, and `random`,
    which `search_recipes_by_ingredients` does not take, is removed.
  - The 367 parameters, across 364 tools in 68 toolsets, that take the
    caller's credential for the service a function calls (an API key, token,
    secret or app ID, which upstream leaves optional) are required. Each is
    described by its service, such as `Your SerpApi key.`, named after the
    host or client library the function calls, and declared under its tool's
    `credentials` (see [Credentials](#credentials)). `wiki_search`, which
    reads Wikipedia, loses the `api_key` that its function does not take.
  - The 6 parameters that take the user's mail or database password,
    `login_password` in the five `Mail` tools and `password` in
    `execute_mysql_query`, are required, described `Your email password.` and
    `Your MySQL password.`, and declared under `credentials`.
- **API-Bank** ([Li et al., 2023](https://aclanthology.org/2023.emnlp-main.187/)):
  100 of the 101 APIs listed in
  [`api-bank/data/all_apis.csv`](https://github.com/AlibabaResearch/DAMO-ConvAI/blob/faf754671f5b850e1b4d1ff870db37a6a2480df6/api-bank/data/all_apis.csv)
  (commit `faf7546`), all except `ImageCaption`. Tool names are the upstream
  class names in lower case (`AddAlarm` becomes `addalarm`); parameter types
  are written as JSON Schema types (`str` as `string`, `int` as `integer`,
  `datetime` as `string`, `List[str]` as `array`); and output parameters are
  removed. A tool requires the inputs that its signature in the `expressions`
  column takes without a default, apart from inputs that upstream text marks
  optional or that the class's `call` method does not take, and eight tools
  gain the `token` input that their signature takes and their `api_info`
  omits. Descriptions and parameter descriptions are upstream's, apart from 15
  corrected against the class's code and the CSV, such as `gettoday`, which
  gets the current date. The `All` toolset holds every API, and nine topical
  toolsets repeat subsets of them. The 14 parameters that take the password of
  an existing account, `password` in `getusertoken` and `modifybankaccount` and
  `old_password` in `modifypassword`, are described `Your account password.` or
  `Your bank account password.` and declared under `credentials`; the 14 that
  set one, `new_password` in `modifypassword` and `password` in `registeruser`
  and `openbankaccount`, are declared under `sets_credentials`.
- **RestGPT** ([Song et al., 2023](https://arxiv.org/abs/2306.06624)):
  all 40 Spotify and 54 TMDB operations in RestBench's OpenAPI specs under
  [`specs`](https://github.com/Yifan-Song793/RestGPT/tree/d99012860cb45323975bf5780152d7aa44f390f5/specs)
  (commit `d990128`), one tool per operation, named after its HTTP method and
  path. A tool's parameters are the operation's own, those declared on its
  path, and the fields of its JSON request body; a body field that repeats a
  query parameter appears once, as the query parameter. A parameter is
  required when upstream marks it so, and the Spotify spec writes that mark as
  the string `"true"` or `"false"`. Three tools differ: `uris` in
  `post_playlists_playlist_id_tracks` and each track's `uri` in
  `delete_playlists_playlist_id_tracks` are required, and `get_recommendations`
  requires none of its three seeds, since it needs at least one, as its
  description says. Parameter schemas keep their type, `enum` and `items`, and
  each `default` has its parameter's type (21 Spotify `limit` and `offset`
  defaults that upstream writes as strings).
  Descriptions are joined onto one line; each tool
  description ends with the operation's method and URL, and each parameter
  description ends with its location (for example `Location: query` or
  `Location: body`). One operation and 16 parameters that have no upstream
  description were given a short one: four query parameters and the 12 TMDB
  path parameters. In TMDB's `get_discover_tv`, the values that `with_status`
  and `with_type` list for a string parameter are written as strings, and
  `with_release_type`, which takes several release types, is a string. 13
  parameters that an operation's description promises and its spec omits are
  added, typed and worded as elsewhere in the same spec: `include_image_language`
  on the TMDB TV, season and episode images tools, `timezone` on
  `get_tv_airing_today`, `append_to_response` on four details tools, and
  `language` on the person and images tools. Eight descriptions that name the
  wrong resource, such as the TV filters of `get_discover_tv`, are rewritten.
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
  where it is sent, `Location: query` or `Location: path`. Upstream's `in`
  field marks a needed parameter `query` or `required` and an optional one
  `optional`: a parameter is required when it fills the endpoint's URL or, in
  the `music`, `game` and `weather` files, when its `in` marks it needed (the
  `food` file marks every parameter `query`). Parameter types are written in
  lower case, `float` as `number` and an empty type as `string`, apart from
  four numeric parameters that are integers, and each default has its
  parameter's type. Parameter descriptions that upstream copies from another
  parameter, such as a menu item's `item_id` described as a page number, are
  written from the endpoint. The sample responses are removed. One `game`
  endpoint, `get_game_reviewed_today`, has its description in the upstream
  `url` field; its tool description is that text followed by
  `HTTP GET https://opencritic-api.p.rapidapi.com/game/reviewed-today`, the
  path the API's other game lists follow.

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
