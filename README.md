# NeMo UserSim: Population-Grounded User Simulation

> Simulate a statistically representative population of users having multi-turn
> conversations with your LLM. Every run yields both evaluation signal and
> curated training data, before you have a real user base to learn from.

**13 probes · 9 shipped locales across 7 countries (+ 22 preview India-language variants) · ~7M+ census-grounded personas · 11 evaluator scorers · 3,100+ tests · offline `usersim smoke` in seconds**

<img src="docs/images/pipeline_architecture.jpg" width="800">

---

## Why simulate users?

Most teams either vibe-test their models or wait for production traffic to
surface real-user feedback. Both are biased toward the populations you
already have, and neither catches the long tail of locales, demographics, or
safety scenarios that matter most.

The parallel to autonomous driving is instructive: Waymo leaned on simulation
long before it had a large fleet, running simulated miles grounded in real
driving distributions to reach rare scenarios faster than road data alone
allowed. Simulation and real data are not a binary choice, but grounded
simulation lets you move first.

Prompting an LLM to "act as a user" does not get you there. It produces
unrealistically articulate, demographically homogeneous users. NeMo UserSim
instead samples from
[Nemotron-Personas](https://huggingface.co/collections/nvidia/nemotron-personas):
millions of synthetic individuals grounded in real census data, generated
through probabilistic graphical models that enforce realistic correlations
between attributes, and written natively in each region's language.

The reasoning behind that design, and where it stops being trustworthy, is in
[`docs/methodology.md`](docs/methodology.md).

What a single simulated trajectory looks like in practice:

![A NeMo UserSim conversation preview: a sov_ai_dynamic probe sending Mark Bastani (a 54-year-old building inspector in Auburn, California, with an impatient interaction style and incremental disclosure) into a 5-turn conversation with the assistant about the U.S. Constitution.](docs/images/conversation_preview.png)

A census-grounded persona, an OCEAN-derived interaction style, a probe-driven
multi-turn conversation, and the trajectory metadata that downstream
evaluation reads off, all from one row of a parquet dataset.

## Who it's for

- **Model evaluators and quality teams** who need to know how an assistant
  performs across demographics, languages, and interaction styles rather than
  on a static benchmark. Runs produce population-stratified evaluations with
  claims grounded in census distributions.
- **Sovereign-AI builders** with no user base yet. You get an in-language
  evaluation surface from day one, drawn from your country's actual census
  distribution: USA, Japan, India, Singapore, Brazil, France and Korea ship
  today.
- **Agent and safety teams** testing multi-turn safety, agentic tool use, and
  reasoning faithfulness in controlled environments. The evaluator runs out of
  sim, so iterating on a judge or rubric never re-simulates.

**Two outputs from one run.** The same conversations answer two questions.
*How is the model doing*: population-stratified scores across demographics,
languages and interaction styles. And *what should it learn next*: the
trajectories that passed your quality gates, curated into a dataset with the
demographic metadata still attached, so you can target the populations or
probes where the model is weakest.

**How it differs.** Benchmarks like MMLU or tau-bench define the task first
and bolt on users second; this inverts that: start with a census-grounded
population, then derive tasks from persona attributes. The test surface grows
through a probe registry rather than being frozen at release, and extensions
plug in through entry points without forking the repository.

## Setup

Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
# Resolve uv.lock and editable-install the project, with the dev tooling
# and the Jupyter kernel the notebooks need.
uv sync
source .venv/bin/activate

# Install the NGC CLI into .venv/bin/ngc (idempotent, sha256-verified).
# Required to download the Nemotron-Personas datasets.
usersim setup-ngc

# Keys for live runs and persona download.
export NVIDIA_API_KEY="your-key-from-build.nvidia.com"   # inference
export NGC_CLI_API_KEY="your-key-from-ngc.nvidia.com"    # persona download

# Verify the install offline: no LLM calls, no network, CI-ready.
usersim smoke
```

`smoke` is the fastest way to confirm an install is sound: it resolves the
model catalogue, every probe's assets, and the plugin entry points without
making a single model call.

## Quickstart

```bash
# 1. Simulate. Pick a locale and a row count; the default probe mix is the
#    three `general` probes.
usersim simulate --locale en_US --num-rows 30 --out output/trajectories

# 2. Score the assistant under test.
usersim eval     --trajectories output/trajectories \
                 --out output/evaluations --judges judge_model

# 3. Render the capability dashboard.
usersim report   --trajectories output/trajectories \
                 --evaluations output/evaluations --out output/report/
```

`report` prints the `index.html` path as a clickable `file://` URL; `--open`
launches it, and `--list-runs` shows which runs are available.

Pick any registered probe with `--probe-mix`:

```bash
usersim simulate --locale ja_JP --num-rows 30 \
  --probe-mix 'sov_ai_facts=1/3,sov_ai_dynamic=1/3,sov_ai_multilingual_parity=1/3' \
  --out output/trajectories
```

### Curate training data

Simulation produces evaluation signal and training data from the same run.
`select` applies quality gates over the evaluator's scores and the
simulator's own health metrics, then writes the passing subset with a funnel
manifest recording why each row survived:

```bash
usersim select --trajectories output/trajectories \
               --evaluations output/evaluations \
               --profile strict --out output/curated
```

Two profiles ship: `generous` and `strict`. Add `--stratify-by` to balance
across locale or demographic cohorts, `--holdout-fraction` to reserve an
evaluation split, and `--hf-repo` to push the result to the Hugging Face Hub
with a generated dataset card.

### Reusing a fixed population

`--locale` samples personas fresh each run. To fix the locales and per-locale
row counts up front (for a matched-pair comparison, say), build a panel and
pass it instead:

```bash
usersim panel    --locale en_US --num-personas 50 --out panel.parquet
usersim simulate --panel panel.parquet --out output/trajectories
```

Note that personas are re-sampled per run either way; a panel sets the shape
of the run, not its cast.

### Models

Inference defaults to [build.nvidia.com](https://build.nvidia.com) through
Data Designer's `nvidia` provider. Five aliases configure a run
(`user_model`, `assistant_model`, `api_response_model`, `judge_model`,
`summary_model`), catalogued in
[`src/usersim/cli/models_default.toml`](src/usersim/cli/models_default.toml);
point `--models` at your own file to change providers, endpoints or sampling.

Every LLM-driven subcommand takes `--dry-run`, which resolves the full plan
and exits before the first network call.

## Documentation

**Using it**

- [`docs/probes.md`](docs/probes.md): the 13 probes and which failure mode each catches
- [`docs/outputs.md`](docs/outputs.md): the parquet columns, evaluation cells and report artifacts a run produces
- [`docs/locales.md`](docs/locales.md): the 9 shipped locales, the 22 India preview variants, persona-language matching
- [`docs/notebooks.md`](docs/notebooks.md): the interactive flow
- [`architecture/selection.md`](architecture/selection.md): curating runs into training datasets
- [`docs/methodology.md`](docs/methodology.md): why grounded simulation works, and where to trust it

**Understanding it**

- [`architecture/overview.md`](architecture/overview.md): the pipeline, the subsystems, the repository layout
- [`architecture/engine.md`](architecture/engine.md) · [`probes.md`](architecture/probes.md) · [`evaluator.md`](architecture/evaluator.md): the simulation substrate
- [`architecture/reporting.md`](architecture/reporting.md) · [`selection.md`](architecture/selection.md): turning trajectories into signal
- [`architecture/cli.md`](architecture/cli.md) · [`assets.md`](architecture/assets.md): the command surface and asset resolution
- [`architecture/plugins.md`](architecture/plugins.md): adding a probe, scorer or backend **without forking**

**Extending it**

- [`docs/engine/AUTHORING_A_PROBE.md`](docs/engine/AUTHORING_A_PROBE.md): end-to-end tutorial for a new probe
- [`docs/engine/README.md`](docs/engine/README.md): the substrate's full API surface
- [`templates/probe/`](templates/probe/): copy-and-edit starters, one per probe shape
- [`CONTRIBUTING.md`](CONTRIBUTING.md): gates, conventions, and how to sign off

## What it's good for

Simulation is a measurement system that reaches populations and scenarios
production traffic won't. It complements real-user data rather than replacing
it, and it's strongest where you have ground truth to check against: tool
calls that either match a schema or don't, facts that are either right or
wrong, red-flag symptoms that were either caught or missed.

Three things are worth knowing before you quote a number.

Behavioural fidelity is bounded by the model doing the simulating. Current
models flatten personality differences relative to what a persona
description implies, so treat interaction-style coverage as breadth rather
than calibrated realism. Closing that gap is the most active area of work.

Judges scoring simulated users is circular for subjective dimensions. A
model's opinion of another model's helpfulness is not an independent
measurement. Lean on the deterministic scorers where ground truth exists, and
reserve human review for taste.

Content review status travels with the numbers. Bank entries authored by a
model rather than reviewed by a domain expert are flagged, and any capability
cell computed from them inherits the flag, so you can always tell which
findings rest on reviewed content.

[`docs/methodology.md`](docs/methodology.md) covers all three in depth.

## License

Apache-2.0. See [LICENSE](LICENSE).
