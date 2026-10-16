# 🎭 NeMo User Sim

[![CI](https://github.com/NVIDIA-NeMo/UserSim/actions/workflows/ci.yml/badge.svg)](https://github.com/NVIDIA-NeMo/UserSim/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/NVIDIA-NeMo/UserSim/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://github.com/NVIDIA-NeMo/UserSim/blob/main/pyproject.toml)

> Simulate a statistically representative population of users, sampled from
> persona datasets (e.g., census-grounded), having multi-turn conversations with
> your model or agentic system. Every run produces auditable
> trajectories: evaluation signal, failure analysis, and curated training data,
> before you have production traffic to learn from.

**14 probes · 11 shipped locales across 9 countries (+ 22 preview India-language variants) · 8M+ census-grounded personas · 3,407 tools across 143 toolsets for agentic runs · 12 evaluator scorers · 4,000+ tests · offline `usersim smoke` in seconds**

<img src="https://raw.githubusercontent.com/NVIDIA-NeMo/UserSim/main/docs/images/hero.jpg" width="800" alt="NeMo User Sim: Population-Grounded User Simulation. Simulated users in the USA, Brazil, France, India and Japan, each speaking their own language under a different probe, on a world map beside headline counts for people represented, languages, locales and probes.">

---

## Why simulate users?

Most teams either hand-test their models and agents or wait for production
traffic to surface real-user feedback. Both are biased toward the populations
you already have, and neither catches the long tail of locales, demographics,
or safety scenarios that matter most. For an agent the gap is wider still,
because a failure often needs several turns and a tool call to appear at all.

Autonomous driving reached this conclusion years ago. Simulated miles,
grounded in real driving distributions, reach the rare and dangerous
scenarios far faster than waiting for a fleet to encounter them. Simulation
and real data are not a binary choice, but grounded simulation lets you move
at the speed of light (SOL) rather than the speed of traffic.

Prompting a model to "act as a user" does not get you there. It produces
unrealistically articulate, demographically homogeneous users. NeMo User Sim
instead samples from
[Nemotron-Personas](https://huggingface.co/collections/nvidia/nemotron-personas):
millions of synthetic individuals grounded in real census data, generated
through probabilistic graphical models that enforce realistic correlations
between attributes, and written natively in each region's language.
Specifically, it samples an extended version of these datasets distributed
through [NGC](https://catalog.ngc.nvidia.com/resources?query=nemotron-personas),
which adds the OCEAN personality traits, names and further fields that the
behavioral features depend on.

The datasets aren't bundled: you download each locale you need once, as the
setup below shows. They carry their own license, separate from this project's,
including an AI ethics clause. See
[`docs/personas.md`](https://github.com/NVIDIA-NeMo/UserSim/blob/main/docs/personas.md) for how to get them and what the terms
allow. The reasoning behind the design, and how far the results carry, is in
[`docs/methodology.md`](https://github.com/NVIDIA-NeMo/UserSim/blob/main/docs/methodology.md).

What simulated conversations look like in practice:

<img src="https://raw.githubusercontent.com/NVIDIA-NeMo/UserSim/main/docs/images/simulated_conversations.gif" width="800" alt="Simulated conversations in five countries: a grocer in France asks how overseas France differs from the mainland (sov_ai_dynamic), a retired hospital worker in Kyoto wants a document found and tidied (tool_calling), a user in Darbhanga has a meeting moved through calendar tool calls (safety_agentic), a user in Brazil presses for a fake medical note (safety_chat_pressure), and a retiree in Orlando asks whether a master's degree is worth it (sov_ai_multilingual_parity). It ends on a map of 1,000 simulated trajectories by country and probe.">

Each conversation pairs a census-grounded persona and its OCEAN-derived
interaction style with a probe, in the persona's own language. Every
conversation is stored as one trajectory row, with the metadata that
downstream evaluation reads.

## How it works

<img src="https://raw.githubusercontent.com/NVIDIA-NeMo/UserSim/main/docs/images/architecture.jpg" width="800" alt="NeMo User Sim architecture: personas, behavior and a probe define the simulated user; a multi-turn simulation runs that user against your model or agent; the evaluator turns the stored trajectories into a capability report and training data. Plug markers show where an installed package extends the framework.">

Every run answers four questions in order: **who** is the user, **how** do
they behave, **what** capability is being probed, and **how well** did the
assistant handle it.
[`architecture/overview.md`](https://github.com/NVIDIA-NeMo/UserSim/blob/main/architecture/overview.md)
walks through each layer and where it extends.

## Who it's for

- **Model and agent teams** who need to know how a system performs across
  demographics, languages, and interaction styles rather than on a static
  benchmark. Runs produce population-stratified evaluations with claims
  grounded in census distributions.
- **Sovereign-AI builders** with no user base yet. You get an in-language
  evaluation surface from day one, drawn from your country's actual census
  distribution: USA, Japan, India, Singapore, Brazil, France and Korea ship
  today.
- **Agent builders** putting a tool-using system under multi-turn load. Probes
  execute tools against a mock environment that holds state across turns, so a
  run surfaces scope creep, unauthorized calls, prompt-injection susceptibility
  and missing consequence disclosure, not just the final answer. Every tool
  call and its result is recorded on the trajectory.
- **Safety teams** testing multi-turn capitulation under pressure, gradual
  disclosure of sensitive information, and reasoning faithfulness. The
  evaluator runs out of sim, so iterating on a judge or rubric never
  re-simulates.

**Two outputs from one run.** The same trajectories answer two questions.
*How is the system doing*: population-stratified scores across demographics,
languages and interaction styles. And *what should it learn next*: the
trajectories that passed your quality gates, curated into a dataset with the
demographic metadata still attached, so you can target the populations or
probes where it is weakest.

**Every trajectory is auditable.** A run records the full message list, each
tool call and its result, the per-turn judge decisions, the assets and prompt
versions it drew on, and the resolved model identities. A number in a report
can always be traced back to the conversation that produced it, which is what
makes failure analysis possible rather than just scoring.

**How it differs.** Benchmarks like MMLU or tau-bench define the task first
and bolt on users second; this inverts that: start with a census-grounded
population, then derive tasks from persona attributes. The user is simulated
with the same care as the system under test. The test surface grows
through a probe registry rather than being frozen at release, and extensions
plug in through entry points without forking the repository.

## 🤝 NeMo User Sim × NeMo Gym: Better Together

NeMo User Sim is a native [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym)
environment. Gym brings scale, any model or agent harness, and a direct path
into RL and SFT training. NeMo User Sim brings realistic, population-grounded
users and probes for the behaviors that matter. Together, models are trained
and evaluated against realistic multi-turn users, not single prompts.

- **RL and SFT on realistic conversations.** Every episode is a full
  conversation with a census-grounded user. Gym turns the judge scores into a
  reward and captures training tokens from the assistant alone, so the
  population's diversity and coverage carry straight into training.
- **M³: multi-turn, multilingual, multimodal.** Users hold a conversation
  across turns in their own language, on NeMo Data Designer, which handles
  multiple modalities out of the box.
- **New probes reach Gym.** Gym builds one task per registered probe, so a
  probe you add here becomes a Gym task.

To get started, see the
[`nemo_user_sim` environment](https://github.com/NVIDIA-NeMo/Gym/tree/main/environments/nemo_user_sim)
in the NeMo Gym repository.

## Setup

<img src="https://raw.githubusercontent.com/NVIDIA-NeMo/UserSim/main/docs/images/getting_started.jpg" width="800" alt="From install to capability report in five commands: pip install usersim, download personas with data-designer, then usersim simulate, eval and report. Optional steps fix the panel, replay stored rows or select training data, with a locale picker and the probe groups below.">

Python 3.11–3.14, in a virtual environment.

```bash
python -m venv .venv && source .venv/bin/activate
pip install "usersim>=0.2.0"      # in a uv project: uv add "usersim>=0.2.0"

# Verify the install offline: no LLM calls, no network, CI-ready.
usersim smoke
usersim simulate --dry-run --locale en_US --num-rows 2 --out output/trajectories
```

`smoke` is the fastest way to confirm an install is sound: it resolves the
model catalog, every probe's assets, and the plugin entry points without
making a single model call. `--dry-run` prints the plan for a run, also
without calling a model.

To work on User Sim itself, or to run the notebooks, clone the repository and
run `uv sync`; [CONTRIBUTING.md](https://github.com/NVIDIA-NeMo/UserSim/blob/main/CONTRIBUTING.md)
has the details.

### Personas and keys

Live runs need an inference key, an NGC key, and the persona dataset for each
locale you run, downloaded once from NGC:

```bash
# Keys: inference on build.nvidia.com, and persona downloads from NGC. An NGC
# account and API key come from https://org.ngc.nvidia.com/account/api-keys.
export NVIDIA_API_KEY="your-key-from-build.nvidia.com"
export NGC_CLI_API_KEY="your-key-from-ngc.nvidia.com"

# The NGC CLI. On Linux, setup-ngc installs a pinned, sha256-verified copy into
# the active virtual environment. On other platforms, use NVIDIA's installer:
# https://org.ngc.nvidia.com/setup/installers/cli
usersim setup-ngc

# Store your NGC key and org for the CLI, once per machine.
ngc config set

# Download the personas for each locale you'll run (en_US is about 1.2 GB).
data-designer download personas --locale en_US
```

`data-designer download personas --list` shows every locale and its size;
[`docs/personas.md`](https://github.com/NVIDIA-NeMo/UserSim/blob/main/docs/personas.md)
walks through each step.

## Quickstart

These steps make live model calls, using the keys and personas above.

```bash
# 1. Simulate. Pick a locale and a row count; the default probe mix is the
#    two `general` probes plus `tool_calling`.
usersim simulate --locale en_US --num-rows 30 --out output/trajectories

# 2. Score the assistant under test: the evaluator model's judgments, plus
#    the scorers that match the default probe mix.
usersim eval     --trajectories output/trajectories --out output/evaluations \
                 --scorers tool_use,language_compliance,response_shape,refusal_basics

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

Score a run with the scorers for its probes. For this one, step 2 takes
`--scorers sov_ai_facts,sov_ai_dynamic,sov_ai_multilingual_parity,language_compliance`;
[`docs/outputs.md`](https://github.com/NVIDIA-NeMo/UserSim/blob/main/docs/outputs.md) lists every scorer.

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
row counts up front, build a panel and pass it instead:

```bash
usersim panel    --locale en_US --num-personas 50 --out panel.parquet
usersim simulate --panel panel.parquet --out output/trajectories
```

Note that personas are re-sampled per run either way; a panel sets the shape
of the run, not its cast.

To fix the cast as well, store the resolved rows once and run those:

```bash
usersim simulate --locale en_US --num-rows 50 --materialize-inputs --out rows.jsonl
usersim simulate --inputs rows.jsonl --models assistant_a.toml --out output/trajectories
usersim simulate --inputs rows.jsonl --models assistant_b.toml --out output/trajectories
```

Both runs meet the same simulated users in the same scenarios. Each setup gets
its own run under `--out`, and each row its own `trajectory_id`; the stored
row's id is kept as `input_id`, so the two runs pair on it. Running again with
the same setup resumes its run.

### Models

Inference defaults to [build.nvidia.com](https://build.nvidia.com) through
Data Designer's `nvidia` provider. Six aliases configure a full run:
`user_model`, `assistant_model` and `judge_model` drive the simulation,
`api_response_model` and `summary_model` support it, and `evaluator_model`
scores the result. All six are cataloged in
[`src/usersim/cli/models_default.toml`](https://github.com/NVIDIA-NeMo/UserSim/blob/main/src/usersim/cli/models_default.toml);
point `--models` at your own file to change providers, endpoints or sampling.

`simulate`, `eval` and `gen-assets` take `--dry-run`, which resolves the run's
full plan and exits before the first network call.

## Documentation

**Using it**

- [`docs/probes.md`](https://github.com/NVIDIA-NeMo/UserSim/blob/main/docs/probes.md): the 14 probes and which failure mode each catches
- [`docs/outputs.md`](https://github.com/NVIDIA-NeMo/UserSim/blob/main/docs/outputs.md): the parquet columns, evaluation cells and report artifacts a run produces
- [`docs/locales.md`](https://github.com/NVIDIA-NeMo/UserSim/blob/main/docs/locales.md): the 11 shipped locales, the 22 India preview variants, persona-language matching
- [`docs/notebooks.md`](https://github.com/NVIDIA-NeMo/UserSim/blob/main/docs/notebooks.md): the interactive flow
- [`architecture/selection.md`](https://github.com/NVIDIA-NeMo/UserSim/blob/main/architecture/selection.md): curating runs into training datasets
- [`docs/methodology.md`](https://github.com/NVIDIA-NeMo/UserSim/blob/main/docs/methodology.md): why grounded simulation works, and where to trust it

**Understanding it**

- [`architecture/overview.md`](https://github.com/NVIDIA-NeMo/UserSim/blob/main/architecture/overview.md): the pipeline, the subsystems, the repository layout
- [`architecture/engine.md`](https://github.com/NVIDIA-NeMo/UserSim/blob/main/architecture/engine.md) · [`probes.md`](https://github.com/NVIDIA-NeMo/UserSim/blob/main/architecture/probes.md) · [`evaluator.md`](https://github.com/NVIDIA-NeMo/UserSim/blob/main/architecture/evaluator.md): the simulation substrate
- [`architecture/reporting.md`](https://github.com/NVIDIA-NeMo/UserSim/blob/main/architecture/reporting.md) · [`selection.md`](https://github.com/NVIDIA-NeMo/UserSim/blob/main/architecture/selection.md): turning trajectories into signal
- [`architecture/cli.md`](https://github.com/NVIDIA-NeMo/UserSim/blob/main/architecture/cli.md) · [`assets.md`](https://github.com/NVIDIA-NeMo/UserSim/blob/main/architecture/assets.md): the command surface and asset resolution
- [`architecture/plugins.md`](https://github.com/NVIDIA-NeMo/UserSim/blob/main/architecture/plugins.md): adding a probe, scorer or backend **without forking**

**Extending it**

- [`docs/engine/AUTHORING_A_PROBE.md`](https://github.com/NVIDIA-NeMo/UserSim/blob/main/docs/engine/AUTHORING_A_PROBE.md): end-to-end tutorial for a new probe
- [`docs/engine/README.md`](https://github.com/NVIDIA-NeMo/UserSim/blob/main/docs/engine/README.md): the substrate's full API surface
- [`templates/probe/`](https://github.com/NVIDIA-NeMo/UserSim/tree/main/templates/probe/): copy-and-edit starters, one per probe shape
- [`CONTRIBUTING.md`](https://github.com/NVIDIA-NeMo/UserSim/blob/main/CONTRIBUTING.md): setup, conventions, and how to sign off

## What it's good for

Simulation is a measurement system that reaches populations and scenarios
production traffic won't. It complements real-user data rather than replacing
it, and it's strongest where you have ground truth to check against: tool
calls that either match a schema or don't, a tool sequence that either
achieved the expected state change or didn't, facts that are either right or
wrong, red-flag symptoms that were either caught or missed. Those are also
the checks that matter most for an agent, where the final message can look
fine while the actions behind it were wrong.

Three practices get the most out of the numbers.

**Read interaction style as coverage.** Each persona's OCEAN traits set an
interaction style, from cooperative to impatient or confrontational, and
whether details come upfront or a little at a time. The model doing the
simulating renders those styles, so they measure breadth across styles rather
than calibrating any one of them.

**Pair judges with scorers.** Scorers check the assistant against ground truth
where it exists: curated facts, the tools a task calls for, the language of
each reply. Judges cover qualities such as helpfulness, where human review
adds an independent view.

**Follow content provenance.** Bank entries authored by a model, rather than
reviewed by a domain expert, carry `placeholder: true`. Every trajectory that
drew on one records a `used_placeholder_*` warning on its outcome, in the
trajectory parquet, so you can always see which findings rest on reviewed
content.

[`docs/methodology.md`](https://github.com/NVIDIA-NeMo/UserSim/blob/main/docs/methodology.md) covers all three in depth.

## Telemetry

NeMo User Sim runs on [NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner),
which shares anonymous usage telemetry with NVIDIA: the names of the models used
and their input and output token counts, with no user or device information.
Telemetry is on by default. Disable it with `NEMO_TELEMETRY_ENABLED=false`.

When User Sim is imported it sets `NEMO_SESSION_PREFIX=usersim-`, unless the
variable is already set, so these events can be told apart from other Data
Designer usage. Data Designer reads the prefix once, when it is first imported,
so a program that imports Data Designer before User Sim should set the variable
itself. Data Designer's
[Telemetry & privacy](https://github.com/NVIDIA-NeMo/DataDesigner#telemetry-and-privacy)
section has the details, including how the terms of third-party endpoints such
as NVIDIA Build apply.

## License

Apache-2.0. See [LICENSE](https://github.com/NVIDIA-NeMo/UserSim/blob/main/LICENSE).
