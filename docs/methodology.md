# Methodology

The case for simulating a grounded population, how behavioural realism is
built, and how far the results carry. Worth reading once before you quote a
number from a run.

## Why this problem is hard

Building and shipping AI models and agentic solutions requires understanding how
real users will interact with them: across languages, cultures, skill levels,
personality types, and an open-ended space of use cases. Today, teams face three
compounding challenges:

**The cold-start problem.** You need realistic user traffic to evaluate and
improve your model, but you don't have traffic until the model is live.
Collecting real-user data is slow, expensive, privacy-sensitive, and inherently
biased toward whichever early-adopter population shows up first.

**The coverage problem.** Even *with* production traffic, real-world data is
structurally incomplete. It over-represents your existing user base and
under-represents everyone else: speakers of languages you haven't launched in
yet, users with low digital literacy, adversarial or safety-probing
interactions, rare but high-stakes agentic workflows, and the long tail of
scenarios that matter but occur infrequently. No amount of patience fully solves
this: the distribution of real users is not the distribution you need to test
against.

**The multilingual and sovereign AI challenge.** These problems are dramatically
harder for teams building models outside the English-dominated ecosystem.
Sovereign AI initiatives (national efforts to build language models that serve
local populations in their own languages) face all of the above with fewer
resources: smaller corpora, sparser benchmarks, and often zero production
traffic in the target language at launch. A Japanese model builder needs to know
how a 70-year-old farmer in Hokkaido will interact differently from a 22-year-old
student in Osaka, in Japanese, with culturally appropriate expectations. A
Brazilian team needs to validate Portuguese performance across São Paulo
professionals and rural Minas Gerais retirees. English-centric evaluation and
training pipelines cannot help here. Multilingual, culturally grounded user
simulation is not a nice-to-have for sovereign AI; it is a prerequisite.

Today's workarounds address none of these well:

- **Vibe-testing**: small manual prompt sets that cover a fraction of the
  interaction surface and reflect the biases of whoever wrote them.
- **"LLM as fake user" prompting**: produces unrealistically articulate,
  cooperative, homogeneous simulated users that fail to stress-test real-world
  failure modes.
- **Waiting for production traffic**: delays feedback loops by weeks or months,
  cannot test rare or safety-critical scenarios proactively, and never delivers
  the demographic or linguistic breadth you actually need.
- **Static benchmarks**: frozen task sets that don't adapt to new capabilities,
  new locales, or new interaction paradigms like multi-step agentic tool use.

## The approach

NeMo UserSim takes a different path: **simulate a statistically representative
population of users interacting with your LLM**, producing structured evaluation
signals and training data automatically, with controllable coverage over
languages, demographics, interaction styles, and probe complexity.

The parallel to autonomous driving is instructive. Waymo leaned heavily on
simulation early on (long before they had a large fleet), running billions of
simulated miles grounded in real driving distributions to test rare scenarios
and iterate faster than road data alone would allow. Crucially, simulation
remained valuable *after* they had real road data: it provides controllable
coverage of corner cases, weather conditions, and traffic patterns that no
amount of driving will encounter frequently enough. Simulation and real-world
data are not a binary choice; both matter.

NeMo UserSim brings this discipline to LLMs by combining
[Nemotron-Personas](https://huggingface.co/collections/nvidia/nemotron-personas)
(census-grounded synthetic personas; the extended NGC version carries the
OCEAN personality traits this depends on) with a **general-purpose probe engine** that can simulate any
interaction pattern: open-ended conversation, tutoring, multi-step tool use,
agentic workflows, safety probing, and more.

## Behavioural realism

NeMo UserSim implements four behavioural-realism features inspired by
[Zhou et al. (2026)](https://arxiv.org/abs/2603.11245). Each is measurable in
the output, and the D1-D4 diagnostics below quantify how far they move a run
away from LLM-as-fake-user homogeneity.

- **Incremental disclosure**: 60% of users withhold details and reveal gradually (real-user-like); 40% provide everything upfront. Controlled by `incremental_disclosure_ratio` (default 0.6).
- **OCEAN-grounded interaction style**: each user receives one of `cooperative` / `neutral` / `impatient` / `frustrated` / `confrontational` derived from their persona's neuroticism / agreeableness / patience scores.
- **Reactive frustration**: when the assistant is judged inadequate, the user agent receives escalating frustration prompts (pushing back, repeating requests, threatening to leave).
- **D1–D4 + MATTR diagnostics**: post-hoc behavioral-realism metrics (front-loading ratio, politeness fraction, verbosity CV, frustration-marker frequency, lexical diversity) surfaced in the dashboard.

Scope:

- **Not a substitute for real-user studies.** Sim2Real gaps remain; simulation is most valuable as an orthogonal measurement system that complements production data, not a replacement for it.
- **Behavioural fidelity is bounded by the simulator-LLM.** Current LLMs flatten OCEAN-derived behavioural differences relative to what the persona descriptions warrant. The D1-D4 diagnostics exist to make that flattening visible in a given run.
- **Review status is tracked per entry.** Bank entries carry a `placeholder` flag, set until a subject-matter expert signs off. Any trajectory drawing on an unreviewed entry records a `used_placeholder_*` warning on its outcome, so a result's provenance is always recoverable from the run.
- **Evaluator circularity for subjective dimensions.** LLM judges evaluating LLM-simulated users interacting with LLMs is structurally circular for subjective quality dimensions; reserve human judgment for those, and prefer verifier-based scoring (code execution, formal verification, step-level checkers) wherever ground truth is available.
