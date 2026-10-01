# Engine

The engine runs conversations. It owns the turn loop, the probe base classes,
the content banks, and the identity scheme that makes a run reproducible and
resumable.

## One conversation, turn by turn

`core/simulation.py` drives a single conversation to completion. Each turn
generates a user message, checks it, calls the assistant, and assesses the
reply, with the probe deciding what each of those steps means.

The check between generation and acceptance is the part that earns its keep.
A user turn that breaks character, parrots the assistant back, or drifts out
of the expected language is regenerated rather than kept. Without it, a long
conversation degrades into two models agreeing with each other, and the
resulting trajectory scores well while measuring nothing.

## Identity you can rely on

Two derived identifiers, both in `core/identity.py`, make runs comparable:

**`persona_uuid`** is a content hash of the persona. The same person is the
same id in every run, forever.

**`trajectory_id`** hashes the persona together with the probe, the resolved
model identities, and the prompt version.

The probe component is the variant the *dispatcher seeds* before the probe
runs: the probe module's first `PROBE_VARIANTS` entry, or an explicit
`probe_variant` column. Seven probes discover a finer-grained variant only
once they have picked a task (a fact category, a pressure strategy, a
sanctioned action). That discovered value lands on the finished row for
stratified reporting, and deliberately does **not** feed the id: identity has
to be knowable before the episode runs, or resume and deduplication cannot
skip work they have already done. `tests/engine/core/test_episode_input.py`
pins the id of every registered probe against this rule.

That combination buys three things at once. A run is **resumable**, because
`simulate` skips any `trajectory_id` already present. A run is
**reproducible**, because changing a model or a prompt produces a different
id rather than silently overwriting. And two assistants are **comparable**,
because their trajectory ids differ while still joining cleanly on
`persona_uuid`, which is what makes matched-pair statistics possible across
model versions.

## Episode construction, in one place

`core/episode_input.py` resolves one episode's inputs (persona-derived
settings, probe metadata, provenance, and trajectory identity) without
calling a model. `ConversationSimulatorGenerator` and the externally hosted
`core/episode_runtime.py` both consume it, so a hosted run and a standalone
run start from the same resolved row rather than two reconstructions that
drift apart. `docs/engine/EXTERNAL_PROBE_RUNTIME.md` covers the host contract.

## Every row explains itself

`core/outcomes.py` writes a structured record for each conversation: the
status, a failure classification, which actor was responsible, per-model
token counts, wall-clock time, and provenance. `core/manifest.py` writes the
run-level companion, pinning the code revision, the asset bank versions, and
the prompt versions.

This is what makes a number auditable months later. A capability score that
cannot say which bank version and which prompt produced it is a number you
have to take on faith.

## Content banks

Probe content (fact banks, query banks, pressure banks, clinical profiles,
finance corpora, tool catalogues) ships with the package and resolves along
a search path, so you can layer your own content over the shipped set without
replacing it. See [assets.md](assets.md).

## Storage that survives interruption

Trajectories land in a Hive-partitioned parquet dataset:

```
output/trajectories/run=<id>/locale=<X>/probe_family=<Y>/
```

Each locale's batch is written as its own set of files. Interrupt a
multi-locale run and you keep complete partitions for the locales that
finished, rather than a dataset you have to throw away.

## One path to a model

`core/llm.py` is the only place the engine talks to a model, and it goes
through Data Designer's model facade rather than calling providers directly.
Provider configuration therefore lives in one file you can point elsewhere,
and nothing in the engine needs to know which vendor is behind an alias.

Debug transcript logging is available through `USERSIM_DEBUG_LOG` and off
unless you ask for it: transcripts contain full conversation content, which
is not something to write to disk by default.
