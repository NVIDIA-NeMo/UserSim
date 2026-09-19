# Selection

Curates trajectories into an exportable dataset. Reads the trajectory and
evaluation parquets, applies a profile, and writes a partitioned subset,
optionally pushed to the Hugging Face Hub with a generated dataset card.

No LLM calls.

## Profiles

`selection/profiles.py` declares what "good" means for an export: which
capabilities must have evidence, which thresholds must be met, and how much
evidence is enough. Profiles reference capability definitions from
[`usersim.taxonomy`](overview.md#taxonomy), so a profile cannot silently
drift from what the report measures.

## The engine

`selection/engine.py` decides row by row. Each decision records *why*, which
is the part that matters: an export whose selection criteria are not
inspectable is not reproducible.

Two behaviours to be aware of when reading results:

- A row whose required scorer errored is recorded with a soft reason and can
  still pass. Treat required-evidence absence as a reason to look, not as a
  pass.
- An evaluation cache hit whose envelope matches is recorded as skipped, and
  profiles that set `require_eval_present` drop those rows rather than counting
  them as evidence.

## Export

`selection/io.py` writes the partitioned dataset and, for Hub uploads,
generates a dataset card. `conversation_messages` is retained, so any stored
`reasoning_content` travels with the export; use `--no-store-reasoning` at
simulation time for runs whose traces should not leave your environment, and
review an export before publishing it as you would any other model internals.
