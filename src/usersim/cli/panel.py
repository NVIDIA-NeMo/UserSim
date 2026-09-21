# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""``usersim panel`` — build a frozen persona seed parquet.

Why a panel? The simulator's identity scheme content-hashes the persona
(``persona_uuid``) so two runs that draw the same persona always
produce the same id. Persisting the panel separately from the simulator
run gives you:

- **Replayability** — re-run the simulator against the same panel after
  fixing a bug or swapping the assistant model; every ``trajectory_id``
  remains stable up to the bits that intentionally changed.
- **Matched-pair evaluation** — feed the same panel to two different
  ``assistant_model`` configurations and join their trajectories on
  ``persona_uuid`` for paired statistical tests.
- **Audit** — stash the panel parquet alongside a model release as
  evidence of the population the model was evaluated against.

The panel is a thin parquet with one row per persona: ``persona``
(JSON dict) plus the seven derived ``persona_*`` expression columns
the simulator already uses, plus ``locale``. No model calls are made;
this is a sampling-only step.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger("usersim.cli.panel")


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "panel",
        help="👥 Build a frozen persona seed parquet (no LLM calls).",
        description=(
            "Sample N personas per locale from Nemotron-Personas and "
            "write a parquet that can be re-fed to `usersim simulate` to "
            "fix the locales and per-locale row counts of a run."
        ),
    )
    p.add_argument(
        "--locale",
        action="append",
        required=True,
        help=(
            "Locale to sample from (repeatable). Examples: en_US, pt_BR, ja_JP, fr_FR, hi_Deva_IN, en_IN, en_SG, ko_KR."
        ),
    )
    p.add_argument(
        "--num-personas",
        type=int,
        required=True,
        help="Number of personas to draw per locale.",
    )
    p.add_argument("--out", type=Path, required=True, help="Output parquet path.")
    p.add_argument(
        "--models",
        type=Path,
        default=None,
        help=(
            "Path to models TOML. Defaults to the bundled "
            "cli/models_default.toml (only providers section is consulted "
            "for sampling; no LLM is invoked)."
        ),
    )
    p.add_argument(
        "--match-persona-language",
        action="store_true",
        help=(
            "Sample only personas whose first language is the conversation "
            "language. Off by default. Note it shifts the demographic makeup "
            "of the sampled population, not just its language, so two "
            "language-matched locales differ by more than language and a "
            "difference measured between them may not be linguistic. "
            "Matches `usersim simulate`: a panel and a run given the same "
            "locale and flags must draw from the same population. For en_IN, "
            "omit this flag to build an English-language panel across the "
            "full India persona corpus; enabling it selects only records "
            "whose current first_language field is recorded as English."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved panel plan and exit without writing.",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    from usersim.cli._models import (
        default_models_path,
        load_models_config,
        to_data_designer_kwargs,
        to_model_configs,
    )
    from usersim.cli._persona_language import (
        DEFAULT_PERSONA_DATASETS_DIR,
        log_persona_language_population,
        persona_language_filter,
        persona_sampler_params,
    )
    from usersim.cli._pipeline import _add_persona_expressions

    models_path = args.models or default_models_path()
    models = load_models_config(models_path)

    if args.dry_run:
        print("usersim panel — dry run")
        print(f"  locales:       {args.locale}")
        print(f"  per-locale:    {args.num_personas}")
        print(f"  total rows:    {len(args.locale) * args.num_personas}")
        print(f"  match_persona_language: {args.match_persona_language}")
        print(f"  models config: {models_path}")
        print(f"  output:        {args.out}")
        return 0

    import data_designer.config as dd
    import pandas as pd
    from data_designer.interface import DataDesigner

    from usersim.engine.core.locale import persona_dataset_locale

    personas_dir = DEFAULT_PERSONA_DATASETS_DIR

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for locale in args.locale:
        # India language-variants reuse an existing India persona dataset:
        # sample from the resolved dataset locale (en_IN today) but tag the
        # panel row with the CONVERSATION locale so downstream simulate/
        # partitioning use the requested language. Non-variant unchanged.
        persona_locale = persona_dataset_locale(locale, datasets_dir=personas_dir)
        language_constraint = persona_language_filter(
            persona_locale,
            locale,
            match_language=args.match_persona_language,
        )
        if language_constraint:
            log_persona_language_population(
                persona_locale=persona_locale,
                conversation_locale=locale,
                constraint=language_constraint,
                datasets_dir=personas_dir,
            )
        logger.info(
            "sampling %d personas for %s (dataset=%s)",
            args.num_personas,
            locale,
            persona_locale,
        )
        dd_kwargs = to_data_designer_kwargs(models)
        data_designer = DataDesigner(**dd_kwargs)
        builder = dd.DataDesignerConfigBuilder(model_configs=to_model_configs(models))
        builder.add_column(
            dd.SamplerColumnConfig(
                name="persona",
                drop=False,
                sampler_type=dd.SamplerType.PERSON,
                # Shared with the simulator rather than rebuilt here: two
                # builders that construct sampler params independently drift,
                # and a panel drawn from a different population than the run
                # it seeds defeats the point of freezing one.
                params=persona_sampler_params(
                    dd,
                    persona_locale,
                    locale,
                    match_language=args.match_persona_language,
                ),
            )
        )
        _add_persona_expressions(builder, dd)
        result = data_designer.create(builder, num_records=args.num_personas)
        df = result.load_dataset()
        df["locale"] = locale
        frames.append(df)

    panel_df = pd.concat(frames, ignore_index=True)
    panel_df.to_parquet(out_path, index=False)
    print(f"wrote {len(panel_df)} personas to {out_path}")
    return 0
