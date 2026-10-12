# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config for the TrajectoryEvaluator DD column generator.

The evaluator is a separate DD plugin (not the simulator). It reads
trajectories from the trajectory store and applies a judge ensemble +
deterministic scorers to score the *assistant under test*. Configured
via ``TrajectoryEvaluatorConfig`` and dispatched as the column type
``"trajectory-evaluator"``.

See ``src/usersim/engine/evaluator/__init__.py``
for the rationale (in-sim / out-of-sim boundary, plugin entry-point
pattern, SLURM compatibility).
"""

from __future__ import annotations

from typing import Literal

from data_designer.config.base import SingleColumnConfig
from pydantic import BaseModel, Field, model_validator

from usersim.engine.evaluator.judges import (
    EnsembleDiversityError,
    JudgeFamily,
    JudgeSpec,
    validate_ensemble_diversity,
)
from usersim.engine.evaluator.prompts import EVAL_PROMPT_VERSION


class JudgeSpecConfig(BaseModel):
    """Pydantic-serializable judge specification.

    Wraps the dataclass ``JudgeSpec`` with a Pydantic model so it
    survives DD config serialization. ``family`` is a string-form
    ``JudgeFamily`` value; ``None`` means "infer from alias / model id
    at run time."
    """

    alias: str = Field(..., description="DD model alias")
    family: str | None = Field(
        default=None,
        description=(
            "Architectural family for the diversity check. "
            "None -> inferred from alias / model id at run time. "
            "See evaluator/judges.py JudgeFamily for valid values."
        ),
    )

    def to_spec(self) -> JudgeSpec:
        fam: JudgeFamily | None = None
        if self.family is not None:
            try:
                fam = JudgeFamily(self.family)
            except ValueError as e:
                raise ValueError(
                    f"Unknown judge family {self.family!r}. Valid: {[f.value for f in JudgeFamily]}"
                ) from e
        return JudgeSpec(alias=self.alias, family=fam)


class TrajectoryEvaluatorConfig(SingleColumnConfig):
    """Configuration for the TrajectoryEvaluator column generator.

    Reads a trajectory row (conversation messages + persona + probe
    metadata + simulation outcome) and writes a single wide JSON cell
    containing the per-axis × per-judge scores plus any deterministic
    scorer outputs. The reporting layer (``reporting/``)
    handles wide -> long via ``pd.melt`` for inter-judge agreement
    and intersectional aggregation.

    Partial-re-run discipline: if ``skip_if_existing`` is True and the
    row already carries a non-null value in this column whose envelope
    (``judge_aliases`` and the ``judge_models`` behind them, ``axes``,
    ``scorers``, ``prompt_version``) matches the current config, the LLM
    judge calls are skipped. See
    ``evaluator/generator.py`` for the envelope check.
    """

    column_type: Literal["trajectory-evaluator"] = "trajectory-evaluator"

    # Every model-generated column config exposes a single
    # ``model_alias``. The evaluator can run a judge ensemble, so this
    # stays pointed at the first judge and ``get_model_aliases()`` below
    # declares the rest.
    model_alias: str = "judge_model"

    # ── Input columns (defaults match the simulator's side-effect columns) ─
    conversation_messages_column: str = "conversation_messages"
    persona_column: str = "persona"
    probe_family_column: str = "probe_family"
    probe_variant_column: str = "probe_variant"
    simulation_outcome_column: str = "simulation_outcome"
    locale_column: str = "locale"
    conversation_language_column: str = "conversation_language"
    trajectory_id_column: str = "trajectory_id"

    # ── Judge ensemble ──────────────────────────────────────────────
    judges: list[JudgeSpecConfig] = Field(
        ...,
        min_length=1,
        description=(
            "Judge ensemble. ≥2 distinct families enforced for "
            "ensembles of size ≥2 (single-judge mode is allowed but "
            "explicitly opts out of inter-judge agreement reporting)."
        ),
    )

    # ── Axes & scorers ──────────────────────────────────────────────
    axes: list[str] | None = Field(
        default=None,
        description=(
            "Subset of axes (by name) to evaluate. None means all applicable axes for the row's probe_family; "
            "an empty list makes no judge call, so only the scorers run."
        ),
    )
    scorers: list[str] = Field(
        default_factory=list,
        description=(
            "Names of deterministic scorers to dispatch (registered in "
            "evaluator/scorers/). Empty by default; opt in per probe."
        ),
    )

    # ── Partial-re-run envelope ─────────────────────────────────────
    prompt_version: str = Field(
        default=EVAL_PROMPT_VERSION,
        description=(
            "Bumping this invalidates prior eval cells whose envelope "
            "carries a different prompt_version — they will be re-judged "
            "on the next run. The mechanism that prevents partial "
            "re-runs from silently masking semantic prompt shifts."
        ),
    )
    skip_if_existing: bool = Field(
        default=True,
        description=(
            "If True (the default), skip re-judging a row whose existing "
            "cell value matches the current envelope. Set False to "
            "force re-evaluation."
        ),
    )

    # ── Operational knobs ───────────────────────────────────────────
    max_judge_tokens: int = 4096
    max_judge_tokens_non_ascii: int = 8192
    verbosity: int = 1

    # ── Validators ─────────────────────────────────────────────────
    @model_validator(mode="after")
    def _validate_judge_diversity(self) -> "TrajectoryEvaluatorConfig":
        """Fail fast at config-validation time if the ensemble lacks family diversity."""
        if self.judges:
            self.model_alias = self.judges[0].alias
        try:
            validate_ensemble_diversity([j.to_spec() for j in self.judges])
        except EnsembleDiversityError as e:
            # Re-raise as ValueError so Pydantic surfaces it cleanly.
            raise ValueError(str(e)) from e
        return self

    def get_model_aliases(self) -> list[str]:
        """Every judge in the ensemble, so all of them are checked at startup.

        A judge whose alias does not resolve drops out of the ensemble
        and the run continues, which quietly produces a score built from
        fewer judges than the evaluation envelope records. Declaring the
        whole ensemble turns that into a startup failure naming the
        alias.
        """
        return list(dict.fromkeys(j.alias for j in self.judges if j.alias))

    @property
    def required_columns(self) -> list[str]:
        # ``persona`` is optional at eval time: historical trajectory
        # datasets keep only projected persona_* columns, and the
        # generator already degrades to an empty persona block when the
        # raw object is absent.
        return [
            self.conversation_messages_column,
            self.probe_family_column,
        ]

    @property
    def side_effect_columns(self) -> list[str]:
        # The evaluator writes this single column; ``name`` (inherited
        # from SingleColumnConfig) is the column it writes to.
        return []
