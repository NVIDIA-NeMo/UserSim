# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Judge family registry + ensemble diversity validator.

Per the design plan §4.1: "judge ensemble diversity is operationalized
via an explicit registry; two variants of the same family do not
satisfy the constraint." This module is that registry.

Why this matters: inter-judge agreement (Cohen's κ / Krippendorff's α)
is meaningful only across architecturally diverse models. Two
fine-tunes of the same base produce spurious agreement that masks
genuine miscalibration. The ensemble diversity check enforces ≥2
distinct families up front, at config-validation time, before any
judging cost is incurred.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional


class JudgeFamily(str, Enum):
    """Architecturally-diverse model families used for judge ensembling.

    Curated to draw real lines, not vendor lines. ``OPENAI`` covers
    GPT-OSS / GPT-5 / etc; ``NVIDIA_NEMOTRON`` covers the Nemotron-3
    family; ``ANTHROPIC`` covers Claude; etc. Two judges with the
    same value here do **not** satisfy the diversity constraint.
    """

    OPENAI = "openai"
    NVIDIA_NEMOTRON = "nvidia_nemotron"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    LLAMA = "llama"
    MISTRAL = "mistral"
    GEMINI = "gemini"
    OTHER = "other"


# Best-effort mapping from common model-name fragments to families.
# Used when the user specifies a judge by alias only and has not
# explicitly set the family. The user can always override.
_FAMILY_HINTS: dict[str, JudgeFamily] = {
    "gpt-oss": JudgeFamily.OPENAI,
    "openai/": JudgeFamily.OPENAI,
    "gpt-5": JudgeFamily.OPENAI,
    "gpt-4": JudgeFamily.OPENAI,
    "o3": JudgeFamily.OPENAI,
    "o4": JudgeFamily.OPENAI,
    "nvidia/": JudgeFamily.NVIDIA_NEMOTRON,
    "nemotron": JudgeFamily.NVIDIA_NEMOTRON,
    "claude": JudgeFamily.ANTHROPIC,
    "anthropic/": JudgeFamily.ANTHROPIC,
    "deepseek": JudgeFamily.DEEPSEEK,
    "qwen": JudgeFamily.QWEN,
    "alibaba/": JudgeFamily.QWEN,
    "llama": JudgeFamily.LLAMA,
    "meta/": JudgeFamily.LLAMA,
    "meta-llama/": JudgeFamily.LLAMA,
    "mistral": JudgeFamily.MISTRAL,
    "mixtral": JudgeFamily.MISTRAL,
    "gemini": JudgeFamily.GEMINI,
    "google/": JudgeFamily.GEMINI,
}


def infer_judge_family(model_id: str) -> JudgeFamily:
    """Best-effort inference of family from a model id or alias.

    Falls back to ``JudgeFamily.OTHER`` when nothing matches; the
    user is then expected to set ``family`` explicitly on the
    ``JudgeSpec`` (otherwise the diversity validator will treat
    multiple OTHER-family judges as non-diverse).
    """
    lowered = model_id.lower()
    for hint, fam in _FAMILY_HINTS.items():
        if hint in lowered:
            return fam
    return JudgeFamily.OTHER


@dataclass(frozen=True)
class JudgeSpec:
    """One judge in the evaluator ensemble.

    ``alias`` is the DD model alias (resolved via the model registry
    at run time). ``family`` is the architectural family for the
    diversity check; if ``None``, ``infer_judge_family`` is consulted
    against ``alias`` (and against the configured model name once
    resolved).
    """

    alias: str
    family: Optional[JudgeFamily] = None

    def resolved_family(self, model_id: Optional[str] = None) -> JudgeFamily:
        if self.family is not None:
            return self.family
        return infer_judge_family(model_id or self.alias)


class EnsembleDiversityError(ValueError):
    """Raised when a judge ensemble fails the diversity constraint."""


def validate_ensemble_diversity(
    judges: Iterable[JudgeSpec],
    *,
    min_distinct_families: int = 2,
    resolved_model_ids: Optional[dict[str, str]] = None,
) -> None:
    """Raise ``EnsembleDiversityError`` if the ensemble lacks family diversity.

    Per the design plan: ``min_distinct_families = 2`` is the default.
    A single-judge ensemble is allowed (it explicitly opts out of κ
    reporting); the constraint only applies once you have ≥2 judges.

    ``resolved_model_ids`` is an optional ``{alias: model_id}`` map
    used to refine family inference at run time when ``JudgeSpec.family``
    is ``None`` and the alias alone is not a useful hint.
    """
    judges_list = list(judges)
    if len(judges_list) < 2:
        return
    families: list[JudgeFamily] = []
    for spec in judges_list:
        model_id = resolved_model_ids.get(spec.alias) if resolved_model_ids else None
        families.append(spec.resolved_family(model_id))
    distinct = set(families)
    other_count = sum(1 for f in families if f == JudgeFamily.OTHER)
    if other_count == len(families):
        raise EnsembleDiversityError(
            f"Judge ensemble of {len(families)} judges all resolved to "
            f"{JudgeFamily.OTHER.value} — set JudgeSpec.family explicitly "
            f"so the diversity check can do its job."
        )
    if len(distinct) < min_distinct_families:
        raise EnsembleDiversityError(
            f"Judge ensemble must span at least {min_distinct_families} distinct "
            f"families (per evaluator/judges.py JudgeFamily); got {sorted(f.value for f in distinct)} "
            f"for aliases {[s.alias for s in judges_list]}. Two variants of the "
            f"same family produce spurious inter-judge agreement and do not "
            f"satisfy this constraint."
        )
