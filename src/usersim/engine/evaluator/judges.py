# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Judge family registry and ensemble diversity check.

Two judges from the same model family tend to share blind spots, so
their scores can agree without being right. An ensemble of two or more
judges must therefore span at least two families from this registry;
two variants of one family do not count. The check runs when the
config is validated, before any judge call, and again for each row
once the judges' model ids are resolved.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


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


# Best-effort mapping from model ids to families, used when the user
# specifies a judge by alias only and has not explicitly set the family.
# The user can always override. Fragments of the model's own name come
# first: a gateway or an OpenAI-compatible route serves other developers'
# models under its own prefix ("openai/google/gemma-3-27b-it"), so a
# provider prefix only decides when the name carries no hint.
_MODEL_NAME_HINTS: dict[str, JudgeFamily] = {
    "gpt-oss": JudgeFamily.OPENAI,
    "gpt-5": JudgeFamily.OPENAI,
    "gpt-4": JudgeFamily.OPENAI,
    "nemotron": JudgeFamily.NVIDIA_NEMOTRON,
    "claude": JudgeFamily.ANTHROPIC,
    "deepseek": JudgeFamily.DEEPSEEK,
    "qwen": JudgeFamily.QWEN,
    "llama": JudgeFamily.LLAMA,
    "mistral": JudgeFamily.MISTRAL,
    "mixtral": JudgeFamily.MISTRAL,
    "gemini": JudgeFamily.GEMINI,
    "gemma": JudgeFamily.GEMINI,
    "o3": JudgeFamily.OPENAI,
    "o4": JudgeFamily.OPENAI,
}
_PROVIDER_HINTS: dict[str, JudgeFamily] = {
    "openai/": JudgeFamily.OPENAI,
    "nvidia/": JudgeFamily.NVIDIA_NEMOTRON,
    "anthropic/": JudgeFamily.ANTHROPIC,
    "alibaba/": JudgeFamily.QWEN,
    "meta/": JudgeFamily.LLAMA,
    "meta-llama/": JudgeFamily.LLAMA,
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
    for hints in (_MODEL_NAME_HINTS, _PROVIDER_HINTS):
        for hint, fam in hints.items():
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
    family: JudgeFamily | None = None

    def resolved_family(self, model_id: str | None = None) -> JudgeFamily:
        if self.family is not None:
            return self.family
        return infer_judge_family(model_id or self.alias)


class EnsembleDiversityError(ValueError):
    """Raised when a judge ensemble fails the diversity constraint."""


def validate_ensemble_diversity(
    judges: Iterable[JudgeSpec],
    *,
    min_distinct_families: int = 2,
    resolved_model_ids: dict[str, str] | None = None,
) -> None:
    """Raise ``EnsembleDiversityError`` if the ensemble lacks family diversity.

    ``min_distinct_families`` defaults to 2. A single judge is allowed;
    the constraint applies once there are two or more judges.

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
            f"same family tend to share blind spots and do not satisfy this "
            f"constraint."
        )
