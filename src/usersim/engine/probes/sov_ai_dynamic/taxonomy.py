# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``sov_ai_dynamic`` probe's taxonomy files.

The taxonomy format, its validator and the cache it shares with
``financial_services`` are in :mod:`usersim.engine.core.probing_taxonomy`.
These name the probe's own files in them: ``<locale>/categories.yaml``
under its asset folder, or the file ``USERSIM_SOV_AI_DYNAMIC_TAXONOMY_<LOCALE>``
names.
"""

from __future__ import annotations

from pathlib import Path

from usersim.engine.core.probing_taxonomy import ProbingTaxonomy, load_probing_taxonomy_for, taxonomy_path_for

_FAMILY = "sov_ai_dynamic"
_FILENAME = "categories.yaml"
_ENV_PREFIX = "USERSIM_SOV_AI_DYNAMIC_TAXONOMY"


def default_probing_taxonomy_path(locale: str) -> Path:
    """The file the probe reads for ``locale``, honoring the environment override."""
    return taxonomy_path_for(_FAMILY, locale, filename=_FILENAME, env_prefix=_ENV_PREFIX)


def probing_taxonomy_path_for(locale: str) -> Path:
    return default_probing_taxonomy_path(locale)


def load_probing_taxonomy_for_locale(locale: str) -> ProbingTaxonomy:
    """The cached taxonomy for ``locale``, the one the probe and its scorer both read."""
    return load_probing_taxonomy_for(_FAMILY, locale, filename=_FILENAME, env_prefix=_ENV_PREFIX)
