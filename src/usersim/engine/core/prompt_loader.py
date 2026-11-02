# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""YAML-based prompt template loader.

Loads prompt templates from per-probe YAML files at
``<asset root>/<probe>/prompts.yaml``, falling back to the compiled-in
Python defaults if YAML files are not found. This allows prompt
iteration and A/B testing without code changes.

Per-probe asset layout: each probe owns a ``<probe>/`` folder under an
asset root and stores its prompt templates as ``prompts.yaml`` inside
that folder, alongside its other assets (e.g.
``tool_calling/prompts.yaml`` lives next to
``tool_calling/toolsets_seed.parquet``). The file is read from the
folder :func:`~usersim.engine.core._assets.probe_assets_dir` resolves,
the one the probe's banks are read from.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger("usersim.engine")


@lru_cache(maxsize=16)
def _load_prompt_file_at(probe: str, assets_root: str) -> dict[str, Any]:
    """Load a probe's ``prompts.yaml`` from a specific asset root."""
    yaml_path = Path(assets_root) / probe / "prompts.yaml"

    if not yaml_path.exists():
        logger.debug("  |-- Prompt YAML not found: %s", yaml_path)
        return {}

    try:
        import yaml
    except ImportError:
        logger.debug("  |-- PyYAML not installed, using compiled-in prompts")
        return {}

    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def load_prompt_file(probe: str) -> dict[str, Any]:
    """Load a probe's ``prompts.yaml`` from the folder its banks resolve to.

    That is the first root on the asset search path holding ``probe``, or
    the ``assets_root`` its class declares, so a root that holds other
    probes' assets does not hide this probe's prompts.

    Returns an empty dict if the file doesn't exist or PyYAML is not
    available — callers should always pass a sensible compiled-in
    default to :func:`get_prompt` to cover that path.
    """
    from usersim.engine.core._assets import probe_assets_dir

    return _load_prompt_file_at(probe, str(probe_assets_dir(probe).parent))


def get_prompt(probe: str, key: str, default: str = "") -> str:
    """Get a single prompt template by probe name and key.

    Falls back to the provided default if the YAML file doesn't exist
    or doesn't contain the requested key.
    """
    prompts = load_prompt_file(probe)
    return prompts.get(key, default)


def render_prompt(template: str, row: dict[str, Any], /, **explicit: Any) -> str:
    """Fill ``template``'s placeholders from ``explicit`` values, then the row.

    Lets a prompt edited in ``assets/<probe>/prompts.yaml`` reference any
    column present on the generated row — ``{uuid}``, ``{seed_prompt}``, a
    column added by a seed dataset — without a matching code change in the
    probe. Values the probe computes (a formatted persona block, a resolved
    topic) are passed as ``explicit`` and take precedence, so a raw row value
    can never shadow a rendered one: the row supplies ``persona`` as a dict,
    for instance, while ``explicit["persona"]`` is the formatted text.

    A placeholder that is in neither raises ``KeyError`` naming it and listing
    what was available, so a typo in a prompt asset is diagnosable instead of
    surfacing as a bare ``KeyError: 'topc'`` from ``str.format``.
    """
    values = {**row, **explicit}
    try:
        return template.format_map(values)
    except KeyError as e:
        missing = e.args[0] if e.args else "?"
        raise KeyError(
            f"prompt placeholder {{{missing}}} is not a probe value "
            f"({sorted(explicit)}) nor a row column ({sorted(row)[:20]})"
        ) from None
