# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The default probes' shipped assets are where their loaders look, and load.

A loader whose default path points at the wrong folder passes every unit test
that hands it a ``tmp_path`` file, so these tests read the shipped files
through the same paths a run uses:

- the topic and subject seeds of ``general_open_ended`` and
  ``general_educational``, for every shipped locale;
- the ``prompts.yaml`` of each default probe;
- the toolset table every ``tool_calling`` row is drawn from.

Every other built-in checks its own default paths in
``tests/probes/<package>/test_default_asset_path.py``, so this file names no
probe outside the default mix.
"""

from __future__ import annotations

import pytest

from usersim.engine.core.locale import SHIPPED_LOCALES
from usersim.engine.core.prompt_loader import load_prompt_file
from usersim.engine.core.seeds import load_seeds

# ---------------------------------------------------------------------------
# Per-probe seed loaders: general_open_ended + general_educational
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("locale", SHIPPED_LOCALES)
@pytest.mark.parametrize(
    "probe,kind",
    [
        ("general_open_ended", "topics"),
        ("general_educational", "subjects"),
    ],
)
def test_seeds_default_path_loads_for_every_locale(
    locale: str,
    probe: str,
    kind: str,
) -> None:
    """``load_seeds`` raises ``FileNotFoundError`` when neither the
    locale's own file nor the ``en_US`` fallback exists. We assert
    a non-empty result so a bug that ships an empty file (or points at
    one) also fails."""
    seeds = load_seeds(locale, probe, kind)
    assert isinstance(seeds, list), f"load_seeds({locale}, {probe}, {kind}) returned {type(seeds).__name__}, not a list"
    assert seeds, (
        f"load_seeds({locale}, {probe}, {kind}) returned an empty list — "
        "either the shipped seed file is empty or the loader landed on "
        "the wrong file."
    )
    # Shape contract every seed obeys (matches test_multi_locale.py).
    for s in seeds:
        assert "type" in s, f"{probe}/{kind} for {locale}: missing 'type' in {s}"
        assert "description" in s, f"{probe}/{kind} for {locale}: missing 'description' in {s}"


# ---------------------------------------------------------------------------
# Per-probe prompts.yaml
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "probe",
    [
        "tool_calling",
        "general_open_ended",
        "general_educational",
    ],
)
def test_prompt_loader_returns_nonempty_for_each_probe(probe: str) -> None:
    """Each probe that overrides defaults via ``assets/<probe>/prompts.yaml``
    must ship a non-empty YAML at that path. The loader silently
    returns ``{}`` on missing file — the per-probe fallback to compiled
    Python defaults would mask a "file moved" regression — so we
    assert non-empty here, which means the file exists AND is
    structurally a non-empty mapping."""
    prompts = load_prompt_file(probe)
    assert prompts, (
        f"assets/{probe}/prompts.yaml is missing or empty — "
        "the probe will silently fall back to compiled-in defaults, "
        "masking the file-moved regression. Ship the file or remove "
        "the get_prompt() call sites in the probe's prompts.py."
    )


# ---------------------------------------------------------------------------
# tool_calling toolsets seed parquet
# ---------------------------------------------------------------------------


def test_tool_calling_toolsets_seed_exists_and_loads() -> None:
    """The toolset table every ``tool_calling`` row is drawn from ships, loads and holds the columns rows read.

    The probe takes a row's tools from its ``tools`` column (the probe's
    ``tools_column`` setting) and refuses a row without it, so a missing or
    malformed table fails every ``tool_calling`` conversation.
    """
    from usersim.engine.core._assets import probe_assets_dir

    p = probe_assets_dir("tool_calling") / "toolsets_seed.parquet"
    assert p.exists(), f"tool_calling toolset seed missing: {p}"

    # Structural health: parquet readable, non-empty, has the columns the
    # data-designer SamplerColumnConfig consumes.
    import pyarrow.parquet as pq

    table = pq.read_table(p)
    assert table.num_rows > 0, f"{p} is empty"
    cols = set(table.column_names)
    expected = {"tools", "toolset_name"}
    missing = expected - cols
    assert not missing, f"{p} is missing expected columns {missing}; present: {sorted(cols)}"
