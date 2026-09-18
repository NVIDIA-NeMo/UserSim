# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-cutting invariant: every shipped probe's default asset path
resolves to a file that **exists** AND **loads cleanly** through its
loader.

This is the test that would have caught the regression where
``core/probing_taxonomy.py`` returned ``assets/sov_ai_dynamic/<locale>/categories.yaml``
while the on-disk directory was still ``assets/population_probing/``.
The per-loader unit-test files all bypass ``default_*_path()`` — they
either write tmp_path fixtures or hardcode ``(packaged_assets_dir() / "...")``
strings — so a default-path constant could point at ``/nonexistent``
and every per-loader suite would still be green.

Coverage axes:

- **Per-locale loaders** (``fact_bank``, ``probing_taxonomy``,
  ``seeds`` for ``general_open_ended`` / ``general_educational``):
  every supported locale × every loader.
- **Locale-agnostic loaders** (``query_bank``, ``pressure_bank``,
  ``agentic_bank``, ``action_taxonomy``): one assertion per loader.
- **Per-probe prompts** (``tool_calling`` / ``general_open_ended`` /
  ``general_educational``): one assertion per probe.

Each assertion checks two things:

1. ``default_*_path()`` (or its loader-internal equivalent) returns a
   path that ``Path.exists()``. Catches "loader points at the wrong
   directory" — the exact bug class above.
2. The file loads through its loader without raising. Catches "loader
   points at a path that exists but the file is malformed / fails
   schema validation" (a different bug class than path mismatch but
   useful coverage).

If a future probe ships, the ``test_marker_set_matches_probe_registry``
invariant in ``test_assets_discovery.py`` will fail (asserting that
``_ASSETS_MARKERS == known_probes()``), forcing the contributor to add
a row here too. The two tests together pin both relationships:
``probe registry == on-disk markers`` and ``each marker == real
loadable asset folder``.
"""

from __future__ import annotations

import pytest

from usersim.engine.core.action_taxonomy import (
    load_action_taxonomy_default,
    reset_action_taxonomy_cache,
)
from usersim.engine.core.agentic_bank import (
    default_agentic_bank_path,
    load_agentic_bank,
    reset_agentic_bank_cache,
)
from usersim.engine.core.clinical_profile_bank import (
    CLIENT_PROBE_LABELS,
    default_clinical_profile_bank_path,
    load_clinical_profile_bank,
    reset_clinical_profile_bank_cache,
)
from usersim.engine.core.fact_bank import (
    default_fact_bank_path,
    load_fact_bank,
    reset_fact_bank_cache,
)
from usersim.engine.core.locale import SHIPPED_LOCALES
from usersim.engine.core.pressure_bank import (
    default_pressure_bank_path,
    load_pressure_bank,
    reset_pressure_bank_cache,
)
from usersim.engine.core.probing_taxonomy import (
    default_probing_taxonomy_path,
    load_probing_taxonomy,
    reset_probing_taxonomy_cache,
)
from usersim.engine.core.prompt_loader import load_prompt_file
from usersim.engine.core.query_bank import (
    default_query_bank_path,
    load_query_bank,
    reset_query_bank_cache,
)
from usersim.engine.core.seeds import load_seeds

# ---------------------------------------------------------------------------
# Per-locale loaders: fact_bank + probing_taxonomy
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_loader_caches():
    """Drop every cached loader state so each test reads from disk
    rather than reusing whatever a sibling test loaded earlier."""
    reset_fact_bank_cache()
    reset_probing_taxonomy_cache()
    reset_pressure_bank_cache()
    reset_query_bank_cache()
    reset_agentic_bank_cache()
    reset_action_taxonomy_cache()
    reset_clinical_profile_bank_cache()
    yield


@pytest.mark.parametrize("locale", SHIPPED_LOCALES)
def test_fact_bank_default_path_exists_and_loads(locale: str) -> None:
    p = default_fact_bank_path(locale)
    assert p.exists(), (
        f"fact_bank default path missing for {locale}: {p}\n"
        "Either ship the file or update default_fact_bank_path() to "
        "point at the right directory."
    )
    bank = load_fact_bank(p)
    assert bank.bank_version, f"fact_bank for {locale} loaded but bank_version is empty"
    assert bank.locale == locale, (
        f"fact_bank at {p} declares locale={bank.locale!r} but the "
        f"directory it lives in declares {locale!r} — drift between "
        "directory layout and file content."
    )


@pytest.mark.parametrize("locale", SHIPPED_LOCALES)
def test_probing_taxonomy_default_path_exists_and_loads(locale: str) -> None:
    p = default_probing_taxonomy_path(locale)
    assert p.exists(), (
        f"probing_taxonomy default path missing for {locale}: {p}\n"
        "Either ship the file or update default_probing_taxonomy_path() to "
        "point at the right directory."
    )
    tax = load_probing_taxonomy(p)
    assert tax.taxonomy_version, f"probing_taxonomy for {locale} loaded but taxonomy_version is empty"
    assert tax.locale == locale, (
        f"probing_taxonomy at {p} declares locale={tax.locale!r} but the "
        f"directory it lives in declares {locale!r} — drift between "
        "directory layout and file content."
    )


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
    locale-specific file nor the ``base/`` fallback exists. We assert
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
# Locale-agnostic loaders
# ---------------------------------------------------------------------------


def test_query_bank_default_path_exists_and_loads() -> None:
    p = default_query_bank_path()
    assert p.exists(), f"query_bank default path missing: {p}"
    bank = load_query_bank(p)
    assert bank.bank_id, "query_bank loaded but bank_id is empty"


def test_pressure_bank_default_path_exists_and_loads() -> None:
    p = default_pressure_bank_path()
    assert p.exists(), f"pressure_bank default path missing: {p}"
    bank = load_pressure_bank(p)
    assert bank.bank_id, "pressure_bank loaded but bank_id is empty"


def test_agentic_bank_default_path_exists_and_loads() -> None:
    p = default_agentic_bank_path()
    assert p.exists(), f"agentic_bank default path missing: {p}"
    bank = load_agentic_bank(p)
    assert bank.bank_id, "agentic_bank loaded but bank_id is empty"


@pytest.mark.parametrize("client", CLIENT_PROBE_LABELS)
def test_clinical_profile_bank_default_path_exists_and_loads(client: str) -> None:
    # Per-client (locale-independent) bundled synthetic banks.
    p = default_clinical_profile_bank_path(client)
    assert p.exists(), (
        f"clinical_profile_bank default path missing for {client}: {p}\n"
        "Either ship the file or update default_clinical_profile_bank_path()."
    )
    bank = load_clinical_profile_bank(p)
    assert bank.bank_version, f"clinical_profile_bank for {client} loaded but bank_version is empty"
    assert bank.client == client, (
        f"clinical_profile_bank at {p} declares client={bank.client!r} but "
        f"lives in the {client!r} directory — drift between layout and content."
    )
    # Bundled banks ship as placeholders until a customer supplies real ground truth.
    assert all(prof.placeholder for prof in bank.profiles), (
        f"bundled clinical_profile_bank for {client} must be all-placeholder"
    )


def test_action_taxonomy_default_path_loads() -> None:
    """``load_action_taxonomy_default`` resolves the path internally
    (no public ``default_*_path`` helper); it raises if the file is
    missing or malformed. We additionally assert non-empty entries
    so an accidentally empty file doesn't pass."""
    tax = load_action_taxonomy_default()
    assert tax.entries, "action_taxonomy loaded but has zero entries"


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
    """``cli/simulate.py`` and ``notebooks/01_simulate.ipynb`` compose the
    toolset-seed path manually as ``assets_dir / "tool_calling" /
    "toolsets_seed.parquet"`` rather than going through a loader. When
    the file is absent both branches silently set ``use_tools=False``
    in ``cli/_pipeline.py``, which sets ``ConversationSimulatorConfig.tools_column=None``,
    which makes every ``tool_calling`` probe fail at construction with
    ``cfg.tools_column not configured``. We pin the existence + structural
    health here so a future move (e.g. somebody renaming the
    ``tool_calling/`` directory without updating the path-composition sites)
    fails the test suite instead of silently producing 100% failed
    ``tool_calling`` trajectories.
    """
    from usersim.engine.core._assets import probe_assets_dir

    p = probe_assets_dir("tool_calling") / "toolsets_seed.parquet"
    assert p.exists(), (
        f"tool_calling toolset seed missing: {p}\n"
        "Without it cli/simulate.py's `use_tools` branch resolves to "
        "False and every tool_calling probe fails at construction with "
        "'cfg.tools_column not configured'."
    )

    # Structural health: parquet readable, non-empty, has the columns the
    # data-designer SamplerColumnConfig consumes.
    import pyarrow.parquet as pq

    table = pq.read_table(p)
    assert table.num_rows > 0, f"{p} is empty"
    cols = set(table.column_names)
    expected = {"tools", "toolset_name"}
    missing = expected - cols
    assert not missing, f"{p} is missing expected columns {missing}; present: {sorted(cols)}"
