# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-probe seed loader with locale-aware replacement.

Seeds for a given probe live under
``assets/<probe>/<kind>/<locale>/<kind>.json``, with a
``<probe>/<kind>/base/<kind>.json`` fallback for locales that don't
ship a translated version. The ``base`` file is treated as the default
(English-language) source; locale-specific files are **complete
replacements** (containing both translated universals and locale-only
additions) so descriptions are always in the target language.

Concrete shipped layouts:

- ``assets/general_open_ended/topics/{base,<locale>}/topics.json``
- ``assets/general_educational/subjects/{base,<locale>}/subjects.json``

API:

>>> load_seeds("pt_BR", "general_open_ended", "topics")  # doctest: +SKIP
[{...}, ...]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_seeds(
    locale: str,
    probe: str,
    kind: str,
    assets_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Load seed data for ``(probe, kind)`` with locale-aware replacement.

    Path resolution:

    1. ``<assets>/<probe>/<kind>/<locale>/<kind>.json`` — used if it exists.
    2. ``<assets>/<probe>/<kind>/base/<kind>.json`` — fallback otherwise
       (covers ``en_US`` and any locale without its own translation).

    Args:
        locale: The locale code (e.g., ``"en_US"``, ``"pt_BR"``).
        probe: The owning probe directory under ``assets/`` (e.g.,
            ``"general_open_ended"``, ``"general_educational"``).
        kind: The seed file basename (and the inner directory name),
            e.g. ``"topics"`` or ``"subjects"``.
        assets_dir: Override the shipped ``assets/`` root. Defaults to
            the discovered shipped directory.

    Returns:
        List of seed dicts in the target language.

    Raises:
        FileNotFoundError: If neither locale-specific nor base seed file
            exists for the requested ``(probe, kind)``.
    """
    if assets_dir is not None:
        base = Path(assets_dir) / probe / kind
    else:
        from usersim.engine.core._assets import probe_assets_dir

        base = probe_assets_dir(probe) / kind

    locale_path = base / locale / f"{kind}.json"
    if locale_path.exists():
        with open(locale_path) as f:
            return json.load(f)

    base_path = base / "base" / f"{kind}.json"
    if not base_path.exists():
        raise FileNotFoundError(
            f"Seed file not found for probe={probe!r} kind={kind!r} "
            f"locale={locale!r}: checked {locale_path} and {base_path}"
        )

    with open(base_path) as f:
        return json.load(f)
