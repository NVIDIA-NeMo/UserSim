# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-probe seed loader, one file per locale.

A probe's seeds live in ``<probe>/<locale>/<kind>.json``, one folder per
locale, and a locale without a file of its own reads the
``FALLBACK_LOCALE`` file. Each locale's file is a complete set
(translated universals and locale-only additions), not additions to the
fallback, so descriptions are always in the target language.

Concrete shipped layouts:

- ``general_open_ended/<locale>/topics.json``
- ``general_educational/<locale>/subjects.json``

API:

>>> load_seeds("pt_BR", "general_open_ended", "topics")  # doctest: +SKIP
[{...}, ...]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FALLBACK_LOCALE = "en_US"
"""The locale whose seed file a locale without its own file reads."""


def load_seeds(
    locale: str,
    probe: str,
    kind: str,
    assets_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Load ``locale``'s ``kind`` seeds for ``probe``.

    Path resolution:

    1. ``<probe>/<locale>/<kind>.json``, used if it exists.
    2. ``<probe>/en_US/<kind>.json`` otherwise, for any locale that
       ships no file of its own.

    Args:
        locale: The locale code (e.g., ``"en_US"``, ``"pt_BR"``).
        probe: The probe's label (e.g., ``"general_open_ended"``,
            ``"general_educational"``).
        kind: The seed file's basename, e.g. ``"topics"`` or ``"subjects"``.
        assets_dir: A root holding ``<probe>/``. Defaults to the folder
            ``probe_assets_dir`` resolves for the probe.

    Returns:
        List of seed dicts in the target language.

    Raises:
        FileNotFoundError: If neither the locale's file nor the
            ``FALLBACK_LOCALE`` file exists for ``(probe, kind)``.
    """
    if assets_dir is not None:
        probe_dir = Path(assets_dir) / probe
    else:
        from usersim.engine.core._assets import probe_assets_dir

        probe_dir = probe_assets_dir(probe)

    locale_path = probe_dir / locale / f"{kind}.json"
    if locale_path.exists():
        with open(locale_path) as f:
            return json.load(f)

    fallback_path = probe_dir / FALLBACK_LOCALE / f"{kind}.json"
    if not fallback_path.exists():
        raise FileNotFoundError(
            f"Seed file not found for probe={probe!r} kind={kind!r} "
            f"locale={locale!r}: checked {locale_path} and {fallback_path}"
        )

    with open(fallback_path) as f:
        return json.load(f)
