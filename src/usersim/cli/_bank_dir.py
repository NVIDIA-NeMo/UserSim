# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The folder an asset command reads or writes when it is given none."""

from __future__ import annotations

from pathlib import Path

from usersim.cli._errors import ConfigError
from usersim.engine.core._assets import shipped_assets_dir


def default_bank_dir(domain: str, locale: str, *, flag: str) -> Path:
    """The ``locale`` bank in the asset folder of the probe that shares ``domain``'s name.

    An asset domain builds the bank of the probe with its name, so that probe's
    own asset folder is where the bank lives. A domain with no such probe has
    no default folder: ``ConfigError`` asks for one with ``flag``.
    """
    try:
        return shipped_assets_dir(domain) / locale
    except LookupError:
        raise ConfigError(
            f"asset domain {domain!r} has no probe of that name whose asset folder could hold its bank; "
            f"pass {flag} to name the folder"
        ) from None
