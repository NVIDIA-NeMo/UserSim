# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The registered probe classes, by label.

``@register_probe`` in :mod:`usersim.engine.core.probes` fills this mapping;
``known_probes()`` and ``resolve_probe()`` there are the way to read it. It
lives in a module that imports nothing else from the engine, so the asset
lookup in :mod:`usersim.engine.core._assets` can see which probes are
registered without importing the probe substrate, and the substrate can
import the asset lookup at its top.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from usersim.engine.core.probes import BaseProbe

#: Each registered probe class, keyed by its ``label``.
PROBE_REGISTRY: dict[str, type[BaseProbe]] = {}
