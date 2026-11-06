# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from usersim.engine.probes.identity_disclosure.probe import IdentityDisclosureProbe
from usersim.extension import register_probe


@register_probe
class FixtureIdentityProbe(IdentityDisclosureProbe):
    """identity_disclosure asked with this package's spec layer, assets/fixture_identity/spec.yaml."""

    label = "fixture_identity"
    assets_root = Path(__file__).parent / "assets"
