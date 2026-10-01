# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from usersim.engine.core.probes import register_probe
from usersim.engine.probes.identity_disclosure.generator import IdentityDisclosureProbe


@register_probe(family="fixture_identity", prompt_version="v1.0", variants=("default",))
class FixtureIdentityProbe(IdentityDisclosureProbe):
    """identity_disclosure asked with this package's spec layer, assets/fixture_identity/spec.yaml."""

    label = "fixture_identity"
