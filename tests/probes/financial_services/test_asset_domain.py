# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An extension that registers an asset domain under financial_services' name cannot displace the built-in one."""

from __future__ import annotations

import test_extension_seams as seam_tests

from usersim.engine.core._extensions import ASSET_DOMAINS

SOURCE = seam_tests.TestAssetDomainDiscovery.SOURCE


def test_builtin_is_not_displaced(extension) -> None:
    """Shadowing financial_services would silently change generated assets."""
    from usersim.asset_gen.registry import clear_domain_cache, get_domain

    extension(
        "ext_domain_mod3.py",
        SOURCE.replace('"partner_domain"', '"financial_services"'),
        ("financial_services", "ext_domain_mod3:DOMAIN", ASSET_DOMAINS),
    )
    clear_domain_cache()
    try:
        domain = get_domain("financial_services")
        assert domain.region_spec_dir is not None, "the extension displaced the built-in financial_services domain"
    finally:
        clear_domain_cache()
