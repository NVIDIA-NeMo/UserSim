# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""financial_services generation: region_spec contract (``spec``) + DD pipeline.

The committed ``region_spec/*.yaml`` files are the generation input; ``spec.py``
validates them and ``pipeline.py`` expands them into a per-institution asset bank.
"""

from usersim.asset_gen.financial_services.spec import (
    RegionSpec,
    RegionSpecError,
    load_region_spec,
    validate_region_spec,
)

__all__ = [
    "RegionSpec",
    "RegionSpecError",
    "load_region_spec",
    "validate_region_spec",
]
