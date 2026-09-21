# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

from usersim.asset_gen.registry import AssetDomain


def _noop(*args, **kwargs):
    return None


DOMAIN = AssetDomain(
    name="fixture_domain",
    region_spec_dir=None,
    add_gen_args=_noop,
    load_spec=_noop,
    describe_plan=_noop,
    generate=_noop,
    validate=_noop,
    evaluate=_noop,
)
