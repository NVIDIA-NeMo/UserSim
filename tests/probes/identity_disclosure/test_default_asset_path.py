# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The identity spec's default path holds a spec its loader accepts."""

from __future__ import annotations

from usersim.engine.probes.identity_disclosure.spec import default_identity_spec_path, load_identity_spec


def test_identity_spec_default_path_exists_and_loads() -> None:
    p = default_identity_spec_path()
    assert p.exists(), f"identity spec default path missing: {p}"
    spec = load_identity_spec(p)
    assert spec.bank_id == "identity_disclosure"
    assert spec.tactics, "identity spec loaded but has no tactics"
