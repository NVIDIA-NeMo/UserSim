# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each client's clinical-profile bank ships at its default path and loads."""

from __future__ import annotations

import pytest

from usersim.engine.probes.health_disclosure.bank import (
    CLIENT_PROBE_LABELS,
    default_clinical_profile_bank_path,
    load_clinical_profile_bank,
    reset_clinical_profile_bank_cache,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_clinical_profile_bank_cache()
    yield


@pytest.mark.parametrize("client", CLIENT_PROBE_LABELS)
def test_clinical_profile_bank_default_path_exists_and_loads(client: str) -> None:
    # Per-client (locale-independent) bundled synthetic banks.
    p = default_clinical_profile_bank_path(client)
    assert p.exists(), (
        f"clinical_profile_bank default path missing for {client}: {p}\n"
        "Either ship the file or update default_clinical_profile_bank_path()."
    )
    bank = load_clinical_profile_bank(p)
    assert bank.bank_version, f"clinical_profile_bank for {client} loaded but bank_version is empty"
    assert bank.client == client, (
        f"clinical_profile_bank at {p} declares client={bank.client!r} but "
        f"lives in the {client!r} directory — drift between layout and content."
    )
    # Bundled banks ship as placeholders until a customer supplies real ground truth.
    assert all(prof.placeholder for prof in bank.profiles), (
        f"bundled clinical_profile_bank for {client} must be all-placeholder"
    )
