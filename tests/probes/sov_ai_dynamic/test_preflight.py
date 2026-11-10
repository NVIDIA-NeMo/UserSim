# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""sov_ai_dynamic runs the India language variants and the Hindi locales on the en_IN bank."""

from __future__ import annotations

from usersim.cli._pipeline import verify_probe_assets


def test_an_india_variant_without_its_own_bank_uses_en_in() -> None:
    verify_probe_assets({"sov_ai_dynamic": 1.0}, ["bn_Beng_IN"], None)


def test_the_hindi_locales_run_on_the_en_in_bank() -> None:
    verify_probe_assets({"sov_ai_dynamic": 1.0}, ["hi_Deva_IN", "hi_Latn_IN"], None)
