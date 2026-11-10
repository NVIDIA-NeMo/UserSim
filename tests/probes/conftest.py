# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures the probes' India-variant tests share (``tests/probes/<package>/test_india_variant.py``)."""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture
def indian_persona() -> dict[str, Any]:
    return {
        "first_name": "Arjun",
        "last_name": "Kumar",
        "age": 34,
        "sex": "male",
        "education_level": "Graduate & above",
        "occupation": "Schoolteacher",
        "country": "India",
        "region": "Tamil Nadu",
        "district": "Chennai",
        "openness": 0.6,
        "conscientiousness": 0.6,
        "extraversion": 0.5,
        "agreeableness": 0.7,
        "neuroticism": 0.4,
        "hobbies_and_interests": "cricket, cooking, local history",
    }


@pytest.fixture
def cfg() -> Any:
    return type("Cfg", (), {"random_seed": 7, "max_turns": 5})()
