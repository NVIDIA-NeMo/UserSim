# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test fixtures for conversation plugin tests."""

import pytest


@pytest.fixture
def full_persona():
    return {
        "first_name": "Maria",
        "last_name": "Silva",
        "age": 35,
        "sex": "Female",
        "education_level": "Bachelor's degree",
        "occupation": "Software Engineer",
        "marital_status": "Married",
        "city": "Sao Paulo",
        "state": "SP",
        "country": "Brazil",
        "openness": 0.7,
        "conscientiousness": 0.8,
        "extraversion": 0.6,
        "agreeableness": 0.75,
        "neuroticism": 0.3,
        "persona": "A tech-savvy professional who enjoys coding and hiking.",
        "skills_and_expertise": "Python, data analysis, machine learning",
        "hobbies_and_interests": "Hiking, reading, photography",
    }


@pytest.fixture
def full_persona_non_indic():
    """A persona explicitly guaranteed to carry no India-only fields."""
    return {
        "first_name": "Maria",
        "last_name": "Silva",
        "age": 35,
        "sex": "Female",
        "education_level": "Bachelor's degree",
        "occupation": "Software Engineer",
        "marital_status": "Married",
        "city": "Sao Paulo",
        "state": "SP",
        "country": "Brazil",
        "openness": 0.7,
        "conscientiousness": 0.8,
        "extraversion": 0.6,
        "agreeableness": 0.75,
        "neuroticism": 0.3,
        "persona": "A tech-savvy professional who enjoys coding and hiking.",
        "skills_and_expertise": "Python, data analysis, machine learning",
        "hobbies_and_interests": "Hiking, reading, photography",
    }


@pytest.fixture
def minimal_persona():
    return {
        "first_name": "Taro",
        "last_name": "Tanaka",
        "age": 72,
        "education_level": "Primary education",
        "occupation": "Retired farmer",
        "country": "Japan",
        "prefecture": "Niigata",
        "openness": 0.3,
        "conscientiousness": 0.6,
        "extraversion": 0.3,
        "agreeableness": 0.8,
        "neuroticism": 0.5,
    }
