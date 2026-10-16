# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The installed harness must collect scenarios, never silently skip all tests."""

import pytest


def pytest_collection_finish(session):
    if not session.items:
        raise pytest.UsageError("installed CLI e2e collected zero cases")
