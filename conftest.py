# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repo-wide pytest configuration.

Lives at the repo root so it applies to every test under ``tests``, the test
root declared in ``pyproject.toml``.

The hermetic harness (the network guard, the extension-discovery and
pytest-asyncio checks, and the tokenizer and persona-dataset stubs) is
``usersim.testing.pytest_plugin``, the plugin an extension's own suite loads,
so built-in probes and extensions are tested under the same rules.
``pytester`` runs the plugin's own tests against a separate test project.
"""

pytest_plugins = ["pytester", "usersim.testing.pytest_plugin"]
