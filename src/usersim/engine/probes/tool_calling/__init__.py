# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Evaluation-oriented tool-calling probe.

Tests the assistant model's native tool-calling capabilities using
persona-grounded, realistic user requests. No majority voting, no
correction loops -- single-shot responses reveal true capability.
"""
