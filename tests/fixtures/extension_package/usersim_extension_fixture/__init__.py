# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""An out-of-tree package that uses every published extension seam.

Nothing here is imported by the project under test. It is installed into a
throwaway environment by ``make check-extensions``, which then asks the CLI
what it can see. That is the difference between testing the wiring and
testing that an actual installed distribution reaches it.
"""
