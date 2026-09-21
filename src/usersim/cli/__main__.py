# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Allow ``python -m cli`` as an alias for the ``usersim`` console script."""

from __future__ import annotations

import sys

from usersim.cli import main

if __name__ == "__main__":
    sys.exit(main())
