# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""usersim — population-grounded user simulation for evaluating conversational LLMs."""

import os

# Data Designer reads this once, when its telemetry module is first imported, and
# prefixes the session id of its anonymous usage events with it. Setting it before
# any usersim module imports Data Designer marks those events as User Sim's; a
# prefix the caller already set is kept.
os.environ.setdefault("NEMO_SESSION_PREFIX", "usersim-")

__version__ = "0.1.0"
