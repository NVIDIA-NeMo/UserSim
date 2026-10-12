# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""usersim — population-grounded user simulation for evaluating conversational LLMs."""

import os

# Data Designer reads this once, when its telemetry module is first imported, and
# prefixes the session id of its anonymous usage events with it. Setting it before
# any usersim module imports Data Designer marks those events as User Sim's; a
# prefix the caller already set is kept.
os.environ.setdefault("NEMO_SESSION_PREFIX", "usersim-")

from usersim.engine.core.provenance import own_distribution


def _installed_version() -> str:
    """The installed distribution's version; ``0+unknown`` for a source tree never installed."""
    distribution = own_distribution()
    return distribution.version if distribution is not None else "0+unknown"


__version__ = _installed_version()
