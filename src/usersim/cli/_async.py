# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Running a coroutine from a synchronous command.

Commands are synchronous entry points, while scoring is awaited. This is
the single place that boundary is crossed, so the notebook case is handled
once rather than at each call site.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Coroutine
from typing import Any, TypeVar

_T = TypeVar("_T")


def run_coroutine(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run ``coro`` to completion and return its result.

    A notebook already owns the loop on its thread, and ``asyncio.run``
    refuses to nest inside one. Where that is the case the work moves to a
    thread with a loop of its own, which is the only way a synchronous call
    can wait for a coroutine from inside a live loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
