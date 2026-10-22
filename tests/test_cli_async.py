# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The synchronous CLI boundary honors the scorer's Awaitable contract."""

from usersim.cli._async import run_coroutine


class Result:
    def __await__(self):
        async def resolve():
            return {"score": 4}

        return resolve().__await__()


def test_cli_awaits_scorer_result_without_a_running_loop():
    assert run_coroutine(Result()) == {"score": 4}


async def test_cli_awaits_scorer_result_inside_a_running_loop():
    assert run_coroutine(Result()) == {"score": 4}
