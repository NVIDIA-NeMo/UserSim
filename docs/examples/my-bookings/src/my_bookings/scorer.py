# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from usersim.extension import register_scorer


async def score(row, models):
    """Whether the assistant booked what the user meant, after asking how many people."""
    if not row.get("my_booking"):
        return {}
    booking = json.loads(row["my_booking"])
    expected, made = booking["expected"], booking["made"]
    if made is None:
        return {"my_booked": 0.0}
    arguments = made["arguments"]
    return {
        "my_booked": 1.0,
        "my_date_correct": float(arguments.get("date") == expected["date"]),
        "my_time_correct": float(arguments.get("time") == expected["time"]),
        "my_people_correct": float(arguments.get("people") == expected["people"]),
        "my_asked_before_booking": float(made["after_user_turns"] > 1),
    }


register_scorer("my_bookings", score, deterministic=True)
