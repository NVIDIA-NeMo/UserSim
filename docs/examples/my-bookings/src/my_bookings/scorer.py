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
        return {"scores": {"my_bookings.booked_rate": {"score": 0.0}}}
    arguments = made["arguments"]
    rates = {
        "my_bookings.booked_rate": 1.0,
        "my_bookings.date_match_rate": float(arguments.get("date") == expected["date"]),
        "my_bookings.time_match_rate": float(arguments.get("time") == expected["time"]),
        "my_bookings.people_match_rate": float(arguments.get("people") == expected["people"]),
        "my_bookings.asked_first_rate": float(made["after_user_turns"] > 1),
    }
    return {"scores": {axis: {"score": value} for axis, value in rates.items()}}


register_scorer("my_bookings", score, deterministic=True)
