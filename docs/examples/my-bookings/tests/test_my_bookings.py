# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import random

import my_bookings.probe  # noqa: F401  registers the probe; discovery is off under test
from my_bookings.probe import Appointment, BookingProbe, draw_booking
from my_bookings.scorer import score

from usersim.extension import ConversationSimulatorConfig
from usersim.testing import (
    ScriptedModel,
    assert_hosted_parity,
    assert_probe_conforms,
    assert_scorer_conforms,
    run_episode,
)

PERSONA = {"first_name": "Ana", "last_name": "Souza", "age": 35, "city": "Recife"}
CONFIG = ConversationSimulatorConfig(name="conversation_messages", max_turns=3, random_seed=7, locale="pt_BR")


def test_the_probe_conforms():
    assert_probe_conforms("my_bookings")


async def test_the_scorer_conforms():
    await assert_scorer_conforms("my_bookings")


async def test_the_user_asks_for_the_drawn_booking_in_their_own_words():
    run = await run_episode("my_bookings", config=CONFIG, persona=PERSONA, language="Portuguese")

    assert run.row["conversation_status"] is True
    booking = json.loads(run.row["my_booking"])
    assert booking["expected"] == {"date": "2027-03-06", "time": "11:00", "people": 6}
    instructions = run.prompts[0]["messages"][0]["content"]
    assert "a visita guiada ao museu da cidade on 2027-03-06 at 11:00" in instructions
    opening = json.loads(run.row["conversation_messages"])[0]
    assert opening == {"role": "user", "content": ScriptedModel.CONTENT["user"]}  # the user model's own words
    assert booking["made"]["after_user_turns"] == 1  # the scripted assistant books on its first reply
    warnings = json.loads(run.row["simulation_outcome"])["warnings"]
    assert [w["kind"] for w in warnings] == [BookingProbe.placeholder_warning_kind]


def test_every_drawn_date_reads_two_ways():
    appointment = Appointment(id="museum_tour", what="the guided tour of the city museum", placeholder=True)
    for seed in range(200):
        _, month, day = draw_booking(appointment, random.Random(seed)).date.split("-")
        assert month != day and int(month) <= 12 and int(day) <= 12


async def test_a_month_first_reading_of_a_day_first_date_is_wrong():
    made = {"arguments": {"date": "2027-06-03", "time": "11:00", "people": 6}, "after_user_turns": 2}
    expected = {"date": "2027-03-06", "time": "11:00", "people": 6}
    result = await score({"my_booking": json.dumps({"expected": expected, "made": made})}, models={})

    assert result == {
        "my_booked": 1.0,
        "my_date_correct": 0.0,
        "my_time_correct": 1.0,
        "my_people_correct": 1.0,
        "my_asked_before_booking": 1.0,
    }


async def test_a_host_reproduces_a_standalone_run():
    await assert_hosted_parity("my_bookings", config=CONFIG, persona=PERSONA, language="Portuguese")
