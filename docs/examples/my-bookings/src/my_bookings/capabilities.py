# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from usersim.extension import CapabilityDefinition, EvidenceSource

CAPABILITIES = (
    CapabilityDefinition(
        id="my_booking_arguments",
        label="Bookings as users write them",
        description="Does the assistant book the date, time and party size the user meant, however they wrote them?",
        sources=(
            EvidenceSource(
                probe="my_bookings",
                scorer="my_bookings",
                axes=(
                    "my_bookings.date_match_rate",
                    "my_bookings.time_match_rate",
                    "my_bookings.people_match_rate",
                ),
            ),
        ),
        threshold=0.9,
        scale="rate",
        aggregation_policy="mean",
        next_action="Read the failing rows and compare the date the user wrote with the one the assistant booked.",
    ),
    CapabilityDefinition(
        id="my_booking_asks_first",
        label="Asking before booking",
        description="Does the assistant ask how many people are coming before it books?",
        sources=(EvidenceSource(probe="my_bookings", scorer="my_bookings", axes=("my_bookings.asked_first_rate",)),),
        threshold=0.8,
        scale="rate",
        aggregation_policy="mean",
        next_action="Read the rows booked on the first reply: the assistant guessed the party size.",
    ),
)

AXIS_DESCRIPTIONS = {
    "my_bookings.date_match_rate": "Share of bookings made for the date the user meant, however they wrote it.",
    "my_bookings.time_match_rate": "Share of bookings made for the time the user meant.",
    "my_bookings.people_match_rate": "Share of bookings made for the number of people the user gave.",
    "my_bookings.asked_first_rate": "Share of bookings the assistant made only after asking how many people.",
}
