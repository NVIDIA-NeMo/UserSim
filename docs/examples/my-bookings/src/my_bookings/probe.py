# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import random
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

from usersim.extension import (
    BankBackedProbe,
    PreviewField,
    ToolExecutionMixin,
    format_persona_for_prompt,
    get_prompt,
    language_instruction,
    load_prompt_file,
    persona_uuid,
    register_probe,
    render_prompt,
)

BOOK_APPOINTMENT = {
    "type": "function",
    "function": {
        "name": "book_appointment",
        "description": "Book an appointment.",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "time": {"type": "string", "description": "HH:MM, 24-hour"},
                "people": {"type": "integer"},
            },
            "required": ["date", "time", "people"],
        },
    },
}


@dataclass(frozen=True)
class Appointment:
    id: str
    what: str
    placeholder: bool


@dataclass(frozen=True)
class BookingBank:
    bank_id: str
    bank_version: str
    appointments: tuple[Appointment, ...]


@dataclass(frozen=True)
class Booking:
    id: str
    what: str
    date: str
    time: str
    people: int
    placeholder: bool


def read_booking_bank(locale_dir: Path) -> BookingBank:
    doc = yaml.safe_load((locale_dir / "bookings.yaml").read_text(encoding="utf-8"))
    return BookingBank(doc["bank_id"], doc["bank_version"], tuple(Appointment(**a) for a in doc["appointments"]))


@cache
def load_booking_bank(locale: str) -> BookingBank:
    return read_booking_bank(BookingProbe.assets_dir() / locale)


def draw_booking(appointment: Appointment, rng: random.Random) -> Booking:
    """A date, a time and a number of people for ``appointment``.

    The day and the month are different numbers no greater than 12, so the date
    reads as two different dates, day first and month first.
    """
    month = rng.randint(1, 12)
    day = rng.choice([d for d in range(1, 13) if d != month])
    return Booking(
        id=appointment.id,
        what=appointment.what,
        date=f"2027-{month:02d}-{day:02d}",
        time=f"{rng.randint(9, 17):02d}:{rng.choice(('00', '30'))}",
        people=rng.randint(2, 6),
        placeholder=appointment.placeholder,
    )


@register_probe
class BookingProbe(ToolExecutionMixin, BankBackedProbe):
    label = "my_bookings"
    description = (
        "Tests whether an assistant books what the user means: a date, a time and a number of people, "
        "written the way people write them in the user's language."
    )
    preview_fields = (PreviewField("my_booking_id", "Appointment"),)
    default_scorers = ("my_bookings",)
    assets_root = Path(__file__).parent / "assets"
    bank_loader = staticmethod(load_booking_bank)
    placeholder_warning_kind = "my_used_placeholder_booking"
    output_columns = ("my_booking_id", "my_booking")

    @classmethod
    def check_locale(cls, locale_dir: Path) -> None:
        read_booking_bank(locale_dir)

    @classmethod
    def asset_problems(cls) -> list[str]:
        problems = super().asset_problems()
        if "user_system_prompt" not in load_prompt_file(cls.label):
            problems.append("prompts.yaml has no user_system_prompt")
        return problems

    def derive_task(self, persona, bank, *, cfg):
        rng = random.Random(f"{persona_uuid(persona)}:{bank.bank_version}:{cfg.random_seed}")
        return draw_booking(rng.choice(bank.appointments), rng)

    def get_user_system_prompt(self) -> str:
        task = self._task
        prompt = render_prompt(
            get_prompt(self.label, "user_system_prompt"),
            self._data,
            persona=format_persona_for_prompt(self._persona),
            what=task.what,
            date=task.date,
            time=task.time,
            people=task.people,
        )
        return f"{prompt}\n\n{language_instruction(self._language, self._locale)}"

    def get_assistant_system_prompt(self) -> str:
        return ""

    def get_tools_for_assistant(self) -> list:
        return [BOOK_APPOINTMENT]

    def allow_early_stop_at_turn(self, turn_idx, state) -> bool:
        return "my_booking" in state.metadata

    async def execute_tool_call(self, name, args, tc, state, models, *, turn_idx, call_idx) -> str:
        if "my_booking" not in state.metadata:
            user_turns = sum(1 for message in state.messages if message.get("role") == "user")
            state.metadata["my_booking"] = {"arguments": args, "after_user_turns": user_turns}
        return json.dumps({"status": "booked"})

    def build_result_extras(self, state) -> dict:
        task = self._task
        booking = {
            "expected": {"date": task.date, "time": task.time, "people": task.people},
            "made": state.metadata.get("my_booking"),
        }
        return {
            **super().build_result_extras(state),
            "my_booking_id": task.id,
            "my_booking": json.dumps(booking, ensure_ascii=False),
        }
