# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The published conformance suite, run against this project's own code.

If ``usersim.testing`` describes a contract the built-ins do not keep, it is
wrong — either the check or the contract. Running it here is what stops it
from becoming aspirational documentation that fails for every extension
author who trusts it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import usersim.engine.generator  # noqa: F401  (populates the registry)
from usersim.engine.core import simulation
from usersim.engine.core.probes import BaseProbe, known_probes
from usersim.engine.evaluator import scorers
from usersim.engine.evaluator.generator import SCORER_FILLED_KEYS
from usersim.engine.evaluator.scorers import list_scorers
from usersim.testing import (
    _metadata_keys_written,
    assert_probe_conforms,
    assert_scorer_conforms,
    probe_conformance_problems,
    scorer_conformance_problems,
)


@pytest.mark.parametrize("probe", known_probes())
def test_shipped_probe_conforms(probe: str) -> None:
    assert_probe_conforms(probe)


def test_the_reserved_metadata_keys_are_the_ones_the_engine_core_writes() -> None:
    """Every key the engine core writes is reserved, so the conformance check covers it."""
    core = Path(simulation.__file__).parent
    written = set().union(*(_metadata_keys_written(path.read_text(encoding="utf-8")) for path in core.glob("*.py")))
    assert written == set(simulation.RESERVED_METADATA_KEYS)


# Defined at module level so their source can be read on every supported Python.
class _WritesEarlyStop(BaseProbe):
    label = "writes_early_stop"

    async def after_assistant_turn(self, models, state, assistant_response, cfg):
        state.metadata["early_stop"] = True
        return ""


class _SeedsJudgeRatings(BaseProbe):
    label = "seeds_judge_ratings"

    def seed_state_metadata(self, state):
        state.metadata.setdefault("user_judge_ratings", [])


class _WritesItsOwnKeys(BaseProbe):
    label = "writes_its_own_keys"

    def seed_state_metadata(self, state):
        state.metadata["acme_checks"] = []
        state.metadata.update({"acme_seen": True})
        state.metadata.setdefault("acme_turns", 0)


#: A row shaped like one the evaluator hands a scorer. Passing it makes the
#: check also score the row with every field the evaluator does not fill set
#: to None.
_SAMPLE_ROW = {
    "trajectory_id": "t0000000000000001",
    "conversation_messages": [
        {"role": "user", "content": "Hello, can you help me?"},
        {"role": "assistant", "content": "Of course. What do you need?"},
    ],
    "persona": {"first_name": "Ana", "last_name": "Silva"},
    "probe_variant": "default",
    "locale": "en_US",
    "language": "English",
    "conversation_language": "English",
    "simulation_outcome": {"status": "ok", "n_turns": 1},
}


@pytest.mark.parametrize("scorer", sorted(list_scorers()))
async def test_shipped_scorer_conforms(scorer: str) -> None:
    await assert_scorer_conforms(scorer)
    for family in known_probes():
        await assert_scorer_conforms(scorer, sample_row={**_SAMPLE_ROW, "probe_family": family})


class TestTheChecksCanFail:
    """A check that cannot fail proves nothing about the ones that pass."""

    async def test_unregistered_probe_is_reported(self) -> None:
        problems = probe_conformance_problems("not_a_probe")
        assert problems and "not registered" in problems[0]
        assert "usersim.probes" in problems[0]

    async def test_unset_label_is_reported(self) -> None:
        from usersim.engine.core.probes import BaseProbe

        class Unlabelled(BaseProbe):
            pass

        problems = probe_conformance_problems(Unlabelled)
        assert any("label" in p for p in problems)

    async def test_missing_overrides_are_reported(self) -> None:
        """A probe inheriting the abstract prompt methods registers fine and
        then raises once a run reaches it."""
        from usersim.engine.core.probes import BaseProbe

        class Hollow(BaseProbe):
            label = "hollow"

        problems = probe_conformance_problems(Hollow)
        assert any("get_user_system_prompt" in p for p in problems)

    async def test_synchronous_hook_override_is_reported(self) -> None:
        """The loop awaits these hooks, so a plain 'def' override raises
        partway through a conversation rather than at registration."""
        from usersim.engine.core.probes import BaseProbe

        class SyncHook(BaseProbe):
            label = "sync_hook"

            def is_capitulation_detected(self, state):  # type: ignore[override]
                return False

        problems = probe_conformance_problems(SyncHook)
        assert any("is_capitulation_detected" in p and "async def" in p for p in problems)

    async def test_a_relative_assets_root_is_reported(self) -> None:
        """A relative root resolves against whatever directory a run starts in."""
        from pathlib import Path

        from usersim.engine.core.probes import BaseProbe

        class RelativeRoot(BaseProbe):
            label = "relative_root"
            assets_root = Path("assets")

        problems = probe_conformance_problems(RelativeRoot)
        assert any("RelativeRoot.assets_root" in p and "relative path" in p for p in problems)

    @pytest.mark.parametrize(
        ("columns", "expected"),
        [
            ("my_score", "must be a tuple of column names"),
            (("my_score", ""), "must be a tuple of non-empty column names"),
            (("my_score", "trajectory_id"), "names ['trajectory_id']"),
        ],
        ids=["bare-string", "empty-name", "core-column"],
    )
    async def test_unusable_output_columns_are_reported(self, columns, expected) -> None:
        """Registration refuses these too; the check reports them on a class that is not registered."""
        from usersim.engine.core.probes import BaseProbe

        class Columns(BaseProbe):
            label = "columns_probe"
            output_columns = columns

        problems = probe_conformance_problems(Columns)
        assert any("Columns.output_columns " in p and expected in p for p in problems), problems

    async def test_a_placeholder_warning_kind_that_marks_no_placeholder_is_reported(self) -> None:
        from usersim.engine.core.probes import BaseProbe

        class Mislabelled(BaseProbe):
            label = "mislabelled_probe"
            placeholder_warning_kind = "acme_slow_tool"

        problems = probe_conformance_problems(Mislabelled)
        assert any("Mislabelled.placeholder_warning_kind must mark placeholder content" in p for p in problems), (
            problems
        )

    @pytest.mark.parametrize(
        ("attributes", "expected"),
        [
            ({"locale_bank_check": "sample.yaml"}, "PreflightProbe.locale_bank_check must be None or a function"),
            (
                {"bank_locale": lambda self, locale, probe_dir: locale},
                "PreflightProbe.bank_locale must be a classmethod",
            ),
            (
                {"authoring_guide": "README.md#locales"},
                "PreflightProbe.authoring_guide must be None or an https:// URL",
            ),
        ],
        ids=["check-not-callable", "bank-locale-not-a-classmethod", "relative-guide"],
    )
    async def test_unusable_preflight_declarations_are_reported(self, attributes, expected) -> None:
        """The pre-flight reads these on the class before any row runs, so a wrong one fails the whole run."""
        cls = type("PreflightProbe", (BaseProbe,), {"label": "preflight_probe", **attributes})

        problems = probe_conformance_problems(cls)
        assert any(expected in p for p in problems), problems

    async def test_usable_preflight_declarations_are_accepted(self) -> None:
        class Declared(BaseProbe):
            label = "declared_probe"
            locale_bank_check = staticmethod(lambda locale_dir: None)
            authoring_guide = "https://example.com/declared-probe/locales"

            @classmethod
            def bank_locale(cls, locale, probe_dir):
                return locale

        declarations = ("locale_bank_check", "bank_locale", "authoring_guide")
        assert not [p for p in probe_conformance_problems(Declared) if any(d in p for d in declarations)]

    @pytest.mark.parametrize(
        ("probe", "key"), [(_WritesEarlyStop, "early_stop"), (_SeedsJudgeRatings, "user_judge_ratings")]
    )
    async def test_writing_a_metadata_key_the_loop_owns_is_reported(self, probe, key) -> None:
        problems = probe_conformance_problems(probe)
        assert any(f"{probe.__qualname__} writes state.metadata[{key!r}]" in p for p in problems), problems

    async def test_a_probes_own_metadata_keys_are_accepted(self) -> None:
        assert not [p for p in probe_conformance_problems(_WritesItsOwnKeys) if "state.metadata" in p]

    async def test_async_hook_override_is_accepted(self) -> None:
        from usersim.engine.core.probes import BaseProbe

        class AsyncHook(BaseProbe):
            label = "async_hook"

            async def is_capitulation_detected(self, state):
                return False

        problems = probe_conformance_problems(AsyncHook)
        assert not any("is_capitulation_detected" in p for p in problems)

    async def test_unregistered_scorer_is_reported(self) -> None:
        problems = await scorer_conformance_problems("not_a_scorer")
        assert problems and "usersim.scorers" in problems[0]

    async def test_raising_scorer_is_reported(self, monkeypatch) -> None:
        """The evaluator calls scorers on rows missing any given field."""

        async def brittle(row, models):
            return {"score": row["definitely_absent"]}

        monkeypatch.setitem(scorers._REGISTRY, "brittle", brittle)
        problems = await scorer_conformance_problems("brittle")
        assert any("KeyError" in p for p in problems)

    async def test_non_dict_scorer_is_reported(self, monkeypatch) -> None:
        async def stringly(row, models):
            return "nope"

        monkeypatch.setitem(scorers._REGISTRY, "stringly", stringly)
        problems = await scorer_conformance_problems("stringly")
        assert any("expected a dict" in p for p in problems)

    async def test_scorer_failing_on_an_empty_cell_is_reported(self, monkeypatch) -> None:
        """A default covers an absent key, but an empty cell arrives present and None."""

        async def defaulted(row, models):
            return {"protocol": row.get("sub_protocol", "").lower()}

        monkeypatch.setitem(scorers._REGISTRY, "defaulted", defaulted)
        problems = await scorer_conformance_problems("defaulted", sample_row={"sub_protocol": "Triage"})
        assert any("AttributeError" in p and "sub_protocol" in p for p in problems)

    async def test_keys_the_evaluator_fills_are_never_emptied(self, monkeypatch) -> None:
        async def relies_on_filled_keys(row, models):
            return {
                "turns": len(row["conversation_messages"]),
                "persona_fields": len(row["persona"]),
                "family": row["probe_family"].lower(),
                "locale": row["locale"].lower(),
                "language": row["language"].lower(),
            }

        monkeypatch.setitem(scorers._REGISTRY, "relies_on_filled_keys", relies_on_filled_keys)
        sample = {
            "conversation_messages": [{"role": "assistant", "content": "Hello."}],
            "persona": {"first_name": "Ana"},
            "probe_family": "general_open_ended",
            "locale": "en_US",
            "language": "English",
        }
        assert sorted(sample) == sorted(SCORER_FILLED_KEYS)
        assert await scorer_conformance_problems("relies_on_filled_keys", sample_row=sample) == []

    def test_assert_form_lists_every_problem(self) -> None:
        with pytest.raises(AssertionError) as exc:
            assert_probe_conforms("not_a_probe")
        assert "does not conform" in str(exc.value)
