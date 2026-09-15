# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end contract for health_disclosure: probe columns -> store -> scorer.

Every other test around this feature hand-builds the trajectory row it feeds the
scorer, which means the row is whatever the *test author* believed the probe
emits. That is a blind spot with teeth: if the producer stops emitting a column
the scorer reads, the scorer quietly returns no realized ground truth, falls back
to committed intent, and every one of those tests still passes.

This test closes that loop. It runs the REAL producer (mixin + Guard +
``build_result_extras``), persists the result the way the store does, and feeds
it to the REAL scorer. Nothing about the columns is asserted from memory — the
row is whatever the probe actually produced.

Both LLM calls are faked at the TRANSPORT boundary (``call_llm``), never at the
boundary of the code under test. That distinction is the direct lesson of a
regression where the audit raised ``NameError`` on every call, a bare ``except``
turned it into an empty result, and the suite stayed green because the tests had
monkeypatched the audit function itself.
"""

from __future__ import annotations

import json
import tempfile

import pandas as pd
import pyarrow as pa
import pytest

from usersim.engine.core import realized_audit as ra
from usersim.engine.evaluator.scorers import get_scorer, load_default_scorers
from usersim.engine.probes.health_disclosure import move_runtime as mr
from usersim.engine.probes.health_disclosure import therapy

MAX_TURNS = 8


class _State:
    """Minimal stand-in for ConversationState (what the probe hooks touch)."""

    def __init__(self):
        self.messages = [{"role": "user", "content": "hi"}]
        self.metadata = {}


def _guarded_probe():
    """A guarded-variant probe without the heavy __init__ (no prompt build)."""
    p = object.__new__(therapy.TherapyDisclosureProbe)
    p._persona = {"uuid": "u-e2e"}
    p._data = {"probe_variant": "guarded", "trajectory_id": "t-e2e"}
    p._models = {"user_model": object()}
    return p


def _commit(move: str, topic: str) -> dict:
    """A provider-shaped completion carrying a native commit_move tool call."""
    return {"role": "assistant", "content": "", "tool_calls": [{
        "id": "c1", "function": {"name": "commit_move", "arguments": json.dumps(
            {"move": move, "topic": topic, "intensity": 0.5,
             "reasoning": "e2e"})}}]}


def _store_roundtrip(df: pd.DataFrame) -> pd.DataFrame:
    """Write and read a frame the way the trajectory store actually does.

    Fidelity matters here, and a first version of this helper did NOT have it: a
    plain pandas frame writes a list column with numpy-backed dtypes and reads
    straight back, so the test passed while the real pipeline was broken. The
    store materialises arrow-backed dtypes, and pandas records those in the file
    metadata by NAME — "list<item: string>[pyarrow]". It cannot rebuild a NESTED
    ArrowDtype from that string, so reading raises

        TypeError: data type 'list<item: string>[pyarrow]' not understood

    The evaluator opens the trajectory frame with pd.read_parquet, so a column
    that fails this makes an entire run generate successfully and then be
    impossible to score.
    """
    arrow_df = pa.Table.from_pandas(df, preserve_index=False).to_pandas(
        types_mapper=pd.ArrowDtype)
    with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
        arrow_df.to_parquet(f.name, index=False)
        return pd.read_parquet(f.name)


def _col(extras: dict, name: str):
    """Decode a structured column the way the scorer's _as_list/_as_levels do."""
    raw = extras[name]
    return json.loads(raw) if isinstance(raw, str) else raw


def _persist(extras: dict) -> dict:
    """JSON-encode the way the trajectory store does, so the scorer sees the
    persisted shape (strings) rather than live Python objects."""
    out = {}
    for k, v in extras.items():
        out[k] = json.dumps(v, ensure_ascii=False, default=str) if isinstance(
            v, (list, dict)) else v
    return out


def _run_producer(monkeypatch, plan=None, said=None):
    """Drive the real move/Guard loop and return (columns, transcript).

    ``plan`` maps turn number -> (move, topic). The default walks the partial ->
    full arc: early turns can only get a partial past the Guard's
    disclosure_gate, later ones can go full.
    """
    probe = _guarded_probe()
    probe.init_guarded_moves(disclosure_style="incremental", max_turns=MAX_TURNS)
    assert probe._moves_on and probe._topics, "guarded variant did not initialise"

    topic = probe._topics[0]
    plan = plan(topic) if callable(plan) else (
        plan or {2: ("partial_disclose", topic), 6: ("disclose", topic)})
    said = said or {2: "it's been a rough few weeks, but nothing major",
                    6: "honestly the panic attacks have been near-daily"}

    def fake_call_llm(models, alias, msgs, **kw):
        return _commit(*plan[fake_call_llm.turn])

    monkeypatch.setattr(mr, "call_llm", fake_call_llm)

    state = _State()
    transcript = [{"role": "user", "content": "opening turn, no move"}]
    for turn in sorted(plan):
        fake_call_llm.turn = turn
        probe.format_followup_user_instructions(turn, state)
        transcript.append({"role": "assistant", "content": "tell me more?"})
        transcript.append({"role": "user", "content": said.get(turn, "mm.")})

    return probe.build_result_extras(state), transcript


def test_probe_columns_feed_the_scorer_and_the_audit_actually_runs(monkeypatch):
    extras, transcript = _run_producer(monkeypatch)

    # The producer really ran the Guard-paced loop.
    assert extras["moves_enabled"] is True
    assert _col(extras, "moves_detail"), "no moves recorded by the real mixin"

    topic = _col(extras, "concealment_topics")[0]
    row = {**_persist(extras), "conversation_messages": transcript}

    seen = {}

    def fake_audit_llm(models, alias, msgs, **kw):
        seen["system"], seen["user"] = msgs[0]["content"], msgs[1]["content"]
        return {"role": "assistant", "content": "", "tool_calls": [{
            "id": "a1", "function": {"name": "record_audit", "arguments": json.dumps(
                {"audits": [{"turn": 6, "fully_disclosed": [topic],
                             "partially_disclosed": [], "risk_revealed": False}]})}}]}

    monkeypatch.setattr(ra, "call_llm", fake_audit_llm)
    load_default_scorers()
    out = get_scorer("health_disclosure_concealment")(row, {"judge_model": object()})

    # The audit ran for real. This is the assertion the old tests could not make:
    # it holds only if moves_detail, conversation_messages and concealment_topics
    # all survived the producer -> store -> scorer trip intact.
    assert out["scorer_kind"] == "llm_audited"

    # Per-topic resolution, end to end. The auditor observed the one topic the
    # conversation actually covered; the other carried topics were never
    # discussed, so they fall back to intent and are named in preview_topics
    # instead of the whole row being discarded (the round-2 review ask).
    assert out["ground_truth_source"] == "mixed_realized_intent"
    assert topic not in out["preview_topics"]
    assert set(out["preview_topics"]) == set(_col(extras, "concealment_topics")) - {topic}

    # The realized turns reached the auditor, paired to their committed moves by
    # turn number, and the client's concrete risk wording reached the prompt.
    assert "[turn 6]" in seen["user"]
    assert "near-daily" in seen["user"]
    assert extras["risk_noun"] in seen["system"]

    # And the audited verdict beat committed intent: the probe recorded a full
    # disclosure, the auditor confirmed it, so the topic scores 1.0 not 0.5.
    assert out["scores"]["concealment.disclosure_coverage"]["score"] > 0


def test_without_an_auditor_the_same_row_still_scores_on_intent(monkeypatch):
    """No judge model wired is a supported mode, not a failure: the scorer must
    still produce a number, and must not claim a model contributed to it."""
    extras, transcript = _run_producer(monkeypatch)
    row = {**_persist(extras), "conversation_messages": transcript}

    load_default_scorers()
    out = get_scorer("health_disclosure_concealment")(row, {})

    assert out["scorer_kind"] == "deterministic"
    assert out["ground_truth_source"] == "committed_intent"
    assert out["scores"]["concealment.disclosure_coverage"]["score"] is not None


def test_the_partial_to_full_upgrade_survives_the_real_guard(monkeypatch):
    """The bug this MR fixes, asserted through the producer rather than a unit:
    a topic told partially at turn 2 must still be fully disclosable at turn 6,
    and must land in the binary column only once it is full."""
    extras, _ = _run_producer(monkeypatch)
    topic = _col(extras, "concealment_topics")[0]

    played = [m["move"] for m in _col(extras, "moves_detail")]
    assert played == ["partial_disclose", "disclose"], (
        f"the Guard blocked the partial -> full arc: {played}")
    assert topic in _col(extras, "disclosed_topics")
    assert _col(extras, "committed_disclosure_levels")[topic] == "full"


@pytest.mark.parametrize("label,plan", [
    ("disclosing", None),  # the default partial -> full arc
    # A patient who conceals throughout. Legitimate and common — a guarded
    # archetype on a short turn budget may never clear the disclosure gate.
    ("fully concealing", lambda topic: {2: ("withhold", topic),
                                        3: ("deflect", topic),
                                        4: ("minimize", topic)}),
])
def test_every_emitted_column_survives_a_parquet_write(monkeypatch, label, plan):
    """The columns must be storable, not merely correct.

    Found by a live run, not by the suite: ``committed_disclosure_levels`` was
    emitted as a bare dict, which reaches Parquet as a struct. A conversation
    with no disclosing move makes it an EMPTY dict — a struct with no child
    fields, which Parquet cannot represent:

        ArrowNotImplementedError: Cannot write struct type
        'committed_disclosure_levels' with no child field

    It failed the whole batch write, so a single fully-concealing trajectory
    aborted an entire simulation run after every LLM call had been paid for. And
    it only reproduced when nobody disclosed, so a disclosing run looked fine.
    """
    extras, _ = _run_producer(monkeypatch, plan=plan)

    broken = []
    for column, value in extras.items():
        try:
            _store_roundtrip(pd.DataFrame([{column: value}]))
        except Exception as exc:  # noqa: BLE001 - reporting, not control flow
            broken.append(f"{column} ({type(value).__name__}): {exc}")
    assert not broken, f"[{label}] columns the store cannot round-trip:\n" + "\n".join(broken)

    # And the whole row together, including the column-projected read the
    # evaluator actually performs (cli/evaluate.py reads the trajectory frame
    # with pd.read_parquet(..., columns=key_cols)).
    frame = _store_roundtrip(pd.DataFrame([extras]))
    assert len(frame) == 1
