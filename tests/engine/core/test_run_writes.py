# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runs are written under one lock per trajectory root, and appear whole.

The invariants: a run becomes visible only with its record and its first rows
in place, two writers never share a run, nothing visible is ever deleted, and
plain ``simulate`` never writes into a run of another kind.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import re
import subprocess
import sys
import textwrap
import threading
import time
import uuid
from pathlib import Path

import pandas as pd
import pytest

from usersim.engine.core import storage
from usersim.engine.core.storage import (
    LEGACY_RUN_ID,
    RUN_KIND_SAMPLED,
    RUN_RECORD_FILENAME,
    existing_trajectory_ids,
    latest_run_where,
    list_runs,
    publish_new_run,
    read_partitioned_dataset,
    read_run_record,
    run_started_at,
    run_write_lock,
    sampled_run_to_resume,
    save_sampled_rows,
    write_locale_partition,
)

#: A kind other than sampled rows, as a later kind of run would record.
_OTHER_KIND = "other"


def _rows(*ids: str, locale: str = "en_US") -> pd.DataFrame:
    return pd.DataFrame(
        {"trajectory_id": list(ids), "locale": [locale] * len(ids), "probe_family": ["general"] * len(ids)}
    )


def _publish(root: Path, *ids: str, kind: str = RUN_KIND_SAMPLED, now: float = 1_800_000_000.0) -> str:
    with run_write_lock(root):
        return publish_new_run(_rows(*ids), root, locale="en_US", record={"kind": kind}, clock=lambda: now)


def test_a_new_run_appears_with_its_record_and_rows(tmp_path):
    run_id = _publish(tmp_path, "a", "b")

    assert list_runs(tmp_path) == [run_id]
    assert read_run_record(tmp_path, run_id) == {"kind": RUN_KIND_SAMPLED}
    assert existing_trajectory_ids(tmp_path, run=run_id) == {"a", "b"}
    assert not list(tmp_path.glob(".claim-*"))


def test_a_taken_second_waits_for_the_next_one(tmp_path):
    taken = _publish(tmp_path, "a", now=1_800_000_000.0)
    ticks = iter([1_800_000_000.2, 1_800_000_000.2, 1_800_000_001.0])
    waits: list[float] = []

    with run_write_lock(tmp_path):
        run_id = publish_new_run(
            _rows("b"),
            tmp_path,
            locale="en_US",
            record={"kind": RUN_KIND_SAMPLED},
            clock=lambda: next(ticks),
            sleep=waits.append,
        )

    assert (taken, run_id) == ("1800000000", "1800000001")
    assert waits == [pytest.approx(0.8)]
    assert existing_trajectory_ids(tmp_path, run=taken) == {"a"}


def test_a_run_is_never_published_empty(tmp_path):
    with run_write_lock(tmp_path), pytest.raises(ValueError, match="never made empty"):
        publish_new_run(_rows(), tmp_path, locale="en_US", record={"kind": RUN_KIND_SAMPLED})

    assert list_runs(tmp_path) == []
    assert not list(tmp_path.glob(".claim-*"))


def test_a_failed_write_leaves_nothing_visible(tmp_path):
    with run_write_lock(tmp_path), pytest.raises(ValueError):
        publish_new_run(pd.DataFrame({"trajectory_id": ["a"]}), tmp_path, locale="en_US", record={})

    assert list_runs(tmp_path) == []
    assert not list(tmp_path.glob(".claim-*"))


#: Two PID scopes on one boot: this process's, and a container's with its own PID namespace.
_SCOPE = "0123456789abcdef0123456789abcdef.4026531836"
_OTHER_NAMESPACE = "0123456789abcdef0123456789abcdef.4026532001"


def _claim(root: Path, *, scope: str, pid: int, age_seconds: float) -> Path:
    """A claim folder made by ``pid`` in ``scope``, last modified ``age_seconds`` ago."""
    claim = root / f".claim-{scope}-{pid}-{uuid.uuid4().hex}"
    claim.mkdir()
    (claim / RUN_RECORD_FILENAME).write_text("{}")
    then = time.time() - age_seconds
    os.utime(claim, (then, then))
    return claim


def _dead_pid() -> int:
    finished = subprocess.Popen([sys.executable, "-c", "pass"])
    finished.wait()
    return finished.pid


def test_the_pid_scope_is_this_boot_and_pid_namespace_where_linux_shows_them():
    scope = storage._pid_scope()
    if Path("/proc/self/ns/pid").exists() and Path("/proc/sys/kernel/random/boot_id").exists():
        assert re.fullmatch(r"[0-9a-f]{32}\.\d+", scope or "")
    else:
        assert scope is None


@pytest.mark.skipif(os.name == "nt", reason="a PID is only probed where os.kill can probe one")
def test_the_lock_clears_only_a_claim_whose_writer_died_in_this_pid_scope(tmp_path, monkeypatch):
    """A slow or paused writer's claim survives, and so does one whose PID this process cannot see."""
    monkeypatch.setattr(storage, "_pid_scope", lambda: _SCOPE)
    two_hours = 2 * 3600
    abandoned = _claim(tmp_path, scope=_SCOPE, pid=_dead_pid(), age_seconds=two_hours)
    just_left = _claim(tmp_path, scope=_SCOPE, pid=_dead_pid(), age_seconds=0)
    paused = _claim(tmp_path, scope=_SCOPE, pid=os.getpid(), age_seconds=two_hours)
    other_namespace = _claim(tmp_path, scope=_OTHER_NAMESPACE, pid=_dead_pid(), age_seconds=two_hours)

    with run_write_lock(tmp_path):
        pass

    assert not abandoned.exists()
    assert just_left.exists() and paused.exists() and other_namespace.exists()


def test_where_the_pid_scope_is_unknown_no_claim_is_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "_pid_scope", lambda: None)
    abandoned = _claim(tmp_path, scope=_SCOPE, pid=_dead_pid(), age_seconds=2 * 3600)

    with run_write_lock(tmp_path):
        pass

    assert abandoned.exists()


def test_a_new_claim_records_its_pid_scope_and_process_when_known(tmp_path, monkeypatch):
    seen = []
    real_rename = os.rename

    def spy(src, dst):
        seen.append(Path(src).name)
        return real_rename(src, dst)

    monkeypatch.setattr(os, "rename", spy)
    monkeypatch.setattr(storage, "_pid_scope", lambda: _SCOPE)
    _publish(tmp_path, "a", now=1_700_000_000.0)
    monkeypatch.setattr(storage, "_pid_scope", lambda: None)
    _publish(tmp_path, "b", now=1_700_000_100.0)

    known, unknown = seen
    assert storage._claim_owner(known) == (_SCOPE, os.getpid())
    assert storage._claim_owner(unknown) is None


def test_a_part_file_appears_whole_or_not_at_all(tmp_path, monkeypatch):
    """A reader never meets a part file still being written, and a failed write leaves nothing behind."""
    import pyarrow.parquet as pq

    real_write = pq.write_table

    def half_written(table, where, **kwargs):
        Path(where).write_bytes(b"PAR1 not yet a parquet file")
        raise OSError("disk full")

    monkeypatch.setattr(pq, "write_table", half_written)
    with pytest.raises(OSError, match="disk full"):
        write_locale_partition(_rows("a"), tmp_path, locale="en_US", run_id="1700000000")
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []

    monkeypatch.setattr(pq, "write_table", real_write)
    previous = os.umask(0o022)
    try:
        write_locale_partition(_rows("b"), tmp_path, locale="en_US", run_id="1700000000")
    finally:
        os.umask(previous)
    (part,) = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert not part.name.startswith(".") and part.stat().st_mode & 0o777 == 0o644
    assert existing_trajectory_ids(tmp_path, run="1700000000") == {"b"}


def test_the_lock_excludes_another_process(tmp_path):
    """A second process waits until the first releases the lock."""
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import time
                from usersim.engine.core.storage import run_write_lock
                with run_write_lock({str(tmp_path)!r}):
                    print("locked", flush=True)
                    time.sleep(1.0)
                """
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "locked"
        started = time.monotonic()
        with run_write_lock(tmp_path):
            waited = time.monotonic() - started
    finally:
        holder.wait(timeout=30)

    assert waited > 0.5


def test_runs_are_searched_newest_first_and_legacy_last(tmp_path):
    (tmp_path / "locale=en_US").mkdir()
    older = _publish(tmp_path, "a", kind=RUN_KIND_SAMPLED, now=1_700_000_000.0)
    newer = _publish(tmp_path, "b", kind=_OTHER_KIND, now=1_800_000_000.0)

    assert latest_run_where(tmp_path, lambda run, record: True) == newer
    assert latest_run_where(tmp_path, lambda run, record: record and record["kind"] == RUN_KIND_SAMPLED) == older


def test_a_run_started_when_its_invocation_did(tmp_path):
    with run_write_lock(tmp_path):
        run_id = publish_new_run(
            _rows("a"),
            tmp_path,
            locale="en_US",
            record={"kind": RUN_KIND_SAMPLED, "started_at": 1_700_000_000.4},
            clock=lambda: 1_700_000_100.0,
        )
    (tmp_path / "run=1600000000").mkdir()

    assert run_id == "1700000100"
    assert run_started_at(tmp_path, run_id) == 1_700_000_000
    assert run_started_at(tmp_path, "1600000000") == 1_600_000_000
    assert run_started_at(tmp_path, LEGACY_RUN_ID) is None


# ---------------------------------------------------------------------------
# Plain ``simulate``
# ---------------------------------------------------------------------------


def test_plain_resume_never_reopens_a_run_of_another_kind(tmp_path):
    sampled = _publish(tmp_path, "a", kind=RUN_KIND_SAMPLED, now=1_700_000_000.0)
    _publish(tmp_path, "b", kind=_OTHER_KIND, now=1_800_000_000.0)

    assert sampled_run_to_resume(tmp_path) == sampled


def test_plain_resume_reopens_a_run_made_before_runs_had_records(tmp_path):
    _publish(tmp_path, "a", kind=_OTHER_KIND, now=1_800_000_000.0)
    old = tmp_path / "run=1700000000"
    old.mkdir()

    assert sampled_run_to_resume(tmp_path) == "1700000000"


def test_a_fresh_plain_run_never_joins_a_run_started_in_the_same_second(tmp_path):
    other = _publish(tmp_path, "s", kind=_OTHER_KIND, now=time.time())

    run_id, written = save_sampled_rows(_rows("p"), tmp_path, locale="en_US", run_id=None, started_at=time.time())

    assert written == 1 and run_id != other
    assert existing_trajectory_ids(tmp_path, run=other) == {"s"}
    assert read_run_record(tmp_path, run_id)["kind"] == RUN_KIND_SAMPLED


def test_rows_saved_in_the_run_since_it_was_chosen_are_not_saved_twice(tmp_path):
    """Whoever saved them: a resume, or the writer that made the run, as with ``--no-skip-existing``."""
    run_id = _publish(tmp_path, "a", kind=RUN_KIND_SAMPLED)

    _, written = save_sampled_rows(_rows("a", "b"), tmp_path, locale="en_US", run_id=run_id, started_at=0.0)

    assert written == 1
    assert sorted(read_partitioned_dataset(tmp_path, run=run_id)["trajectory_id"]) == ["a", "b"]


def test_a_fresh_plain_run_records_when_it_started(tmp_path):
    run_id, _ = save_sampled_rows(_rows("a"), tmp_path, locale="en_US", run_id=None, started_at=1_700_000_000.0)

    assert json.loads((tmp_path / f"run={run_id}" / RUN_RECORD_FILENAME).read_text()) == {
        "kind": RUN_KIND_SAMPLED,
        "started_at": 1_700_000_000.0,
    }


def test_sampled_rows_never_go_into_a_run_of_another_kind(tmp_path):
    other = _publish(tmp_path, "s", kind=_OTHER_KIND)

    with pytest.raises(ValueError, match="run of their own"):
        save_sampled_rows(_rows("p"), tmp_path, locale="en_US", run_id=other, started_at=0.0)
    assert existing_trajectory_ids(tmp_path, run=other) == {"s"}


def test_a_new_run_gets_the_usual_permissions(tmp_path):
    """Others sharing the root can read a run, as they could before runs were published."""
    previous = os.umask(0o022)
    try:
        run_id = _publish(tmp_path, "a")
    finally:
        os.umask(previous)

    assert (tmp_path / f"run={run_id}").stat().st_mode & 0o777 == 0o755


def test_the_lock_excludes_another_thread(tmp_path):
    """Two threads of one process exclude each other, as two processes do."""
    held = threading.Event()

    def hold():
        with run_write_lock(tmp_path):
            held.set()
            time.sleep(1.0)

    holder = threading.Thread(target=hold)
    holder.start()
    try:
        assert held.wait(timeout=30)
        started = time.monotonic()
        with run_write_lock(tmp_path):
            waited = time.monotonic() - started
    finally:
        holder.join(timeout=30)

    assert waited > 0.5


def _without_flock(monkeypatch, code: int = errno.ENOSYS) -> None:
    """Make ``flock`` fail as it does on a filesystem without usable locks."""
    fcntl = pytest.importorskip("fcntl")

    def unsupported(*args, **kwargs):
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(fcntl, "flock", unsupported)


@pytest.mark.parametrize(
    "code",
    [errno.ENOSYS, errno.ENOLCK, errno.EOPNOTSUPP, errno.ENOTSUP],
    ids=["ENOSYS", "ENOLCK", "EOPNOTSUPP", "ENOTSUP"],
)
def test_where_flock_is_unsupported_runs_are_still_written(tmp_path, monkeypatch, caplog, recwarn, code):
    """Written without the shared lock, with one warning per root, and never through a soft lock.

    Lustre without ``flock`` fails with ``ENOSYS``, NFS without its lock
    service with ``ENOLCK``, and some network mounts with ``EOPNOTSUPP``.
    """
    _without_flock(monkeypatch, code)

    with caplog.at_level(logging.WARNING, logger="usersim.engine"):
        run_id, _ = save_sampled_rows(_rows("a"), tmp_path, locale="en_US", run_id=None, started_at=0.0)
        save_sampled_rows(_rows("b"), tmp_path, locale="en_US", run_id=run_id, started_at=0.0)

    assert existing_trajectory_ids(tmp_path, run=run_id) == {"a", "b"}
    assert sum("not supported" in record.getMessage() for record in caplog.records) == 1
    assert not [warning for warning in recwarn if "SoftFileLock" in str(warning.message)]


def test_where_flock_is_unsupported_the_threads_of_a_process_still_take_turns(tmp_path, monkeypatch):
    _without_flock(monkeypatch)
    held = threading.Event()

    def hold():
        with run_write_lock(tmp_path):
            held.set()
            time.sleep(1.0)

    holder = threading.Thread(target=hold)
    holder.start()
    try:
        assert held.wait(timeout=30)
        started = time.monotonic()
        with run_write_lock(tmp_path):
            waited = time.monotonic() - started
    finally:
        holder.join(timeout=30)

    assert waited > 0.5
