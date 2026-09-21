# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Where a simulation run executes.

Running in-process is one choice among several: a large sweep may belong on a
scheduler that takes a job, runs it on its own compute, and hands back results
later. That difference is narrow enough to sit behind a small protocol, and
keeping it behind one is what lets a cluster integration live in a separate
distribution instead of a fork of this project.

The protocol is deliberately three calls. ``submit`` starts work and returns a
handle, ``poll`` reports on it, and ``fetch`` materializes the output. A
backend that runs locally collapses all three into "already done", which is
the honest description of what in-process execution is.

**Credentials are passed to ``submit``, never stored on the backend.** A
backend built once and reused across runs that held provider keys as state
would push every integration toward writing those keys into a config file to
construct it. Passing them per call removes the reason to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Protocol, runtime_checkable

#: Terminal and non-terminal states a submitted job can be in.
JobState = Literal["pending", "running", "succeeded", "failed"]


@dataclass(frozen=True)
class JobSpec:
    """What to run, independent of where it runs.

    ``callable_`` is the unit of work. A local backend calls it directly; a
    remote backend uses ``name`` and ``params`` to reconstruct the same work
    on its own compute, since a live Python callable does not cross a
    scheduler boundary.
    """

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    callable_: Callable[[], int] | None = None


@dataclass(frozen=True)
class JobHandle:
    """Backend-specific reference to submitted work."""

    backend: str
    job_id: str
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JobStatus:
    """Where a job is, and why it stopped if it did.

    ``error`` carries the live exception for backends that ran the work in
    this process. A caller can re-raise it and keep the exact failure
    semantics it would have had without a backend in the path; one that ran
    the work elsewhere leaves it ``None`` and reports through ``message``.
    """

    state: JobState
    exit_code: int | None = None
    message: str = ""
    error: BaseException | None = None

    @property
    def done(self) -> bool:
        return self.state in ("succeeded", "failed")


@runtime_checkable
class ExecutionBackend(Protocol):
    """Somewhere a :class:`JobSpec` can run."""

    name: str

    def submit(
        self,
        spec: JobSpec,
        *,
        credentials: Mapping[str, str] | None = None,
    ) -> JobHandle:
        """Start the work and return a handle to it."""
        ...

    def poll(self, handle: JobHandle) -> JobStatus:
        """Report the current state of submitted work."""
        ...

    def fetch(self, handle: JobHandle, dest: Path) -> Path:
        """Materialize the job's output under ``dest`` and return the path."""
        ...


class LocalBackend:
    """Runs the job in this process, before ``submit`` returns.

    Output is written where the job itself writes it, so ``fetch`` reports
    that location rather than copying anything.
    """

    name = "local"

    def __init__(self) -> None:
        self._results: dict[str, JobStatus] = {}
        self._counter = 0

    def submit(
        self,
        spec: JobSpec,
        *,
        credentials: Mapping[str, str] | None = None,
    ) -> JobHandle:
        self._counter += 1
        job_id = f"{spec.name}-{self._counter}"
        if spec.callable_ is None:
            self._results[job_id] = JobStatus(
                state="failed",
                message=f"JobSpec {spec.name!r} has no callable to run locally",
            )
            return JobHandle(backend=self.name, job_id=job_id)
        try:
            code = int(spec.callable_() or 0)
        except Exception as exc:
            # Recorded rather than raised, so a caller written against the
            # protocol learns the outcome the same way for every backend.
            # The exception itself is kept so the CLI can re-raise it and
            # preserve the exit code its error handling already assigns.
            self._results[job_id] = JobStatus(
                state="failed",
                exit_code=1,
                message=str(exc),
                error=exc,
            )
        else:
            self._results[job_id] = JobStatus(
                state="succeeded" if code == 0 else "failed",
                exit_code=code,
            )
        return JobHandle(backend=self.name, job_id=job_id)

    def poll(self, handle: JobHandle) -> JobStatus:
        if handle.job_id not in self._results:
            raise KeyError(f"unknown job {handle.job_id!r} for backend {self.name!r}")
        return self._results[handle.job_id]

    def fetch(self, handle: JobHandle, dest: Path) -> Path:
        return Path(dest)


def _builtin_backends() -> dict[str, Any]:
    return {LocalBackend.name: LocalBackend}


def known_backends() -> tuple[str, ...]:
    """Names of every available backend, built-in and contributed."""
    from usersim.engine.core._extensions import BACKENDS, merge_extensions

    return tuple(sorted(merge_extensions(BACKENDS, _builtin_backends(), what="backend")))


def get_backend(name: str) -> ExecutionBackend:
    """Construct a backend by name.

    An entry point may resolve to a class or to a zero-argument factory; both
    are called to produce the instance.
    """
    from usersim.engine.core._extensions import BACKENDS, merge_extensions

    available = merge_extensions(BACKENDS, _builtin_backends(), what="backend")
    if name not in available:
        raise KeyError(f"unknown execution backend {name!r}; available: {', '.join(sorted(available))}")
    backend = available[name]
    return backend() if callable(backend) else backend
