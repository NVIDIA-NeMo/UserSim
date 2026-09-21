# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

from usersim.engine.core.backends import JobHandle, JobStatus


class FixtureBackend:
    name = "fixture_backend"

    def submit(self, spec, *, credentials=None):
        return JobHandle(backend=self.name, job_id="fixture-1")

    def poll(self, handle):
        return JobStatus(state="succeeded", exit_code=0)

    def fetch(self, handle, dest):
        return dest
