# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""asset_gen — offline Data Designer generation of financial_services assets.

Repo-root package (parallels ``reporting/`` and ``selection/``) that authors the
committed ``assets/financial_services/<locale>/`` banks the runtime probe/scorer
consume. It imports the plugin (for the shared gold-derivation resolver + bank
schema) but the plugin never imports it — the dependency arrow is one-way.

Layout:

- ``asset_gen/financial_services/spec.py`` — the region_spec contract as Pydantic
  models (the shared vocabulary + validation).
- ``asset_gen/financial_services/pipeline.py`` — the whole Data Designer flow:
  schemas -> prompt -> seed plan -> column DAG -> explode -> serialize.
- ``asset_gen/financial_services/region_spec/*.yaml`` — the committed structured
  seed per region (the generation input; data, not code).

See the probe design doc:
``src/usersim/engine/probes/financial_services/README.md``.
"""
