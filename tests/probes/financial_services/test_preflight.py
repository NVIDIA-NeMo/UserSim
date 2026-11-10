# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""financial_services before a run: a locale without a bank that loads is refused, with the next step."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_missing_locale_subtree_is_rejected() -> None:
    from usersim.cli._pipeline import verify_probe_assets
    from usersim.engine.errors import ConfigError

    with pytest.raises(ConfigError) as exc:
        verify_probe_assets({"financial_services": 1.0}, ["en_SG"], None)
    message = str(exc.value)
    assert "en_SG: no bank" in message
    # Naming the locales that can run turns a dead end into a next step.
    assert "en_US" in message.split("Locales that can run: ")[1].splitlines()[0]


def test_a_locale_whose_bank_does_not_load_is_rejected(finance_assets_without_fr_tasks: Path) -> None:
    """A locale folder that exists is not enough: without tasks, every row aborts."""
    from usersim.cli._pipeline import verify_probe_assets
    from usersim.engine.errors import ConfigError

    with pytest.raises(ConfigError) as exc:
        verify_probe_assets({"financial_services": 1.0}, ["fr_FR"], finance_assets_without_fr_tasks)
    message = str(exc.value)
    assert "fr_FR: its bank does not load" in message and "tasks.yaml" in message
    assert message.split("Locales that can run: ")[1].splitlines()[0] == "en_US"
    verify_probe_assets({"financial_services": 1.0}, ["en_US"], finance_assets_without_fr_tasks)


def test_india_variant_without_its_own_bank_uses_en_in() -> None:
    """A registered variant runs on its base locale's bank, as its probe and scorer resolve it."""
    from usersim.cli._pipeline import verify_probe_assets

    verify_probe_assets({"financial_services": 1.0}, ["bn_Beng_IN"], None)


def test_india_variant_is_rejected_when_its_base_bank_is_absent(tmp_path: Path) -> None:
    from usersim.cli._pipeline import verify_probe_assets
    from usersim.engine.errors import ConfigError

    (tmp_path / "financial_services" / "en_US").mkdir(parents=True)
    (tmp_path / "financial_services" / "en_US" / "bank.yaml").write_text("{}")
    with pytest.raises(ConfigError) as exc:
        verify_probe_assets({"financial_services": 1.0}, ["bn_Beng_IN"], tmp_path)
    assert "bn_Beng_IN" in str(exc.value)


def test_the_hindi_locales_run_finance_on_the_en_in_bank() -> None:
    """hi_Deva_IN ships a corpus without tasks and hi_Latn_IN no bank; both run on en_IN's."""
    from usersim.cli._pipeline import verify_probe_assets

    verify_probe_assets({"financial_services": 1.0}, ["hi_Deva_IN", "hi_Latn_IN"], None)


def test_the_hindi_locales_are_rejected_when_the_en_in_bank_is_absent(tmp_path: Path) -> None:
    from usersim.cli._pipeline import verify_probe_assets
    from usersim.engine.errors import ConfigError

    (tmp_path / "financial_services" / "en_US").mkdir(parents=True)
    (tmp_path / "financial_services" / "en_US" / "bank.yaml").write_text("{}")
    with pytest.raises(ConfigError) as exc:
        verify_probe_assets({"financial_services": 1.0}, ["hi_Deva_IN", "hi_Latn_IN"], tmp_path)
    assert "hi_Deva_IN (through en_IN): no bank; hi_Latn_IN (through en_IN): no bank" in str(exc.value)


def test_a_refusal_names_the_locales_that_run_on_another_bank() -> None:
    from usersim.cli._pipeline import verify_probe_assets
    from usersim.engine.errors import ConfigError

    with pytest.raises(ConfigError) as exc:
        verify_probe_assets({"financial_services": 1.0}, ["pt_BR"], None)
    can_run = str(exc.value).split("Locales that can run: ")[1].splitlines()[0]
    assert "en_US" in can_run
    assert can_run.endswith(", plus hi_Deva_IN, hi_Latn_IN and the India language variants through en_IN")


def test_a_locale_without_a_bank_says_how_to_ask_for_one() -> None:
    from usersim.cli._pipeline import verify_probe_assets
    from usersim.engine.errors import ConfigError

    with pytest.raises(ConfigError) as exc:
        verify_probe_assets({"financial_services": 1.0}, ["pt_BR"], None)
    assert str(exc.value).split("\n\n", 1)[1].splitlines() == [
        "Drop the probe from --probe-mix or the locale from the run, or point --assets-dir at a tree that has them.",
        "To ask for a locale, open a feature request: "
        "https://github.com/NVIDIA-NeMo/UserSim/issues/new?template=feature-request.yml",
        "To build a financial_services bank yourself, see https://github.com/NVIDIA-NeMo/UserSim/blob/main/"
        "src/usersim/engine/probes/financial_services/README.md#authoring-a-new-locale",
    ]


def test_the_probe_links_a_guide_to_building_a_bank() -> None:
    """The refusal for a locale without a bank links this guide, and the shared tests check the link resolves."""
    from usersim.engine.core.probes import resolve_probe

    assert resolve_probe("financial_services").authoring_guide
