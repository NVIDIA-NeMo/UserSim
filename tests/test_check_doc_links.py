# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``scripts/check_doc_links.py`` resolves links against the checkout."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_doc_links.py"
_spec = importlib.util.spec_from_file_location("check_doc_links", _SCRIPT)
check_doc_links = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_doc_links)


def _broken(doc: Path, text: str) -> list[str]:
    doc.write_text(text)
    return [target for _, target in check_doc_links.broken_links(doc)]


def test_links_into_the_main_branch_resolve_against_the_checkout(tmp_path: Path) -> None:
    gone_page = "https://github.com/NVIDIA-NeMo/UserSim/blob/main/docs/no-such-page.md"
    gone_image = "https://raw.githubusercontent.com/NVIDIA-NeMo/UserSim/main/docs/images/no-such.png"
    text = (
        "[here](https://github.com/NVIDIA-NeMo/UserSim/blob/main/README.md#setup)\n"
        "[dir](https://github.com/NVIDIA-NeMo/UserSim/tree/main/templates/probe/)\n"
        f"[gone]({gone_page})\n"
        f'<img src="{gone_image}">\n'
        "[elsewhere](https://example.com/anything)\n"
    )
    assert _broken(tmp_path / "doc.md", text) == [gone_page, gone_image]


def test_relative_links_resolve_against_the_file(tmp_path: Path) -> None:
    (tmp_path / "there.md").write_text("")
    assert _broken(tmp_path / "doc.md", "[ok](there.md)\n[gone](missing.md#anchor)\n") == ["missing.md#anchor"]


def test_tracked_markdown_preserves_spaces_and_private_scope(tmp_path: Path, monkeypatch) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for name in ["guide with spaces.md", "_private/notes.md"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("documentation\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    monkeypatch.setattr(check_doc_links, "REPO", tmp_path)
    assert check_doc_links.tracked_markdown() == [tmp_path / "guide with spaces.md"]
