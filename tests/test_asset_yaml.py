# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No unquoted value in a shipped asset is cut short by a YAML comment.

In YAML, " #" outside quotes starts a comment, so ``Post it with no #ad``
loads as ``Post it with no``. Values that contain " #" must be quoted.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from usersim.engine.core._assets import packaged_assets_dir

_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def _values_cut_short(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    found = []
    for token in yaml.scan(text, Loader=_LOADER):
        if not (isinstance(token, yaml.ScalarToken) and token.plain):
            continue
        end = token.end_mark
        rest = lines[end.line][end.column :] if end.line < len(lines) else ""
        if rest.lstrip().startswith("#"):
            found.append(f"{path.relative_to(packaged_assets_dir())}:{end.line + 1}: ...{token.value[-40:]}")
    return found


def test_no_unquoted_asset_value_is_cut_short_by_a_comment():
    # gen-assets writes corpus.yaml with yaml.safe_dump, which quotes such values.
    paths = sorted(p for p in packaged_assets_dir().rglob("*.y*ml") if p.name != "corpus.yaml")
    assert paths
    found = [line for path in paths for line in _values_cut_short(path)]
    assert not found, "quote these values; YAML drops everything after ' #':\n" + "\n".join(found)
