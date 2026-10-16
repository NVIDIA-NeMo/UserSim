# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Staged SPDX fixes preserve executable content and only touch named files."""

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/update_license_headers.py"


def test_explicit_paths_fix_only_named_file_and_preserve_shebang(tmp_path: Path) -> None:
    selected = tmp_path / "selected.py"
    untouched = tmp_path / "untouched.py"
    body = '#!/usr/bin/env python3\nprint("example")\n'
    selected.write_text(body)
    untouched.write_text(body)
    selected.chmod(0o755)
    result = subprocess.run([sys.executable, str(SCRIPT), "--check", str(selected)], capture_output=True)
    assert result.returncode == 1
    subprocess.run([sys.executable, str(SCRIPT), str(selected)], check=True)
    text = selected.read_text()
    assert text.startswith("#!/usr/bin/env python3\n")
    assert "SPDX-License-Identifier: Apache-2.0" in text
    assert text.endswith('print("example")\n')
    assert selected.stat().st_mode & 0o111 == 0o111
    assert untouched.read_text() == body
    subprocess.run([sys.executable, str(SCRIPT), "--check", str(selected)], check=True)


def test_directory_names_containing_skip_tokens_do_not_hide_sources(tmp_path: Path) -> None:
    # .github is not .git; suffixes such as .pyc in a parent name are not caches.
    selected = tmp_path / ".github.pyc-source" / "script.py"
    selected.parent.mkdir()
    selected.write_text('print("source")\n')
    subprocess.run([sys.executable, str(SCRIPT), str(selected)], check=True)
    assert "SPDX-License-Identifier:" in selected.read_text()
