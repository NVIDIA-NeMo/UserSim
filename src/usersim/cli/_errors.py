# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Error types shared by the CLI entry points and the library modules
they call.

Why this exists
---------------
The argparse entry points under ``cli/`` report user errors by raising
``SystemExit``. That works because ``SystemExit`` is not an ``Exception``
subclass, so it bypasses ``main``'s ``try`` and the interpreter prints
the message without a traceback. It is the right shape for a module that
IS the program.

``cli/_pipeline.py`` is not. It builds the Data Designer config for the
notebooks as well as the CLI, so a ``SystemExit``
from there would kill a notebook kernel over a mistyped path. Those
call sites raise instead, and this is what they raise.

Subclassing ``ValueError`` keeps that promise cheap: a library caller
that already catches ``ValueError`` around config construction keeps
working, and existing ``pytest.raises(ValueError)`` assertions still
pass.
"""

from __future__ import annotations


class ConfigError(ValueError):
    """A run configuration is unusable, and the user can fix it.

    Raise this for anything a user could correct by changing a flag,
    a path, or an asset tree: a seed file that is not there,
    two candidate files where one was expected, an unknown probe name.

    ``main`` catches it and prints the message alone, so the text must
    stand on its own as the entire error output: say what was wrong and
    what to do about it, without assuming a traceback for context.

    NOT for programming errors. A helper called with a nonsensical
    argument combination, or an internal branch that should be
    unreachable, should stay a plain ``ValueError`` so it surfaces as a
    traceback — which is the correct output for a bug.
    """
