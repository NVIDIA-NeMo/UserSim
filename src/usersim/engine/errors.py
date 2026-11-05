# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The error a run raises when its configuration is unusable and the user can fix it.

The ``usersim`` entry points report user errors by raising ``SystemExit``,
which bypasses ``main``'s ``try`` so the interpreter prints the message
without a traceback. Library code, which notebooks call as well as the CLI,
raises ``ConfigError`` instead: a ``SystemExit`` there would end a notebook
kernel over a mistyped path.

Subclassing ``ValueError`` keeps a library caller that already catches
``ValueError`` around config construction working.
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
    traceback, which is the correct output for a bug.
    """
