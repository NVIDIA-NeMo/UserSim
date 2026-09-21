# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Per-locale renderings of asset-bank text fields.

Several asset banks ship text fields whose content the user-LLM
(simulation infrastructure) emits verbatim or that the judge LLM
grounds its rubric against — and where running an English string
against a non-English persona produces a confounded test signal:

- ``safety_chat_pressure``: ``target_requests[].initial_prompt`` is
  injected verbatim as the turn-1 user message; if the persona is
  Brazilian and the prompt is English, the assistant responds in
  English to a Portuguese-speaking persona, contaminating the
  refusal-under-pressure signal.
- ``safety_agentic``: ``action_requests[].initial_user_prompt``
  has the same problem (single user turn, even worse).
- The judge LLM grounds per-turn rubrics on
  ``target.expected_refusal`` /
  ``action_request.expected_behavior`` text; the assistant
  response is in the persona's locale, so the rubric anchor should
  match.

This module provides :class:`LocalizedText` — a small wrapper that
the bank loaders use to absorb a polymorphic YAML shape:

- A plain string ``"hello"`` is treated as
  ``{DEFAULT_LOCALE: "hello"}`` so existing single-language banks
  keep working unchanged (back-compat).
- A dict ``{en_US: "hello", pt_BR: "olá"}`` is used as-is; missing
  locales fall back to ``DEFAULT_LOCALE`` (and missing
  ``DEFAULT_LOCALE`` falls back to any rendering present, with a
  warning the loader logs at parse time).

Bank consumers call :meth:`LocalizedText.for_locale` to pick the
right rendering at simulator time. The judge-rubric scorers do the
same so the rubric anchor matches the assistant's operating
language.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

logger = logging.getLogger("usersim.engine")


class LocalizedTextError(ValueError):
    """A YAML value couldn't be parsed as a localised string.

    Message format mirrors the asset-bank conventions:
    ``"<bank-path>::<entry id>::<field>: <what's wrong>"``.
    """


@dataclass(frozen=True)
class LocalizedText:
    """Locale-keyed text with a default-locale fallback.

    ``renderings`` is ``{locale_code: text}`` for every locale the
    asset has been authored in. ``for_locale(locale)`` is the only
    interface bank consumers should use — never reach into
    ``renderings`` directly, so adding a new fallback semantic is a
    one-spot change.
    """

    renderings: Mapping[str, str]

    # Default locale used by ``from_yaml_value`` when the YAML value
    # is a bare string (back-compat) and by ``for_locale`` as the
    # fallback when a requested locale isn't authored. ``en_US`` is
    # the framework's choice because it's the most-shipped locale
    # and most reviewers can audit it; consumers can override
    # per-call by passing ``fallback=...``.
    DEFAULT_LOCALE: ClassVar[str] = "en_US"

    @classmethod
    def from_yaml_value(
        cls,
        value: Any,
        *,
        field_path: str = "<unknown>",
    ) -> "LocalizedText":
        """Parse a YAML field value into a ``LocalizedText``.

        Accepts either:

        - ``str`` — wrapped as ``{DEFAULT_LOCALE: <str>}``. This is
          the back-compat path; existing single-language asset banks
          load unchanged.
        - ``dict[str, str]`` — used as-is after validating that
          every value is a non-empty string.

        Raises :class:`LocalizedTextError` for any other shape so
        bank reviewers catch malformed entries at load time rather
        than discovering the silent default at simulator time.
        """
        if isinstance(value, str):
            if not value.strip():
                raise LocalizedTextError(f"{field_path}: empty string")
            return cls(renderings={cls.DEFAULT_LOCALE: value})

        if isinstance(value, dict):
            if not value:
                raise LocalizedTextError(f"{field_path}: empty dict (need at least one locale)")
            cleaned: dict[str, str] = {}
            for loc, text in value.items():
                if not isinstance(loc, str) or not loc.strip():
                    raise LocalizedTextError(
                        f"{field_path}: locale key must be a non-empty string, got {type(loc).__name__}={loc!r}"
                    )
                if not isinstance(text, str) or not text.strip():
                    raise LocalizedTextError(
                        f"{field_path}::{loc}: rendering must be a non-empty string, got {type(text).__name__}"
                    )
                cleaned[loc] = text
            if cls.DEFAULT_LOCALE not in cleaned:
                # Allow it (fall back to "any rendering"), but log
                # at INFO so reviewers notice. Most field
                # authoring will include en_US since that's the
                # primary review-audit locale.
                logger.info(
                    f"{field_path}: no rendering for default locale "
                    f"{cls.DEFAULT_LOCALE!r}; for_locale will fall "
                    f"back to one of {sorted(cleaned)}"
                )
            return cls(renderings=cleaned)

        raise LocalizedTextError(f"{field_path}: must be a string or a dict[locale, text], got {type(value).__name__}")

    def for_locale(
        self,
        locale: str,
        *,
        fallback: str = "",
    ) -> str:
        """Return the rendering for ``locale``, falling back per the
        documented semantics.

        Resolution order:
        1. ``self.renderings[locale]`` if present.
        2. ``self.renderings[fallback]`` if non-empty and present.
        3. ``self.renderings[DEFAULT_LOCALE]`` if present.
        4. The first rendering in iteration order (which is YAML
           insertion order under Python 3.7+).

        Empty string is returned only when ``renderings`` is empty,
        which the loader prevents.
        """
        if locale in self.renderings:
            return self.renderings[locale]
        if fallback and fallback in self.renderings:
            return self.renderings[fallback]
        if self.DEFAULT_LOCALE in self.renderings:
            return self.renderings[self.DEFAULT_LOCALE]
        # Last-ditch: any rendering. The loader prevents empty
        # ``renderings``, so this is non-empty by construction.
        return next(iter(self.renderings.values()), "")

    def available_locales(self) -> tuple[str, ...]:
        """Sorted tuple of locales this field has renderings for."""
        return tuple(sorted(self.renderings.keys()))

    def is_localized(self) -> bool:
        """True iff the field carries renderings for more than just
        the default locale (i.e. some translation work has been
        done). Useful for the placeholder-discipline reporting:
        a bank with all-default-locale entries surfaces as a
        placeholder in the multilingual readiness scorecard."""
        return len(self.renderings) > 1
