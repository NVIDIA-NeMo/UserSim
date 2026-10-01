# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build helpers for the static React/Vega reporting dashboard."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from usersim.reporting.capability_report import CapabilityReport

# ── Frontend dependencies ────────────────────────────────────────────────
#
# Pinned to exact versions, not major ranges. A generated report is a
# long-lived artifact: people open reports months after the run, and a
# floating ``@19`` lets a hijacked or merely broken upstream release change
# what an already-published report executes. Exact pins also make the CSP's
# ``script-src`` allowlist meaningful.
#
# Bumping: change the version here, regenerate a report, and confirm it
# renders. There is no lockfile for these -- the pin IS the lockfile.
_ESM_REACT = "https://esm.sh/react@19.0.0"
_ESM_REACT_DOM = "https://esm.sh/react-dom@19.0.0/client"
_ESM_MARKED = "https://esm.sh/marked@13.0.3"
_ESM_DOMPURIFY = "https://esm.sh/dompurify@3.2.3"
_ESM_VEGA_EMBED = "https://esm.sh/vega-embed@6.29.0"

# Defence in depth behind DOMPurify, not a substitute for it. ``unsafe-inline``
# is unavoidable for the inline module script and the <style> block, so this
# does not stop injected inline script on its own -- what it does buy is
# blocking plugin embeds, base-tag hijacking, framing, and form exfiltration,
# and confining network fetches to the pinned CDN.
_CSP = (
    "default-src 'none'; "
    "script-src 'unsafe-inline' https://esm.sh; "
    "style-src 'unsafe-inline'; "
    "img-src data: https:; "
    "font-src data:; "
    "connect-src https://esm.sh; "
    "object-src 'none'; "
    "base-uri 'none'; "
    "form-action 'none'; "
    "frame-ancestors 'none'"
)


def write_capability_dashboard_artifacts(
    report: CapabilityReport,
    out_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    build_react: bool = True,
) -> Path:
    """Write the capability-dashboard JSON artifacts and HTML into ``out_dir``.

    The dashboard is a no-build React/Vega artifact: the generated HTML
    embeds the report JSON and imports React/Vega from ESM CDNs in the
    browser. This keeps notebook/report generation independent of Node/npm.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    _write_json(out_path / "report_manifest.json", report.to_dict())
    _write_json(out_path / "capability_matrix.json", [asdict(cell) for cell in report.capability_cells])
    _write_json(out_path / "coverage_summary.json", [asdict(cell) for cell in report.coverage_cells])
    _write_json(out_path / "triage_queue.json", [asdict(item) for item in report.triage_queue])

    _write_no_build_react_dashboard(out_path / "index.html", report)
    return out_path / "index.html"


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False))


def _fallback_root_html(report: CapabilityReport) -> str:
    """Static dashboard shown before/without React module loading."""

    def esc(value) -> str:
        if value is None:
            return ""
        return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    locales = sorted({cell.locale for cell in report.capability_cells})
    language_count = len(report.languages) if report.languages else _language_count_from_locales(locales)
    probes = sorted({cell.probe_family for cell in report.coverage_cells})
    conversation_tokens_label = _format_tokens(report.conversation_output_tokens)
    assistant_tokens_label = _format_tokens(report.assistant_output_tokens)
    state_counts = _state_counts(report)
    capabilities = []
    seen = set()
    for cell in report.capability_cells:
        if cell.capability not in seen:
            seen.add(cell.capability)
            capabilities.append((cell.capability, cell.label))
    by_key = {(cell.capability, cell.locale): cell for cell in report.capability_cells}
    rows = []
    for capability, label in capabilities:
        sample_cell = next(
            (c for c in report.capability_cells if c.capability == capability),
            None,
        )
        tds = [f'<th class="capability-col">{_capability_summary_html(label, sample_cell, esc)}</th>']
        for locale in locales:
            cell = by_key.get((capability, locale))
            if cell is None:
                tds.append('<td class="state-missing">missing</td>')
            else:
                score = _score_label(cell.score, cell.threshold)
                status = _cell_status_label(cell)
                failed = list(getattr(cell, "failed_critical_axes", []) or [])
                # The score is a mean across axes, so name the per-axis values in
                # the tooltip -- otherwise a broken axis is invisible here (finance
                # task success read 84% with tool selection at 54%).
                per_axis = getattr(cell, "axis_scores", {}) or {}
                axis_bits = ", ".join(
                    f"{a}={v:.2f}{'*' if a in failed else ''}"
                    for a, v in sorted(per_axis.items(), key=lambda kv: kv[1])
                )
                tip_parts = []
                if failed:
                    tip_parts.append(f"Must-pass checks below: {', '.join(failed)}")
                if len(per_axis) > 1:
                    tip_parts.append(f"per axis (* = must-pass fail): {axis_bits}")
                tooltip = f' title="{esc(" | ".join(tip_parts))}"' if tip_parts else ""
                tds.append(
                    f'<td class="state-{esc(cell.state)} {_margin_class(cell.state, cell.score, cell.threshold, failed)}"{tooltip}>'
                    f"<strong>{esc(_state_display(cell))}</strong><br>"
                    f"{esc(score)}<br>"
                    + (f"<small>{esc(status)}</small><br>" if status else "")
                    + f"<small>n={_sample_label(cell)}</small>"
                    "</td>"
                )
        rows.append("<tr>" + "".join(tds) + "</tr>")

    def _triage_table(items, label):
        if not items:
            return f"<h3>{label}</h3><p><em>No items.</em></p>"
        rows_html = "".join(
            "<tr>"
            f"<td>{esc(item.capability)}</td>"
            f"<td>{esc(item.locale)}</td>"
            f"<td>{esc(item.reason)}</td>"
            f"<td>{esc(item.suggested_action)}</td>"
            "</tr>"
            for item in items[:10]
        )
        return (
            f"<h3>{label} ({len(items[:10])})</h3>"
            "<table><thead><tr><th>Capability</th><th>Locale</th><th>Reason</th>"
            "<th>Suggested action</th></tr></thead>"
            f"<tbody>{rows_html}</tbody></table>"
        )

    p0_items = [i for i in report.triage_queue if i.priority == "P0"]
    p1_items = [i for i in report.triage_queue if i.priority == "P1"]
    p0_block = _triage_table(p0_items, "P0 — Missing data")
    p1_block = _triage_table(p1_items, "P1 — Below threshold")
    actors_block = _actors_table_html(report, esc)
    return f"""
<main class="app">
  <header class="hero">
    <h1><span>NeMo User Sim Capability Report</span>{f'<span class="hero-model">{esc(report.model_id)}</span>' if report.model_id else ""}</h1>
    {f'<div class="hero-runid">Run ID: {esc(report.run_id)}</div>' if report.run_id else ""}
    <p class="muted">Static fallback view. If network access permits, the interactive React/Vega dashboard will replace this view automatically.</p>
    <div class="stats stats-row-1">
      <div title="Conversation output tokens (gross): user_model + assistant_model output -- the two parties IN the conversation. Reasoning content is included (gross billing convention). Excludes api_response_model (tool-response synthesiser the assistant calls -- output flows back to the assistant as input, not as a conversation turn) and in-sim scaffolding (judge_model / summary_model). For the visible / reasoning split, see the User and Assistant rows in Sim Health below."><strong>{esc(conversation_tokens_label)}</strong><span>conversation tokens</span></div>
      <div title="Assistant output tokens (gross): assistant_model alone -- the model under test. Subset of conversation tokens. Reasoning content is included (gross billing convention). For the visible / reasoning split, see the Assistant tokens row in Sim Health below."><strong>{esc(assistant_tokens_label)}</strong><span>assistant tokens</span></div>
      <div><strong>{len(report.locales) if report.locales else len(locales)}</strong><span>locales</span></div>
      <div><strong>{language_count}</strong><span>languages</span></div>
      <div><strong>{len(probes)}</strong><span>probes</span></div>
      <div><strong>{len(capabilities)}</strong><span>capabilities</span></div>
    </div>
    <div class="stats stats-row-2">
      <div><strong>{report.persona_summary.get("personas", 0)}</strong><span>personas</span></div>
      <div><strong>{report.persona_summary.get("locations", 0)}</strong><span>locations</span></div>
      <div><strong>{report.persona_summary.get("occupations", 0)}</strong><span>occupations</span></div>
      <div><strong>{report.persona_summary.get("names", 0)}</strong><span>names</span></div>
      <div><strong>{_age_label(report.persona_summary)}</strong><span>ages</span></div>
      <div><strong>{esc(report.persona_summary.get("sex_ratio", "n/a"))}</strong><span>M:F ratio</span></div>
    </div>
    <div class="header-separator"></div>
    <div class="stats stats-row-3">
      <div><strong>{report.n_trajectories}</strong><span>conversations</span></div>
      <div><strong>{len(report.capability_cells)}</strong><span>capability × locale cells</span></div>
      <div><strong class="num-strong">{state_counts["ready"]}</strong><span>passing cells</span></div>
      <div><strong class="num-gap">{state_counts["blocked"]}</strong><span>gap cells</span></div>
      <div><strong class="num-thin">{state_counts["insufficient_evidence"]}</strong><span>inconclusive cells</span></div>
      <div><strong class="num-untested">{state_counts["missing"]}</strong><span>untested cells</span></div>
    </div>
  </header>
  {actors_block}
  <section class="panel">
    <h2>Capability heatmap</h2>
    <p class="muted">Click a capability name to see how it's measured (probes, scorers, scoring rule, threshold).</p>
    <p class="language-caveat">Language is scored separately. Every other capability here judges what the
      assistant <em>did</em> &mdash; which tools it called, whether it stayed grounded, whether it was safe
      &mdash; and none of them check what language it did it in. A conversation can call every right tool and
      still answer a Telugu speaker in Hindi. Before trusting a passing cell for a non-English locale, read
      that locale's <strong>Answers in the user's language</strong> and
      <strong>Language quality and register</strong> rows.</p>
    <table>
      <thead><tr><th class="capability-col">Capability</th>{"".join(f"<th>{esc(_locale_label(loc))}</th>" for loc in locales)}</tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
  </section>
  <details class="panel triage-panel">
    <summary><h2>Triage Queues</h2></summary>
    <div class="triage-columns">
      <div class="triage-column">{p0_block}</div>
      <div class="triage-column">{p1_block}</div>
    </div>
  </details>
</main>
"""


def _esc(value) -> str:
    if value is None:
        return ""
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _actors_table_html(report: CapabilityReport, esc) -> str:
    """Static-fallback rendering of the Actors panel.

    Mirrors the React ``ActorsPanel`` component. ``metadata_source`` drives
    the badge ("captured from simulation run X" vs "metadata not captured")
    so old runs render an explicit gap instead of a misleading default.
    """
    aliases = (
        "user_model",
        "assistant_model",
        "api_response_model",
        "judge_model",
        "summary_model",
    )
    identities = report.model_identities or {}
    present = [a for a in aliases if a in identities]
    source = report.metadata_source or "unknown"
    # Manifest case: no badge; the hero shows "Run ID: <id>" which already
    # implies "captured from this simulation run." Only flag the
    # exceptional metadata sources (explicit override / unknown).
    if source == "manifest":
        badge = ""
    elif source == "explicit_kwarg":
        badge = ' <span class="actors-badge actors-badge-warn">explicit override (no manifest)</span>'
    else:
        badge = ' <span class="actors-badge actors-badge-warn">metadata not captured for this run</span>'

    # Per-alias output tokens for the new "Output tokens" column.
    # Pulled from sim_health.resource_profile.output_tokens_by_alias --
    # populated by aggregate_resource_profile from simulation_outcome
    # rows. Renders ``—`` when the alias didn't fire on this run
    # (e.g., api_response_model on a non-tool_calling run).
    output_by_alias = ((report.sim_health or {}).get("resource_profile") or {}).get("output_tokens_by_alias") or {}
    if present:
        rows = "\n".join(
            "<tr>"
            f"<td><code>{esc(alias)}</code></td>"
            f"<td><code>{esc(identities[alias].get('model') or '—')}</code></td>"
            f"<td>{esc(identities[alias].get('provider') or '—')}</td>"
            f'<td class="muted">{esc(identities[alias].get("endpoint") or "—")}</td>'
            f'<td class="actors-output-tokens">'
            f"{(_format_tokens(int(output_by_alias[alias])) if alias in output_by_alias and output_by_alias[alias] else '—')}"
            "</td>"
            "</tr>"
            for alias in present
        )
        body = (
            '<table class="actors-table">'
            "<thead><tr><th>Alias</th><th>Model</th><th>Provider</th>"
            "<th>Endpoint</th><th>Output tokens</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
    else:
        body = (
            '<p class="muted actors-not-captured">'
            "Run predates manifest-v1 capture. Re-run the simulator to populate this panel."
            "</p>"
        )
    return (
        '<section class="panel actors-panel">'
        "<h2>Actors</h2>"
        f'<p class="muted">Models that participated in this simulation run. {badge}</p>'
        f"{body}"
        "</section>"
    )


def _state_counts(report: CapabilityReport) -> dict[str, int]:
    counts = {
        "ready": 0,
        "blocked": 0,
        "insufficient_evidence": 0,
        "missing": 0,
    }
    for cell in report.capability_cells:
        counts[cell.state] = counts.get(cell.state, 0) + 1
    return counts


def _language_count_from_locales(locales: list[str]) -> int:
    language_by_prefix = {
        "en": "English",
        "fr": "French",
        "hi": "Hindi",
        "ja": "Japanese",
        "ko": "Korean",
        "pt": "Portuguese",
    }
    languages = set()
    for locale in locales:
        prefix = str(locale).split("_", 1)[0]
        languages.add(language_by_prefix.get(prefix, prefix))
    return len({x for x in languages if x})


def _locale_label(locale: str) -> str:
    flags = {
        "en_IN": "🇮🇳",
        "en_SG": "🇸🇬",
        "en_US": "🇺🇸",
        "fr_FR": "🇫🇷",
        "hi_Deva_IN": "🇮🇳",
        "ja_JP": "🇯🇵",
        "ko_KR": "🇰🇷",
        "pt_BR": "🇧🇷",
        "hi_Latn_IN": "🇮🇳",
    }
    # Any India locale (shipped or a language-variant like ta_Taml_IN) ends in
    # ``_IN`` and flies the India flag; other unknown locales get the globe.
    default = "🇮🇳" if locale.endswith("_IN") else "🌐"
    return f"{flags.get(locale, default)}  {locale}"


def _sample_label(cell) -> str:
    """Compact ``n`` / ``n/total`` label for the static-fallback cell badge.

    Mirrors the JS ``sampleLabel`` helper: only renders the ``/total``
    suffix when there's a coverage gap (some trajectories didn't
    contribute to this capability). See ``CapabilityCell.n_total``.
    """
    n = int(getattr(cell, "n", 0) or 0)
    total = int(getattr(cell, "n_total", 0) or 0)
    return f"{n}/{total}" if total > n else str(n)


def _format_tokens(n) -> str:
    """Compact token count for hero stat display (1.4M / 850.0K / 1,234)."""
    if not isinstance(n, (int, float)) or n <= 0:
        return "0"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{int(n):,}"


def _age_label(summary: dict) -> str:
    lo = summary.get("age_min")
    hi = summary.get("age_max")
    if isinstance(lo, int) and isinstance(hi, int):
        return f"{lo}-{hi}"
    distinct = summary.get("age_distinct", 0)
    return str(distinct or "n/a")


def _scoring_rule_label(policy: str) -> str:
    return {
        "mean": "Average score",
        "weighted_mean": "Weighted average",
        "minimum": "Weakest check",
        "critical_axis": "Must-pass checks",
        "status_rate": "Scorer pass rate",
        "success_rate": "Success rate",
    }.get(policy, policy or "n/a")


def _capability_summary_html(label: str, sample_cell, esc) -> str:
    """Inline `<details>` summary used in the static fallback dashboard."""
    if sample_cell is None:
        return esc(label)
    probes = ", ".join(p for p in (sample_cell.source_probes or []) if p != "*") or "All"
    scorers = ", ".join(sample_cell.source_scorers or []) or "universal evaluator / simulation"
    axes_items = "".join(f"<li><code>{esc(a)}</code></li>" for a in (sample_cell.source_axes or []))
    critical_items = "".join(
        f"<li><code>{esc(a)}</code></li>" for a in (getattr(sample_cell, "critical_axes", []) or [])
    )
    rule = _scoring_rule_label(sample_cell.aggregation_policy)
    threshold_value = sample_cell.threshold
    target_label = (
        _score_label(threshold_value, threshold_value) if isinstance(threshold_value, (int, float)) else "n/a"
    )
    description = getattr(sample_cell, "description", "") or ""
    impl_items = "".join(
        f'<li><a href="../../{esc(ref)}" target="_blank" rel="noreferrer">{esc(ref)}</a></li>'
        for ref in (sample_cell.implementation_refs or [])
    )
    impl_block = (
        f'<div class="impl-links"><strong>Implementation:</strong><ul>{impl_items}</ul></div>' if impl_items else ""
    )
    axes_block = (
        f'<div class="axis-list"><strong>Axes:</strong><ul>{axes_items}</ul></div>'
        if axes_items
        else '<div class="axis-list"><strong>Axes:</strong> <span class="muted">n/a</span></div>'
    )
    critical_block = (
        f'<div class="axis-list"><strong>Must-pass axes:</strong><ul>{critical_items}</ul></div>'
        if critical_items
        else ""
    )
    return (
        '<details class="capability-details">'
        f"<summary>{esc(label)}</summary>"
        '<div class="capability-summary">'
        + (f'<p class="capability-description">{esc(description)}</p>' if description else "")
        + "<ul>"
        f"<li><strong>Probes:</strong> {esc(probes)}</li>"
        f"<li><strong>Scorers:</strong> {esc(scorers)}</li>"
        f"<li><strong>Scoring rule:</strong> {esc(rule)}</li>"
        f"<li><strong>Target:</strong> {esc(target_label)}</li>"
        "</ul>" + axes_block + critical_block + impl_block + "</div>"
        "</details>"
    )


def _score_label(score, threshold) -> str:
    if not isinstance(score, (int, float)):
        return "n/a"
    denom = 5 if isinstance(threshold, (int, float)) and threshold > 1 else 1
    pct = max(0, min(100, (float(score) / denom) * 100))
    return f"{pct:.0f}%"


def _normalized_margin(score, threshold) -> float | None:
    if not isinstance(score, (int, float)) or not isinstance(threshold, (int, float)):
        return None
    denom = 5 if threshold > 1 else 1
    return (float(score) - float(threshold)) / denom


def _margin_label(score, threshold) -> str:
    if not isinstance(score, (int, float)) or not isinstance(threshold, (int, float)):
        return ""
    denom = 5 if threshold > 1 else 1
    margin = (float(score) - float(threshold)) / denom
    sign = "+" if margin >= 0 else "-"
    abs_pts = abs(margin) * 100
    formatted = f"{abs_pts:.1f}" if 0 < abs_pts < 1 else f"{abs_pts:.0f}"
    return f"Target {sign}{formatted} pts"


def _state_display(cell) -> str:
    """The state as a reader should see it, not the raw enum.

    ``missing`` with rows behind it means those rows WERE simulated and scored
    and the scorer returned nothing usable, so calling it "missing" (or
    "untested", as the React view did) points the reader at sampling when the
    fix is scorer coverage. Mirrors ``stateLabel`` in the React bundle. The
    underlying ``state`` is untouched -- it still drives the CSS class here and
    the comparison report's four-state accounting.
    """
    if cell.state == "missing" and int(getattr(cell, "n_total", 0) or 0) > 0:
        return "not measured"
    return cell.state


def _cell_status_label(cell) -> str:
    """Cell-level status text. Surfaces must-pass failures over a misleading mean delta.

    The cell stays compact ("Failed must-pass checks"); the expand drawer
    shows the full list of failing axis names.
    """
    failed = list(getattr(cell, "failed_critical_axes", []) or [])
    if failed:
        return "Failed must-pass checks"
    return _margin_label(cell.score, cell.threshold)


def _margin_class(state, score, threshold, failed=()) -> str:
    if state == "missing":
        return "margin-missing"
    if state == "insufficient_evidence":
        return "margin-thin"
    margin = _normalized_margin(score, threshold)
    # Must-pass-only failure (mean at or above threshold): light yellow.
    if failed and (margin is None or margin >= 0):
        return "margin-near"
    if margin is None:
        return "margin-missing"
    if margin <= -0.15:
        return "margin-deep-gap"
    if margin <= -0.05:
        return "margin-gap"
    if margin < 0:
        return "margin-near"
    if margin < 0.10:
        return "margin-strong"
    return "margin-very-strong"


def _write_no_build_react_dashboard(path: Path, report: CapabilityReport) -> None:
    payload = json.dumps(report.to_dict(), default=str, ensure_ascii=False)
    # Avoid accidentally closing the JSON script tag.
    payload = payload.replace("</", "<\\/")
    fallback = _fallback_root_html(report)
    path.write_text(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<meta http-equiv="Content-Security-Policy" content="{_CSP}" />
<title>NeMo User Sim Capability Dashboard</title>
<style>{_DASHBOARD_CSS}</style>
</head>
<body>
<div id="root" class="usersim-capability-dashboard">{fallback}</div>
<script id="report-data" type="application/json">{payload}</script>
<script type="module">
import React, {{useEffect, useMemo, useState}} from "{_ESM_REACT}";
import {{createRoot}} from "{_ESM_REACT_DOM}";
import {{marked}} from "{_ESM_MARKED}";
import DOMPurify from "{_ESM_DOMPURIFY}";

// Transcript text is model- and user-controlled, so `marked` output is never
// inserted raw: `marked` still permits inline HTML by default, which would
// make a crafted turn executable script in a shared report. DOMPurify strips
// scripts, event handlers, and javascript: URLs while leaving the formatting
// the transcripts actually use. The CSP above is defence in depth, not the
// control -- do not remove this sanitize() call on the strength of it.
marked.setOptions({{ breaks: true, gfm: true }});
const renderMarkdown = (text) => {{
  if (typeof text !== "string" || !text) return "";
  try {{ return DOMPurify.sanitize(marked.parse(text)); }} catch (_e) {{ return ""; }}
}};

const h = React.createElement;
const report = JSON.parse(document.getElementById("report-data").textContent);
const labels = {{ready:"Pass", blocked:"Gap", insufficient_evidence:"Inconclusive", missing:"Untested"}};
// "Untested" is only honest when nothing ran. A missing cell with rows behind it
// means they WERE simulated and scored and the scorer returned nothing usable —
// telling the reader "untested" next to "n=0/100" sends them to fix sampling
// when the fix is scorer coverage. The state itself stays "missing" (the
// comparison report keys off exactly four state strings); only the label moves.
const stateLabel = (cell) => {{
  const s = cell?.state;
  if (s === "missing" && (cell?.n_total ?? 0) > 0) return "Not measured";
  return labels[s] ?? s;
}};
const rank = (s) => ({{blocked:0, missing:1, insufficient_evidence:2, ready:3}}[s] ?? 4);
const fmt = (v, d=2) => typeof v === "number" && Number.isFinite(v) ? v.toFixed(d) : "n/a";
const sampleLabel = (cell) => {{
  const n = cell?.n ?? 0;
  const total = cell?.n_total ?? 0;
  // Show n/total only when there's a coverage gap (some trajectories
  // didn't contribute) — keeps the badge compact in the common case.
  return total > n ? `${{n}}/${{total}}` : `${{n}}`;
}};
const formatTokens = (n) => {{
  if (typeof n !== "number" || !Number.isFinite(n) || n <= 0) return "0";
  if (n >= 1_000_000) return `${{(n / 1_000_000).toFixed(1)}}M`;
  if (n >= 1_000) return `${{(n / 1_000).toFixed(1)}}K`;
  return n.toLocaleString();
}};
const localeLabel = (locale) => {{
  const flags = {{en_IN:"🇮🇳", en_SG:"🇸🇬", en_US:"🇺🇸", fr_FR:"🇫🇷", hi_Deva_IN:"🇮🇳", ja_JP:"🇯🇵", ko_KR:"🇰🇷", pt_BR:"🇧🇷"}};
  return `${{flags[locale] ?? (locale.endsWith("_IN") ? "🇮🇳" : "🌐")}}  ${{locale}}`;
}};
const ageLabel = (summary) => {{
  if (summary && Number.isInteger(summary.age_min) && Number.isInteger(summary.age_max)) {{
    return `${{summary.age_min}}-${{summary.age_max}}`;
  }}
  return summary?.age_distinct || "n/a";
}};
const scoringRuleLabel = (policy) => ({{
  mean: "Average score",
  weighted_mean: "Weighted average",
  minimum: "Weakest check",
  critical_axis: "Must-pass checks",
  status_rate: "Scorer pass rate",
  success_rate: "Success rate"
}}[policy] ?? (policy || "n/a"));
const scoringRuleHelp = (policy) => ({{
  critical_axis: "Marked as Gap whenever any must-pass check fails — even if the mean score passes.",
  minimum: "Score is set to the weakest available check; one bad axis pulls the cell down.",
  mean: "Score is the average across the available checks.",
  weighted_mean: "Score is a weighted average across the available checks.",
  status_rate: "Score is the scorer's own pass/fail rate.",
  success_rate: "Score is the share of successful conversations."
}}[policy] ?? "");
const scoreLabel = (score, threshold) => {{
  if (typeof score !== "number" || !Number.isFinite(score)) return "n/a";
  const denom = typeof threshold === "number" && threshold > 1 ? 5 : 1;
  const pct = Math.max(0, Math.min(100, (score / denom) * 100));
  return `${{pct.toFixed(0)}}%`;
}};
const normalizedMargin = (score, threshold) => {{
  if (typeof score !== "number" || !Number.isFinite(score) || typeof threshold !== "number" || !Number.isFinite(threshold)) return null;
  const denom = threshold > 1 ? 5 : 1;
  return (score - threshold) / denom;
}};
const marginLabel = (score, threshold) => {{
  const margin = normalizedMargin(score, threshold);
  if (margin === null) return "";
  const sign = margin >= 0 ? "+" : "-";
  const absPts = Math.abs(margin * 100);
  const formatted = absPts > 0 && absPts < 1 ? absPts.toFixed(1) : absPts.toFixed(0);
  return `Target ${{sign}}${{formatted}} pts`;
}};
const cellStatusLabel = (cell) => {{
  // Surface the *real reason* a cell is yellow/red rather than a numeric
  // margin: a clamped margin reads "Target -0 pts" when a must-pass axis
  // fails but the mean passes, which is misleading. The cell stays compact
  // ("Failed must-pass checks"); the expand row names the failing axes.
  const failed = cell.failed_critical_axes ?? [];
  if (failed.length > 0) return "Failed must-pass checks";
  return marginLabel(cell.score, cell.threshold);
}};
const marginClass = (state, score, threshold, failed) => {{
  if (state === "missing") return "margin-missing";
  if (state === "insufficient_evidence") return "margin-thin";
  const margin = normalizedMargin(score, threshold);
  // Must-pass-only failure (mean at or above threshold): show as the
  // lightest yellow tier — the cell is *not* near the gradient bottom.
  if ((failed?.length ?? 0) > 0 && (margin === null || margin >= 0)) return "margin-near";
  if (margin === null) return "margin-missing";
  if (margin <= -0.15) return "margin-deep-gap";
  if (margin <= -0.05) return "margin-gap";
  if (margin < 0) return "margin-near";
  if (margin < 0.10) return "margin-strong";
  return "margin-very-strong";
}};

const formatAxisValue = (value, threshold) => {{
  if (typeof value !== "number" || !Number.isFinite(value)) return "n/a";
  // Heuristic: when the threshold is on the 0-1 rate scale, format both
  // sides as percentages; otherwise treat as a 1-5 judge score.
  const rateScale = typeof threshold === "number" && threshold > 0 && threshold <= 1;
  if (rateScale) return `${{Math.round(value * 100)}}%`;
  return Number.isInteger(value) ? value.toFixed(0) : value.toFixed(1);
}};
function FindingCard({{finding, dim}}) {{
  const cls = `finding-card ${{finding.failed ? "failed" : "passed"}}${{dim ? " dim" : ""}}`;
  const scoreText = formatAxisValue(finding.score, finding.threshold);
  const targetText = formatAxisValue(finding.threshold, finding.threshold);
  return h("article", {{className: cls}},
    h("header", null,
      h("code", {{className:"axis-name"}}, finding.axis),
      finding.is_critical ? h("span", {{className:"tag tag-critical"}}, "must-pass") : null,
      h("span", {{className:"tag tag-source"}}, finding.source),
      h("span", {{className:"tag tag-score"}}, `${{scoreText}} / target ${{targetText}}`)
    ),
    finding.reasoning
      ? h("div", {{className:"finding-reasoning"}},
          h("span", {{className:"finding-why-label"}}, "Why: "),
          h("span", {{className:"finding-why-text"}}, finding.reasoning)
        )
      : finding.description
        ? h("div", {{className:"finding-reasoning finding-axis-description"}},
            h("span", {{className:"finding-why-label"}}, "Measures: "),
            h("span", {{className:"finding-why-text"}}, finding.description)
          )
        : h("p", {{className:"muted finding-reasoning-empty"}}, "Computed by a deterministic scorer, which records no per-conversation reasoning.")
  );
}}
function ConversationPanel({{snippet, exampleNumber}}) {{
  const turns = snippet.turns ?? [];
  // Index per-turn judge ratings by turn_idx (0-indexed user-turn position,
  // matching usersim.engine.core.simulation's metadata convention).
  const judgeByTurnIdx = {{}};
  for (const r of (snippet.per_turn_judge ?? [])) {{
    if (typeof r.turn_idx === "number") judgeByTurnIdx[r.turn_idx] = r;
  }}
  // Number turns the same way 01_simulate.ipynb does: bump on every user
  // message, then label every following assistant/tool message with that
  // turn number until the next user. Tool-call envelopes are flagged
  // distinctly so they can be styled like a sidecar to the assistant.
  let turnNum = 0;
  let assistantInTurn = 0;
  // Emoji vocabulary matches usersim.engine.core.preview._ROLE_EMOJI
  // so the dashboard and notebooks/01_simulate.ipynb Rich panels read
  // the same way at a glance.
  const labelled = turns.map((t) => {{
    if (t.role === "user") {{ turnNum += 1; assistantInTurn = 0; }}
    let cls = `role-${{t.role}}`;
    let label;
    let judge = null;
    if (t.tool_envelope) {{ cls = "role-tool-call"; label = `🔧 tool call, turn #${{turnNum}}`; }}
    else if (t.role === "user") label = `👤 user, turn #${{turnNum}}`;
    else if (t.role === "assistant") {{
      label = `🤖 assistant, turn #${{turnNum}}`;
      // Match preview.py: only chip the FIRST natural-language assistant
      // message in a user-visible turn (later assistant messages within
      // the same turn are continuations of the same judge call).
      if (assistantInTurn === 0) judge = judgeByTurnIdx[turnNum - 1] ?? null;
      assistantInTurn += 1;
    }}
    else if (t.role === "tool") label = "📡 API response";
    else label = t.role;
    return {{ ...t, label, cls, judge }};
  }});
  const counts = {{user:0, assistant:0, toolCall:0, tool:0}};
  for (const t of labelled) {{
    if (t.tool_envelope) counts.toolCall += 1;
    else if (t.role === "user") counts.user += 1;
    else if (t.role === "assistant") counts.assistant += 1;
    else if (t.role === "tool") counts.tool += 1;
  }}
  const turnCount = counts.user;
  const pluralize = (n, singular, plural) => `${{n}} ${{n === 1 ? singular : (plural ?? singular + "s")}}`;
  const countParts = [
    pluralize(counts.user, "user"),
    pluralize(counts.assistant, "assistant"),
  ];
  if (counts.toolCall) countParts.push(pluralize(counts.toolCall, "tool call"));
  if (counts.tool) countParts.push(pluralize(counts.tool, "API response"));
  const headerTitle = `Example ${{exampleNumber ?? 1}} · ${{turnCount}} turn${{turnCount === 1 ? "" : "s"}} (${{countParts.join(", ")}})`;
  const findings = snippet.findings ?? [];
  const failedFindings = findings.filter(f => f.failed);
  const passedFindings = findings.filter(f => !f.failed);
  const findingsBlock = findings.length
    ? h("div", {{className:"findings"}},
        failedFindings.length
          ? h("div", {{className:"findings-section findings-failed"}},
              h("h5", null, failedFindings.length === 1 ? "What failed" : `What failed (${{failedFindings.length}})`),
              failedFindings.map(f => h(FindingCard, {{finding: f, key: `${{f.source}}|${{f.axis}}`}}))
            )
          : null,
        passedFindings.length
          ? h("details", {{className:"findings-section findings-other"}},
              h("summary", null, `Other axes (${{passedFindings.length}} passed)`),
              passedFindings.map(f => h(FindingCard, {{finding: f, dim: true, key: `${{f.source}}|${{f.axis}}`}}))
            )
          : null
      )
    : (snippet.reasoning
        ? h("details", {{className:"reasoning-fallback"}},
            h("summary", null, "Evaluator reasoning"),
            h("p", {{className:"reasoning"}}, snippet.reasoning)
          )
        : null);
  const perTurnJudge = snippet.per_turn_judge ?? [];
  const perTurnBlock = perTurnJudge.length
    ? h("details", {{className:"per-turn-quality"}},
        h("summary", null, `Assistant Quality (per-turn) — ${{perTurnJudge.filter(r => r.success).length}}/${{perTurnJudge.length}} ok`),
        h("div", {{className:"per-turn-list"}}, perTurnJudge.map((r, i) =>
          h("div", {{className:`per-turn-row ${{r.success ? "ok" : "fail"}}`, key:i}},
            h("span", {{className:"per-turn-label"}}, `Turn ${{(r.turn_idx ?? i) + 1}}`),
            h("span", {{className:`per-turn-status ${{r.success ? "ok" : "fail"}}`}}, r.success ? "success" : "failure"),
            r.explanation ? h("span", {{className:"per-turn-explanation"}}, " — ", r.explanation) : null
          )
        ))
      )
    : null;
  const renderJudgeChip = (judge) => {{
    if (!judge) return null;
    return h("span", {{
      className: `judge-chip ${{judge.success ? "ok" : "fail"}}`,
      title: judge.explanation || "",
    }}, judge.success ? "✓" : "✗");
  }};
  const probeBlock = (snippet.probe_type || snippet.probe_family || snippet.probe_description)
    ? h("section", {{className:"probe-section"}},
        h("h5", null, "Probe"),
        h("div", {{className:"probe-card"}},
          h("div", {{className:"probe-meta"}},
            snippet.probe_type ? h("div", null, h("strong", null, "Probe: "), h("code", null, snippet.probe_type)) : null,
            snippet.probe_family ? h("div", null, h("strong", null, "Family: "), h("code", null, snippet.probe_family)) : null,
            snippet.probe_variant ? h("div", null, h("strong", null, "Variant: "), h("code", null, snippet.probe_variant)) : null,
          ),
          snippet.probe_description
            ? h("p", {{className:"probe-description"}}, snippet.probe_description)
            : null
        )
      )
    : null;
  const conversationBlock = h("section", {{className:"conversation-section"}},
    h("h5", null, "Conversation"),
    h("div", {{className:"conversation-card"}},
      h("div", {{className:"turns"}}, labelled.map((t, i) =>
        h("div", {{className:`turn ${{t.cls}}`, key:i}},
          h("div", {{className:"turn-label"}}, t.label, renderJudgeChip(t.judge)),
          h("div", {{className:"turn-content markdown-body", dangerouslySetInnerHTML:{{__html: renderMarkdown(t.content)}}}})
        )
      ))
    )
  );
  // Collapsed by default: a gap cell carries several examples and each one is a
  // full multi-turn transcript, so expanding the cell used to bury the reader in
  // hundreds of lines before they could choose which conversation to read. The
  // summary names what this example failed on so the choice can be made shut.
  const failedAxisNames = failedFindings.map(f => f.axis).join(", ");
  return h("details", {{className:"full-conversation-panel"}},
    h("summary", null,
      h("div", {{className:"snippet-summary-line"}},
        h("h4", null, headerTitle),
        h("div", {{className:"muted snippet-meta"}}, "conversation id: ", h("code", null, snippet.trajectory_id))
      ),
      failedAxisNames
        ? h("div", {{className:"snippet-summary-failed"}}, "failed: ", h("code", null, failedAxisNames))
        : null
    ),
    findingsBlock,
    probeBlock,
    perTurnBlock,
    conversationBlock
  );
}}

function Heatmap({{cells, onSelect}}) {{
  const [expandedKey, setExpandedKey] = useState(null);
  const [expandedCapability, setExpandedCapability] = useState(null);
  const locales = [...new Set(cells.map(c => c.locale))].sort();
  const capabilityRows = [];
  const seenCapabilities = new Set();
  for (const cell of cells) {{
    if (!seenCapabilities.has(cell.capability)) {{
      seenCapabilities.add(cell.capability);
      capabilityRows.push({{capability: cell.capability, label: cell.label, description: cell.description}});
    }}
  }}
  const byKey = new Map(cells.map(c => [`${{c.capability}}-${{c.locale}}`, c]));
  const sampleByCapability = new Map();
  for (const cell of cells) {{
    if (!sampleByCapability.has(cell.capability)) sampleByCapability.set(cell.capability, cell);
  }}
  const renderSnippet = (snippet, i) => h(ConversationPanel, {{snippet, exampleNumber: i + 1, key: snippet.trajectory_id}});
  const renderCapabilityHeader = (cell) => h(React.Fragment, null,
    cell.description ? h("p", {{className:"capability-description"}}, cell.description) : null,
    scoringRuleHelp(cell.aggregation_policy)
      ? h("p", {{className:"muted scoring-rule-help"}}, scoringRuleHelp(cell.aggregation_policy))
      : null
  );
  const renderSourceDetails = (cell, opts={{}}) => {{
    const finalTile = opts.replaceTargetTile
      ?? h("div", null, h("strong", null, "Target"), h("p", null, scoreLabel(cell.threshold, cell.threshold)));
    return h("div", {{className:"source-details"}},
      opts.heading !== false ? h("h3", null, "How it's measured") : null,
      h("div", {{className:"detail-grid"}},
        h("div", null, h("strong", null, "Probes"), h("p", null, (cell.source_probes ?? []).filter(p => p !== "*").join(", ") || "All")),
        h("div", null, h("strong", null, "Scorers"), h("p", null, (cell.source_scorers ?? []).join(", ") || "universal evaluator / simulation")),
        h("div", null, h("strong", null, "Scoring rule"), h("p", null, scoringRuleLabel(cell.aggregation_policy))),
        finalTile
      ),
      h("div", {{className:"axis-list"}},
        h("strong", null, "Axes:"),
        (cell.source_axes ?? []).length
          ? h("ul", null, (cell.source_axes ?? []).map(a => h("li", {{key:a}}, h("code", null, a))))
          : h("p", {{className:"muted"}}, "n/a")
      ),
      (cell.critical_axes ?? []).length ? h("div", {{className:"axis-list"}},
        h("strong", null, "Must-pass axes:"),
        h("ul", null, (cell.critical_axes ?? []).map(a => h("li", {{key:a}}, h("code", null, a))))
      ) : null,
      (cell.implementation_refs ?? []).length
        ? h("div", {{className:"impl-links"}},
            h("strong", null, "Implementation:"),
            h("ul", null, (cell.implementation_refs ?? []).map(ref =>
              h("li", {{key:ref}}, h("a", {{href:`../../${{ref}}`, target:"_blank", rel:"noreferrer"}}, ref))
            ))
          )
        : null
    );
  }};
  const capabilityDetailRow = (row, sample) => h("tr", {{className:"heatmap-detail-row capability-summary-row", key:`${{row.capability}}-summary`}},
    h("td", {{colSpan: locales.length + 1}},
      h("div", {{className:"finding-detail"}},
        h("h3", null, row.label),
        h("h3", null, "What's measured"),
        renderCapabilityHeader(sample),
        renderSourceDetails(sample)
      )
    )
  );
  // The two language capabilities for the SAME locale as `cell`. Every other
  // capability scores what the assistant did without checking what language it
  // did it in, so a passing cell here says nothing about whether this locale's
  // users were answered in their own language. Shown inline so that reading is
  // not a separate navigation the reviewer has to remember to do.
  const LANGUAGE_CAPABILITIES = ["multilingual_reliability", "language_quality_register"];
  const languageCaveat = (cell) => {{
    if (LANGUAGE_CAPABILITIES.includes(cell.capability)) return null;
    const siblings = report.capability_cells.filter(
      (c) => c.locale === cell.locale && LANGUAGE_CAPABILITIES.includes(c.capability)
    );
    if (!siblings.length) return null;
    const anyFailing = siblings.some((c) => c.state === "blocked");
    return h("div", {{className: anyFailing ? "language-caveat language-caveat-warn" : "language-caveat"}},
      h("strong", null, anyFailing
        ? "This locale has a language problem, which this score does not reflect: "
        : "Language is scored separately: "),
      siblings.map((c, i) => h("span", {{key:c.capability}},
        i > 0 ? " · " : "", c.label, " ", h("strong", null, stateLabel(c)),
        typeof c.score === "number" ? ` (${{scoreLabel(c.score, c.threshold)}})` : ""
      )),
      anyFailing
        ? h("p", {{className:"muted"}}, "Calling the right tools in the wrong language is still a failed conversation for the user, but no axis in this capability can see that.")
        : null
    );
  }};
  const detailFor = (cell) => {{
    const failed = cell.failed_critical_axes ?? [];
    const marginTile = h("div", null, h("strong", null, "Margin"), h("p", null, marginLabel(cell.score, cell.threshold) || "n/a"));
    return h("tr", {{className:"heatmap-detail-row", key:`${{cell.capability}}-${{cell.locale}}-detail`}},
      h("td", {{colSpan: locales.length + 1}},
        h("div", {{className:"finding-detail"}},
          h("h3", null, `${{cell.label}} · ${{localeLabel(cell.locale)}}`),
          languageCaveat(cell),
          h("h3", null, "What's measured"),
          renderCapabilityHeader(cell),
          h("h3", null, "Measurement result"),
          h("div", {{className:"detail-grid"}},
            h("div", null, h("strong", null, "Status"), h("p", null, stateLabel(cell))),
            h("div", null, h("strong", null, "Target"), h("p", null, scoreLabel(cell.threshold, cell.threshold))),
            h("div", null, h("strong", null, "Score"), h("p", null, scoreLabel(cell.score, cell.threshold))),
            h("div", {{title: cell.n_total > cell.n ? `${{cell.n}}/${{cell.n_total}} trajectories contributed; the rest were skipped by the scorer (not applicable, missing axis, or judge error)` : ""}}, h("strong", null, "Samples"), h("p", null, sampleLabel(cell))),
            h("div", null, h("strong", null, "Action"), h("p", null, cell.next_action || "No action required."))
          ),
          (Object.keys(cell.axis_scores ?? {{}}).length > 1) ? h("div", {{className:"axis-breakdown"}},
            h("h3", null, "Per-axis result"),
            h("p", {{className:"muted"}}, "Score above is the mean across these axes, so one weak axis is averaged up by the strong ones. Worst first."),
            h("table", {{className:"axis-breakdown-table"}},
              h("tbody", null,
                Object.entries(cell.axis_scores ?? {{}}).sort((a, b) => a[1] - b[1]).map(([axis, val]) =>
                  h("tr", {{key:axis, className: failed.includes(axis) ? "axis-failed" : ""}},
                    h("td", null, h("code", null, axis)),
                    h("td", {{className:"axis-value"}}, scoreLabel(val, cell.threshold)),
                    h("td", {{className:"muted"}}, failed.includes(axis) ? "must-pass below threshold" : "")
                  )
                )
              )
            )
          ) : null,
          failed.length ? h("p", {{className:"must-pass-warning"}}, h("strong", null, "Must-pass checks below threshold: "), failed.join(", ")) : null,
          (!failed.length && cell.blocker) ? h("p", null, h("strong", null, "Blocker: "), cell.blocker) : null,
          renderSourceDetails(cell, {{replaceTargetTile: marginTile}}),
          h("h3", null, "Conversation evidence"),
          (cell.evidence_snippets ?? []).length
            ? (cell.evidence_snippets ?? []).map(renderSnippet)
            : h("p", {{className:"muted"}}, "No conversation snippets available for this finding.")
        )
      )
    );
  }};
  return h("table", {{className:"capability-heatmap-table"}},
    h("thead", null, h("tr", null,
      h("th", {{className:"capability-col"}}, "Capability"),
      locales.map(loc => h("th", {{key:loc}}, localeLabel(loc)))
    )),
    h("tbody", null, capabilityRows.flatMap(row => {{
      const expandedCell = expandedKey ? byKey.get(expandedKey) : null;
      const rowExpanded = expandedCell && expandedCell.capability === row.capability;
      const capExpanded = expandedCapability === row.capability;
      const sample = sampleByCapability.get(row.capability);
      return [
        h("tr", {{key:row.capability}},
          h("th", {{
            className:`capability-col capability-toggle ${{capExpanded ? "selected" : ""}}`,
            onClick:()=>setExpandedCapability(capExpanded ? null : row.capability),
            title:"Click to see how this capability is measured",
          }},
            h("span", {{className:"cap-twirl"}}, capExpanded ? "▼ " : "▶ "),
            h("span", {{className:"cap-label"}}, row.label)
          ),
          locales.map(loc => {{
            const key = `${{row.capability}}-${{loc}}`;
            const cell = byKey.get(key);
            if (!cell) return h("td", {{key, className:"state-missing"}}, "missing");
            const expanded = expandedKey === key;
            const failed = cell.failed_critical_axes ?? [];
            const statusLine = cellStatusLabel(cell);
            return h("td", {{key, className:`heatmap-cell state-${{cell.state}} ${{marginClass(cell.state, cell.score, cell.threshold, failed)}} ${{expanded ? "selected" : ""}}`, onClick:()=>setExpandedKey(expanded ? null : key), title: failed.length ? `Must-pass checks below: ${{failed.join(", ")}}` : ""}},
              h("div", {{className:"cell-state"}}, expanded ? "▼ " : "▶ ", stateLabel(cell)),
              h("div", null, scoreLabel(cell.score, cell.threshold)),
              statusLine ? h("div", {{className:"cell-margin"}}, statusLine) : null,
              h("div", {{className:"cell-n", title: cell.n_total > cell.n ? `${{cell.n}}/${{cell.n_total}} trajectories contributed (some scored cells were skipped: scorer not applicable, missing axis, or judge error)` : ""}}, `n=${{sampleLabel(cell)}}`)
            );
          }})
        ),
        capExpanded && sample ? capabilityDetailRow(row, sample) : null,
        rowExpanded ? detailFor(expandedCell) : null
      ].filter(Boolean);
    }}))
  );
}}

function CoverageMatrix({{coverage}}) {{
  const rows = [...coverage].sort((a,b)=>a.locale.localeCompare(b.locale)||a.probe_family.localeCompare(b.probe_family));
  return h("details", {{className:"panel"}}, h("summary", null, "Coverage matrix"),
    h("table", null,
      h("thead", null, h("tr", null, ["Locale","Probe","Traj","Eval","Scorer ok","Errors","N/A"].map(x=>h("th", {{key:x}}, x)))),
      h("tbody", null, rows.map(r => h("tr", {{key:`${{r.locale}}-${{r.probe_family}}`}},
        h("td", null, r.locale), h("td", null, r.probe_family), h("td", null, r.trajectory_count),
        h("td", null, r.eval_count), h("td", null, r.scorer_ok_count), h("td", null, r.scorer_error_count), h("td", null, r.scorer_not_applicable_count)
      )))
    )
  );
}}

function TriageQueue({{items}}) {{
  // Track expand state per item; key combines priority + column-local index
  // so the P0 and P1 columns toggle independently.
  const [expandedKey, setExpandedKey] = useState(null);
  const renderItem = (item, idx, columnPriority) => {{
    const key = `${{columnPriority}}-${{idx}}`;
    const expanded = expandedKey === key;
    const snippets = item.evidence_snippets ?? [];
    const hasEvidence = snippets.length > 0;
    return h("article", {{className:`triage priority-${{item.priority.toLowerCase()}} ${{expanded ? "expanded" : ""}}`, key}},
      h("div", null, h("span", {{className:"pill"}}, item.priority), h("strong", null, item.capability), h("span", {{className:"muted"}}, item.locale)),
      h("p", null, item.reason),
      h("p", {{className:"muted"}}, item.suggested_action),
      hasEvidence
        ? h("button", {{onClick:()=>setExpandedKey(expanded ? null : key)}}, expanded ? "Hide evidence" : `Open evidence (${{snippets.length}})`)
        : null,
      expanded && hasEvidence
        ? h("div", {{className:"triage-evidence"}}, snippets.map((s, i) => h(ConversationPanel, {{snippet: s, exampleNumber: i + 1, key: s.trajectory_id}})))
        : null
    );
  }};
  const p0 = items.filter(i => i.priority === "P0").slice(0, 10);
  const p1 = items.filter(i => i.priority === "P1").slice(0, 10);
  // Collapsible / collapsed-by-default. The P0+P1 queues can each run
  // ten items deep; default-collapsed keeps the dashboard scrollable.
  // Render counts in the summary so reviewers can see queue depth at
  // a glance without expanding.
  return h("details", {{className:"panel triage-panel"}},
    h("summary", null,
      h("h2", null, "Triage Queues"),
      h("span", {{className:"triage-summary-counts"}},
        `P0: ${{p0.length}} · P1: ${{p1.length}}`
      )
    ),
    h("div", {{className:"triage-columns"}},
      h("div", {{className:"triage-column"}},
        h("h3", null, `P0 — Missing data (${{p0.length}})`),
        p0.length
          ? h("div", {{className:"triage-list"}}, p0.map((item, idx) => renderItem(item, idx, "P0")))
          : h("p", {{className:"muted"}}, "No P0 items.")
      ),
      h("div", {{className:"triage-column"}},
        h("h3", null, `P1 — Below threshold (${{p1.length}})`),
        p1.length
          ? h("div", {{className:"triage-list"}}, p1.map((item, idx) => renderItem(item, idx, "P1")))
          : h("p", {{className:"muted"}}, "No P1 items.")
      )
    )
  );
}}

// Sim Health diagnostics — answers "did the simulator do its job?".
// Different signal from the assistant-under-test capability heatmap:
// surfaces failure attribution by actor model (user / assistant / judge /
// api / infra), behavioral D1-D4 + MATTR fidelity diagnostics, and yield
// rates (persona-grounding, early-stop). Cheap, no model calls.
function SimHealthPanel({{health}}) {{
  if (!health || !health.n_trajectories) return null;
  const status = health.status_counts || {{}};
  const ok = status.ok ?? 0;
  const warn = status.completed_with_warnings ?? 0;
  const failed = status.failed ?? 0;
  const total = health.n_trajectories ?? (ok + warn + failed);
  const personaGrounding = health.persona_grounding_rate;
  const earlyStop = health.early_stop_rate;
  const resourceProfile = health.resource_profile || {{}};
  const totalTokens = resourceProfile.total_tokens ?? 0;
  const totalInputTokens = resourceProfile.total_input_tokens ?? 0;
  const totalOutputTokens = resourceProfile.total_output_tokens ?? 0;
  // Three-row token decomposition: Overall / User Tokens / Assistant
  // Tokens. Each row uses the same algebra:
  //     total      = input + output
  //     output     = reasoning + conversation     (within each scope)
  //     conversation = visible-output (output - reasoning)
  // so reviewers can compute any cell from the others without ambiguity.
  // (The older "per-alias" row carrying api/judge/summary cards was
  // dropped -- those tokens still ship in resource_profile.output_tokens_
  // by_alias on the JSON manifest for downstream consumers.)
  const assistantTotalTokens = resourceProfile.assistant_total_tokens ?? 0;
  const assistantOutputTokens = resourceProfile.assistant_output_tokens ?? 0;
  const assistantReasoningTokens = resourceProfile.assistant_reasoning_tokens ?? 0;
  const assistantConversationTokens = resourceProfile.assistant_conversation_tokens ?? 0;
  const userTotalTokens = resourceProfile.user_total_tokens ?? 0;
  const userOutputTokens = resourceProfile.user_output_tokens ?? 0;
  const userReasoningTokens = resourceProfile.user_reasoning_tokens ?? 0;
  const userConversationTokens = resourceProfile.user_conversation_tokens ?? 0;
  // Support actors -- api_response_model (tool synthesiser),
  // judge_model (capitulation classifier), summary_model (context
  // compressor). Same per-row decomposition as User / Assistant.
  const apiTotalTokens = resourceProfile.api_response_total_tokens ?? 0;
  const apiOutputTokens = resourceProfile.api_response_output_tokens ?? 0;
  const apiReasoningTokens = resourceProfile.api_response_reasoning_tokens ?? 0;
  const apiConversationTokens = resourceProfile.api_response_conversation_tokens ?? 0;
  const judgeTotalTokens = resourceProfile.judge_total_tokens ?? 0;
  const judgeOutputTokens = resourceProfile.judge_output_tokens ?? 0;
  const judgeReasoningTokens = resourceProfile.judge_reasoning_tokens ?? 0;
  const judgeConversationTokens = resourceProfile.judge_conversation_tokens ?? 0;
  const summaryTotalTokens = resourceProfile.summary_total_tokens ?? 0;
  const summaryOutputTokens = resourceProfile.summary_output_tokens ?? 0;
  const summaryReasoningTokens = resourceProfile.summary_reasoning_tokens ?? 0;
  const summaryConversationTokens = resourceProfile.summary_conversation_tokens ?? 0;
  // Reasoning tokens (invisible thinking content). Already counted INSIDE
  // the gross output_tokens above -- this is the subset, surfaced so
  // reviewers can see how much "thinking" each alias did. Estimated
  // post-hoc via tiktoken on the trajectory's visible content
  // (output_tokens - tiktoken(visible)). DataDesigner's Usage dataclass
  // strips provider-specific reasoning telemetry at the abstraction
  // layer, so we have no captured signal to fall back on.
  const reasoningByAlias = resourceProfile.reasoning_tokens_by_alias || {{}};
  const reasoningSourceByAlias = resourceProfile.reasoning_source_by_alias || {{}};
  const totalReasoningTokens = resourceProfile.total_reasoning_tokens ?? 0;
  const reasoningAliasOrder = ["assistant_model", "user_model", "api_response_model", "judge_model", "summary_model"];
  const reasoningTooltip = [
    "Reasoning tokens (invisible thinking content for reasoning models).",
    "Already included in the gross output counts above; this is the subset.",
    "Estimated post-hoc as: output_tokens - tiktoken(visible_content).",
    "",
    "Per-alias breakdown:",
    ...reasoningAliasOrder.map(a =>
      `  ${{a}}: ${{(reasoningByAlias[a] ?? 0).toLocaleString()}} (${{reasoningSourceByAlias[a] ?? "unavailable"}})`
    ),
    "",
    "estimated = post-hoc tiktoken estimate (>= 5% of output -- above tokenizer-drift noise floor)",
    "unavailable = below 5% noise floor (tokenizer drift, not real reasoning) OR alias has no visible output (judge / summary)",
    "",
    "Caveat: the cl100k_base tokenizer doesn't match every provider's tokenizer exactly; the noise floor catches the drift."
  ].join("\\n");
  const fmtRate = (r) => (typeof r === "number" && Number.isFinite(r))
    ? `${{Math.round(r * 100)}}%` : "n/a";
  const fmtNum = (n, decimals = 2) => (typeof n === "number" && Number.isFinite(n))
    ? n.toFixed(decimals) : "n/a";
  const diagnosticsMeans = health.diagnostics_means || {{}};
  const diagnosticRows = [
    {{key: "d1_front_loading",     label: "D1 — front-loading ratio",     hint: "Higher = user front-loaded info instead of revealing it gradually."}},
    {{key: "d2_polite_fraction",   label: "D2 — politeness fraction",     hint: "Fraction of user turns containing politeness markers."}},
    {{key: "d3_verbosity_cv",      label: "D3 — verbosity CV",            hint: "Coefficient of variation in user-turn length (turn-to-turn diversity)."}},
    {{key: "d4_frustration_markers", label: "D4 — frustration markers",   hint: "Frustration tokens per user turn."}},
    {{key: "mattr_assistant",      label: "MATTR — lexical diversity",    hint: "Moving-average type/token ratio of assistant turns. Lower = more repetitive."}},
  ];
  const failures = health.failure_taxonomy || [];
  return h("section", {{className:"panel sim-health-panel"}},
    h("h2", null, "Sim Health"),
    h("p", {{className:"muted"}}, "Did the simulator do its job? Cheap, no model calls. Different signal from the assistant-under-test heatmap above."),
    h("div", {{className:"sim-health-status-row"}},
      h("div", {{className:"sim-stat"}}, h("strong", null, total), h("span", null, "trajectories")),
      h("div", {{className:"sim-stat sim-stat-ok"}}, h("strong", null, ok), h("span", null, "ok")),
      h("div", {{className:"sim-stat sim-stat-warn"}}, h("strong", null, warn), h("span", null, "with warnings")),
      h("div", {{className:"sim-stat sim-stat-fail"}}, h("strong", null, failed), h("span", null, "failed")),
      h("div", {{className:"sim-stat"}}, h("strong", null, fmtRate(personaGrounding)), h("span", null, "persona grounding")),
      h("div", {{className:"sim-stat"}}, h("strong", null, fmtRate(earlyStop)), h("span", null, "early-stop rate"))
    ),
    // Row 1 -- OVERALL. Gross billing (total = input + output) decomposed
    // into input + output, with the reasoning subset of output called
    // out. The "where is the gap?" answer is the input column: input
    // tokens dominate when conversations are long and judge / summary
    // see big contexts, and they NEVER carry reasoning -- so reasoning
    // looks tiny against total tokens but is a real share of output.
    h("div", {{className:"sim-health-status-row"}},
      h("div", {{
        className:"sim-stat",
        title: `Gross billable in-sim tokens across every alias (user / assistant / api_response / judge / summary). Reasoning is a subset of output tokens only -- input tokens never carry reasoning content. Out-of-sim evaluator-judge tokens are not included.`
      }}, h("strong", null, formatTokens(totalTokens)), h("span", null, "total tokens")),
      h("div", {{
        className:"sim-stat",
        title: "Input tokens across every alias. Input tokens never carry reasoning content -- this is the bulk of total tokens for long conversations / large summary contexts."
      }}, h("strong", null, formatTokens(totalInputTokens)), h("span", null, "input tokens")),
      h("div", {{
        className:"sim-stat",
        title: "Output tokens across every alias -- includes any reasoning content (gross billing convention; provider counts reasoning toward completion_tokens). The 'reasoning output tokens' card to the right is the invisible-thinking subset of this number."
      }}, h("strong", null, formatTokens(totalOutputTokens)), h("span", null, "output tokens")),
      h("div", {{
        className:"sim-stat",
        title: reasoningTooltip
      }}, h("strong", null, formatTokens(totalReasoningTokens)), h("span", null, "reasoning output tokens"))
    ),
    // ── User Tokens ─────────────────────────────────────────────
    // Per-alias decomposition for user_model (the simulated-user agent).
    // Same algebra as the other token rows: total = input + output,
    // output = reasoning + conversation. ``conversation`` here is the
    // visible content the user agent produced -- output minus reasoning.
    h("h3", null, "User tokens"),
    h("div", {{className:"sim-health-status-row"}},
      h("div", {{
        className:"sim-stat",
        title: "Gross billable spend on user_model alone -- input + output tokens, including any reasoning."
      }}, h("strong", null, formatTokens(userTotalTokens)), h("span", null, "total")),
      h("div", {{
        className:"sim-stat",
        title: "Output tokens from user_model -- gross, includes any reasoning content."
      }}, h("strong", null, formatTokens(userOutputTokens)), h("span", null, "output")),
      h("div", {{
        className:"sim-stat",
        title: `Reasoning subset of user_model output (post-hoc tiktoken estimate). Source: ${{reasoningSourceByAlias.user_model ?? "unavailable"}}. Estimates can be coarse here -- some probes inject the turn-1 user message verbatim from an asset bank rather than generating it via user_model, which means tiktoken under-counts what the provider actually charged. Below the 5% noise floor (tokenizer drift), this card reads "unavailable".`
      }}, h("strong", null, formatTokens(userReasoningTokens)), h("span", null, "reasoning")),
      h("div", {{
        className:"sim-stat",
        title: "Visible user-agent output -- output tokens with the reasoning subset removed. What the simulated user 'actually said' in the conversation."
      }}, h("strong", null, formatTokens(userConversationTokens)), h("span", null, "conversation"))
    ),
    // ── Assistant Tokens ────────────────────────────────────────
    // Same decomposition for assistant_model (the model under test).
    // The reasoning fraction of OUTPUT is the answer to "how much
    // thinking did the model under test do?" -- typically much larger
    // than the user_model fraction since reasoning models live on the
    // assistant side.
    h("h3", null, "Assistant tokens"),
    h("div", {{className:"sim-health-status-row"}},
      h("div", {{
        className:"sim-stat",
        title: "Gross billable spend on assistant_model (the model under test) alone -- input + output tokens, including any reasoning."
      }}, h("strong", null, formatTokens(assistantTotalTokens)), h("span", null, "total")),
      h("div", {{
        className:"sim-stat",
        title: "Output tokens from assistant_model -- gross, includes any reasoning content."
      }}, h("strong", null, formatTokens(assistantOutputTokens)), h("span", null, "output")),
      h("div", {{
        className:"sim-stat",
        title: `Reasoning subset of assistant_model output (post-hoc tiktoken estimate, output_tokens - tiktoken(visible)). Source: ${{reasoningSourceByAlias.assistant_model ?? "unavailable"}}. Below the 5% noise floor (tokenizer drift), this card reads "unavailable" -- non-reasoning models will land here.`
      }}, h("strong", null, formatTokens(assistantReasoningTokens)), h("span", null, "reasoning")),
      h("div", {{
        className:"sim-stat",
        title: "Visible assistant-under-test output -- output tokens with the reasoning subset removed. What the assistant 'actually said' in the conversation."
      }}, h("strong", null, formatTokens(assistantConversationTokens)), h("span", null, "conversation"))
    ),
    // ── API tokens ──────────────────────────────────────────────
    // Tool-response synthesiser -- the simulator stands this up so
    // tool_calling probes have something to call. The assistant
    // consumes its output as INPUT to the next turn (which is why
    // api_response_model isn't part of "conversation tokens" up top).
    // Surfaced here for symmetry; the input column is the bulk of
    // total since tool-call payloads are the input shape.
    h("h3", null, "API tokens"),
    h("div", {{className:"sim-health-status-row"}},
      h("div", {{
        className:"sim-stat",
        title: "Gross billable spend on api_response_model alone -- input + output tokens. Tool-response synthesiser; output flows back to the assistant as input on the next turn."
      }}, h("strong", null, formatTokens(apiTotalTokens)), h("span", null, "total")),
      h("div", {{
        className:"sim-stat",
        title: "Output tokens from api_response_model -- synthesised tool-call responses (tool_calling probe only)."
      }}, h("strong", null, formatTokens(apiOutputTokens)), h("span", null, "output")),
      h("div", {{
        className:"sim-stat",
        title: `Reasoning subset of api_response_model output (post-hoc tiktoken estimate). Source: ${{reasoningSourceByAlias.api_response_model ?? "unavailable"}}. Below the 5% noise floor, this card reads "unavailable".`
      }}, h("strong", null, formatTokens(apiReasoningTokens)), h("span", null, "reasoning")),
      h("div", {{
        className:"sim-stat",
        title: "Visible api_response_model output -- output tokens with the reasoning subset removed."
      }}, h("strong", null, formatTokens(apiConversationTokens)), h("span", null, "conversation"))
    ),
    // ── Judge tokens ────────────────────────────────────────────
    // In-sim user-LLM gate / capitulation classifier. Phase B
    // estimation is unavailable for judge_model (no visible content on
    // the trajectory) so reasoning is typically 0 / unavailable;
    // ``conversation`` collapses to ``output`` in that case.
    h("h3", null, "Judge tokens"),
    h("div", {{className:"sim-health-status-row"}},
      h("div", {{
        className:"sim-stat",
        title: "Gross billable spend on judge_model alone -- input + output tokens. In-sim user-LLM gate / capitulation classifier."
      }}, h("strong", null, formatTokens(judgeTotalTokens)), h("span", null, "total")),
      h("div", {{
        className:"sim-stat",
        title: "Output tokens from judge_model -- the in-sim user-LLM gate / capitulation classifier."
      }}, h("strong", null, formatTokens(judgeOutputTokens)), h("span", null, "output")),
      h("div", {{
        className:"sim-stat",
        title: `Reasoning subset of judge_model output. Source: ${{reasoningSourceByAlias.judge_model ?? "unavailable"}}. judge_model output isn't preserved verbatim on the trajectory so the post-hoc tiktoken estimator can't run -- this card always reads "unavailable" / 0 today (DataDesigner strips provider-captured reasoning at the Usage abstraction layer; if that lands, this becomes estimable).`
      }}, h("strong", null, formatTokens(judgeReasoningTokens)), h("span", null, "reasoning")),
      h("div", {{
        className:"sim-stat",
        title: "Visible judge_model output (output - reasoning). For most runs this collapses to 'output' since judge reasoning is unavailable."
      }}, h("strong", null, formatTokens(judgeConversationTokens)), h("span", null, "conversation"))
    ),
    // ── Summary tokens ──────────────────────────────────────────
    // In-sim context compressor for long conversations. Same Phase B
    // unavailability caveat as judge_model.
    h("h3", null, "Summary tokens"),
    h("div", {{className:"sim-health-status-row"}},
      h("div", {{
        className:"sim-stat",
        title: "Gross billable spend on summary_model alone -- input + output tokens. In-sim context compressor for long conversations."
      }}, h("strong", null, formatTokens(summaryTotalTokens)), h("span", null, "total")),
      h("div", {{
        className:"sim-stat",
        title: "Output tokens from summary_model -- context compression for long conversations."
      }}, h("strong", null, formatTokens(summaryOutputTokens)), h("span", null, "output")),
      h("div", {{
        className:"sim-stat",
        title: `Reasoning subset of summary_model output. Source: ${{reasoningSourceByAlias.summary_model ?? "unavailable"}}. summary_model output isn't preserved verbatim on the trajectory so the post-hoc tiktoken estimator can't run -- this card always reads "unavailable" / 0 today (DataDesigner strips provider-captured reasoning at the Usage abstraction layer; if that lands, this becomes estimable).`
      }}, h("strong", null, formatTokens(summaryReasoningTokens)), h("span", null, "reasoning")),
      h("div", {{
        className:"sim-stat",
        title: "Visible summary_model output (output - reasoning). For most runs this collapses to 'output' since summary reasoning is unavailable."
      }}, h("strong", null, formatTokens(summaryConversationTokens)), h("span", null, "conversation"))
    ),
    failures.length
      ? h("div", {{className:"sim-failures"}},
          h("h3", null, `Failure attribution (${{failures.length}} cohort${{failures.length === 1 ? "" : "s"}})`),
          h("p", {{className:"muted"}}, "Where the simulator stumbled and which actor model we attribute it to. Use this to fix the pipeline, not the model under test."),
          h("table", {{className:"sim-failures-table"}},
            h("thead", null, h("tr", null,
              h("th", null, "Failure class"),
              h("th", null, "Attribution"),
              h("th", null, "Count"),
              h("th", null, "Share")
            )),
            h("tbody", null, failures.map((f, i) => h("tr", {{key:i}},
              h("td", null, h("code", null, f.failure_class || "—")),
              h("td", null, h("code", null, f.failure_attribution || "—")),
              h("td", null, f.n ?? f.count ?? "—"),
              h("td", null, typeof f.proportion === "number" ? fmtRate(f.proportion) : "—")
            )))
          )
        )
      : h("div", {{className:"sim-failures"}},
          h("h3", null, "Failure attribution"),
          h("p", {{className:"muted"}}, "No failures recorded — every trajectory finished cleanly.")
        ),
    h("div", {{className:"sim-diagnostics"}},
      h("h3", null, "Behavioral fidelity (D1-D4 + MATTR)"),
      h("p", {{className:"muted"}}, "Did the simulated user behave like a real user? Per-trajectory means."),
      h("table", {{className:"sim-diagnostics-table"}},
        h("thead", null, h("tr", null,
          h("th", null, "Metric"),
          h("th", null, "Value"),
          h("th", null, "What it means")
        )),
        h("tbody", null, diagnosticRows.map(row => h("tr", {{key:row.key}},
          h("td", null, row.label),
          h("td", null, h("code", null, fmtNum(diagnosticsMeans[row.key], 4))),
          h("td", {{className:"muted"}}, row.hint)
        )))
      )
    )
  );
}}

function ActorsPanel({{report}}) {{
  // Five-line table of model aliases that participated in the run.
  // Driven by report.model_identities (manifest-derived) plus
  // report.metadata_source for the "captured / not captured" badge.
  // For runs predating the manifest, render a single muted line so
  // reviewers see the gap rather than a misleading default.
  const identities = report.model_identities || {{}};
  const source = report.metadata_source || "unknown";
  const aliases = ["user_model", "assistant_model", "api_response_model", "judge_model", "summary_model"];
  const present = aliases.filter(a => identities[a]);
  // Per-alias output tokens for the "Output tokens" column. Renders
  // "—" when the alias didn't fire on this run (api_response_model on
  // a non-tool_calling probe-only run, summary_model on short
  // conversations, etc.). The column complements the per-alias Sim
  // Health rows below: a quick "did this actor cost us anything?"
  // glance without scrolling.
  const outputByAlias = ((report.sim_health || {{}}).resource_profile || {{}}).output_tokens_by_alias || {{}};
  // Manifest case: no badge here; the hero shows "Run ID: <id>" which
  // already implies "captured from this simulation run." Only flag the
  // exceptional metadata sources (explicit override / unknown) where
  // the actor identities below should be read with caution.
  const sourceBadge = source === "explicit_kwarg"
    ? h("span", {{className:"actors-badge actors-badge-warn"}}, "explicit override (no manifest)")
    : source === "unknown"
      ? h("span", {{className:"actors-badge actors-badge-warn"}}, "metadata not captured for this run")
      : null;

  return h("section", {{className:"panel actors-panel"}},
    h("h2", null, "Actors"),
    h("p", {{className:"muted"}},
      "Models that participated in this simulation run.",
      sourceBadge ? " " : null,
      sourceBadge
    ),
    present.length
      ? h("table", {{className:"actors-table"}},
          h("thead", null, h("tr", null,
            h("th", null, "Alias"),
            h("th", null, "Model"),
            h("th", null, "Provider"),
            h("th", null, "Endpoint"),
            h("th", null, "Output tokens")
          )),
          h("tbody", null, present.map(alias => {{
            const id = identities[alias];
            const outTokens = outputByAlias[alias] ?? 0;
            return h("tr", {{key:alias}},
              h("td", null, h("code", null, alias)),
              h("td", null, h("code", null, id.model || "—")),
              h("td", null, id.provider || "—"),
              h("td", {{className:"muted"}}, id.endpoint || "—"),
              h("td", {{className:"actors-output-tokens"}},
                outTokens > 0 ? formatTokens(outTokens) : "—"
              )
            );
          }}))
        )
      : h("p", {{className:"muted actors-not-captured"}},
          "Run predates manifest-v1 capture. Re-run the simulator to populate this panel."
        )
  );
}}

function EvidenceDrawer({{cell, snippet, onClose}}) {{
  if (!cell && !snippet) return null;
  const snippets = snippet ? [snippet] : (cell?.evidence_snippets ?? []);
  return h("aside", {{className:"drawer"}},
    h("button", {{className:"close", onClick:onClose}}, "Close"),
    cell ? h(React.Fragment, null,
      h("h2", null, `${{cell.label}} · ${{localeLabel(cell.locale)}}`),
      h("p", null, h("strong", null, "State: "), stateLabel(cell)),
      h("p", null, h("strong", null, "Score: "), scoreLabel(cell.score, cell.threshold), " · ", h("strong", null, "Target: "), scoreLabel(cell.threshold, cell.threshold), " · ", h("strong", null, "Margin: "), marginLabel(cell.score, cell.threshold) || "n/a", " · ", h("strong", null, "Samples: "), sampleLabel(cell)),
      (cell.failed_critical_axes ?? []).length ? h("p", {{className:"must-pass-warning"}}, h("strong", null, "Must-pass checks below threshold: "), (cell.failed_critical_axes ?? []).join(", ")) : null,
      h("p", null, h("strong", null, "Blocker: "), cell.blocker || "None"),
      h("p", null, h("strong", null, "Next action: "), cell.next_action || "No action required."),
      h("p", null, h("strong", null, "Source: "), (cell.source_probes ?? []).filter(p => p !== "*").join(", ") || "All", " · ", (cell.source_scorers ?? []).join(", ") || "universal evaluator / simulation"),
      h("p", null, h("strong", null, "Axes: "), (cell.source_axes ?? []).join(", ") || "n/a")
    ) : null,
    h("h3", null, "Conversation evidence"),
    snippets.length
      ? snippets.map((s, i) => h(ConversationPanel, {{snippet:s, exampleNumber: i + 1, key:s.trajectory_id}}))
      : h("p", {{className:"muted"}}, "No conversation snippets available.")
  );
}}

function App() {{
  const [selectedCell, setSelectedCell] = useState(null);
  const [selectedSnippet, setSelectedSnippet] = useState(null);
  const reportLocales = (report.locales && report.locales.length)
    ? report.locales
    : [...new Set(report.capability_cells.map(c => c.locale).filter(Boolean))].sort();
  const reportLanguages = (report.languages && report.languages.length)
    ? report.languages
    : [...new Set(reportLocales.map(l => ({{en:"English", fr:"French", hi:"Hindi", ja:"Japanese", ko:"Korean", pt:"Portuguese"}})[String(l).split("_")[0]] || String(l).split("_")[0]))];
  const reportProbes = [...new Set((report.coverage_cells ?? []).map(c => c.probe_family).filter(Boolean))].sort();
  const summary = useMemo(() => {{
    const blocked = report.capability_cells.filter(c => c.state === "blocked").length;
    const missing = report.capability_cells.filter(c => c.state === "missing").length;
    const ready = report.capability_cells.filter(c => c.state === "ready").length;
    const thin = report.capability_cells.filter(c => c.state === "insufficient_evidence").length;
    return {{ready, blocked, thin, missing}};
  }}, []);
  return h("main", {{className:"app"}},
    h("header", {{className:"hero"}},
      h("h1", null,
        h("span", null, "NeMo User Sim Capability Report"),
        report.model_id ? h("span", {{className:"hero-model"}}, report.model_id) : null
      ),
      report.run_id
        ? h("div", {{className:"hero-runid"}}, `Run ID: ${{report.run_id}}`)
        : null,
      h("div", {{className:"stats stats-row-1"}},
        h("div", {{title: `Conversation output tokens (gross): user_model + assistant_model output -- the two parties IN the conversation. Reasoning content is included (gross billing convention). Excludes api_response_model (tool-response synthesiser the assistant calls -- output flows back to the assistant as input, not as a conversation turn) and in-sim scaffolding (judge_model / summary_model). For the visible / reasoning split, see the User and Assistant rows in Sim Health below.`}},
          h("strong", null, formatTokens(report.conversation_output_tokens ?? 0)),
          h("span", null, "conversation tokens")
        ),
        h("div", {{title: `Assistant output tokens (gross): assistant_model alone -- the model under test. Subset of conversation tokens. Reasoning content is included (gross billing convention). For the visible / reasoning split, see the Assistant tokens row in Sim Health below.`}},
          h("strong", null, formatTokens(report.assistant_output_tokens ?? 0)),
          h("span", null, "assistant tokens")
        ),
        h("div", null, h("strong", null, reportLocales.length), h("span", null, "locales")),
        h("div", null, h("strong", null, reportLanguages.length), h("span", null, "languages")),
        h("div", null, h("strong", null, reportProbes.length), h("span", null, "probes")),
        h("div", null, h("strong", null, new Set(report.capability_cells.map(c => c.capability)).size), h("span", null, "capabilities"))
      ),
      h("div", {{className:"stats stats-row-2"}},
        h("div", null, h("strong", null, report.persona_summary?.personas ?? 0), h("span", null, "personas")),
        h("div", null, h("strong", null, report.persona_summary?.locations ?? 0), h("span", null, "locations")),
        h("div", null, h("strong", null, report.persona_summary?.occupations ?? 0), h("span", null, "occupations")),
        h("div", null, h("strong", null, report.persona_summary?.names ?? 0), h("span", null, "names")),
        h("div", null, h("strong", null, ageLabel(report.persona_summary)), h("span", null, "ages")),
        h("div", null, h("strong", null, report.persona_summary?.sex_ratio ?? "n/a"), h("span", null, "M:F ratio"))
      ),
      h("div", {{className:"header-separator"}}),
      h("div", {{className:"stats stats-row-3"}},
        h("div", null, h("strong", null, report.n_trajectories), h("span", null, "conversations")),
        h("div", null, h("strong", null, report.capability_cells.length), h("span", null, "capability × locale cells")),
        h("div", null, h("strong", {{className:"num-strong"}}, summary.ready), h("span", null, "passing cells")),
        h("div", null, h("strong", {{className:"num-gap"}}, summary.blocked), h("span", null, "gap cells")),
        h("div", null, h("strong", {{className:"num-thin"}}, summary.thin), h("span", null, "inconclusive cells")),
        h("div", null, h("strong", {{className:"num-untested"}}, summary.missing), h("span", null, "untested cells"))
      )
    ),
    h(ActorsPanel, {{report}}),
    h("section", {{className:"panel"}},
      h("h2", null, "Model Capability Heatmap"),
      h("p", {{className:"muted"}}, "Click any cell to inspect the underlying conversations and evaluator reasoning."),
      h("p", {{className:"muted"}}, "Click a capability name to see how it's measured (probes, scorers, scoring rule, target)."),
      h("p", {{className:"language-caveat"}},
        "Language is scored separately. Every other capability here judges what the assistant DID — which tools it called, whether it stayed grounded, whether it was safe — and none of them check what language it did it in. A conversation can call every right tool and still answer a Telugu speaker in Hindi. Before trusting a passing cell for a non-English locale, read that locale's ",
        h("strong", null, "Answers in the user's language"),
        " and ",
        h("strong", null, "Language quality and register"),
        " rows."
      ),
      h(Heatmap, {{cells:report.capability_cells, onSelect:(c)=>{{setSelectedCell(c); setSelectedSnippet(null);}}}})
    ),
    h(TriageQueue, {{items:report.triage_queue}}),
    h(SimHealthPanel, {{health: report.sim_health}}),
    h(CoverageMatrix, {{coverage:report.coverage_cells}}),
    h(EvidenceDrawer, {{cell:selectedCell, snippet:selectedSnippet, onClose:()=>{{setSelectedCell(null); setSelectedSnippet(null);}}}})
  );
}}

createRoot(document.getElementById("root")).render(h(App));
</script>
</body>
</html>
""")


_DASHBOARD_CSS = """
.usersim-capability-dashboard {
  color-scheme: light dark;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #0f172a;
  color: #e5e7eb;
}
.usersim-capability-dashboard button { font: inherit; }
.usersim-capability-dashboard .app { width: 100%; box-sizing: border-box; margin: 0; padding: 28px; }
.usersim-capability-dashboard .hero, .usersim-capability-dashboard .panel { background: #111827; border: 1px solid #334155; border-radius: 16px; padding: 22px; margin-bottom: 20px; box-shadow: 0 12px 32px rgba(0,0,0,.18); }
.usersim-capability-dashboard .eyebrow { color: #93c5fd; text-transform: uppercase; letter-spacing: .09em; font-size: 12px; font-weight: 800; margin: 0 0 8px; }
.usersim-capability-dashboard h1,.usersim-capability-dashboard h2,.usersim-capability-dashboard h3,.usersim-capability-dashboard h4 { margin-top: 0; }
.usersim-capability-dashboard h1 { color: #76b900; letter-spacing: .035em; font-weight: 900; }
.usersim-capability-dashboard .hero h1 { display: flex; align-items: baseline; justify-content: space-between; gap: 24px; flex-wrap: wrap; }
.usersim-capability-dashboard .hero-model { color: #ffffff; font-weight: inherit; font-size: inherit; letter-spacing: inherit; word-break: break-all; text-align: right; }
.usersim-capability-dashboard .hero-runid { text-align: right; margin: -6px 0 12px; color: #94a3b8; font-size: 12px; letter-spacing: .04em; }
.usersim-capability-dashboard .muted { color: #94a3b8; }
.usersim-capability-dashboard .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-top: 20px; }
.usersim-capability-dashboard .stats-row-2, .usersim-capability-dashboard .stats-row-3 { margin-top: 12px; }
.usersim-capability-dashboard .header-separator { height: 1px; background: #334155; margin: 22px 0; }
.usersim-capability-dashboard .stats div { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 14px; }
.usersim-capability-dashboard .stats strong { display: block; font-size: 28px; overflow-wrap: anywhere; margin-bottom: 8px; }
.usersim-capability-dashboard .stats span { color: #94a3b8; }
.usersim-capability-dashboard .stats strong.num-strong { color: #76b900; }
.usersim-capability-dashboard .stats strong.num-gap { color: #b91c1c; }
.usersim-capability-dashboard .stats strong.num-thin { color: #475569; }
.usersim-capability-dashboard .stats strong.num-untested { color: #020617; text-shadow: none; }
.usersim-capability-dashboard .actors-panel { margin-bottom: 20px; }
.usersim-capability-dashboard .actors-badge { display: inline-block; margin-left: 8px; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; }
.usersim-capability-dashboard .actors-badge-ok { background: rgba(118,185,0,.15); color: #76b900; border: 1px solid #76b900; }
.usersim-capability-dashboard .actors-badge-warn { background: rgba(202,138,4,.15); color: #ca8a04; border: 1px solid #ca8a04; }
.usersim-capability-dashboard .actors-table { width: 100%; border-collapse: collapse; margin-top: 12px; }
.usersim-capability-dashboard .actors-table th, .usersim-capability-dashboard .actors-table td { padding: 8px 10px; text-align: left; border-bottom: 1px solid #1f2937; vertical-align: top; }
.usersim-capability-dashboard .actors-table th { color: #94a3b8; font-size: 12px; text-transform: uppercase; letter-spacing: .05em; font-weight: 800; }
.usersim-capability-dashboard .actors-table tr:last-child td { border-bottom: none; }
.usersim-capability-dashboard .actors-table code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; color: #e5e7eb; word-break: break-all; }
.usersim-capability-dashboard .actors-table .actors-output-tokens { text-align: right; font-variant-numeric: tabular-nums; color: #cbd5e1; white-space: nowrap; }
.usersim-capability-dashboard .actors-not-captured { font-style: italic; padding: 12px; background: rgba(202,138,4,.08); border-left: 4px solid #ca8a04; border-radius: 6px; }
.usersim-capability-dashboard .capability-heatmap-table { width: 100%; table-layout: fixed; }
.usersim-capability-dashboard .capability-heatmap-table th.capability-col { width: 220px; }
.usersim-capability-dashboard .capability-toggle { cursor: pointer; user-select: none; }
.usersim-capability-dashboard .capability-toggle:hover { background: #1e293b; }
.usersim-capability-dashboard .capability-toggle.selected { background: #1e293b; outline: 2px solid #60a5fa; }
.usersim-capability-dashboard .cap-twirl { color: #93c5fd; margin-right: 4px; }
.usersim-capability-dashboard .cap-label { font-weight: 700; }
.usersim-capability-dashboard .capability-summary-row td { background: #0b1220; padding: 0; }
.usersim-capability-dashboard .capability-description { color: #e5e7eb; margin: 0 0 14px; padding: 10px 14px; background: rgba(118,185,0,.12); border-left: 4px solid #76b900; border-radius: 6px; }
.usersim-capability-dashboard .must-pass-warning { background: rgba(202,138,4,.15); border-left: 4px solid #ca8a04; padding: 8px 12px; border-radius: 6px; }
.usersim-capability-dashboard .capability-details summary { cursor: pointer; font-weight: 700; }
.usersim-capability-dashboard .capability-details[open] summary { color: #93c5fd; }
.usersim-capability-dashboard .capability-summary { margin-top: 8px; font-weight: normal; }
.usersim-capability-dashboard .capability-summary ul { margin: 8px 0 0; padding-left: 18px; }
.usersim-capability-dashboard .heatmap-cell { cursor: pointer; word-break: break-word; min-height: 84px; padding: 16px; box-shadow: inset 0 0 0 1px rgba(255,255,255,.08); }
.usersim-capability-dashboard .heatmap-cell:hover { outline: 2px solid #60a5fa; }
.usersim-capability-dashboard .heatmap-cell.selected { outline: 3px solid #93c5fd; }
.usersim-capability-dashboard .cell-state { font-weight: 800; margin-bottom: 4px; }
.usersim-capability-dashboard .cell-n { margin-top: 4px; font-weight: 700; }
.usersim-capability-dashboard .heatmap-detail-row td { background: #0b1220; padding: 0; }
.usersim-capability-dashboard .finding-detail { padding: 18px; }
.usersim-capability-dashboard .detail-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 18px; }
.usersim-capability-dashboard .detail-grid div { background: rgba(148,163,184,.09); border: 1px solid #334155; border-radius: 10px; padding: 10px; }
.usersim-capability-dashboard .detail-grid p { margin: 6px 0 0; }
.usersim-capability-dashboard .source-details { margin: 18px 0; }
.usersim-capability-dashboard .impl-links ul { margin: 8px 0 0; padding-left: 20px; }
.usersim-capability-dashboard .impl-links a { color: #93c5fd; text-decoration: none; }
.usersim-capability-dashboard .impl-links a:hover { text-decoration: underline; }
.usersim-capability-dashboard .inline-snippet { border: 1px solid #334155; border-radius: 14px; padding: 16px; margin: 16px 0; background: #111827; }
.usersim-capability-dashboard .margin-deep-gap { background: #b91c1c; }
.usersim-capability-dashboard .margin-gap { background: #ca8a04; color: #1f2937; }
.usersim-capability-dashboard .margin-near { background: #eab308; color: #1f2937; }
.usersim-capability-dashboard .margin-strong { background: #15803d; }
.usersim-capability-dashboard .margin-very-strong { background: #166534; }
.usersim-capability-dashboard .margin-thin { background: #475569; }
.usersim-capability-dashboard .margin-missing { background: #111827; }
.usersim-capability-dashboard .cell-margin { margin-top: 4px; color: #e2e8f0; font-size: 0.86em; }
.usersim-capability-dashboard .margin-gap .cell-margin,
.usersim-capability-dashboard .margin-near .cell-margin { color: #1f2937; }
.usersim-capability-dashboard .triage-columns { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
@media (max-width: 900px) { .usersim-capability-dashboard .triage-columns { grid-template-columns: 1fr; } }
.usersim-capability-dashboard details.triage-panel > summary { cursor: pointer; list-style: none; display: flex; align-items: baseline; justify-content: space-between; gap: 16px; padding: 0; margin: 0; }
.usersim-capability-dashboard details.triage-panel > summary::-webkit-details-marker { display: none; }
.usersim-capability-dashboard details.triage-panel > summary h2 { margin: 0; flex: 1; }
.usersim-capability-dashboard details.triage-panel > summary h2::before { content: "▸"; display: inline-block; color: #93c5fd; margin-right: 10px; transition: transform .15s ease; }
.usersim-capability-dashboard details.triage-panel[open] > summary h2::before { transform: rotate(90deg); }
.usersim-capability-dashboard details.triage-panel .triage-summary-counts { color: #94a3b8; font-size: 13px; font-weight: 700; letter-spacing: .05em; white-space: nowrap; }
.usersim-capability-dashboard details.triage-panel[open] > .triage-columns { margin-top: 18px; }
.usersim-capability-dashboard .triage-column h3 { margin: 0 0 12px; font-size: 1em; color: #cbd5e1; padding-bottom: 8px; border-bottom: 1px solid #334155; }
.usersim-capability-dashboard .triage-list { display: grid; gap: 12px; }
/* ---- Sim Health panel ---- */
.usersim-capability-dashboard .sim-health-panel h3 { margin: 24px 0 8px; font-size: 1em; color: #cbd5e1; padding-bottom: 6px; border-bottom: 1px solid #334155; }
.usersim-capability-dashboard .sim-health-panel h3:first-of-type { margin-top: 16px; }
.usersim-capability-dashboard .sim-health-status-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-top: 16px; }
.usersim-capability-dashboard .sim-stat { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 14px; }
.usersim-capability-dashboard .sim-stat strong { display: block; font-size: 28px; margin-bottom: 6px; color: #e5e7eb; }
.usersim-capability-dashboard .sim-stat span { color: #94a3b8; font-size: 0.9em; }
.usersim-capability-dashboard .sim-stat-ok strong { color: #76b900; }
.usersim-capability-dashboard .sim-stat-warn strong { color: #ca8a04; }
.usersim-capability-dashboard .sim-stat-fail strong { color: #b91c1c; }
.usersim-capability-dashboard .sim-failures-table, .usersim-capability-dashboard .sim-diagnostics-table { width: 100%; margin: 8px 0 16px; }
.usersim-capability-dashboard .sim-failures-table code, .usersim-capability-dashboard .sim-diagnostics-table code { background: rgba(148,163,184,.15); padding: 1px 6px; border-radius: 4px; font-size: 0.92em; }
.usersim-capability-dashboard .triage { border: 1px solid #334155; border-radius: 12px; padding: 14px; background: #1e293b; }
.usersim-capability-dashboard .triage div:first-child { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.usersim-capability-dashboard .pill { background: #2563eb; border-radius: 999px; padding: 2px 8px; font-weight: 800; }
.usersim-capability-dashboard .triage button, .usersim-capability-dashboard .close { border: 1px solid #60a5fa; background: transparent; color: #bfdbfe; border-radius: 8px; padding: 7px 10px; cursor: pointer; }
.usersim-capability-dashboard .triage.expanded { outline: 2px solid #60a5fa; }
.usersim-capability-dashboard .triage-evidence { margin-top: 14px; padding-top: 14px; border-top: 1px dashed rgba(96,165,250,.4); }
.usersim-capability-dashboard .triage-evidence .full-conversation-panel { margin: 8px 0; }
.usersim-capability-dashboard table { width: 100%; border-collapse: collapse; }
.usersim-capability-dashboard th, .usersim-capability-dashboard td { border: 1px solid #334155; padding: 12px 14px; text-align: left; }
.usersim-capability-dashboard th { background: #1e293b; }
.usersim-capability-dashboard .drawer { position: fixed; top: 0; right: 0; bottom: 0; width: min(920px, 92vw); overflow-y: auto; background: #0f172a; border-left: 1px solid #475569; padding: 24px; box-shadow: -18px 0 48px rgba(0,0,0,.35); z-index: 10; }
.usersim-capability-dashboard .close { float: right; }
.usersim-capability-dashboard .reasoning { border-left: 4px solid #60a5fa; padding: 10px 12px; background: rgba(96,165,250,.1); margin: 0 16px 12px; border-radius: 6px; }
/* ---- Per-axis structured findings (replaces single-blob reasoning) ---- */
.usersim-capability-dashboard .full-conversation-panel .findings { padding: 12px 16px 4px; border-bottom: 1px solid rgba(51,65,85,.6); }
.usersim-capability-dashboard .findings-section h5 { margin: 0 0 8px; font-size: 0.92em; text-transform: uppercase; letter-spacing: .06em; color: #f87171; }
.usersim-capability-dashboard .findings-section.findings-other summary { cursor: pointer; padding: 8px 0; color: #94a3b8; font-size: 0.92em; }
.usersim-capability-dashboard .findings-section.findings-other[open] summary { color: #cbd5e1; }
.usersim-capability-dashboard .finding-card { border: 1px solid #334155; border-radius: 8px; padding: 10px 12px; margin: 8px 0; background: #0f172a; }
.usersim-capability-dashboard .finding-card.failed { border-color: #b91c1c; background: rgba(185,28,28,.08); border-left: 4px solid #b91c1c; }
.usersim-capability-dashboard .finding-card.passed { border-left: 4px solid #15803d; }
.usersim-capability-dashboard .finding-card.dim { opacity: 0.85; }
.usersim-capability-dashboard .finding-card header { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-bottom: 6px; }
.usersim-capability-dashboard .finding-card .axis-name { background: rgba(148,163,184,.18); padding: 2px 8px; border-radius: 4px; font-size: 0.92em; }
.usersim-capability-dashboard .finding-card .tag { font-size: 0.78em; padding: 2px 8px; border-radius: 999px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; }
.usersim-capability-dashboard .finding-card .tag-critical { background: rgba(202,138,4,.2); color: #fde68a; border: 1px solid #ca8a04; }
.usersim-capability-dashboard .finding-card .tag-source { background: rgba(96,165,250,.15); color: #bfdbfe; border: 1px solid rgba(96,165,250,.4); }
.usersim-capability-dashboard .finding-card .tag-score { background: rgba(148,163,184,.18); color: #e2e8f0; border: 1px solid rgba(148,163,184,.4); }
.usersim-capability-dashboard .finding-card.failed .tag-score { background: rgba(185,28,28,.15); color: #fecaca; border-color: rgba(185,28,28,.5); }
.usersim-capability-dashboard .finding-card .finding-reasoning { margin: 6px 0 0; padding: 8px 12px; background: rgba(15,23,42,.5); border-radius: 6px; line-height: 1.55; }
.usersim-capability-dashboard .finding-card .finding-why-label { color: #93c5fd; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; font-size: 0.78em; margin-right: 6px; }
.usersim-capability-dashboard .finding-card .finding-why-text { color: #e5e7eb; white-space: pre-wrap; word-break: break-word; }
.usersim-capability-dashboard .finding-card.failed .finding-why-label { color: #fca5a5; }
.usersim-capability-dashboard .finding-reasoning-empty { margin: 6px 0 0; font-style: italic; }
.usersim-capability-dashboard .reasoning-fallback { margin: 0 16px 12px; }
.usersim-capability-dashboard .reasoning-fallback summary { cursor: pointer; color: #94a3b8; padding: 8px 0; }
.usersim-capability-dashboard .language-caveat { margin: 10px 0 16px; padding: 9px 12px; border-left: 3px solid #38bdf8; background: rgba(56,189,248,.08); border-radius: 3px; font-size: 13px; line-height: 1.5; }
.usersim-capability-dashboard .language-caveat.language-caveat-warn { border-left-color: #fbbf24; background: rgba(202,138,4,.13); }
.usersim-capability-dashboard .language-caveat p { margin: 6px 0 0; }
.usersim-capability-dashboard .axis-breakdown { margin: 12px 0 18px; }
.usersim-capability-dashboard .axis-breakdown-table { border-collapse: collapse; width: 100%; }
.usersim-capability-dashboard .axis-breakdown-table td { padding: 5px 10px; border-bottom: 1px solid #1e293b; vertical-align: middle; }
.usersim-capability-dashboard .axis-breakdown-table .axis-value { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.usersim-capability-dashboard .axis-breakdown-table tr.axis-failed td { background: rgba(202,138,4,.13); }
.usersim-capability-dashboard .axis-breakdown-table tr.axis-failed .axis-value { color: #fbbf24; font-weight: 600; }
.usersim-capability-dashboard .axis-list { margin: 12px 0; }
.usersim-capability-dashboard .axis-list ul { margin: 6px 0 0; padding: 0; list-style: none; display: flex; flex-wrap: wrap; gap: 4px 18px; }
.usersim-capability-dashboard .axis-list li { display: inline-flex; align-items: center; gap: 6px; padding: 2px 0; }
.usersim-capability-dashboard .axis-list li::before { content: "\\2022"; color: #94a3b8; line-height: 1; }
.usersim-capability-dashboard .axis-list code { background: rgba(148,163,184,.15); padding: 1px 6px; border-radius: 4px; font-size: 0.9em; }
/* ---- Full Conversation panel (mirrors 01_simulate.ipynb Rich panel) ---- */
.usersim-capability-dashboard .full-conversation-panel { border: 1px solid #334155; border-radius: 12px; margin: 18px 0; background: #0f172a; overflow: hidden; }
.usersim-capability-dashboard .full-conversation-panel header { background: #1e293b; padding: 12px 16px; border-bottom: 1px solid #334155; display: flex; align-items: baseline; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
.usersim-capability-dashboard .full-conversation-panel header h4 { margin: 0; font-size: 1em; color: #e5e7eb; }
.usersim-capability-dashboard .full-conversation-panel > summary { background: #1e293b; padding: 12px 16px; cursor: pointer; list-style: none; }
.usersim-capability-dashboard .full-conversation-panel[open] > summary { border-bottom: 1px solid #334155; }
.usersim-capability-dashboard .full-conversation-panel > summary::-webkit-details-marker { display: none; }
.usersim-capability-dashboard .full-conversation-panel > summary::before { content: "\\25B6"; color: #94a3b8; font-size: 0.8em; margin-right: 8px; }
.usersim-capability-dashboard .full-conversation-panel[open] > summary::before { content: "\\25BC"; }
.usersim-capability-dashboard .full-conversation-panel .snippet-summary-line { display: inline-flex; align-items: baseline; justify-content: space-between; gap: 12px; flex-wrap: wrap; width: calc(100% - 24px); }
.usersim-capability-dashboard .full-conversation-panel .snippet-summary-line h4 { margin: 0; font-size: 1em; color: #e5e7eb; }
.usersim-capability-dashboard .full-conversation-panel .snippet-summary-failed { margin: 4px 0 0 24px; font-size: 0.85em; color: #fca5a5; }
.usersim-capability-dashboard .finding-axis-description .finding-why-label { color: #94a3b8; }
.usersim-capability-dashboard .full-conversation-panel .snippet-meta { font-size: 0.85em; }
.usersim-capability-dashboard .full-conversation-panel .turns { padding: 8px 0; }
.usersim-capability-dashboard .full-conversation-panel .turn { display: grid; grid-template-columns: 180px 1fr; gap: 16px; padding: 12px 18px; }
.usersim-capability-dashboard .full-conversation-panel .turn + .turn { border-top: 1px solid rgba(51,65,85,.5); }
.usersim-capability-dashboard .full-conversation-panel .turn-label { font-weight: 700; font-size: 0.92em; padding-top: 2px; }
.usersim-capability-dashboard .full-conversation-panel .role-user .turn-label { color: #94a3b8; }
.usersim-capability-dashboard .full-conversation-panel .role-assistant .turn-label { color: #76b900; }
.usersim-capability-dashboard .full-conversation-panel .role-tool .turn-label { color: #fb923c; }
.usersim-capability-dashboard .full-conversation-panel .role-tool-call .turn-label { color: #c084fc; }
/* Probe section — h5 outside, colored card inside (matches WHAT FAILED layout) */
.usersim-capability-dashboard .probe-section { padding: 12px 16px 4px; border-bottom: 1px solid rgba(51,65,85,.6); }
.usersim-capability-dashboard .probe-section h5 { margin: 0 0 8px; font-size: 0.92em; text-transform: uppercase; letter-spacing: .06em; color: #c084fc; }
.usersim-capability-dashboard .probe-card { padding: 12px 14px; margin: 8px 0; background: rgba(192,132,252,.08); border: 1px solid rgba(192,132,252,.4); border-left: 4px solid #c084fc; border-radius: 8px; }
.usersim-capability-dashboard .probe-meta { display: flex; flex-wrap: wrap; gap: 4px 24px; margin-bottom: 8px; }
.usersim-capability-dashboard .probe-meta code { background: rgba(148,163,184,.18); padding: 1px 6px; border-radius: 4px; font-size: 0.92em; }
.usersim-capability-dashboard .probe-description { margin: 0; color: #cbd5e1; line-height: 1.5; font-style: italic; }
/* Conversation section — h5 outside, blue-bordered card around the turns */
.usersim-capability-dashboard .conversation-section { padding: 12px 16px 16px; }
.usersim-capability-dashboard .conversation-section h5 { margin: 0 0 8px; font-size: 0.92em; text-transform: uppercase; letter-spacing: .06em; color: #60a5fa; }
.usersim-capability-dashboard .conversation-card { background: rgba(96,165,250,.06); border: 1px solid rgba(96,165,250,.3); border-left: 4px solid #60a5fa; border-radius: 8px; overflow: hidden; }
.usersim-capability-dashboard .conversation-card .turns { padding: 4px 0; }
/* In-sim per-turn assistant judge chips (mirrors preview.py _judge_chip_for_turn) */
.usersim-capability-dashboard .judge-chip { display: inline-block; margin-left: 8px; font-weight: 800; font-size: 1em; }
.usersim-capability-dashboard .judge-chip.ok { color: #4ade80; }
.usersim-capability-dashboard .judge-chip.fail { color: #f87171; }
/* Per-turn assistant quality breakdown panel */
.usersim-capability-dashboard .per-turn-quality { padding: 12px 16px; border-bottom: 1px solid rgba(51,65,85,.6); }
.usersim-capability-dashboard .per-turn-quality summary { cursor: pointer; padding: 4px 0; color: #93c5fd; font-weight: 700; font-size: 0.92em; }
.usersim-capability-dashboard .per-turn-quality[open] summary { color: #bfdbfe; margin-bottom: 8px; }
.usersim-capability-dashboard .per-turn-list { display: flex; flex-direction: column; gap: 6px; padding-left: 12px; }
.usersim-capability-dashboard .per-turn-row { line-height: 1.55; }
.usersim-capability-dashboard .per-turn-label { font-weight: 700; color: #e5e7eb; margin-right: 8px; }
.usersim-capability-dashboard .per-turn-status { font-weight: 700; text-transform: uppercase; font-size: 0.82em; padding: 1px 8px; border-radius: 4px; }
.usersim-capability-dashboard .per-turn-status.ok { color: #14532d; background: #4ade80; }
.usersim-capability-dashboard .per-turn-status.fail { color: #fff1f2; background: #b91c1c; }
.usersim-capability-dashboard .per-turn-explanation { color: #cbd5e1; }
/* ---- Markdown body inside conversation turns ---- */
.usersim-capability-dashboard .markdown-body { color: #e5e7eb; line-height: 1.55; word-wrap: break-word; }
.usersim-capability-dashboard .markdown-body > *:first-child { margin-top: 0; }
.usersim-capability-dashboard .markdown-body > *:last-child { margin-bottom: 0; }
.usersim-capability-dashboard .markdown-body p { margin: 0 0 0.6em; white-space: pre-wrap; }
.usersim-capability-dashboard .markdown-body h1, .usersim-capability-dashboard .markdown-body h2, .usersim-capability-dashboard .markdown-body h3, .usersim-capability-dashboard .markdown-body h4 { margin: 0.8em 0 0.4em; line-height: 1.3; }
.usersim-capability-dashboard .markdown-body ul, .usersim-capability-dashboard .markdown-body ol { margin: 0.4em 0 0.6em; padding-left: 22px; }
.usersim-capability-dashboard .markdown-body li { margin: 0.15em 0; }
.usersim-capability-dashboard .markdown-body code { background: rgba(148,163,184,.18); padding: 1px 6px; border-radius: 4px; font-size: 0.92em; }
.usersim-capability-dashboard .markdown-body pre { background: #0b1220; border: 1px solid #334155; border-radius: 8px; padding: 12px 14px; overflow-x: auto; margin: 0.6em 0; }
.usersim-capability-dashboard .markdown-body pre code { background: transparent; padding: 0; font-size: 0.88em; line-height: 1.5; }
.usersim-capability-dashboard .markdown-body blockquote { border-left: 3px solid #475569; margin: 0.4em 0; padding: 0.2em 0 0.2em 12px; color: #cbd5e1; }
.usersim-capability-dashboard .markdown-body table { border-collapse: collapse; margin: 0.6em 0; }
.usersim-capability-dashboard .markdown-body table th, .usersim-capability-dashboard .markdown-body table td { border: 1px solid #334155; padding: 6px 10px; }
.usersim-capability-dashboard .markdown-body table th { background: #1e293b; }
.usersim-capability-dashboard .markdown-body a { color: #93c5fd; }
.usersim-capability-dashboard .markdown-body strong { color: #f1f5f9; }
"""
