#!/usr/bin/env python3

"""exp2_figures.py

Essential visualizations for Exp2 results.

Current exp2 result files in this repo are shaped as:
  out/exp2_results_{dataset}.json
where each file contains a dict:
  {strategy_name: {accuracy, correct_count, total_count, results: [ ... ]}}

Each result item contains (at minimum):
  - is_correct (bool)
  - misinformation_strategy (str)
  - mallm_log (dict) with debate metadata such as agreements / turns / voting.

This script focuses on essential evaluations:
- Accuracy by misinformation strategy (overall + per dataset)
- Agreement rate by strategy (overall + per dataset)
- Accuracy vs agreement (do disagreements correlate with correctness?)
- Misinformed opinion maintenance (how often do agents end in *agreed but incorrect* outcomes?)

It loads files matching:
- out/exp2_results_*.json
- out/exp2_ablation_results_*.json (only used for the separate ablation comparison figures)
"""

from __future__ import annotations

import argparse
import os
import json
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
import seaborn as sns

from shared_utils import load_results, find_result_files, extract_dataset_name, infer_model_from_result_path
from shared_visualization import (
    setup_plot_style,
    save_plot,
    get_named_colors,
    get_sequential_cmap,
    get_diverging_cmap,
    get_palette as get_project_palette,
)


setup_plot_style()

MISINFO_CATEGORY_LABEL = "Misinformation Category"


def _safe_load_results(file_path: str) -> Optional[dict]:
    """Load JSON result file safely; return None if unreadable/malformed."""
    try:
        data = load_results(file_path)
        if not isinstance(data, dict):
            print(f"[WARN] Skipping non-dict results file: {file_path}")
            return None
        return data
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        print(f"[WARN] Skipping unreadable results file: {file_path} ({e})")
        return None

STRATEGY_ORDER = [
    "false_fact",
    "clickbait",
    "conspiracy",
    "framing",
    "hoax",
    "other",
    "propaganda",
    "rumor",
    "satire",
]

# Display labels
_DATASET_LABEL = {
    "complex_web_questions": "Complex Web Questions",
    "ethics": "Ethics",
    "logiqa": "LogiQA",
    "winogrande": "WinoGrande",
    "cwq": "Complex Web Questions",
}


def _display_dataset_name(ds: str) -> str:
    ds = str(ds)
    if ds in _DATASET_LABEL:
        return _DATASET_LABEL[ds]
    return ds.replace("_", " ").strip().title()

# Semantic labels/colors used across figures
_COND_LABEL = {
    "single_baseline": "Single-agent\nuninformed",
    "single_misinformed": "Single-agent\nmisinformed",
    "multi_baseline": "Multi-agent\nuninformed",
    "multi_misinformed": "Multi-agent\nmisinformed",
}

_COMPACT_DATASET_LABEL = {
    "complex_web_questions": "CWQ",
    "cwq": "CWQ",
    "ethics": "Ethics",
    "winogrande": "WinoGrande",
    "logiqa": "LogiQA",
}

_COND_MARKERS = {
    "single_baseline": "o",
    "single_misinformed": "s",
    "multi_baseline": "^",
    "multi_misinformed": "D",
}


def _panel_dataset_title(ds: str) -> str:
    ds_key = str(ds).strip().lower()
    if ds_key in {"complex_web_questions", "cwq"}:
        return "ComplexWebQ"
    return _display_dataset_name(ds)


def _compact_dataset_label(ds: str) -> str:
    ds_key = str(ds).strip().lower()
    if ds_key in _COMPACT_DATASET_LABEL:
        return _COMPACT_DATASET_LABEL[ds_key]
    return _display_dataset_name(ds)


def _model_short_label(m: str) -> str:
    raw = str(m).strip().lower()
    if "llama" in raw and "3.3" in raw:
        return "Llama-3.3"
    if "glm" in raw and "4.7" in raw:
        return "GLM-4.7"
    if "gpt-oss" in raw:
        return "GPT-OSS-120B"
    s = str(m).replace("_", "/", 1)
    return s.split("/", 1)[1] if "/" in s else s


def _model_order_key(m: str) -> tuple[int, str]:
    s = str(m).lower()
    if "llama" in s:
        return (0, s)
    if "glm" in s:
        return (1, s)
    if "gpt-oss" in s:
        return (2, s)
    return (3, s)


def _cond_legend_label(cond_key: str) -> str:
    return _COND_LABEL[cond_key].replace("\n", " ")


def plot_exp1_vs_exp2_overall_accuracy_by_dataset_dumbbell(
    pv: pd.DataFrame,
    datasets: List[str],
    conds: List[str],
    models: List[str],
    cond_color: Dict[str, str],
    output_dir: str,
) -> None:
    """Horizontal dumbbell variant of exp1 vs exp2 overall accuracy by dataset."""
    multi_model = len(models) > 1
    row_specs: List[tuple[str, str, str, float]] = []
    model_groups: List[tuple[str, float]] = []
    group_gap = 0.55
    y_cur = 0.0

    def _get_acc(ds: str, model: str, cond: str) -> float:
        key = (ds, model)
        try:
            if key in pv.index and cond in pv.columns and pd.notna(pv.loc[key, cond]):
                return float(pv.loc[key, cond])
        except Exception:
            pass
        return float("nan")

    if multi_model:
        for m_i, model in enumerate(models):
            if m_i > 0:
                y_cur += group_gap
            group_ys: List[float] = []
            for ds in datasets:
                row_specs.append((ds, model, _compact_dataset_label(ds), y_cur))
                group_ys.append(y_cur)
                y_cur += 1.0
            model_groups.append((_model_short_label(model), float(np.mean(group_ys))))
    else:
        model = models[0] if models else ""
        for ds in datasets:
            row_specs.append((ds, model, _compact_dataset_label(ds), y_cur))
            y_cur += 1.0

    y_max = max((spec[3] for spec in row_specs), default=0.0)
    base_fs = float(plt.rcParams.get("font.size", 12))
    title_fs = max(21.0, base_fs * 1.75)
    axis_label_fs = max(22.0, base_fs * 1.65)
    tick_fs = max(18.0, base_fs * 1.40)
    group_header_fs = max(17.0, base_fs * 1.30)
    legend_fs = max(17.0, base_fs * 1.35)
    val_fs = max(16.0, base_fs * 1.25)

    fig_h = max(7.4, 0.88 * (y_max + 1.0) + 3.4)
    fig, ax = plt.subplots(figsize=(15.5, fig_h))

    for ds, model, _row_label, y in row_specs:
        row_points: List[tuple[float, str, str, str]] = []
        for cond in conds:
            vv = _get_acc(ds, model, cond)
            if pd.isna(vv):
                continue
            row_points.append((vv, cond_color.get(cond, "#cccccc"), _COND_MARKERS.get(cond, "o"), cond))

        points_by_cond = {cond: vv for vv, _color, _marker, cond in row_points}
        for group in (
            ("single_baseline", "single_misinformed"),
            ("multi_baseline", "multi_misinformed"),
        ):
            group_vals = [points_by_cond[c] for c in group if c in points_by_cond]
            if len(group_vals) >= 2:
                ax.plot(
                    [min(group_vals), max(group_vals)],
                    [y, y],
                    color="#bdbdbd",
                    linewidth=2.0,
                    solid_capstyle="round",
                    zorder=1,
                )

        for vv, color, marker, _cond in row_points:
            ax.scatter(
                [vv],
                [y],
                s=220,
                marker=marker,
                color=color,
                edgecolors="#2e2e2e",
                linewidths=0.9,
                zorder=3,
            )

        for pt_idx, (vv, color, _marker, _cond) in enumerate(sorted(row_points, key=lambda p: p[0])):
            if pt_idx % 2 == 0:
                xytext = (0, -14)
                va = "top"
            else:
                xytext = (0, 14)
                va = "bottom"
            ax.annotate(
                f"{vv:.2f}",
                (vv, y),
                xytext=xytext,
                textcoords="offset points",
                ha="center",
                va=va,
                fontsize=val_fs,
                color=color,
            )

    ax.set_title(
        "Single- and Multi-Agent Accuracy by Dataset",
        fontweight="bold",
        fontsize=title_fs,
    )
    ax.set_xlabel("Accuracy", fontsize=axis_label_fs)
    ax.set_ylabel("")
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(-0.85, y_max + 0.85)
    ax.set_yticks([spec[3] for spec in row_specs])
    ax.set_yticklabels([spec[2] for spec in row_specs], fontsize=tick_fs)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=tick_fs)
    ax.grid(axis="x", alpha=0.25, linestyle="--", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    if multi_model and model_groups:
        from matplotlib.transforms import blended_transform_factory

        group_label_trans = blended_transform_factory(ax.transAxes, ax.transData)
        for model_label, y_center in model_groups:
            ax.text(
                -0.24,
                y_center,
                model_label,
                rotation=90,
                ha="center",
                va="center",
                fontsize=group_header_fs,
                fontweight="bold",
                color=get_named_colors().get("slate", "#4A4A4A"),
                transform=group_label_trans,
                clip_on=False,
            )

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=_COND_MARKERS.get(cond, "o"),
            color="w",
            markerfacecolor=cond_color.get(cond, "#cccccc"),
            markeredgecolor="#2e2e2e",
            markeredgewidth=0.6,
            markersize=16,
            linestyle="None",
            label=_cond_legend_label(cond),
        )
        for cond in conds
    ]
    fig.legend(
        handles=legend_handles,
        labels=[h.get_label() for h in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=2,
        frameon=True,
        fontsize=legend_fs,
        handlelength=1.2,
        handletextpad=0.45,
        columnspacing=1.6,
        labelspacing=0.45,
        borderpad=0.5,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.87))
    if multi_model:
        fig.subplots_adjust(left=0.24)
    save_plot(output_dir, "exp1_vs_exp2_overall_accuracy_by_dataset_dumbbell.pdf")


def _plot_gradient_segment(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color_start: str,
    color_end: str,
    *,
    linewidth: float = 2.8,
    linestyle: str = "-",
    n: int = 48,
    zorder: int = 2,
) -> None:
    """Draw a line segment with a color gradient from start to end."""
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    points = np.column_stack([xs, ys]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    cmap = mcolors.LinearSegmentedColormap.from_list("", [color_start, color_end])
    lc = LineCollection(
        segments,
        cmap=cmap,
        linewidths=linewidth,
        linestyle=linestyle,
        capstyle="round",
        zorder=zorder,
    )
    lc.set_array(np.linspace(0.0, 1.0, len(segments)))
    ax.add_collection(lc)


def plot_exp1_vs_exp2_overall_accuracy_by_dataset_gradient(
    pv: pd.DataFrame,
    datasets: List[str],
    models: List[str],
    cond_color: Dict[str, str],
    output_dir: str,
) -> None:
    """Faceted line plot: Clean (uninformed) -> Misinf. showing accuracy change per setup."""
    multi_model = len(models) > 1
    x_clean, x_mis = 0.0, 1.0
    x_ticks = [x_clean, x_mis]
    x_labels = ["Clean", "Misinf."]

    setup_series = (
        ("single", "single_baseline", "single_misinformed"),
        ("multi", "multi_baseline", "multi_misinformed"),
    )
    setup_line_color = {
        "single": "#8a8a8a",
        "multi": cond_color.get("multi_misinformed", "#ffbb6f"),
    }

    def _get_acc(ds: str, model: str, cond: str) -> float:
        key = (ds, model)
        try:
            if key in pv.index and cond in pv.columns and pd.notna(pv.loc[key, cond]):
                return float(pv.loc[key, cond])
        except Exception:
            pass
        return float("nan")

    def _model_linestyle(model: str) -> str:
        return "--" if "glm" in str(model).lower() else "-"

    base_fs = float(plt.rcParams.get("font.size", 12))
    axis_label_fs = max(18.0, base_fs * 1.40)
    tick_fs = max(15.0, base_fs * 1.20)
    panel_title_fs = max(16.0, base_fs * 1.30)
    val_fs = max(13.0, base_fs * 1.05)
    legend_fs = max(14.0, base_fs * 1.10)

    n_ds = max(1, len(datasets))
    fig_w = max(11.0, 3.8 * n_ds)
    fig, axes = plt.subplots(1, n_ds, figsize=(fig_w, 5.2), sharey=True)
    if n_ds == 1:
        axes = [axes]

    for ax_i, ds in enumerate(datasets):
        ax = axes[ax_i]
        for model in models:
            ls = _model_linestyle(model) if multi_model else "-"
            for setup_key, base_cond, mis_cond in setup_series:
                y0 = _get_acc(ds, model, base_cond)
                y1 = _get_acc(ds, model, mis_cond)
                if pd.isna(y0) and pd.isna(y1):
                    continue
                if pd.isna(y0):
                    y0 = y1
                if pd.isna(y1):
                    y1 = y0

                c0 = cond_color.get(base_cond, setup_line_color[setup_key])
                c1 = cond_color.get(mis_cond, setup_line_color[setup_key])
                _plot_gradient_segment(
                    ax,
                    x_clean,
                    y0,
                    x_mis,
                    y1,
                    c0,
                    c1,
                    linewidth=2.8,
                    linestyle=ls,
                )
                ax.scatter(
                    [x_clean, x_mis],
                    [y0, y1],
                    s=52,
                    color=[c0, c1],
                    edgecolors="#2e2e2e",
                    linewidths=0.7,
                    zorder=3,
                )
                for x_pt, y_pt, pt_color in (
                    (x_clean, y0, c0),
                    (x_mis, y1, c1),
                ):
                    ax.annotate(
                        f"{y_pt:.2f}",
                        (x_pt, y_pt),
                        xytext=(0, 7),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=val_fs,
                        color=pt_color,
                    )

        ax.set_title(_panel_dataset_title(ds), fontweight="bold", fontsize=panel_title_fs)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=tick_fs)
        ax.set_xlim(-0.12, 1.12)
        ax.set_ylim(0.3, 1.0)
        ax.set_yticks([0.3, 0.5, 0.7, 0.9, 1.0])
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.grid(axis="y", alpha=0.28, linestyle="-", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Accuracy", fontsize=axis_label_fs)

    legend_handles = [
        plt.Line2D([0], [0], color=setup_line_color["single"], linewidth=2.8, label="Single-agent"),
        plt.Line2D([0], [0], color=setup_line_color["multi"], linewidth=2.8, label="Multi-agent"),
    ]
    legend_labels = [h.get_label() for h in legend_handles]
    if multi_model:
        legend_handles.extend(
            [
                plt.Line2D([0], [0], color="#4a4a4a", linewidth=2.8, linestyle="-", label="Llama-3.3"),
                plt.Line2D([0], [0], color="#4a4a4a", linewidth=2.8, linestyle="--", label="GLM-4.7"),
            ]
        )
        legend_labels = [h.get_label() for h in legend_handles]

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(legend_handles),
        frameon=True,
        fontsize=legend_fs,
        handlelength=2.0,
        columnspacing=1.4,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    save_plot(output_dir, "exp1_vs_exp2_overall_accuracy_by_dataset_gradient.pdf")


class _LegendHeaderHandler(HandlerBase):
    """Legend handler for header rows: label only, no handle whitespace."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        p = Rectangle((0, 0), 0, 0, linewidth=0, edgecolor="none", facecolor="none")
        p.set_transform(trans)
        return [p]


def _apply_axis_style(ax: plt.Axes) -> None:
    """Small cosmetics to make plots more publication-friendly."""
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.18)


def _append_legend_top_row(
    fig: plt.Figure,
    handles: List[object],
    labels: List[str],
    *,
    title: str = "",
    y: float = 1.18,
    ncol: Optional[int] = None,
    frameon: bool = True,
    fontsize: Optional[float] = None,
    title_fontsize: Optional[float] = None,
) -> None:
    """Place a figure-level legend at the top in a single row."""
    if not handles or not labels:
        return
    ncol = int(ncol if ncol is not None else len(labels))
    fig.legend(
        handles,
        labels,
        title=title,
        loc="upper center",
        bbox_to_anchor=(0.5, float(y)),
        ncol=ncol,
        frameon=frameon,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
    )


def _move_ax_legend_to_top_row(ax: plt.Axes, *, title: str = "", y: float = 1.18) -> None:
    """Remove axis legend and re-create it as a top-row figure legend."""
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    _append_legend_top_row(ax.figure, handles, labels, title=title, y=y, ncol=len(labels))


def _style_rotated_xticklabels(ax: plt.Axes, rotation: float = 35, *, pad: float = 2.0) -> None:
    """Rotate x tick labels and anchor their RIGHT edge at the tick position."""
    ax.tick_params(axis="x", pad=pad)
    for lab in ax.get_xticklabels():
        lab.set_rotation(rotation)
        lab.set_ha("right")
        lab.set_rotation_mode("anchor")


def _annotate_bars(ax: plt.Axes, labels: List[str], values: List[float], ns: List[int]) -> None:
    """Annotate bars with accuracy and n on top (kept compact)."""
    for i, (lab, v, n) in enumerate(zip(labels, values, ns)):
        if v is None or pd.isna(v):
            continue
        ax.text(i, min(0.98, float(v) + 0.02), f"{float(v):.3f}\n(n={int(n)})", ha="center", va="bottom", fontsize=11)


def _tint(hex_color: str, amount: float) -> str:
    """Lighten a hex color by mixing with white (amount in [0,1])."""
    try:
        r, g, b = mcolors.to_rgb(hex_color)
    except Exception:
        return hex_color
    amount = max(0.0, min(1.0, float(amount)))
    r2, g2, b2 = (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)
    return mcolors.to_hex((r2, g2, b2))


def _canonical_dataset_name(name: str) -> str:
    # Keep filenames stable but make plots readable.
    mapping = {
        "cwq": "complex_web_questions",
    }
    return mapping.get(name, name)


def _extract_agreement(mallm_log: Optional[Dict]) -> Optional[bool]:
    """Return last recorded agreement if present."""
    if not isinstance(mallm_log, dict):
        return None

    agreements = mallm_log.get("agreements")
    if not isinstance(agreements, list) or not agreements:
        return None

    last = agreements[-1]
    if isinstance(last, dict) and "agreement" in last:
        try:
            return bool(last["agreement"])
        except Exception:
            return None

    return None


def _display_strategy_name(strategy: str) -> str:
    """Human-friendly strategy names for figures."""
    s = str(strategy).strip().lower()
    if s in {"", "none", "baseline"}:
        return "Baseline"
    if s == "false_fact":
        return "Neutral"
    if strategy == "avg_all_strategies":
        return "Average"
    return s.replace("_", " ").strip().title()


def _strategy_color_key(strategy: str) -> str:
    """Map key strategies to requested semantic colors."""
    s = str(strategy).strip().lower()
    if s in {"none", "baseline"}:
        return "baseline"
    if s == "false_fact":
        return "relevant"
    if s.startswith("irrelevant_"):
        return "irrelevant"
    return "other"


_SEMANTIC_COLORS = {
    "baseline": "#2E7D32",   # green
    "relevant": "#C62828",   # red
    "irrelevant": "#1565C0", # blue
}

# Prefer the canonical mapping defined in plot_config.py (single source of truth).
try:
    import plot_config  # type: ignore

    _SEMANTIC_COLORS = dict(getattr(plot_config, "MISINFO_RELEVANCE_COLORS", {})) or _SEMANTIC_COLORS
except Exception:
    pass


def _canon_solution(sol: Optional[object]) -> Optional[str]:
    """Canonicalize a 'solution' string for equality comparisons across turns."""
    if sol is None:
        return None
    try:
        s = str(sol).strip()
    except Exception:
        return None
    if not s:
        return None

    # Common prefixes from outputs
    for prefix in ("Final Solution:", "Final Answer:", "Solution:", "Answer:"):
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix) :].strip()
            break

    # Collapse whitespace + normalize case for robust matching
    s = " ".join(s.split())
    return s.lower() or None


def _infer_agent_types(mallm_log: Optional[Dict]) -> Dict[str, str]:
    """Infer agent types from persona descriptions.

    Returns mapping: agent_id -> {"misinformed", "correct_info"}.
    """
    if not isinstance(mallm_log, dict):
        return {}
    personas = mallm_log.get("personas")
    if not isinstance(personas, list):
        return {}

    out: Dict[str, str] = {}
    for p in personas:
        if not isinstance(p, dict):
            continue
        aid = p.get("agentId") or p.get("agent_id")
        if not isinstance(aid, str) or not aid.strip():
            continue
        desc = str(p.get("personaDescription") or "").lower()
        # Heuristic: the misinformed agent is the one with injected "following information"
        # (this is how exp2 creates the adversarial personaDescription).
        if ("following information" in desc) or ("extra information" in desc):
            out[aid] = "misinformed"
        else:
            out[aid] = "correct_info"
    return out


def _extract_solutions_by_agent_turn(mallm_log: Optional[Dict]) -> Dict[str, Dict[int, Optional[str]]]:
    """Extract per-agent per-turn canonical solutions from globalMemory.

    Output: agent_id -> {turn -> canonical_solution}
    """
    if not isinstance(mallm_log, dict):
        return {}
    mem = mallm_log.get("globalMemory")
    if not isinstance(mem, list) or not mem:
        return {}

    # Some turns can have multiple messages from the same agent; keep the latest by message_id.
    latest: Dict[tuple[str, int], tuple[int, Optional[str]]] = {}
    for m in mem:
        if not isinstance(m, dict):
            continue
        aid = m.get("agent_id") or m.get("agentId")
        if not isinstance(aid, str) or not aid.strip():
            continue
        try:
            turn = int(m.get("turn"))
        except Exception:
            continue
        mid = m.get("message_id")
        try:
            mid_i = int(mid) if mid is not None else -1
        except Exception:
            mid_i = -1

        sol = _canon_solution(m.get("solution"))
        key = (aid, turn)
        prev = latest.get(key)
        if prev is None or mid_i >= prev[0]:
            latest[key] = (mid_i, sol)

    out: Dict[str, Dict[int, Optional[str]]] = {}
    for (aid, turn), (_mid, sol) in latest.items():
        out.setdefault(aid, {})[turn] = sol
    return out


def load_exp2_opinion_persistence_long(out_dir: str = "out", model: Optional[str] = None) -> pd.DataFrame:
    """Pair-level opinion persistence across turns.

    Each row is one comparison:
      turn t: agent X supports solution A
      turn t+1: agent X supports solution A

    Columns:
      - dataset, strategy
      - agent_type: {"misinformed", "correct_info"}
      - persisted: bool (same canonical solution across consecutive turns)
      - turn_from, turn_to (ints)
    """
    files = find_result_files(out_dir=out_dir, prefix="exp2_results_")
    rows: List[Dict] = []

    for file_path in files:
        if not os.path.basename(file_path).startswith("exp2_results_"):
            continue
        dataset = _canonical_dataset_name(extract_dataset_name(file_path))
        model_key = infer_model_from_result_path(file_path, out_dir=out_dir)
        if model is not None and str(model_key) != str(model):
            continue
        data = _safe_load_results(file_path)
        if data is None:
            continue

        for strategy_key, payload in data.items():
            if not isinstance(payload, dict) or "results" not in payload:
                continue
            results = payload.get("results") or []
            if not isinstance(results, list):
                continue

            for r in results:
                if not isinstance(r, dict):
                    continue
                ml = r.get("mallm_log")
                agent_types = _infer_agent_types(ml)
                sols = _extract_solutions_by_agent_turn(ml)
                if not sols:
                    continue

                for aid, by_turn in sols.items():
                    a_type = agent_types.get(aid, "correct_info")
                    turns = sorted(t for t in by_turn.keys() if isinstance(t, int))
                    if len(turns) < 2:
                        continue
                    for t in turns:
                        t2 = t + 1
                        if t2 not in by_turn:
                            continue
                        s1 = by_turn.get(t)
                        s2 = by_turn.get(t2)
                        if s1 is None or s2 is None:
                            continue
                        rows.append(
                            {
                                "dataset": dataset,
                                "model": model_key,
                                "strategy": str(strategy_key).replace("strategy_", ""),
                                "agent_type": a_type,
                                "persisted": bool(s1 == s2),
                                "turn_from": int(t),
                                "turn_to": int(t2),
                            }
                        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["dataset"] = df["dataset"].astype(str)
    df["model"] = df["model"].astype(str)
    df["dataset_display"] = df["dataset"].map(_display_dataset_name)
    df["strategy"] = df["strategy"].astype(str)
    df["agent_type"] = df["agent_type"].astype(str)
    df["persisted"] = df["persisted"].astype(bool)
    return df


def _load_exp1_baseline(out_dir: str, model: Optional[str] = None) -> tuple[float, Dict[str, float]]:
    """Return (overall_baseline_accuracy, baseline_accuracy_by_dataset) from exp1 results."""
    import exp1_figures as exp1

    df1 = exp1.load_exp1_results_long(out_dir=out_dir)
    if model is not None and not df1.empty and "model" in df1.columns:
        df1 = df1[df1["model"].astype(str) == str(model)].copy()
    if df1.empty:
        return float("nan"), {}

    base = df1[df1["relevance"] == "none"].copy()
    if base.empty:
        return float("nan"), {}

    overall = float(base["is_correct"].mean())
    by_ds = base.groupby("dataset", observed=True)["is_correct"].mean().to_dict()
    by_ds = {str(k): float(v) for k, v in by_ds.items()}
    return overall, by_ds


def load_exp2_results_long(out_dir: str = "out", model: Optional[str] = None) -> pd.DataFrame:
    """Load all exp2 result files and return a normalized long-form DataFrame."""

    files = find_result_files(out_dir=out_dir, prefix="exp2_results_")
    rows: List[Dict] = []

    for file_path in files:
        if not os.path.basename(file_path).startswith("exp2_results_"):
            continue

        dataset = _canonical_dataset_name(extract_dataset_name(file_path))
        model_key = infer_model_from_result_path(file_path, out_dir=out_dir)
        if model is not None and str(model_key) != str(model):
            continue
        data = _safe_load_results(file_path)
        if data is None:
            continue

        for strategy_key, payload in data.items():
            if not isinstance(payload, dict) or "results" not in payload:
                continue

            results = payload.get("results") or []
            if not isinstance(results, list):
                continue

            for r in results:
                if not isinstance(r, dict):
                    continue

                ml = r.get("mallm_log")

                rows.append(
                    {
                        "dataset": dataset,
                        "model": model_key,
                        "strategy": str(strategy_key),
                        "misinformation_strategy": r.get("misinformation_strategy"),
                        "is_correct": bool(r.get("is_correct", False)),
                        "turns": (ml or {}).get("turns") if isinstance(ml, dict) else None,
                        "decision_success": (ml or {}).get("decisionSuccess") if isinstance(ml, dict) else None,
                        "agreed": _extract_agreement(ml),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["dataset"] = df["dataset"].astype(str)
    df["model"] = df["model"].astype(str)
    df["dataset_display"] = df["dataset"].map(_display_dataset_name)
    df["strategy"] = df["strategy"].astype(str)
    # Normalize some common strategy prefixes/variants if they ever appear.
    df["strategy"] = df["strategy"].str.replace("strategy_", "", regex=False)

    # Keep turns numeric if present
    df["turns"] = pd.to_numeric(df["turns"], errors="coerce")

    return df


def load_exp2_ablation_results_long(out_dir: str = "out", model: Optional[str] = None) -> pd.DataFrame:
    """Load Exp2 ablation result files and return a normalized long-form DataFrame.

    Expected files:
      out/exp2_ablation_results_{dataset}.json
    with the same schema as exp2_results_{dataset}.json.
    """

    files = find_result_files(out_dir=out_dir, prefix="exp2_ablation_results_")
    rows: List[Dict] = []

    for file_path in files:
        if not os.path.basename(file_path).startswith("exp2_ablation_results_"):
            continue

        model_key = infer_model_from_result_path(file_path, out_dir=out_dir)
        if model is not None and str(model_key) != str(model):
            continue
        # shared_utils.extract_dataset_name() is tailored to exp2_results_* filenames.
        # For ablation filenames, parse directly.
        base = os.path.basename(file_path)
        ds_raw = base.replace("exp2_ablation_results_", "").replace(".json", "")
        dataset = _canonical_dataset_name(ds_raw)
        data = _safe_load_results(file_path)
        if data is None:
            continue

        for strategy_key, payload in data.items():
            if not isinstance(payload, dict) or "results" not in payload:
                continue

            results = payload.get("results") or []
            if not isinstance(results, list):
                continue

            for r in results:
                if not isinstance(r, dict):
                    continue

                ml = r.get("mallm_log")

                rows.append(
                    {
                        "dataset": dataset,
                        "model": model_key,
                        "strategy": str(strategy_key),
                        "misinformation_strategy": r.get("misinformation_strategy"),
                        "is_correct": bool(r.get("is_correct", False)),
                        "turns": (ml or {}).get("turns") if isinstance(ml, dict) else None,
                        "decision_success": (ml or {}).get("decisionSuccess") if isinstance(ml, dict) else None,
                        "agreed": _extract_agreement(ml),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["dataset"] = df["dataset"].astype(str)
    df["model"] = df["model"].astype(str)
    df["dataset_display"] = df["dataset"].map(_display_dataset_name)
    df["strategy"] = df["strategy"].astype(str)
    df["strategy"] = df["strategy"].str.replace("strategy_", "", regex=False)
    df["turns"] = pd.to_numeric(df["turns"], errors="coerce")
    return df


def plot_accuracy_heatmap(
    df: pd.DataFrame, output_dir: str, *, results_out_dir: str = "out", model: Optional[str] = None
) -> None:
    """Accuracy by dataset x strategy."""
    if df.empty:
        print("No data available.")
        return

    pivot = df.groupby(["strategy", "dataset"], observed=True)["is_correct"].mean().unstack("dataset")
    pivot = pivot.rename(columns={c: _display_dataset_name(c) for c in pivot.columns})

    # Add exp1 baseline ("none") as a reference row, if available.
    _, base_by_ds = _load_exp1_baseline(results_out_dir, model=model)
    if base_by_ds:
        # Align to displayed dataset labels (columns are already prettified above).
        pivot.loc["none"] = pd.Series({_display_dataset_name(ds): acc for ds, acc in base_by_ds.items()})
        pivot = pivot.sort_index()

    # Prettify strategy labels (incl. false_fact -> Neutral, none -> Baseline).
    pivot = pivot.rename(index={s: _display_strategy_name(s) for s in pivot.index})

    plt.figure(figsize=(10.5, max(4, 0.45 * len(pivot.index))))
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap=get_diverging_cmap(),
        vmin=0,
        vmax=1,
        linewidths=0.4,
        linecolor=(0, 0, 0, 0.08),
        cbar_kws={"label": "Accuracy"},
    )
    ax.set_title("Accuracy by Strategy and Dataset", fontweight="bold")
    ax.set_xlabel("Dataset")
    ax.set_ylabel(MISINFO_CATEGORY_LABEL)
    plt.tight_layout()
    save_plot(output_dir, "exp2_accuracy_by_strategy_dataset.pdf")


def plot_accuracy_overall(
    df: pd.DataFrame, output_dir: str, *, results_out_dir: str = "out", model: Optional[str] = None
) -> None:
    """Overall accuracy per strategy (across all datasets)."""
    if df.empty:
        print("No data available.")
        return

    model_list = []
    if "model" in df.columns:
        model_list = sorted(df["model"].dropna().astype(str).unique().tolist())
    multi_model = len(model_list) > 1

    group_cols = ["strategy"] + (["model"] if multi_model else [])
    stats = (
        df.groupby(group_cols, observed=True)["is_correct"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "accuracy", "count": "n"})
    )
    # Keep strategy order stable by overall accuracy (across models when present).
    order = (
        df.groupby("strategy", observed=True)["is_correct"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # Add exp1 baseline ("none") as a reference bar, if available.
    if multi_model:
        base_rows = []
        for m in model_list:
            base_m, _ = _load_exp1_baseline(results_out_dir, model=m)
            if pd.notna(base_m):
                base_rows.append({"strategy": "none", "model": m, "accuracy": base_m, "n": 0})
        if base_rows:
            stats = pd.concat([pd.DataFrame(base_rows), stats], ignore_index=True)
            order = ["none"] + order
    else:
        base_overall, _ = _load_exp1_baseline(results_out_dir, model=model)
        if pd.notna(base_overall):
            stats = pd.concat(
                [
                    pd.DataFrame([{"strategy": "none", "accuracy": base_overall, "n": 0}]),
                    stats,
                ],
                ignore_index=True,
            )
            order = ["none"] + order

    from figure_style import get_palette, cycle_palette
    pal3 = get_palette(3)
    stats["strategy_display"] = stats["strategy"].map(_display_strategy_name)
    order_display = [_display_strategy_name(s) for s in order]

    if multi_model:
        stats["model_display"] = stats["model"].astype(str).str.replace("_", "/", n=1, regex=False)
        model_display = [m.replace("_", "/", 1) for m in model_list]
        model_palette_base = get_project_palette()
        model_palette = {m: model_palette_base[i % len(model_palette_base)] for i, m in enumerate(model_display)}
        plt.figure(figsize=(14.8, 6.0))
        ax = sns.barplot(
            data=stats,
            x="strategy_display",
            y="accuracy",
            hue="model_display",
            order=order_display,
            hue_order=model_display,
            palette=model_palette,
        )
    else:
        # Use the 8-color palette for many strategies, but pin the baseline bar to neutral gray.
        bar_colors = cycle_palette(8, len(order_display))
        for i, s in enumerate(order):
            if str(s).strip().lower() in {"none", "baseline"}:
                bar_colors[i] = pal3[1]  # gray
        palette = {lab: c for lab, c in zip(order_display, bar_colors)}
        plt.figure(figsize=(13.2, 5.6))
        ax = sns.barplot(
            data=stats,
            x="strategy_display",
            y="accuracy",
            order=order_display,
            hue="strategy_display",
            hue_order=order_display,
            dodge=False,
            palette=palette,
            legend=False,
        )
    _apply_axis_style(ax)
    ax.set_title("Overall Accuracy by Strategy", fontweight="bold")
    ax.set_xlabel(MISINFO_CATEGORY_LABEL)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    _style_rotated_xticklabels(ax, rotation=35)

    if multi_model:
        _move_ax_legend_to_top_row(ax, title="", y=1.20)
    else:
        # Add n labels
        for i, row in stats.reset_index(drop=True).iterrows():
            if int(row["n"]) > 0:
                ax.text(i, min(0.98, row["accuracy"] + 0.02), f"n={int(row['n'])}", ha="center", va="bottom", fontsize=11)

    if multi_model:
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    else:
        plt.tight_layout()
    save_plot(output_dir, "exp2_accuracy_by_strategy_overall.pdf")


def plot_accuracy_by_strategy_by_dataset(
    df: pd.DataFrame, output_dir: str, *, results_out_dir: str = "out", model: Optional[str] = None
) -> None:
    """Grouped bars: accuracy by strategy, split by dataset."""
    if df.empty:
        print("No data available.")
        return

    stats = (
        df.groupby(["strategy", "dataset"], observed=True)["is_correct"]
        .mean()
        .reset_index()
        .rename(columns={"is_correct": "accuracy"})
    )
    stats["dataset_display"] = stats["dataset"].map(_display_dataset_name)

    # Add exp1 baseline ("none") as an additional strategy, per dataset.
    _, base_by_ds = _load_exp1_baseline(results_out_dir, model=model)
    if base_by_ds:
        base_rows = [{"strategy": "none", "dataset": ds, "accuracy": acc} for ds, acc in base_by_ds.items()]
        stats = pd.concat([pd.DataFrame(base_rows), stats], ignore_index=True)

    # Stable order by overall exp2 accuracy
    order = (
        df.groupby("strategy", observed=True)["is_correct"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    if base_by_ds:
        order = ["none"] + order

    stats["strategy_display"] = stats["strategy"].map(_display_strategy_name)
    order_display = [_display_strategy_name(s) for s in order]

    plt.figure(figsize=(15.2, 6.6))
    ax = sns.barplot(data=stats, x="strategy_display", y="accuracy", hue="dataset_display", order=order_display)
    ax.set_title("Accuracy by Strategy (Split by Dataset)", fontweight="bold")
    ax.set_xlabel(MISINFO_CATEGORY_LABEL)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    _style_rotated_xticklabels(ax, rotation=35)
    _move_ax_legend_to_top_row(ax, title="", y=1.20)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    save_plot(output_dir, "exp2_accuracy_by_strategy_by_dataset.pdf")


def plot_overall_accuracy_by_dataset(df: pd.DataFrame, output_dir: str) -> None:
    """Overall accuracy per dataset (across all strategies)."""
    if df.empty:
        print("No data available.")
        return

    stats = (
        df.groupby("dataset", observed=True)["is_correct"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "accuracy", "count": "n"})
        .sort_values("accuracy", ascending=False)
    )
    stats["dataset_display"] = stats["dataset"].map(_display_dataset_name)

    colors = get_named_colors()
    plt.figure(figsize=(8.5, 5))
    ax = sns.barplot(
        data=stats,
        x="dataset_display",
        y="accuracy",
        order=stats["dataset_display"].tolist(),
        color=colors["sky"],
    )
    _apply_axis_style(ax)
    ax.set_title("Overall Accuracy by Dataset", fontweight="bold")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    for i, row in stats.reset_index(drop=True).iterrows():
        ax.text(i, min(0.98, row["accuracy"] + 0.02), f"n={int(row['n'])}", ha="center", va="bottom", fontsize=11)
    plt.tight_layout()
    save_plot(output_dir, "exp2_overall_accuracy_by_dataset.pdf")


def plot_agreement_heatmap(df: pd.DataFrame, output_dir: str) -> None:
    """Agreement rate by dataset x strategy."""
    if df.empty:
        print("No data available.")
        return

    sub = df[df["agreed"].notna()].copy()
    if sub.empty:
        print("No agreement data available (mallm_log.agreements missing).")
        return

    pivot = sub.groupby(["strategy", "dataset"], observed=True)["agreed"].mean().unstack("dataset")
    pivot = pivot.rename(columns={c: _display_dataset_name(c) for c in pivot.columns})
    pivot = pivot.rename(index={s: _display_strategy_name(s) for s in pivot.index})
    plt.figure(figsize=(10.5, max(4, 0.45 * len(pivot.index))))
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap=get_sequential_cmap(),
        vmin=0,
        vmax=1,
        linewidths=0.4,
        linecolor=(0, 0, 0, 0.08),
        cbar_kws={"label": "Agreement rate"},
    )
    ax.set_title("Agreement Rate by Strategy and Dataset", fontweight="bold")
    ax.set_xlabel("Dataset")
    ax.set_ylabel(MISINFO_CATEGORY_LABEL)
    plt.tight_layout()
    save_plot(output_dir, "exp2_agreement_rate_by_strategy_dataset.pdf")


def plot_accuracy_vs_agreement(df: pd.DataFrame, output_dir: str) -> None:
    """Compare accuracy when agents agreed vs disagreed."""
    if df.empty:
        print("No data available.")
        return

    sub = df[df["agreed"].notna()].copy()
    if sub.empty:
        print("No agreement data available (mallm_log.agreements missing).")
        return

    stats = sub.groupby("agreed", observed=True)["is_correct"].agg(["mean", "count"]).reset_index()
    stats["agreed"] = stats["agreed"].map({True: "agreed", False: "disagreed"})

    from figure_style import get_palette
    pal2 = get_palette(2)
    plt.figure(figsize=(6, 5))
    ax = sns.barplot(
        data=stats,
        x="agreed",
        y="mean",
        order=["agreed", "disagreed"],
        hue="agreed",
        hue_order=["agreed", "disagreed"],
        dodge=False,
        palette={"agreed": pal2[0], "disagreed": pal2[1]},
        legend=False,
    )
    ax.set_title("Accuracy vs Agreement", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)

    for i, r in stats.iterrows():
        ax.text(i, min(0.98, r["mean"] + 0.02), f"n={int(r['count'])}", ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    save_plot(output_dir, "exp2_accuracy_vs_agreement.pdf")


def plot_accuracy_vs_agreement_by_dataset(df: pd.DataFrame, output_dir: str) -> None:
    """Accuracy vs agreement, split by dataset."""
    if df.empty:
        print("No data available.")
        return

    sub = df[df["agreed"].notna()].copy()
    if sub.empty:
        print("No agreement data available (mallm_log.agreements missing).")
        return

    stats = (
        sub.groupby(["dataset", "agreed"], observed=True)["is_correct"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "accuracy", "count": "n"})
    )
    stats["agreement"] = stats["agreed"].map({True: "agreed", False: "disagreed"})
    stats["dataset_display"] = stats["dataset"].map(_display_dataset_name)

    from figure_style import get_palette
    pal2 = get_palette(2)
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=stats,
        x="dataset_display",
        y="accuracy",
        hue="agreement",
        hue_order=["agreed", "disagreed"],
        palette={"agreed": pal2[0], "disagreed": pal2[1]},
    )
    ax.set_title("Accuracy vs Agreement (Split by Dataset)", fontweight="bold")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    _move_ax_legend_to_top_row(ax, title="", y=1.20)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    save_plot(output_dir, "exp2_accuracy_vs_agreement_by_dataset.pdf")


def plot_opinion_persistence(dfp: pd.DataFrame, output_dir: str) -> None:
    """Plot opinion persistence for misinformed vs correct-info agents.

    Uses df from load_exp2_opinion_persistence_long(): pair-level persisted over consecutive turns.
    """
    if dfp.empty:
        print("No persistence data available (missing globalMemory/solutions).")
        return

    from figure_style import get_palette
    pal2 = get_palette(2)
    colors = get_named_colors()

    # Overall by strategy
    overall = (
        dfp.groupby(["agent_type", "strategy"], observed=True)["persisted"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "rate", "count": "n"})
    )
    overall["strategy_display"] = overall["strategy"].map(_display_strategy_name)

    # Strategy order: sort by misinformed persistence desc (fallback overall)
    if (overall["agent_type"] == "misinformed").any():
        order = (
            overall[overall["agent_type"] == "misinformed"]
            .sort_values("rate", ascending=False)["strategy"]
            .tolist()
        )
    else:
        order = overall.groupby("strategy", observed=True)["rate"].mean().sort_values(ascending=False).index.tolist()
    order_display = [_display_strategy_name(s) for s in order]

    # 1) Misinformed-only
    mis = overall[overall["agent_type"] == "misinformed"].copy()
    if not mis.empty:
        plt.figure(figsize=(12, max(4, 0.45 * len(order))))
        ax = sns.barplot(data=mis, x="rate", y="strategy_display", order=order_display, color=pal2[1])
        ax.set_title("Opinion Persistence (Misinformation Agent)\nP(Same Solution t→t+1)", fontweight="bold")
        ax.set_xlabel("Persistence rate")
        ax.set_ylabel(MISINFO_CATEGORY_LABEL)
        ax.set_xlim(0, 1)
        for i, row in mis.set_index("strategy_display").reindex(order_display).dropna().reset_index().iterrows():
            ax.text(min(0.98, float(row["rate"]) + 0.02), i, f"{row['rate']:.3f} (n={int(row['n'])})", va="center", fontsize=11)
        plt.tight_layout()
        save_plot(output_dir, "exp2_opinion_persistence_misinformed.pdf")

    # 2) Correct-info-only
    cor = overall[overall["agent_type"] == "correct_info"].copy()
    if not cor.empty:
        plt.figure(figsize=(12, max(4, 0.45 * len(order))))
        ax = sns.barplot(data=cor, x="rate", y="strategy_display", order=order_display, color=pal2[0])
        ax.set_title("Opinion Persistence (Correct-Info Agents)\nP(Same Solution t→t+1)", fontweight="bold")
        ax.set_xlabel("Persistence rate")
        ax.set_ylabel(MISINFO_CATEGORY_LABEL)
        ax.set_xlim(0, 1)
        for i, row in cor.set_index("strategy_display").reindex(order_display).dropna().reset_index().iterrows():
            ax.text(min(0.98, float(row["rate"]) + 0.02), i, f"{row['rate']:.3f} (n={int(row['n'])})", va="center", fontsize=11)
        plt.tight_layout()
        save_plot(output_dir, "exp2_opinion_persistence_correct_info.pdf")

    # 3) Comparison (grouped)
    plt.figure(figsize=(14, 6))
    comp = overall.copy()
    comp["agent_type"] = comp["agent_type"].map({"misinformed": "misinformation", "correct_info": "correct_info"}).fillna(comp["agent_type"])
    ax = sns.barplot(
        data=comp,
        x="strategy_display",
        y="rate",
        hue="agent_type",
        order=order_display,
        hue_order=["misinformation", "correct_info"],
        palette={"misinformation": pal2[1], "correct_info": pal2[0]},
    )
    ax.set_title("Opinion Persistence Comparison\nP(Same Solution t→t+1)", fontweight="bold")
    ax.set_xlabel(MISINFO_CATEGORY_LABEL)
    ax.set_ylabel("Persistence rate")
    ax.set_ylim(0, 1)
    _style_rotated_xticklabels(ax, rotation=35)
    _move_ax_legend_to_top_row(ax, title="", y=1.20)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    save_plot(output_dir, "exp2_opinion_persistence_comparison.pdf")


def plot_opinion_persistence_by_dataset(dfp: pd.DataFrame, output_dir: str) -> None:
    """Heatmap view: persistence rate by dataset x strategy, split by agent_type."""
    if dfp.empty:
        print("No persistence data available (missing globalMemory/solutions).")
        return

    sub = (
        dfp.groupby(["agent_type", "strategy", "dataset"], observed=True)["persisted"]
        .mean()
        .reset_index()
    )
    if sub.empty:
        print("No persistence rows available.")
        return

    # Two-panel heatmap: misinformed vs correct-info
    pivot_m = sub[sub["agent_type"] == "misinformed"].pivot(index="strategy", columns="dataset", values="persisted")
    pivot_c = sub[sub["agent_type"] == "correct_info"].pivot(index="strategy", columns="dataset", values="persisted")
    pivot_m = pivot_m.rename(columns={c: _display_dataset_name(c) for c in pivot_m.columns})
    pivot_c = pivot_c.rename(columns={c: _display_dataset_name(c) for c in pivot_c.columns})

    def _with_avg_row(piv: pd.DataFrame) -> pd.DataFrame:
        if piv.empty:
            return piv
        avg = piv.mean(axis=0, skipna=True)  # average across strategies, per dataset
        piv2 = piv.copy()
        piv2.loc["avg_all_strategies"] = avg
        return piv2

    pivot_m = _with_avg_row(pivot_m)
    pivot_c = _with_avg_row(pivot_c)

    def _ordered_with_labels(piv: pd.DataFrame) -> pd.DataFrame:
        if piv.empty:
            return piv
        # Force 'false_fact' first, then the rest of STRATEGY_ORDER, then any extras, then 'average' row.
        idx = [str(i) for i in piv.index.tolist()]
        base = [s for s in STRATEGY_ORDER if s in idx]
        extras = sorted([s for s in idx if s not in set(STRATEGY_ORDER) and s != "avg_all_strategies"])
        order = []
        if "false_fact" in base:
            order.append("false_fact")
        order += [s for s in base if s != "false_fact"]
        order += extras
        if "avg_all_strategies" in idx:
            order.append("avg_all_strategies")
        piv2 = piv.reindex(order)
        piv2.index = [_display_strategy_name(str(s)) for s in piv2.index.tolist()]
        return piv2

    pivot_m = _ordered_with_labels(pivot_m)
    pivot_c = _ordered_with_labels(pivot_c)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(4, 0.5 * max(len(pivot_m.index), len(pivot_c.index), 8))))
    if not pivot_m.empty:
        from figure_style import get_colormaps
        sns.heatmap(pivot_m, annot=True, fmt=".3f", cmap=get_colormaps().sequential, vmin=0, vmax=1, ax=ax1)
    ax1.set_title("Misinformation agent\npersistence", fontweight="bold")
    ax1.set_xlabel("Dataset")
    ax1.set_ylabel(MISINFO_CATEGORY_LABEL)

    if not pivot_c.empty:
        from figure_style import get_colormaps
        sns.heatmap(pivot_c, annot=True, fmt=".3f", cmap=get_colormaps().sequential, vmin=0, vmax=1, ax=ax2)
    ax2.set_title("Correct-info agents\npersistence", fontweight="bold")
    ax2.set_xlabel("Dataset")
    ax2.set_ylabel("")

    fig.suptitle("Opinion Persistence by Dataset\nP(Same Solution t→t+1)", y=1.02, fontweight="bold")
    plt.tight_layout()
    save_plot(output_dir, "exp2_opinion_persistence_by_dataset.pdf")


def plot_opinion_persistence_delta_barplot(dfp: pd.DataFrame, output_dir: str) -> None:
    """Delta bar plot: (correct-info persistence) − (misinformation persistence).

    This is a companion to `exp2_opinion_persistence_by_dataset.pdf`, but focuses on the *gap*
    between the two agent types rather than absolute rates.
    """
    if dfp.empty:
        print("No persistence data available (missing globalMemory/solutions).")
        return

    sub = (
        dfp.groupby(["agent_type", "strategy", "dataset"], observed=True)["persisted"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "rate", "count": "n"})
    )
    if sub.empty:
        print("No persistence rows available.")
        return

    # Compute delta per strategy×dataset.
    piv = sub.pivot_table(index=["strategy", "dataset"], columns="agent_type", values="rate", aggfunc="mean")
    if "misinformed" not in piv.columns or "correct_info" not in piv.columns:
        print("Missing one of agent types (misinformed/correct_info); cannot compute delta.")
        return

    delta = (piv["correct_info"] - piv["misinformed"]).reset_index().rename(columns={0: "delta", "correct_info": "delta"})

    # Add per-dataset average (across strategies) row for parity with the heatmap figure.
    avg_rows = []
    for ds, g in delta.groupby("dataset", observed=True):
        avg_rows.append({"strategy": "avg_all_strategies", "dataset": ds, "delta": float(g["delta"].mean())})
    if avg_rows:
        delta = pd.concat([delta, pd.DataFrame(avg_rows)], ignore_index=True)

    # Enforce strategy ordering + display labels.
    # Paper order request (displayed): neutral, clickbait, hoax, rumor, satire, propaganda, framing, conspiracy, other.
    _DELTA_STRATEGY_ORDER = [
        "false_fact",   # Neutral
        "clickbait",
        "hoax",
        "rumor",
        "satire",
        "propaganda",
        "framing",
        "conspiracy",
        "other",
    ]
    def _order_with_labels(strategies: List[str]) -> tuple[list[str], list[str]]:
        idx = [str(i) for i in strategies]
        base = [s for s in _DELTA_STRATEGY_ORDER if s in idx]
        extras = sorted([s for s in idx if s not in set(_DELTA_STRATEGY_ORDER) and s != "avg_all_strategies"])
        order = base + extras
        if "avg_all_strategies" in idx:
            order.append("avg_all_strategies")
        return order, [_display_strategy_name(s) for s in order]

    strat_order, strat_display_order = _order_with_labels(delta["strategy"].unique().tolist())
    delta["strategy"] = delta["strategy"].astype(str)
    delta["strategy_display"] = delta["strategy"].map(_display_strategy_name)

    datasets = sorted(delta["dataset"].astype(str).unique().tolist())
    from figure_style import get_palette
    pal6 = get_palette(6)
    slate = get_named_colors().get("slate", "#4A4A4A")

    fig, axes = plt.subplots(1, len(datasets), figsize=(6.3 * len(datasets), 5.8), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    from matplotlib.ticker import PercentFormatter

    # Choose a sensible symmetric y-range around 0.
    finite = delta["delta"].astype(float)
    finite = finite[pd.notna(finite)]
    m = float(finite.abs().max()) if not finite.empty else 0.2
    ylim = (-(m + 0.05), (m + 0.05))

    for ax, ds in zip(axes, datasets):
        dsd = delta[delta["dataset"].astype(str) == str(ds)].copy()
        # Align bars to consistent order, inserting NaNs for missing strategies.
        dsd = dsd.set_index("strategy").reindex(strat_order).reset_index()
        dsd["strategy_display"] = dsd["strategy"].map(_display_strategy_name)
        vals = dsd["delta"].astype(float).tolist()
        xs = list(range(len(strat_order)))
        # Positive deltas: correct-info more persistent; negative: misinformed more persistent.
        pos_c = pal6[2]  # sage
        neg_c = pal6[3]  # rose
        bar_colors = [pos_c if (v is not None and pd.notna(v) and v >= 0) else neg_c for v in vals]

        ax.bar(xs, [0.0 if (v is None or pd.isna(v)) else float(v) for v in vals], color=bar_colors, alpha=0.85)
        ax.axhline(0, color="black", linewidth=1.2, alpha=0.6)
        ax.set_title(_display_dataset_name(ds), fontweight="bold")
        ax.set_xlabel(MISINFO_CATEGORY_LABEL)
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
        ax.set_xticks(xs)
        ax.set_xticklabels([_display_strategy_name(s) for s in strat_order], rotation=35, ha="right", rotation_mode="anchor")
        ax.grid(axis="y", alpha=0.18)

        # Small value labels.
        for x, v in zip(xs, vals):
            if v is None or pd.isna(v):
                continue
            vv = float(v)
            ax.text(
                x,
                vv + (0.01 if vv >= 0 else -0.01),
                f"{vv * 100:.1f}%",
                ha="center",
                va="bottom" if vv >= 0 else "top",
                fontsize=10,
                color=slate,
            )

    axes[0].set_ylabel("Δ Persistence (%)")
    # Mixed-weight title: main line bold, definition line regular.
    fig.text(
        0.5,
        1.10,
        "Opinion Persistence Delta by Misinformation Category (Turn-to-Turn)",
        ha="center",
        va="top",
        fontweight="bold",
    )
    fig.text(
        0.5,
        1.03,
        "Δ Persistence = Persistence(Uninformed) − Persistence(Misinformed)",
        ha="center",
        va="top",
        fontweight="normal",
    )
    plt.tight_layout()
    save_plot(output_dir, "exp2_opinion_persistence_delta_barplot.pdf")


def plot_opinion_persistence_delta_barplot_grid(dfp: pd.DataFrame, output_dir: str) -> None:
    """Alternative delta bar plot: small-multiples grid.

    Rows  = misinformation categories (strategies).
    Cols  = datasets.
    Each cell is a single horizontal bar; positive = green, negative = red.
    """
    if dfp.empty:
        print("No persistence data available (missing globalMemory/solutions).")
        return

    sub = (
        dfp.groupby(["agent_type", "strategy", "dataset"], observed=True)["persisted"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "rate", "count": "n"})
    )
    if sub.empty:
        print("No persistence rows available.")
        return

    piv = sub.pivot_table(index=["strategy", "dataset"], columns="agent_type", values="rate", aggfunc="mean")
    if "misinformed" not in piv.columns or "correct_info" not in piv.columns:
        print("Missing one of agent types (misinformed/correct_info); cannot compute delta.")
        return

    delta = (piv["correct_info"] - piv["misinformed"]).reset_index().rename(columns={0: "delta", "correct_info": "delta"})

    avg_rows = []
    for ds, g in delta.groupby("dataset", observed=True):
        avg_rows.append({"strategy": "avg_all_strategies", "dataset": ds, "delta": float(g["delta"].mean())})
    if avg_rows:
        delta = pd.concat([delta, pd.DataFrame(avg_rows)], ignore_index=True)

    _DELTA_STRATEGY_ORDER = [
        "false_fact",
        "clickbait",
        "hoax",
        "rumor",
        "satire",
        "propaganda",
        "framing",
        "conspiracy",
        "other",
    ]
    strategies_present = [str(s) for s in delta["strategy"].unique().tolist()]
    base = [s for s in _DELTA_STRATEGY_ORDER if s in strategies_present]
    extras = sorted([s for s in strategies_present if s not in set(_DELTA_STRATEGY_ORDER) and s != "avg_all_strategies"])
    strat_order = base + extras
    if "avg_all_strategies" in strategies_present:
        strat_order.append("avg_all_strategies")

    datasets = sorted(delta["dataset"].astype(str).unique().tolist())
    if not datasets or not strat_order:
        print("Nothing to plot for delta grid.")
        return

    n_rows = len(strat_order)
    n_cols = len(datasets)

    xlim = (-0.20, 0.10)

    from figure_style import get_palette
    pal6 = get_palette(6)
    pos_color = pal6[2]
    neg_color = pal6[3]
    slate = get_named_colors().get("slate", "#4A4A4A")

    fig_w = max(10.0, 2.8 * n_cols + 4.0)
    fig_h = max(6.5, 0.85 * n_rows + 2.5)

    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(fig_w, fig_h),
        sharey=True,
        sharex=True,
    )
    if n_cols == 1:
        axes = [axes]

    strat_labels = [_display_strategy_name(s) for s in strat_order]
    y_positions = np.arange(n_rows)
    bar_height = 0.65

    for col_idx, (ax, ds) in enumerate(zip(axes, datasets)):
        dsd = delta[delta["dataset"].astype(str) == str(ds)].copy()
        dsd = dsd.set_index("strategy").reindex(strat_order).reset_index()
        vals = dsd["delta"].astype(float).tolist()

        bar_vals = [0.0 if (v is None or pd.isna(v)) else float(v) for v in vals]
        bar_colors = [pos_color if bv >= 0 else neg_color for bv in bar_vals]

        ax.barh(
            y_positions,
            bar_vals,
            height=bar_height,
            color=bar_colors,
            alpha=0.85,
            edgecolor="none",
        )
        ax.axvline(0, color="black", linewidth=1.0, alpha=0.55)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.6, n_rows - 0.4)
        ax.invert_yaxis()

        ax.set_title(_compact_dataset_label(ds), fontweight="bold", fontsize=24, pad=14)
        ax.set_xlabel("Δ Persistence (%)", fontsize=22)
        ax.tick_params(axis="x", labelsize=20)
        ax.grid(axis="x", alpha=0.18, linewidth=0.6)
        for spine_name in ("top", "right"):
            ax.spines[spine_name].set_visible(False)

        from matplotlib.ticker import FuncFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v * 100:+.0f}"))

        if col_idx == 0:
            ax.set_yticks(y_positions)
            ax.set_yticklabels(strat_labels, fontsize=22)
        ax.tick_params(
            axis="y",
            length=0,
            labelleft=(col_idx == 0),
        )

        for y, v in zip(y_positions, vals):
            if v is None or pd.isna(v):
                ax.text(
                    0.0,
                    y,
                    " n/a",
                    ha="left",
                    va="center",
                    fontsize=20,
                    color=slate,
                )
                continue
            vv = float(v)
            text_color = pos_color if vv >= 0 else neg_color
            if vv >= 0:
                ax.text(
                    vv + (xlim[1] - xlim[0]) * 0.012,
                    y,
                    f"{vv * 100:+.1f}",
                    ha="left",
                    va="center",
                    fontsize=20,
                    color=text_color,
                )
            else:
                ax.text(
                    vv - (xlim[1] - xlim[0]) * 0.012,
                    y,
                    f"{vv * 100:+.1f}",
                    ha="right",
                    va="center",
                    fontsize=20,
                    color=text_color,
                )

    axes[0].set_ylabel(MISINFO_CATEGORY_LABEL, fontsize=22)

    fig.text(
        0.5,
        1.04,
        "Opinion Persistence Delta by Misinformation Category (Turn-to-Turn)",
        ha="center",
        va="top",
        fontweight="bold",
        fontsize=26,
    )
    fig.text(
        0.5,
        0.99,
        "Δ Persistence = Persistence(Uninformed) − Persistence(Misinformed)",
        ha="center",
        va="top",
        fontweight="normal",
        fontsize=24,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.subplots_adjust(wspace=0.08)
    save_plot(output_dir, "exp2_opinion_persistence_delta_barplot_grid.pdf")


def plot_misinformed_opinion_maintenance(df: pd.DataFrame, output_dir: str) -> None:
    """How often do agents maintain a misinformed opinion during debate?

    Operationalization (log-derived, dataset-agnostic):
    - "maintained misinformed opinion" := agents reached agreement AND the final answer was incorrect.
      i.e. (agreed == True) AND (is_correct == False)

    This captures *misinformed consensus* at the end of debate.
    """
    if df.empty:
        print("No data available.")
        return

    sub = df[df["agreed"].notna()].copy()
    if sub.empty:
        print("No agreement data available (mallm_log.agreements missing).")
        return

    sub["maintained_misinformed"] = (sub["agreed"] == True) & (sub["is_correct"] == False)  # noqa: E712

    overall = (
        sub.groupby("strategy", observed=True)["maintained_misinformed"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "rate", "count": "n"})
        .sort_values("rate", ascending=False)
    )
    overall["strategy_display"] = overall["strategy"].map(_display_strategy_name)

    by_ds = sub.groupby(["strategy", "dataset"], observed=True)["maintained_misinformed"].mean().unstack("dataset")
    by_ds = by_ds.rename(columns={c: _display_dataset_name(c) for c in by_ds.columns})
    by_ds = by_ds.rename(index={s: _display_strategy_name(s) for s in by_ds.index})

    # One consolidated figure: overall bars + dataset heatmap
    fig_h = max(4.5, 0.45 * max(8, len(by_ds.index)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, fig_h), gridspec_kw={"width_ratios": [1.1, 1.4]})

    sns.barplot(data=overall, x="rate", y="strategy_display", ax=ax1, color=get_named_colors()["slate"])
    ax1.set_title("Misinformed Opinion Maintained\n(Agreed + Incorrect)", fontweight="bold")
    ax1.set_xlabel("Rate")
    ax1.set_ylabel(MISINFO_CATEGORY_LABEL)
    ax1.set_xlim(0, 1)

    # Add value + n labels to bar ends
    for i, row in overall.reset_index(drop=True).iterrows():
        rate = float(row["rate"])
        ax1.text(
            min(0.98, rate + 0.02),
            i,
            f"{rate:.3f}  (n={int(row['n'])})",
            va="center",
            fontsize=11,
        )

    sns.heatmap(by_ds, annot=True, fmt=".3f", cmap="Reds", vmin=0, vmax=1, ax=ax2)
    ax2.set_title("Misinformed Opinion Maintained by Dataset", fontweight="bold")
    ax2.set_xlabel("Dataset")
    ax2.set_ylabel("")

    fig.suptitle("How Often Do Agents Maintain a Misinformed Opinion?", y=1.02, fontweight="bold")
    plt.tight_layout()
    save_plot(output_dir, "exp2_misinformed_opinion_maintenance.pdf")


def plot_exp1_vs_exp2_strategy_comparison(out_dir: str, output_dir: str, model: Optional[str] = None) -> None:
    """Compare Exp1 (single-agent) vs Exp2 (multi-agent) accuracy by strategy and overall.

    - Exp1: uses *relevant* misinformation rows only (excludes baseline + irrelevant_misinformed).
    - Exp2: uses all rows in exp2_results_* (all are strategies).
    """

    # Import lazily to avoid coupling unless requested.
    import exp1_figures as exp1

    df2 = load_exp2_results_long(out_dir=out_dir, model=model)
    df1 = exp1.load_exp1_results_long(out_dir=out_dir)
    if not df1.empty and model is not None and "model" in df1.columns:
        df1 = df1[df1["model"].astype(str) == str(model)].copy()

    if df2.empty or df1.empty:
        print("Missing exp1/exp2 data; cannot compare.")
        return

    # Exp1: relevant-only strategies
    df1s = df1[(df1["relevance"] == "relevant") & (df1["base_strategy"] != "none")].copy()
    df1s["strategy"] = df1s["base_strategy"].astype(str)

    # Exp2: strategies already normalized in load_exp2_results_long
    df2s = df2.copy()

    s1 = df1s.groupby("strategy", observed=True)["is_correct"].agg(["mean", "count"]).reset_index()
    s1["experiment"] = "exp1_single"
    s1.rename(columns={"mean": "accuracy", "count": "n"}, inplace=True)

    s2 = df2s.groupby("strategy", observed=True)["is_correct"].agg(["mean", "count"]).reset_index()
    s2["experiment"] = "exp2_multi"
    s2.rename(columns={"mean": "accuracy", "count": "n"}, inplace=True)

    merged = pd.concat([s1, s2], ignore_index=True)

    # Keep only strategies present in both for a fair side-by-side.
    common = sorted(set(s1["strategy"]).intersection(set(s2["strategy"])))
    merged = merged[merged["strategy"].isin(common)].copy()
    if merged.empty:
        print("No overlapping strategies between exp1 and exp2.")
        return

    # Order by exp2 accuracy (descending) for readability
    order = (
        merged[merged["experiment"] == "exp2_multi"]
        .sort_values("accuracy", ascending=False)["strategy"]
        .tolist()
    )

    from figure_style import get_palette
    pal2 = get_palette(2)
    colors = get_named_colors()
    pretty_exp = {"exp1_single": "Single-agent", "exp2_multi": "Multi-agent"}
    merged_plot = merged.copy()
    merged_plot["experiment_label"] = merged_plot["experiment"].map(pretty_exp).fillna(merged_plot["experiment"])
    merged_plot["strategy_display"] = merged_plot["strategy"].map(_display_strategy_name)
    order_display = [_display_strategy_name(s) for s in order]

    plt.figure(figsize=(14.5, 6))
    ax = sns.barplot(
        data=merged_plot,
        x="strategy_display",
        y="accuracy",
        hue="experiment_label",
        hue_order=["Single-agent", "Multi-agent"],
        order=order_display,
        palette={"Single-agent": pal2[0], "Multi-agent": pal2[1]},
    )
    _apply_axis_style(ax)
    ax.set_title("Accuracy by Strategy (Single vs Multi-Agent)", fontweight="bold")
    ax.set_xlabel(MISINFO_CATEGORY_LABEL)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    _style_rotated_xticklabels(ax, rotation=35)
    _move_ax_legend_to_top_row(ax, title="", y=1.20)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    save_plot(output_dir, "exp1_vs_exp2_accuracy_by_strategy.pdf")

    # Overall comparison (4 bins):
    #  1) single agent baseline (no misinformation)
    #  2) single agent misinformed (relevant strategies only)
    #  3) multi agent baseline (no misinformation ablation)
    #  4) multi agent misinformed (exp2 strategies)
    base_exp1_overall, _ = _load_exp1_baseline(out_dir, model=model)
    df1_base = df1[df1["relevance"] == "none"].copy()

    df2a = load_exp2_ablation_results_long(out_dir=out_dir, model=model)

    overall_rows: List[Dict] = [
        {"condition": "single_baseline", "accuracy": base_exp1_overall, "n": len(df1_base)},
        {"condition": "single_misinformed", "accuracy": float(df1s["is_correct"].mean()), "n": len(df1s)},
    ]
    if not df2a.empty:
        overall_rows.append(
            {"condition": "multi_baseline", "accuracy": float(df2a["is_correct"].mean()), "n": len(df2a)}
        )
    else:
        print("No exp2 ablation rows found; multi_baseline bin will be omitted.")
    overall_rows.append({"condition": "multi_misinformed", "accuracy": float(df2s["is_correct"].mean()), "n": len(df2s)})

    overall = pd.DataFrame(overall_rows)
    cond_order = ["single_baseline", "single_misinformed", "multi_baseline", "multi_misinformed"]
    cond_order = [c for c in cond_order if c in set(overall["condition"])]

    from figure_style import get_palette
    pal3 = get_palette(3)
    palette = {
        "single_baseline": pal3[1],      # gray
        "single_misinformed": pal2[0],   # purple
        "multi_baseline": pal3[1],       # gray
        "multi_misinformed": pal2[1],    # gold
    }
    palette_label = {_COND_LABEL.get(k, k): v for k, v in palette.items()}

    plt.figure(figsize=(8.5, 5))
    ax2 = sns.barplot(
        data=overall,
        x="condition",
        y="accuracy",
        hue="condition",
        order=cond_order,
        hue_order=cond_order,
        dodge=False,
        palette=palette,
    )
    if getattr(ax2, "legend_", None) is not None:
        ax2.legend_.remove()
    _apply_axis_style(ax2)
    ax2.set_title("Overall Accuracy (Single vs Multi-Agent)", fontweight="bold")
    ax2.set_xlabel("")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    # Make x labels human-readable
    ax2.set_xticks(list(range(len(cond_order))))
    ax2.set_xticklabels([_COND_LABEL.get(c, c) for c in cond_order])
    # De-emphasize baselines
    for patch, cond in zip(ax2.patches, cond_order):
        if "baseline" in cond:
            patch.set_alpha(0.55)
    _annotate_bars(
        ax2,
        labels=cond_order,
        values=[float(overall.set_index("condition").loc[c, "accuracy"]) for c in cond_order],
        ns=[int(overall.set_index("condition").loc[c, "n"]) for c in cond_order],
    )
    plt.tight_layout()
    save_plot(output_dir, "exp1_vs_exp2_overall_accuracy.pdf")

    # One consolidated figure: all datasets, fixed strategy order (false_fact leftmost)
    colors = get_named_colors()
    common_datasets = sorted(set(df1s["dataset"]).intersection(set(df2s["dataset"])))
    base_overall, base_by_ds = _load_exp1_baseline(out_dir, model=model)
    exp2_base_by_ds: Dict[str, float] = {}
    if not df2a.empty:
        exp2_base_by_ds = (
            df2a.groupby("dataset", observed=True)["is_correct"].mean().astype(float).to_dict()
        )

    combo_rows = []
    for ds in common_datasets:
        d1 = df1s[df1s["dataset"] == ds]
        d2 = df2s[df2s["dataset"] == ds]
        if d1.empty or d2.empty:
            continue

        a1 = d1.groupby("strategy", observed=True)["is_correct"].mean().to_dict()
        a2 = d2.groupby("strategy", observed=True)["is_correct"].mean().to_dict()

        for strat in STRATEGY_ORDER:
            if strat not in a1 or strat not in a2:
                continue
            combo_rows.append({"dataset": ds, "strategy": strat, "experiment": "exp1_single", "accuracy": a1[strat]})
            combo_rows.append({"dataset": ds, "strategy": strat, "experiment": "exp2_multi", "accuracy": a2[strat]})

    if combo_rows:
        cdf = pd.DataFrame(combo_rows)
        # Ensure fixed ordering
        cdf["strategy"] = pd.Categorical(cdf["strategy"], categories=STRATEGY_ORDER, ordered=True)
        cdf["dataset_display"] = cdf["dataset"].map(_display_dataset_name)
        cdf["strategy_display"] = cdf["strategy"].map(_display_strategy_name)
        cdf["experiment_label"] = cdf["experiment"].map(pretty_exp).fillna(cdf["experiment"])
        order_display = [_display_strategy_name(s) for s in STRATEGY_ORDER]

        from figure_style import get_palette
        pal2 = get_palette(2)
        g = sns.catplot(
            data=cdf,
            kind="bar",
            x="strategy_display",
            y="accuracy",
            hue="experiment_label",
            col="dataset_display",
            col_order=[_display_dataset_name(d) for d in common_datasets],
            order=order_display,
            height=5.2,
            aspect=1.1,
            palette={"Single-agent": pal2[0], "Multi-agent": pal2[1]},
            sharey=True,
        )
        base_fs = float(plt.rcParams.get("font.size", 12))
        panel_title_fs = max(16.0, base_fs * 0.95)
        axis_label_fs = max(15.0, base_fs * 0.90)
        tick_fs = max(13.0, base_fs * 0.82)
        legend_fs = max(13.0, base_fs * 0.82)
        baseline_text_fs = max(12.0, base_fs * 0.75)
        suptitle_fs = max(18.0, base_fs * 1.00)

        g.set_titles("{col_name}", size=panel_title_fs, weight="bold")
        g.set_axis_labels(MISINFO_CATEGORY_LABEL, "Accuracy")
        g.set(ylim=(0, 1))

        # Legend: figure-level, top, single row (extract from seaborn's legend).
        handles: List[object] = []
        labels: List[str] = []
        leg = getattr(g, "_legend", None)
        if leg is not None:
            handles = list(getattr(leg, "legend_handles", None) or getattr(leg, "legendHandles", None) or [])
            labels = [t.get_text() for t in getattr(leg, "texts", [])]
            leg.remove()
        if not handles or not labels:
            handles, labels = g.axes.flat[0].get_legend_handles_labels()
        # Keep legend INSIDE figure bounds to avoid extra whitespace from bbox_inches="tight".
        _append_legend_top_row(g.fig, handles, labels, title="", y=0.90, ncol=len(labels), fontsize=legend_fs)

        # Baseline reference line per dataset panel (match experiment colors)
        for ax, ds in zip(g.axes.flat, common_datasets):
            ax.set_xlabel(MISINFO_CATEGORY_LABEL, fontsize=axis_label_fs)
            ax.set_ylabel("Accuracy", fontsize=axis_label_fs)
            ax.tick_params(axis="both", labelsize=tick_fs)
            _style_rotated_xticklabels(ax, rotation=35)
            if ds in base_by_ds:
                b = base_by_ds[ds]
                ax.axhline(b, color=pal2[0], linestyle="--", linewidth=1.6, alpha=0.95)
                ax.text(
                    0.02,
                    b + 0.02,
                    f"single-agent uninformed={b:.3f}",
                    transform=ax.get_yaxis_transform(),
                    fontsize=baseline_text_fs,
                )
            if ds in exp2_base_by_ds:
                b2 = float(exp2_base_by_ds[ds])
                ax.axhline(b2, color=pal2[1], linestyle=":", linewidth=2.0, alpha=0.95)
                # Place label with minimal overlap with exp1 baseline label.
                dy = -0.06 if b2 > 0.10 else 0.05
                ax.text(
                    0.02,
                    min(0.98, max(0.02, b2 + dy)),
                    f"multi-agent uninformed={b2:.3f}",
                    transform=ax.get_yaxis_transform(),
                    fontsize=baseline_text_fs,
                    color=pal2[1],
                )

        g.fig.suptitle(
            "Accuracy by Misinformation Strategy per Dataset",
            y=0.75,
            fontweight="bold",
            fontsize=suptitle_fs,
        )
        g.fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
        save_plot(output_dir, "exp1_vs_exp2_strategies_all_datasets_fixed_order.pdf")

    # Per-dataset strategy comparison
    common_datasets = sorted(set(df1s["dataset"]).intersection(set(df2s["dataset"])))
    for ds in common_datasets:
        d1 = df1s[df1s["dataset"] == ds]
        d2 = df2s[df2s["dataset"] == ds]
        if d1.empty or d2.empty:
            continue

        s1d = d1.groupby("strategy", observed=True)["is_correct"].agg(["mean", "count"]).reset_index()
        s1d["experiment"] = "exp1_single"
        s1d.rename(columns={"mean": "accuracy", "count": "n"}, inplace=True)

        s2d = d2.groupby("strategy", observed=True)["is_correct"].agg(["mean", "count"]).reset_index()
        s2d["experiment"] = "exp2_multi"
        s2d.rename(columns={"mean": "accuracy", "count": "n"}, inplace=True)

        md = pd.concat([s1d, s2d], ignore_index=True)
        common_ds_strats = sorted(set(s1d["strategy"]).intersection(set(s2d["strategy"])))
        md = md[md["strategy"].isin(common_ds_strats)]
        if md.empty:
            continue

        # Fixed order with false_fact leftmost (plus any extras appended)
        extras = [s for s in common_ds_strats if s not in STRATEGY_ORDER]
        order_ds = [s for s in STRATEGY_ORDER if s in common_ds_strats] + sorted(extras)

        from figure_style import get_palette
        pal2 = get_palette(2)
        plt.figure(figsize=(14.5, 6))
        mdp = md.copy()
        mdp["experiment_label"] = mdp["experiment"].map(pretty_exp).fillna(mdp["experiment"])
        mdp["strategy_display"] = mdp["strategy"].map(_display_strategy_name)
        order_ds_display = [_display_strategy_name(s) for s in order_ds]
        axd = sns.barplot(
            data=mdp,
            x="strategy_display",
            y="accuracy",
            hue="experiment_label",
            hue_order=["Single-agent", "Multi-agent"],
            order=order_ds_display,
            palette={"Single-agent": pal2[0], "Multi-agent": pal2[1]},
        )
        _apply_axis_style(axd)
        axd.set_title(f"Accuracy by Strategy ({_display_dataset_name(ds)})", fontweight="bold")
        axd.set_xlabel(MISINFO_CATEGORY_LABEL)
        axd.set_ylabel("Accuracy")
        axd.set_ylim(0, 1)
        _style_rotated_xticklabels(axd, rotation=35)
        # Add exp1 baseline reference line for this dataset if available.
        if ds in base_by_ds:
            b = base_by_ds[ds]
            axd.axhline(b, color=pal2[0], linestyle="--", linewidth=1.5, alpha=0.9)
            axd.text(0.01, b + 0.02, f"exp1 baseline={b:.3f}", transform=axd.get_yaxis_transform())
        _move_ax_legend_to_top_row(axd, title="", y=1.20)
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
        save_plot(output_dir, f"exp1_vs_exp2_accuracy_by_strategy_{ds}.pdf")

    # Overall per-dataset comparison (4 bins), split by model when available.
    overall_ds: List[Dict] = []
    if model is None and "model" in df2s.columns and "model" in df1s.columns:
        model_candidates = sorted(set(df2s["model"].astype(str)).intersection(set(df1s["model"].astype(str))))
    elif model is not None:
        model_candidates = [str(model)]
    else:
        model_candidates = []

    for m in model_candidates:
        _, base_by_ds_m = _load_exp1_baseline(out_dir, model=m)
        d1_base_m = df1_base[df1_base["model"].astype(str) == m].copy() if "model" in df1_base.columns else df1_base.copy()
        d1_mis_m = df1s[df1s["model"].astype(str) == m].copy() if "model" in df1s.columns else df1s.copy()
        d2_mis_m = df2s[df2s["model"].astype(str) == m].copy() if "model" in df2s.columns else df2s.copy()
        d2a_m = df2a[df2a["model"].astype(str) == m].copy() if (not df2a.empty and "model" in df2a.columns) else df2a.copy()

        ds_common_all = sorted(set(d1_mis_m["dataset"]).intersection(set(d2_mis_m["dataset"])))
        for ds in ds_common_all:
            d1_mis = d1_mis_m[d1_mis_m["dataset"] == ds]
            d2_mis = d2_mis_m[d2_mis_m["dataset"] == ds]
            if d1_mis.empty or d2_mis.empty:
                continue

            # single baseline (from exp1 baseline)
            if ds in base_by_ds_m:
                overall_ds.append(
                    {
                        "dataset": ds,
                        "model": m,
                        "condition": "single_baseline",
                        "accuracy": float(base_by_ds_m[ds]),
                        "n": int(len(d1_base_m[d1_base_m["dataset"] == ds])),
                    }
                )
            # single misinformed
            overall_ds.append(
                {
                    "dataset": ds,
                    "model": m,
                    "condition": "single_misinformed",
                    "accuracy": float(d1_mis["is_correct"].mean()),
                    "n": int(len(d1_mis)),
                }
            )
            # multi baseline:
            # Prefer ablation rows; fallback to false_fact as the closest non-injected baseline proxy.
            d2_base = d2a_m[d2a_m["dataset"] == ds] if not d2a_m.empty else pd.DataFrame()
            if not d2_base.empty:
                overall_ds.append(
                    {
                        "dataset": ds,
                        "model": m,
                        "condition": "multi_baseline",
                        "accuracy": float(d2_base["is_correct"].mean()),
                        "n": int(len(d2_base)),
                    }
                )
            else:
                d2_base_proxy = d2_mis[d2_mis["strategy"].astype(str) == "false_fact"]
                if not d2_base_proxy.empty:
                    overall_ds.append(
                        {
                            "dataset": ds,
                            "model": m,
                            "condition": "multi_baseline",
                            "accuracy": float(d2_base_proxy["is_correct"].mean()),
                            "n": int(len(d2_base_proxy)),
                        }
                    )
            # multi misinformed
            overall_ds.append(
                {
                    "dataset": ds,
                    "model": m,
                    "condition": "multi_misinformed",
                    "accuracy": float(d2_mis["is_correct"].mean()),
                    "n": int(len(d2_mis)),
                }
            )

    if overall_ds:
        odf = pd.DataFrame(overall_ds)
        datasets = sorted(odf["dataset"].astype(str).unique().tolist())
        conds = ["single_baseline", "single_misinformed", "multi_baseline", "multi_misinformed"]
        conds = [c for c in conds if c in set(odf["condition"].astype(str))]
        models = sorted(odf["model"].astype(str).unique().tolist()) if "model" in odf.columns else ["combined"]
        models = sorted(models, key=_model_order_key)

        # Lookup table for fast value retrieval.
        pv = odf.pivot_table(index=["dataset", "model"], columns="condition", values="accuracy", aggfunc="mean")

        # Restore old visual semantics:
        # - Single-agent: blueish-gray shades
        # - Multi-agent: orange shades
        # - Baseline is lighter than misinformed within each setup
        single_dark = "#665567"
        single_light = _tint(single_dark, 0.55)
        multi_dark = "#ffbb6f"
        multi_light = "#ffe0bd"
        cond_color = {
            "single_baseline": single_light,
            "single_misinformed": single_dark,
            "multi_baseline": multi_light,
            "multi_misinformed": multi_dark,
        }

        fig, axo = plt.subplots(figsize=(11.8, 5.8))  # keep width unchanged
        x = np.arange(len(datasets))
        n_slots = max(1, len(conds) * len(models))
        # Wider bars with visible gaps within/between model groups (matches exp1 relevance figure).
        bar_w = min(0.082, 0.90 / max(1, n_slots))
        inner_gap = 0.020
        group_gap = 0.055 if len(models) > 1 else 0.0
        total_span = (
            n_slots * bar_w
            + max(0, n_slots - len(models)) * inner_gap
            + max(0, len(models) - 1) * group_gap
        )
        left = -total_span / 2.0 + bar_w / 2.0
        slot_offsets = []
        cur = left
        for _m_i, _m in enumerate(models):
            for c_i, _c in enumerate(conds):
                slot_offsets.append(cur)
                cur += bar_w
                if c_i < len(conds) - 1:
                    cur += inner_gap
            if _m_i < len(models) - 1:
                cur += group_gap
        model_hatches = ["", "//", "xx", "\\\\", "..", "++"]
        label_color = get_named_colors().get("slate", "#4A4A4A")
        base_fs = float(plt.rcParams.get("font.size", 12))
        val_fs = max(7.0, base_fs * 0.62)

        slot_i = 0
        for m_i, m in enumerate(models):
            for cond in conds:
                heights = []
                xpos = []
                for d_i, ds in enumerate(datasets):
                    key = (ds, m)
                    v = float("nan")
                    try:
                        if key in pv.index and cond in pv.columns and pd.notna(pv.loc[key, cond]):
                            v = float(pv.loc[key, cond])
                    except Exception:
                        v = float("nan")
                    heights.append(v)
                    xpos.append(float(x[d_i] + slot_offsets[slot_i]))
                bars = axo.bar(
                    xpos,
                    [0.0 if pd.isna(v) else v for v in heights],
                    width=bar_w,
                    color=cond_color.get(cond, "#cccccc"),
                    edgecolor="#2e2e2e",
                    linewidth=0.35,
                    hatch=model_hatches[m_i % len(model_hatches)],
                    alpha=0.95,
                    zorder=3,
                )
                for b, v in zip(bars, heights):
                    if pd.isna(v):
                        continue
                    axo.text(
                        b.get_x() + b.get_width() / 2.0,
                        min(0.985, float(v) + 0.010),
                        f"{float(v):.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=val_fs,
                        color=label_color,
                    )
                slot_i += 1

        _apply_axis_style(axo)
        title_fs = max(15.0, base_fs * 1.30)
        axis_label_fs = max(16.0, base_fs * 1.35)
        tick_fs = max(15.0, base_fs * 1.20)
        legend_fs = max(12.0, base_fs * 1.0)

        axo.set_title(
            "Single- and Multi-Agent Accuracy by Dataset",
            fontweight="bold",
            fontsize=title_fs,
        )
        axo.set_xlabel("")
        axo.set_ylabel("Accuracy", fontsize=axis_label_fs)
        axo.set_ylim(0, 1.1)
        axo.set_yticks(np.arange(0.0, 1.01, 0.2))
        axo.set_xticks(x)
        axo.set_xticklabels([_display_dataset_name(d) for d in datasets], fontsize=tick_fs)
        axo.tick_params(axis="y", labelsize=tick_fs)

        def _cond_legend_label(cond_key: str) -> str:
            return _COND_LABEL[cond_key].replace("\n", " ")

        # Legend layout: 4 condition labels in a 2x2 block, spacer column, then
        # 2 model labels in a 1x2 column (matches exp1 relevance figure).
        condition_handles = [
            Patch(facecolor=cond_color["single_baseline"], edgecolor="#2e2e2e", label=_cond_legend_label("single_baseline")),
            Patch(facecolor=cond_color["single_misinformed"], edgecolor="#2e2e2e", label=_cond_legend_label("single_misinformed")),
            Patch(facecolor=cond_color["multi_baseline"], edgecolor="#2e2e2e", label=_cond_legend_label("multi_baseline")),
            Patch(facecolor=cond_color["multi_misinformed"], edgecolor="#2e2e2e", label=_cond_legend_label("multi_misinformed")),
        ]

        def _model_short_label(m: str) -> str:
            raw = str(m).strip().lower()
            if "llama" in raw and "3.3" in raw:
                return "Llama-3.3"
            if "glm" in raw and "4.7" in raw:
                return "GLM-4.7"
            s = str(m).replace("_", "/", 1)
            return s.split("/", 1)[1] if "/" in s else s

        model_handles = [
            Patch(
                facecolor="#f4f4f4",
                edgecolor="#2e2e2e",
                hatch=model_hatches[i % len(model_hatches)],
                label=_model_short_label(m),
            )
            for i, m in enumerate(models)
        ]
        spacer_handle = Patch(facecolor="none", edgecolor="none", label="")
        combined_handles = (
            condition_handles
            + [spacer_handle, spacer_handle]
            + model_handles
        )
        combined_labels = [h.get_label() for h in combined_handles]

        fig.legend(
            handles=combined_handles,
            labels=combined_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=4,
            frameon=True,
            fontsize=legend_fs,
            handlelength=1.2,
            handletextpad=0.45,
            columnspacing=0.95,
            labelspacing=0.55,
            borderpad=0.5,
            borderaxespad=0.0,
        )

        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
        save_plot(output_dir, "exp1_vs_exp2_overall_accuracy_by_dataset.pdf")
        plot_exp1_vs_exp2_overall_accuracy_by_dataset_dumbbell(
            pv, datasets, conds, models, cond_color, output_dir
        )
        plot_exp1_vs_exp2_overall_accuracy_by_dataset_gradient(
            pv, datasets, models, cond_color, output_dir
        )


def plot_exp1_vs_exp2_ablation_vs_exp2_strategy_comparison(
    out_dir: str, output_dir: str, model: Optional[str] = None
) -> None:
    """Compare Exp1 vs Exp2 ablation vs Exp2 (multi-agent) accuracy by strategy and overall.

    - Exp1: uses *relevant* misinformation rows only (excludes baseline + irrelevant_misinformed).
    - Exp2 / Exp2 ablation: use all rows in exp2_results_* / exp2_ablation_results_* (all are strategies).
    """

    import exp1_figures as exp1

    df2 = load_exp2_results_long(out_dir=out_dir, model=model)
    df2a = load_exp2_ablation_results_long(out_dir=out_dir, model=model)
    df1 = exp1.load_exp1_results_long(out_dir=out_dir)
    if not df1.empty and model is not None and "model" in df1.columns:
        df1 = df1[df1["model"].astype(str) == str(model)].copy()

    if df2.empty or df2a.empty or df1.empty:
        print("Missing exp1/exp2/exp2_ablation data; cannot create ablation comparison plots.")
        return

    # Exp1: relevant-only strategies
    df1s = df1[(df1["relevance"] == "relevant") & (df1["base_strategy"] != "none")].copy()
    df1s["strategy"] = df1s["base_strategy"].astype(str)

    df2s = df2.copy()
    df2as = df2a.copy()

    from figure_style import get_palette
    pal3 = get_palette(3)  # purple / gray / gold

    # Overall comparison (strategy rows only for exp1)
    overall = pd.DataFrame(
        [
            {"experiment": "exp1_single", "accuracy": df1s["is_correct"].mean(), "n": len(df1s)},
            {"experiment": "exp2_ablation", "accuracy": df2as["is_correct"].mean(), "n": len(df2as)},
            {"experiment": "exp2_multi", "accuracy": df2s["is_correct"].mean(), "n": len(df2s)},
        ]
    )

    plt.figure(figsize=(7, 5))
    ax2 = sns.barplot(
        data=overall,
        x="experiment",
        y="accuracy",
        hue="experiment",
        dodge=False,
        order=["exp1_single", "exp2_ablation", "exp2_multi"],
        palette={"exp1_single": pal3[0], "exp2_ablation": pal3[1], "exp2_multi": pal3[2]},
    )
    if getattr(ax2, "legend_", None) is not None:
        ax2.legend_.remove()
    _apply_axis_style(ax2)
    ax2.set_title("Overall Accuracy (Single vs Multi-Agent)", fontweight="bold")
    ax2.set_xlabel("")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(["Single-agent\nmisinformed", "Multi-agent\nbaseline", "Multi-agent\nmisinformed"])
    _annotate_bars(
        ax2,
        labels=["exp1_single", "exp2_ablation", "exp2_multi"],
        values=[float(overall.iloc[i]["accuracy"]) for i in range(len(overall))],
        ns=[int(overall.iloc[i]["n"]) for i in range(len(overall))],
    )
    plt.tight_layout()
    save_plot(output_dir, "exp1_vs_exp2_ablation_vs_exp2_overall_accuracy.pdf")

    # Overall per-dataset comparison
    common_datasets = sorted(set(df1s["dataset"]).intersection(set(df2as["dataset"])).intersection(set(df2s["dataset"])))
    overall_ds = []
    for ds in common_datasets:
        d1 = df1s[df1s["dataset"] == ds]
        d2a = df2as[df2as["dataset"] == ds]
        d2 = df2s[df2s["dataset"] == ds]
        if d1.empty or d2a.empty or d2.empty:
            continue
        overall_ds.append({"dataset": ds, "experiment": "exp1_single", "accuracy": d1["is_correct"].mean(), "n": len(d1)})
        overall_ds.append({"dataset": ds, "experiment": "exp2_ablation", "accuracy": d2a["is_correct"].mean(), "n": len(d2a)})
        overall_ds.append({"dataset": ds, "experiment": "exp2_multi", "accuracy": d2["is_correct"].mean(), "n": len(d2)})

    if overall_ds:
        odf = pd.DataFrame(overall_ds)
        odf["dataset_display"] = odf["dataset"].map(_display_dataset_name)
        plt.figure(figsize=(11.6, 6.0))
        axo = sns.barplot(
            data=odf,
            x="dataset_display",
            y="accuracy",
            hue="experiment",
            hue_order=["exp1_single", "exp2_ablation", "exp2_multi"],
            palette={"exp1_single": pal3[0], "exp2_ablation": pal3[1], "exp2_multi": pal3[2]},
        )
        _apply_axis_style(axo)
        axo.set_title("Overall Accuracy by Dataset (Single vs Multi-Agent)", fontweight="bold")
        axo.set_xlabel("Dataset")
        axo.set_ylabel("Accuracy")
        axo.set_ylim(0, 1)
        axo.legend(title="", loc="upper center", bbox_to_anchor=(0.5, 1.26), ncol=3, frameon=True)
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
        save_plot(output_dir, "exp1_vs_exp2_ablation_vs_exp2_overall_accuracy_by_dataset.pdf")

    s1 = df1s.groupby("strategy", observed=True)["is_correct"].agg(["mean", "count"]).reset_index()
    s1["experiment"] = "exp1_single"
    s1.rename(columns={"mean": "accuracy", "count": "n"}, inplace=True)

    s2a = df2as.groupby("strategy", observed=True)["is_correct"].agg(["mean", "count"]).reset_index()
    s2a["experiment"] = "exp2_ablation"
    s2a.rename(columns={"mean": "accuracy", "count": "n"}, inplace=True)

    s2 = df2s.groupby("strategy", observed=True)["is_correct"].agg(["mean", "count"]).reset_index()
    s2["experiment"] = "exp2_multi"
    s2.rename(columns={"mean": "accuracy", "count": "n"}, inplace=True)

    merged = pd.concat([s1, s2a, s2], ignore_index=True)

    # Keep only strategies present in all three for a fair side-by-side.
    common = sorted(set(s1["strategy"]).intersection(set(s2a["strategy"])).intersection(set(s2["strategy"])))
    merged = merged[merged["strategy"].isin(common)].copy()
    if merged.empty:
        print(
            "No overlapping strategies between exp1, exp2_ablation, and exp2; "
            "skipping by-strategy ablation comparison plots."
        )
        return

    # Order by exp2 accuracy (descending) for readability
    order = (
        merged[merged["experiment"] == "exp2_multi"]
        .sort_values("accuracy", ascending=False)["strategy"]
        .tolist()
    )

    plt.figure(figsize=(14, 6))
    ax = sns.barplot(
        data=merged,
        x="strategy",
        y="accuracy",
        hue="experiment",
        hue_order=["exp1_single", "exp2_ablation", "exp2_multi"],
        order=order,
        palette={"exp1_single": pal3[0], "exp2_ablation": pal3[1], "exp2_multi": pal3[2]},
    )
    ax.set_title("Accuracy by Strategy (Single vs Multi-Agent)", fontweight="bold")
    ax.set_xlabel(MISINFO_CATEGORY_LABEL)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    _style_rotated_xticklabels(ax, rotation=35)
    _move_ax_legend_to_top_row(ax, title="", y=1.20)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    save_plot(output_dir, "exp1_vs_exp2_ablation_vs_exp2_accuracy_by_strategy.pdf")

    # One consolidated figure: all datasets, fixed strategy order (false_fact leftmost)
    base_overall, base_by_ds = _load_exp1_baseline(out_dir, model=model)

    combo_rows = []
    for ds in common_datasets:
        d1 = df1s[df1s["dataset"] == ds]
        d2a = df2as[df2as["dataset"] == ds]
        d2 = df2s[df2s["dataset"] == ds]
        if d1.empty or d2a.empty or d2.empty:
            continue

        a1 = d1.groupby("strategy", observed=True)["is_correct"].mean().to_dict()
        a2a = d2a.groupby("strategy", observed=True)["is_correct"].mean().to_dict()
        a2 = d2.groupby("strategy", observed=True)["is_correct"].mean().to_dict()

        for strat in STRATEGY_ORDER:
            if strat not in a1 or strat not in a2a or strat not in a2:
                continue
            combo_rows.append({"dataset": ds, "strategy": strat, "experiment": "exp1_single", "accuracy": a1[strat]})
            combo_rows.append({"dataset": ds, "strategy": strat, "experiment": "exp2_ablation", "accuracy": a2a[strat]})
            combo_rows.append({"dataset": ds, "strategy": strat, "experiment": "exp2_multi", "accuracy": a2[strat]})

    if combo_rows:
        cdf = pd.DataFrame(combo_rows)
        cdf["strategy"] = pd.Categorical(cdf["strategy"], categories=STRATEGY_ORDER, ordered=True)
        cdf["strategy_display"] = cdf["strategy"].map(_display_strategy_name)
        order_display = [_display_strategy_name(s) for s in STRATEGY_ORDER]

        from figure_style import get_palette
        pal3 = get_palette(3)

        g = sns.catplot(
            data=cdf,
            kind="bar",
            x="strategy_display",
            y="accuracy",
            hue="experiment",
            hue_order=["exp1_single", "exp2_ablation", "exp2_multi"],
            col="dataset",
            col_order=common_datasets,
            order=order_display,
            height=5.2,
            aspect=1.1,
            palette={"exp1_single": pal3[0], "exp2_ablation": pal3[1], "exp2_multi": pal3[2]},
            sharey=True,
        )
        g.set_titles("{col_name}")
        g.set_axis_labels(MISINFO_CATEGORY_LABEL, "Accuracy")
        g.set(ylim=(0, 1))

        # Legend: figure-level, top, single row (extract from seaborn's legend).
        handles: List[object] = []
        labels: List[str] = []
        leg = getattr(g, "_legend", None)
        if leg is not None:
            handles = list(getattr(leg, "legend_handles", None) or getattr(leg, "legendHandles", None) or [])
            labels = [t.get_text() for t in getattr(leg, "texts", [])]
            leg.remove()
        if not handles or not labels:
            handles, labels = g.axes.flat[0].get_legend_handles_labels()
        # Keep legend INSIDE figure bounds to avoid extra whitespace from bbox_inches="tight".
        _append_legend_top_row(g.fig, handles, labels, title="", y=0.99, ncol=len(labels))

        for axp, ds in zip(g.axes.flat, common_datasets):
            _style_rotated_xticklabels(axp, rotation=35)
            if ds in base_by_ds:
                b = base_by_ds[ds]
                axp.axhline(b, color=pal3[1], linestyle="--", linewidth=1.5, alpha=0.9)
                axp.text(0.02, b + 0.02, f"exp1 baseline={b:.3f}", transform=axp.get_yaxis_transform(), fontsize=12)

        g.fig.suptitle(
            "Accuracy by Misinformation Strategy per Dataset",
            y=0.885,
            fontweight="bold",
        )
        g.fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
        save_plot(output_dir, "exp1_vs_exp2_ablation_vs_exp2_strategies_all_datasets_fixed_order.pdf")

    # Per-dataset strategy comparison
    for ds in common_datasets:
        d1 = df1s[df1s["dataset"] == ds]
        d2a = df2as[df2as["dataset"] == ds]
        d2 = df2s[df2s["dataset"] == ds]
        if d1.empty or d2a.empty or d2.empty:
            continue

        s1d = d1.groupby("strategy", observed=True)["is_correct"].agg(["mean", "count"]).reset_index()
        s1d["experiment"] = "exp1_single"
        s1d.rename(columns={"mean": "accuracy", "count": "n"}, inplace=True)

        s2ad = d2a.groupby("strategy", observed=True)["is_correct"].agg(["mean", "count"]).reset_index()
        s2ad["experiment"] = "exp2_ablation"
        s2ad.rename(columns={"mean": "accuracy", "count": "n"}, inplace=True)

        s2d = d2.groupby("strategy", observed=True)["is_correct"].agg(["mean", "count"]).reset_index()
        s2d["experiment"] = "exp2_multi"
        s2d.rename(columns={"mean": "accuracy", "count": "n"}, inplace=True)

        md = pd.concat([s1d, s2ad, s2d], ignore_index=True)
        common_ds_strats = sorted(
            set(s1d["strategy"]).intersection(set(s2ad["strategy"])).intersection(set(s2d["strategy"]))
        )
        md = md[md["strategy"].isin(common_ds_strats)]
        if md.empty:
            continue

        extras = [s for s in common_ds_strats if s not in STRATEGY_ORDER]
        order_ds = [s for s in STRATEGY_ORDER if s in common_ds_strats] + sorted(extras)

        from figure_style import get_palette
        pal3 = get_palette(3)
        plt.figure(figsize=(14, 6))
        axd = sns.barplot(
            data=md,
            x="strategy",
            y="accuracy",
            hue="experiment",
            hue_order=["exp1_single", "exp2_ablation", "exp2_multi"],
            order=order_ds,
            palette={"exp1_single": pal3[0], "exp2_ablation": pal3[1], "exp2_multi": pal3[2]},
        )
        axd.set_title(f"Exp1 vs Exp2 ablation vs Exp2: Accuracy by strategy ({ds})", fontweight="bold")
        axd.set_xlabel(MISINFO_CATEGORY_LABEL)
        axd.set_ylabel("Accuracy")
        axd.set_ylim(0, 1)
        _style_rotated_xticklabels(axd, rotation=35)

        if ds in base_by_ds:
            b = base_by_ds[ds]
            axd.axhline(b, color=pal3[1], linestyle="--", linewidth=1.5, alpha=0.7)
            axd.text(0.01, b + 0.02, f"exp1 baseline={b:.3f}", transform=axd.get_yaxis_transform())

        _move_ax_legend_to_top_row(axd, title="", y=1.20)
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
        save_plot(output_dir, f"exp1_vs_exp2_ablation_vs_exp2_accuracy_by_strategy_{ds}.pdf")


def plot_model_comparison_overall(df: pd.DataFrame, output_dir: str) -> None:
    """Compare overall exp2 accuracy across models."""
    if df.empty or "model" not in df.columns:
        return
    models = sorted(df["model"].dropna().astype(str).unique().tolist())
    if len(models) < 2:
        return

    stats = (
        df.groupby("model", observed=True)["is_correct"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "accuracy", "count": "n"})
        .sort_values("accuracy", ascending=False)
    )
    palette = get_project_palette()
    color_map = {m: palette[i % len(palette)] for i, m in enumerate(stats["model"].tolist())}
    stats["model_display"] = stats["model"].astype(str).str.replace("_", "/", n=1, regex=False)

    plt.figure(figsize=(11.2, 5.6))
    ax = sns.barplot(
        data=stats,
        x="model_display",
        y="accuracy",
        hue="model",
        hue_order=stats["model"].tolist(),
        dodge=False,
        palette=color_map,
        legend=False,
    )
    ax.set_title("Model Comparison (Overall Accuracy)", fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    for i, row in stats.reset_index(drop=True).iterrows():
        ax.text(i, min(0.98, float(row["accuracy"]) + 0.02), f"n={int(row['n'])}", ha="center", va="bottom", fontsize=11)
    for lab in ax.get_xticklabels():
        lab.set_rotation(20)
        lab.set_ha("right")
        lab.set_rotation_mode("anchor")
    plt.tight_layout()
    save_plot(output_dir, "exp2_model_comparison_overall_accuracy.pdf")


def plot_model_comparison_by_strategy(df: pd.DataFrame, output_dir: str) -> None:
    """Compare exp2 strategy accuracies across models in one figure."""
    if df.empty or "model" not in df.columns:
        return
    models = sorted(df["model"].dropna().astype(str).unique().tolist())
    if len(models) < 2:
        return

    stats = (
        df.groupby(["strategy", "model"], observed=True)["is_correct"]
        .mean()
        .reset_index()
        .rename(columns={"is_correct": "accuracy"})
    )
    if stats.empty:
        return

    order = (
        df.groupby("strategy", observed=True)["is_correct"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    stats["strategy_display"] = stats["strategy"].map(_display_strategy_name)
    order_display = [_display_strategy_name(s) for s in order]
    stats["model_display"] = stats["model"].astype(str).str.replace("_", "/", n=1, regex=False)
    model_display = [str(m).replace("_", "/", 1) for m in models]

    model_palette_base = get_project_palette()
    model_palette = {m: model_palette_base[i % len(model_palette_base)] for i, m in enumerate(model_display)}

    plt.figure(figsize=(15.2, 6.4))
    ax = sns.barplot(
        data=stats,
        x="strategy_display",
        y="accuracy",
        hue="model_display",
        order=order_display,
        hue_order=model_display,
        palette=model_palette,
    )
    _apply_axis_style(ax)
    ax.set_title("Model Comparison by Strategy", fontweight="bold")
    ax.set_xlabel(MISINFO_CATEGORY_LABEL)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    _style_rotated_xticklabels(ax, rotation=35)
    _move_ax_legend_to_top_row(ax, title="", y=1.20)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    save_plot(output_dir, "exp2_model_comparison_by_strategy.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create essential visualizations for exp2 results")
    parser.add_argument("--out_dir", type=str, default="out", help="Directory containing exp2_results_*.json")
    parser.add_argument("--output_dir", type=str, default="out/figures/exp2", help="Output directory for figures")

    parser.add_argument("--all_plots", action="store_true", help="Generate all essential plots")
    parser.add_argument("--accuracy", action="store_true", help="Accuracy plots (heatmap + overall bar)")
    parser.add_argument("--agreement", action="store_true", help="Agreement plots (heatmap + accuracy vs agreement)")
    parser.add_argument(
        "--misinfo_maintenance",
        action="store_true",
        help="Misinformed opinion maintenance plot (agreed + incorrect rate)",
    )
    parser.add_argument(
        "--opinion_persistence",
        action="store_true",
        help="Opinion persistence plots (t to t+1 same-solution) for misinformed vs uninformed agents",
    )
    parser.add_argument("--compare_exp1", action="store_true", help="Compare exp2 (multi-agent) vs exp1 (single-agent)")

    args = parser.parse_args()

    df = load_exp2_results_long(out_dir=args.out_dir)
    if df.empty:
        print("No exp2 result rows found. Make sure you have out/exp2_results_*.json")
        return

    print(f"Loaded {len(df)} rows from exp2 results.")
    print("Datasets:", sorted(df["dataset"].unique().tolist()))
    print("Models:", sorted(df["model"].unique().tolist()))
    print("Strategies:", sorted(df["strategy"].unique().tolist()))

    if not any(
        [
            args.all_plots,
            args.accuracy,
            args.agreement,
            args.misinfo_maintenance,
            args.opinion_persistence,
            args.compare_exp1,
        ]
    ):
        args.all_plots = True

    if args.all_plots or args.accuracy:
        print("Creating accuracy plots (combined + per-model)...")

    if args.all_plots or args.agreement:
        print("Creating agreement plots (combined + per-model)...")

    if args.all_plots or args.misinfo_maintenance:
        print("Creating misinformed opinion maintenance plots (combined + per-model)...")

    os.makedirs(args.output_dir, exist_ok=True)
    combined_dir = os.path.join(args.output_dir, "combined")
    model_root = os.path.join(args.output_dir, "models")
    os.makedirs(combined_dir, exist_ok=True)
    os.makedirs(model_root, exist_ok=True)

    # Combined (all available models together).
    if args.all_plots or args.accuracy:
        plot_accuracy_heatmap(df, combined_dir, results_out_dir=args.out_dir)
        plot_accuracy_overall(df, combined_dir, results_out_dir=args.out_dir)
        plot_accuracy_by_strategy_by_dataset(df, combined_dir, results_out_dir=args.out_dir)
        plot_overall_accuracy_by_dataset(df, combined_dir)
        plot_model_comparison_overall(df, combined_dir)
        plot_model_comparison_by_strategy(df, combined_dir)

    if args.all_plots or args.agreement:
        plot_agreement_heatmap(df, combined_dir)
        plot_accuracy_vs_agreement(df, combined_dir)
        plot_accuracy_vs_agreement_by_dataset(df, combined_dir)

    if args.all_plots or args.misinfo_maintenance:
        plot_misinformed_opinion_maintenance(df, combined_dir)

    if args.all_plots or args.opinion_persistence:
        print("Computing opinion persistence (combined + per-model)...")
        dfp = load_exp2_opinion_persistence_long(out_dir=args.out_dir)
        print(f"Loaded {len(dfp)} persistence comparisons (combined).")
        plot_opinion_persistence(dfp, combined_dir)
        plot_opinion_persistence_by_dataset(dfp, combined_dir)
        plot_opinion_persistence_delta_barplot(dfp, combined_dir)
        plot_opinion_persistence_delta_barplot_grid(dfp, combined_dir)

    # Per-model outputs.
    for model in sorted(df["model"].dropna().astype(str).unique().tolist()):
        dmf = df[df["model"].astype(str) == model].copy()
        if dmf.empty:
            continue
        model_dir = os.path.join(model_root, model)
        os.makedirs(model_dir, exist_ok=True)
        print(f"Creating per-model exp2 figures for {model}...")
        if args.all_plots or args.accuracy:
            plot_accuracy_heatmap(dmf, model_dir, results_out_dir=args.out_dir, model=model)
            plot_accuracy_overall(dmf, model_dir, results_out_dir=args.out_dir, model=model)
            plot_accuracy_by_strategy_by_dataset(dmf, model_dir, results_out_dir=args.out_dir, model=model)
            plot_overall_accuracy_by_dataset(dmf, model_dir)
        if args.all_plots or args.agreement:
            plot_agreement_heatmap(dmf, model_dir)
            plot_accuracy_vs_agreement(dmf, model_dir)
            plot_accuracy_vs_agreement_by_dataset(dmf, model_dir)
        if args.all_plots or args.misinfo_maintenance:
            plot_misinformed_opinion_maintenance(dmf, model_dir)
        if args.all_plots or args.opinion_persistence:
            dfp_m = load_exp2_opinion_persistence_long(out_dir=args.out_dir, model=model)
            print(f"Loaded {len(dfp_m)} persistence comparisons for model {model}.")
            plot_opinion_persistence(dfp_m, model_dir)
            plot_opinion_persistence_by_dataset(dfp_m, model_dir)
            plot_opinion_persistence_delta_barplot(dfp_m, model_dir)
            plot_opinion_persistence_delta_barplot_grid(dfp_m, model_dir)

    if args.all_plots or args.compare_exp1:
        print("Creating exp1 vs exp2 comparison plots (combined + per-model)...")
        plot_exp1_vs_exp2_strategy_comparison(args.out_dir, combined_dir, model=None)
        # Also create an ablation comparison if ablation results are present.
        df2a = load_exp2_ablation_results_long(out_dir=args.out_dir, model=None)
        if df2a.empty:
            print("No exp2 ablation rows found; skipping ablation comparison plots.")
        else:
            print("Creating exp1 vs exp2 ablation vs exp2 comparison plots...")
            plot_exp1_vs_exp2_ablation_vs_exp2_strategy_comparison(args.out_dir, combined_dir, model=None)

        for model in sorted(df["model"].dropna().astype(str).unique().tolist()):
            model_dir = os.path.join(model_root, model)
            plot_exp1_vs_exp2_strategy_comparison(args.out_dir, model_dir, model=model)
            df2a_m = load_exp2_ablation_results_long(out_dir=args.out_dir, model=model)
            if df2a_m.empty:
                continue
            plot_exp1_vs_exp2_ablation_vs_exp2_strategy_comparison(args.out_dir, model_dir, model=model)

    print(f"All plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
