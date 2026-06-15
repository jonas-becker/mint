#!/usr/bin/env python3

"""exp1_figures.py

Essential visualizations for Exp1 (single-agent) results.

This script intentionally focuses on the core evaluations:
- Accuracy by dataset and misinformation relevance (baseline vs relevant vs irrelevant misinfo vs irrelevant true info)
- Accuracy comparison across misinformation strategies (relevant vs irrelevant variants)
- Strategy impact relative to baseline (delta accuracy heatmaps)

Loads files matching: out/**/exp1_results_*.json and out/**/exp1a_results_*.json
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import re
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Patch, Rectangle

from shared_utils import (
    load_results,
    find_result_files,
    extract_dataset_name,
    extract_answer_from_response,
    infer_model_from_result_path,
)
from shared_visualization import setup_plot_style, save_plot, add_value_labels, get_named_colors, get_palette as get_project_palette


setup_plot_style()

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


class _LegendHeaderHandler(HandlerBase):
    """Legend handler for header rows: label only, no handle whitespace."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        p = Rectangle((0, 0), 0, 0, linewidth=0, edgecolor="none", facecolor="none")
        p.set_transform(trans)
        return [p]


def _display_model_name(model: str) -> str:
    m = str(model).strip()
    if m.lower() == "combined":
        return "Combined"
    if "_" in m:
        left, right = m.split("_", 1)
        if left and right:
            return f"{left}/{right}"
    return m


# Requested categorical colors (paper-friendly, consistent across figures)
_RELEVANCE_ORDER = ["none", "relevant", "irrelevant", "irrelevant_true"]

_RELEVANCE_DISPLAY = {
    "none": "Uninformed",
    "relevant": "Misinformed",
    "irrelevant": "Irrelevantly misinformed",
    "irrelevant_true": "Irrelevantly truly informed",
}
try:
    import plot_config  # type: ignore

    _RELEVANCE_COLORS_RAW = dict(getattr(plot_config, "MISINFO_RELEVANCE_COLORS", {})) or {
        "baseline": "#999999",
        "relevant": "#ffbb6f",
        "irrelevant": "#5e4c5f",
        "irrelevant_true": "#6f8fa6",
    }
except Exception:
    _RELEVANCE_COLORS_RAW = {
        "baseline": "#999999",
        "relevant": "#ffbb6f",
        "irrelevant": "#5e4c5f",
        "irrelevant_true": "#6f8fa6",
    }

# Map colors onto the displayed legend labels.
_RELEVANCE_COLORS = {
    _RELEVANCE_DISPLAY[k]: _RELEVANCE_COLORS_RAW.get(
        "baseline" if k == "none" else k,
        "#999999",
    )
    for k in _RELEVANCE_DISPLAY
}


def _display_strategy_name(strategy: str) -> str:
    s = str(strategy).strip().lower()
    if s in {"", "none", "baseline"}:
        return "Baseline"
    if s == "false_fact":
        return "Neutral"
    return s.replace("_", " ").strip().title()


def _normalize_strategy(misinformation_strategy: Optional[str], condition: str) -> Tuple[str, str]:
    """Return (relevance, base_strategy).

    - relevance in {"none", "relevant", "irrelevant", "irrelevant_true"}
    - base_strategy in {"none", "false_fact", "clickbait", ...}

    We prefer the per-sample misinformation_strategy field, and fall back to condition name.
    """

    strat = (misinformation_strategy or "").strip().lower()

    # Baseline / no misinfo
    if condition == "baseline" or strat in {"", "none", "baseline"}:
        return "none", "none"

    # Irrelevant true information (exp1a)
    if strat == "irrelevant_true_information" or condition == "irrelevant_true_information":
        return "irrelevant_true", "irrelevant_true_information"

    # Irrelevant misinfo (false claims from other items)
    if strat.startswith("irrelevant_"):
        return "irrelevant", strat[len("irrelevant_") :]
    if condition == "irrelevant_misinformed":
        # If a run forgot to store the strategy, keep it as unknown-but-irrelevant.
        return "irrelevant", strat or "unknown"

    # Relevant misinfo
    if condition == "false_fact" or strat == "false_fact":
        return "relevant", "false_fact"
    if condition.startswith("strategy_"):
        return "relevant", condition[len("strategy_") :]

    return "relevant", strat or "unknown"


def _evaluate_is_correct(row: Dict) -> Tuple[Optional[str], bool]:
    """Compute (predicted_answer, is_correct) robustly.

    - For multiple-choice: re-extract predicted answer from the response and compare to correct_answer.
    - For free-form: fall back to stored is_correct.

    Returns:
      predicted_answer may be None for free-form.
    """

    is_mc = bool(row.get("is_multiple_choice", True))
    response = row.get("response") or ""
    correct_answer = row.get("correct_answer")
    options = row.get("options")

    if is_mc and isinstance(options, list) and options:
        predicted = extract_answer_from_response(response, options)
        correct_text: Optional[str] = None

        # Some datasets store correct_answer as an index (e.g. ethics: 0/1).
        if isinstance(correct_answer, int):
            if 0 <= correct_answer < len(options):
                correct_text = options[correct_answer]
            elif 1 <= correct_answer <= len(options):
                correct_text = options[correct_answer - 1]
        elif isinstance(correct_answer, str):
            ca = correct_answer.strip()
            if ca.isdigit():
                idx = int(ca)
                if 0 <= idx < len(options):
                    correct_text = options[idx]
                elif 1 <= idx <= len(options):
                    correct_text = options[idx - 1]
            else:
                # Some datasets (e.g., LogiQA) store the correct answer as a letter like "C".
                m = re.match(r"^\s*([A-E])\s*[\)\.\:\-]?\s*$", ca, flags=re.IGNORECASE)
                if m:
                    letter = m.group(1).upper()
                    idx = ord(letter) - ord("A")
                    if 0 <= idx < len(options):
                        correct_text = options[idx]
                    else:
                        correct_text = ca
                else:
                    correct_text = ca

        # If we couldn't map the correct answer text, fall back to stored evaluation.
        if not predicted or correct_text is None:
            return predicted, bool(row.get("is_correct", False))

        is_correct = predicted == correct_text
        return predicted, is_correct

    # Free-form or missing options: trust stored evaluation
    return None, bool(row.get("is_correct", False))


def _load_results_files_long(
    files: List[str],
    *,
    out_dir: str,
    required_prefix: str,
) -> List[Dict]:
    """Parse result JSON files into normalized row dicts."""
    rows: List[Dict] = []

    for file_path in files:
        if not os.path.basename(file_path).startswith(required_prefix):
            continue

        dataset = extract_dataset_name(file_path)
        if dataset == "unknown":
            continue

        model = infer_model_from_result_path(file_path, out_dir=out_dir)
        data = load_results(file_path)

        for condition, payload in data.items():
            if not isinstance(payload, dict) or "results" not in payload:
                continue
            results = payload.get("results") or []
            if not isinstance(results, list):
                continue

            for r in results:
                if not isinstance(r, dict):
                    continue

                relevance, base_strategy = _normalize_strategy(r.get("misinformation_strategy"), condition)

                row = {
                    "dataset": dataset,
                    "model": model,
                    "condition": condition,
                    "relevance": relevance,
                    "base_strategy": base_strategy,
                    "misinformation_strategy": r.get("misinformation_strategy"),
                    "is_multiple_choice": r.get("is_multiple_choice", True),
                    "response": r.get("response", ""),
                    "correct_answer": r.get("correct_answer", ""),
                    "options": r.get("options", []),
                    "is_correct": r.get("is_correct", False),
                }

                predicted, computed_is_correct = _evaluate_is_correct(row)
                row["predicted_answer"] = predicted
                row["is_correct"] = computed_is_correct

                rows.append(row)

    return rows


def load_exp1_results_long(out_dir: str = "out") -> pd.DataFrame:
    """Load exp1 and exp1a result files into a normalized long-form DataFrame."""

    exp1_files = find_result_files(out_dir=out_dir, prefix="exp1_results_")
    exp1a_files = find_result_files(out_dir=out_dir, prefix="exp1a_results_")

    rows = _load_results_files_long(exp1_files, out_dir=out_dir, required_prefix="exp1_results_")
    rows += _load_results_files_long(exp1a_files, out_dir=out_dir, required_prefix="exp1a_results_")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Make plots stable and readable
    df["dataset"] = df["dataset"].astype(str)
    df["model"] = df["model"].astype(str)
    df["condition"] = df["condition"].astype(str)
    df["relevance"] = pd.Categorical(df["relevance"], categories=_RELEVANCE_ORDER, ordered=True)
    df["base_strategy"] = df["base_strategy"].astype(str)

    # Exclude LogiQA from exp1 figures (paper figure set).
    df = df[df["dataset"].str.lower().ne("logiqa")].copy()

    return df


def plot_model_comparison_relevance(df: pd.DataFrame, output_dir: str) -> None:
    """Compare model accuracies with relevance split."""
    if df.empty or "model" not in df.columns:
        return
    models = sorted(df["model"].dropna().astype(str).unique().tolist())
    if len(models) < 2:
        return

    agg = df.groupby(["model", "relevance"], observed=True)["is_correct"].mean().reset_index()
    agg["model_display"] = agg["model"].map(_display_model_name)
    agg["relevance_display"] = agg["relevance"].astype(str).map(_RELEVANCE_DISPLAY).fillna(agg["relevance"].astype(str))

    plt.figure(figsize=(13.5, 6.6))
    ax = sns.barplot(
        data=agg,
        x="model_display",
        y="is_correct",
        hue="relevance_display",
        hue_order=[_RELEVANCE_DISPLAY[h] for h in _RELEVANCE_ORDER],
        palette=_RELEVANCE_COLORS,
    )
    ax.set_title("Model Comparison by Misinformation Relevance", fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(title="", frameon=True)
    for lab in ax.get_xticklabels():
        lab.set_rotation(20)
        lab.set_ha("right")
        lab.set_rotation_mode("anchor")
    plt.tight_layout()
    save_plot(output_dir, "exp1_model_comparison_by_relevance.pdf")


def plot_accuracy_by_relevance(df: pd.DataFrame, output_dir: str) -> None:
    """Baseline vs relevant vs irrelevant misinformation, per dataset."""

    if df.empty:
        print("No data available.")
        return

    model_list = []
    if "model" in df.columns:
        model_list = sorted(df["model"].dropna().astype(str).unique().tolist())
    multi_model = len(model_list) > 1

    group_cols = ["dataset", "relevance"] + (["model"] if multi_model else [])
    agg = df.groupby(group_cols, observed=True)["is_correct"].mean().reset_index()
    agg["dataset_display"] = agg["dataset"].map(_display_dataset_name)
    # Baseline per dataset for delta labels
    baseline_by_ds = (
        agg[agg["relevance"] == "none"]
        .set_index("dataset")["is_correct"]
        .to_dict()
    )

    ds_order_raw = sorted(agg["dataset"].unique().tolist())
    ds_order_display = [_display_dataset_name(d) for d in ds_order_raw]
    agg_plot = agg.copy()
    agg_plot["relevance_display"] = (
        agg_plot["relevance"].astype(str).map(_RELEVANCE_DISPLAY).fillna(agg_plot["relevance"].astype(str))
    )

    hue_order = list(_RELEVANCE_ORDER)
    hue_order_display = [_RELEVANCE_DISPLAY[h] for h in hue_order]

    if multi_model:
        agg_plot["model_display"] = agg_plot["model"].map(_display_model_name)
        model_display = [_display_model_name(m) for m in model_list]
        fig, ax = plt.subplots(figsize=(12.0, 6.8))
        x = np.arange(len(ds_order_display))
        n_rel = len(hue_order)
        n_models = len(model_display)
        n_slots = n_rel * n_models
        # Per dataset: model 1 relevances (left), gap, model 2 relevances (right).
        bar_w = min(0.082, 0.90 / max(1, n_slots))
        inner_gap = 0.020  # visible space between adjacent bars
        group_gap = 0.055 if n_models > 1 else 0.0  # space between model clusters
        total_span = (
            n_slots * bar_w
            + max(0, n_slots - n_models) * inner_gap
            + max(0, n_models - 1) * group_gap
        )
        left = -total_span / 2.0 + bar_w / 2.0
        slot_offsets: List[float] = []
        cur = left
        for m_i in range(n_models):
            for _r_i in range(n_rel):
                slot_offsets.append(cur)
                cur += bar_w
                if _r_i < n_rel - 1:
                    cur += inner_gap
            if m_i < n_models - 1:
                cur += group_gap
        model_hatches = ["", "//", "xx", "\\\\", "..", "++"]

        val_map = (
            agg_plot.set_index(["dataset_display", "relevance", "model_display"])["is_correct"].to_dict()
        )
        label_color = get_named_colors().get("slate", "#4A4A4A")
        base_fs = float(plt.rcParams.get("font.size", 12))
        val_fs = max(7.0, base_fs * 0.62)

        slot_i = 0
        for m_i, m_disp in enumerate(model_display):
            for rel in hue_order:
                color = _RELEVANCE_COLORS[_RELEVANCE_DISPLAY[rel]]
                heights = []
                xpos = []
                for d_i, ds_disp in enumerate(ds_order_display):
                    heights.append(float(val_map.get((ds_disp, rel, m_disp), np.nan)))
                    xpos.append(float(x[d_i] + slot_offsets[slot_i]))
                bars = ax.bar(
                    xpos,
                    [0.0 if pd.isna(v) else v for v in heights],
                    width=bar_w,
                    color=color,
                    edgecolor="#2e2e2e",
                    linewidth=0.35,
                    hatch=model_hatches[m_i % len(model_hatches)],
                    alpha=0.95,
                    zorder=3,
                )
                for b, v in zip(bars, heights):
                    if pd.isna(v):
                        continue
                    ax.text(
                        b.get_x() + b.get_width() / 2.0,
                        min(0.985, float(v) + 0.010),
                        f"{float(v):.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=val_fs,
                        color=label_color,
                    )
                slot_i += 1

        title_fs = max(15.0, base_fs * 1.30)
        axis_label_fs = max(16.0, base_fs * 1.35)
        tick_fs = max(15.0, base_fs * 1.20)
        legend_fs = max(12.0, base_fs * 1.0)

        ax.set_title(
            "Accuracy by Dataset and Misinformation Relevance",
            fontweight="bold",
            fontsize=title_fs,
        )
        ax.set_xlabel("")
        ax.set_ylabel("Accuracy", fontsize=axis_label_fs)
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(ds_order_display, fontsize=tick_fs)
        ax.tick_params(axis="y", labelsize=tick_fs)
        ax.grid(axis="y", alpha=0.2, zorder=0)

        def _model_short_label(m: str) -> str:
            raw = str(m).strip().lower()
            if "llama" in raw and "3.3" in raw:
                return "Llama-3.3"
            if "glm" in raw and "4.7" in raw:
                return "GLM-4.7"
            s = str(m).strip()
            return s.split("/", 1)[1] if "/" in s else s

        relevance_handles = [
            Patch(facecolor=_RELEVANCE_COLORS[_RELEVANCE_DISPLAY[rel]], edgecolor="#2e2e2e", label=_RELEVANCE_DISPLAY[rel])
            for rel in hue_order
        ]
        model_handles = [
            Patch(
                facecolor="#f4f4f4",
                edgecolor="#2e2e2e",
                hatch=model_hatches[i % len(model_hatches)],
                label=_model_short_label(m),
            )
            for i, m in enumerate(model_display)
        ]
        # Column-major layout: 4 relevance handles fill the first two columns
        # (2x2), an invisible spacer column adds a gap, and the two model
        # handles fill the final column (1x2).
        spacer_handle = Patch(facecolor="none", edgecolor="none", label="")
        combined_handles = (
            relevance_handles
            + [spacer_handle, spacer_handle]
            + model_handles
        )
        combined_labels = [h.get_label() for h in combined_handles]

        fig.legend(
            handles=combined_handles,
            labels=combined_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.005),
            ncol=4,
            frameon=True,
            fontsize=legend_fs,
            handlelength=1.2,
            handletextpad=0.45,
            columnspacing=1.4,
            labelspacing=0.55,
            borderpad=0.5,
            borderaxespad=0.0,
        )
    else:
        plt.figure(figsize=(15.5, 6.8))
        ax = sns.barplot(
            data=agg_plot,
            x="dataset_display",
            y="is_correct",
            hue="relevance_display",
            order=ds_order_display,
            hue_order=hue_order_display,
            palette=_RELEVANCE_COLORS,
        )
        base_fs_single = float(plt.rcParams.get("font.size", 12))
        title_fs = max(18.0, base_fs_single * 1.55)
        axis_label_fs = max(16.0, base_fs_single * 1.35)
        tick_fs = max(15.0, base_fs_single * 1.20)
        legend_fs = max(15.0, base_fs_single * 1.20)

        ax.set_title(
            "Accuracy by Dataset and Misinformation Relevance",
            fontweight="bold",
            fontsize=title_fs,
        )
        ax.set_xlabel("")
        ax.set_ylabel("Accuracy", fontsize=axis_label_fs)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", pad=2, labelsize=tick_fs)
        ax.tick_params(axis="y", labelsize=tick_fs)
        for lab in ax.get_xticklabels():
            lab.set_rotation(0)
            lab.set_ha("center")
        ax.legend(title="", frameon=True, fontsize=legend_fs)

        n_x = len(ds_order_raw)
        label_color = get_named_colors().get("slate", "#4A4A4A")
        base_fs = float(plt.rcParams.get("font.size", 12))
        abs_fs = max(10.0, base_fs * 0.95)
        pct_fs = max(9.0, base_fs * 0.75)
        for h_i, rel in enumerate(hue_order):
            for x_i, ds in enumerate(ds_order_raw):
                patch_idx = h_i * n_x + x_i
                if patch_idx >= len(ax.patches):
                    continue
                p = ax.patches[patch_idx]
                acc = float(p.get_height())
                base = baseline_by_ds.get(ds)
                if rel in ("relevant", "irrelevant", "irrelevant_true") and base is not None:
                    delta_pp = (acc - float(base)) * 100.0
                    x = p.get_x() + p.get_width() / 2.0
                    y_abs = min(0.985, acc + 0.05)
                    y_pct = min(y_abs - 0.035, acc + 0.012)
                    ax.text(
                        x,
                        y_abs,
                        f"{acc:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=abs_fs,
                        rotation=0,
                        color=label_color,
                    )
                    ax.text(
                        x,
                        y_pct,
                        f"({delta_pp:+.1f}%)",
                        ha="center",
                        va="bottom",
                        fontsize=pct_fs,
                        rotation=0,
                        color=label_color,
                    )
                else:
                    ax.text(
                        p.get_x() + p.get_width() / 2.0,
                        min(0.985, acc + 0.01),
                        f"{acc:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=abs_fs,
                        rotation=0,
                        color=label_color,
                    )

    if multi_model:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    else:
        plt.tight_layout()
    save_plot(output_dir, "exp1_accuracy_by_relevance.pdf")


def _model_short_label(m: str) -> str:
    raw = str(m).strip().lower()
    if "llama" in raw and "3.3" in raw:
        return "Llama-3.3"
    if "glm" in raw and "4.7" in raw:
        return "GLM-4.7"
    if "gpt-oss" in raw:
        return "GPT-OSS-120B"
    s = str(m).strip()
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


_RELEVANCE_MARKERS = {
    "none": "o",
    "relevant": "s",
    "irrelevant": "D",
    "irrelevant_true": "^",
}


_COMPACT_DATASET_LABEL = {
    "complex_web_questions": "CWQ",
    "cwq": "CWQ",
    "ethics": "Ethics",
    "winogrande": "WinoGrande",
    "logiqa": "LogiQA",
}


def _compact_dataset_label(ds: str) -> str:
    ds_key = str(ds).strip().lower()
    if ds_key in _COMPACT_DATASET_LABEL:
        return _COMPACT_DATASET_LABEL[ds_key]
    return _display_dataset_name(ds)


def plot_accuracy_by_relevance_dumbbell(df: pd.DataFrame, output_dir: str) -> None:
    """Horizontal dumbbell plot: accuracy by dataset/model row and relevance condition."""
    if df.empty:
        print("No data available.")
        return

    model_list = []
    if "model" in df.columns:
        model_list = sorted(df["model"].dropna().astype(str).unique().tolist(), key=_model_order_key)
    multi_model = len(model_list) > 1

    group_cols = ["dataset", "relevance"] + (["model"] if multi_model else [])
    agg = df.groupby(group_cols, observed=True)["is_correct"].mean().reset_index()

    ds_order_raw = sorted(agg["dataset"].unique().tolist())
    hue_order = list(_RELEVANCE_ORDER)

    val_map: Dict[tuple, float] = {}
    for row in agg.itertuples(index=False):
        ds = str(getattr(row, "dataset"))
        rel = str(getattr(row, "relevance"))
        model = str(getattr(row, "model")) if multi_model else ""
        val_map[(ds, rel, model)] = float(row.is_correct)

    row_specs: List[tuple[str, str, str, float]] = []
    model_groups: List[tuple[str, float]] = []
    group_gap = 0.55
    y_cur = 0.0

    if multi_model:
        for m_i, model in enumerate(model_list):
            if m_i > 0:
                y_cur += group_gap
            group_ys: List[float] = []
            for ds in ds_order_raw:
                row_specs.append((ds, model, _compact_dataset_label(ds), y_cur))
                group_ys.append(y_cur)
                y_cur += 1.0
            model_groups.append((_model_short_label(model), float(np.mean(group_ys))))
    else:
        for ds in ds_order_raw:
            row_specs.append((ds, "", _compact_dataset_label(ds), y_cur))
            y_cur += 1.0

    y_positions = [spec[3] for spec in row_specs]
    y_max = max(y_positions) if y_positions else 0.0
    base_fs = float(plt.rcParams.get("font.size", 12))
    title_fs = max(21.0, base_fs * 1.75)
    axis_label_fs = max(22.0, base_fs * 1.65)
    tick_fs = max(18.0, base_fs * 1.40)
    group_header_fs = max(17.0, base_fs * 1.30)
    legend_fs = max(17.0, base_fs * 1.35)
    val_fs = max(16.0, base_fs * 1.25)
    label_color = get_named_colors().get("slate", "#4A4A4A")

    fig_h = max(7.4, 0.88 * (y_max + 1.0) + 3.4)
    fig, ax = plt.subplots(figsize=(15.5, fig_h))

    for ds, model, row_label, y in row_specs:
        row_points: List[tuple[float, str, str, str]] = []
        for rel in hue_order:
            key = (ds, rel, model if multi_model else "")
            v = val_map.get(key, np.nan)
            if pd.isna(v):
                continue
            vv = float(v)
            disp = _RELEVANCE_DISPLAY[rel]
            row_points.append((vv, _RELEVANCE_COLORS[disp], _RELEVANCE_MARKERS.get(rel, "o"), rel))

        if row_points:
            xs = [p[0] for p in row_points]
            ax.plot(
                [min(xs), max(xs)],
                [y, y],
                color="#bdbdbd",
                linewidth=2.0,
                solid_capstyle="round",
                zorder=1,
            )

        for vv, color, marker, rel in row_points:
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

        # Alternate below/above left-to-right by score on each row (first = below).
        for pt_idx, (vv, color, _marker, _rel) in enumerate(sorted(row_points, key=lambda p: p[0])):
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
        "Accuracy by Dataset, Model, and Information Condition",
        fontweight="bold",
        fontsize=title_fs,
    )
    ax.set_xlabel("Accuracy", fontsize=axis_label_fs)
    ax.set_ylabel("")
    ax.set_xlim(0.3, 0.8)
    ax.set_ylim(-0.85, y_max + 0.85)
    ax.set_yticks(y_positions)
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
                color=label_color,
                transform=group_label_trans,
                clip_on=False,
            )

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=_RELEVANCE_MARKERS.get(rel, "o"),
            color="w",
            markerfacecolor=_RELEVANCE_COLORS[_RELEVANCE_DISPLAY[rel]],
            markeredgecolor="#2e2e2e",
            markeredgewidth=0.6,
            markersize=16,
            linestyle="None",
            label=_RELEVANCE_DISPLAY[rel],
        )
        for rel in hue_order
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
    save_plot(output_dir, "exp1_accuracy_by_relevance_dumbbell.pdf")


def plot_strategy_comparison_overall(df: pd.DataFrame, output_dir: str) -> None:
    """Compare strategies overall (relevant vs irrelevant variants), with baseline reference."""

    if df.empty:
        print("No data available.")
        return

    # Baseline reference
    baseline = df[df["relevance"] == "none"]["is_correct"].mean()

    # Strategy accuracies (exclude 'none')
    strat = df[df["base_strategy"] != "none"].copy()
    if strat.empty:
        print("No strategy data available.")
        return

    model_list = []
    if "model" in df.columns:
        model_list = sorted(df["model"].dropna().astype(str).unique().tolist())
    multi_model = len(model_list) > 1
    group_cols = ["base_strategy", "relevance"] + (["model"] if multi_model else [])
    stats = (
        strat.groupby(group_cols, observed=True)["is_correct"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "accuracy", "count": "n"})
    )

    # Keep only relevant/irrelevant for this view
    stats = stats[stats["relevance"].isin(["relevant", "irrelevant"])]

    # Order strategies by relevant accuracy (descending) for readability
    rel = stats[stats["relevance"] == "relevant"].groupby("base_strategy", observed=True)["accuracy"].mean()
    order = rel.sort_values(ascending=False).index.tolist()

    stats["strategy_display"] = stats["base_strategy"].map(_display_strategy_name)
    order_display = [_display_strategy_name(s) for s in order]

    rel_plot = stats.copy()
    rel_plot["relevance_display"] = rel_plot["relevance"].astype(str).map(_RELEVANCE_DISPLAY).fillna(rel_plot["relevance"].astype(str))

    if multi_model:
        rel_plot["model_display"] = rel_plot["model"].map(_display_model_name)
        model_display = [_display_model_name(m) for m in model_list]
        model_palette_base = get_project_palette()
        model_palette = {m: model_palette_base[i % len(model_palette_base)] for i, m in enumerate(model_display)}
        fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.6), sharey=True)
        for ax, rel in zip(axes, ["relevant", "irrelevant"]):
            sub = rel_plot[rel_plot["relevance"] == rel].copy()
            sns.barplot(
                data=sub,
                x="strategy_display",
                y="accuracy",
                hue="model_display",
                order=order_display,
                hue_order=model_display,
                palette=model_palette,
                ax=ax,
            )
            ax.set_title(_RELEVANCE_DISPLAY[rel], fontweight="bold")
            ax.set_xlabel("Misinformation Category")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1)
            for lab in ax.get_xticklabels():
                lab.set_rotation(35)
                lab.set_ha("right")
                lab.set_rotation_mode("anchor")
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title="", loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=max(1, len(labels)))
        fig.suptitle("Strategy Accuracy (Relevant vs Irrelevant)", fontweight="bold", y=1.10)
        ax = axes[0]
    else:
        plt.figure(figsize=(15.0, 6.6))
        from figure_style import get_palette
        pal2 = get_palette(2)
        rel_pal = {
            _RELEVANCE_DISPLAY["relevant"]: pal2[1],   # warm gold
            _RELEVANCE_DISPLAY["irrelevant"]: pal2[0], # cool purple
        }

        ax = sns.barplot(
            data=rel_plot,
            x="strategy_display",
            y="accuracy",
            hue="relevance_display",
            order=order_display,
            hue_order=[_RELEVANCE_DISPLAY["relevant"], _RELEVANCE_DISPLAY["irrelevant"]],
            palette=rel_pal,
        )
        ax.set_title("Strategy Accuracy (Relevant vs Irrelevant)", fontweight="bold")
        ax.set_xlabel("Misinformation Category")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", pad=2)
        for lab in ax.get_xticklabels():
            lab.set_rotation(35)
            lab.set_ha("right")
            lab.set_rotation_mode("anchor")
        ax.legend(title="Relevance", frameon=True)

    # Baseline line
    if pd.notna(baseline):
        if multi_model:
            for ax_i in axes:
                ax_i.axhline(baseline, color="black", linestyle="--", linewidth=1, alpha=0.6)
                ax_i.text(0.01, baseline + 0.02, f"Baseline = {baseline:.3f}", transform=ax_i.get_yaxis_transform())
        else:
            ax.axhline(baseline, color="black", linestyle="--", linewidth=1, alpha=0.6)
            ax.text(0.01, baseline + 0.02, f"Baseline = {baseline:.3f}", transform=ax.get_yaxis_transform())

    plt.tight_layout()
    save_plot(output_dir, "exp1_strategy_comparison_overall.pdf")


def plot_strategy_delta_heatmaps(df: pd.DataFrame, output_dir: str) -> None:
    """Heatmaps of delta accuracy vs baseline, split by relevance."""

    if df.empty:
        print("No data available.")
        return

    baseline = df[df["relevance"] == "none"].groupby("dataset", observed=True)["is_correct"].mean()
    if baseline.empty:
        print("No baseline data found; cannot compute deltas.")
        return

    for relevance in ["relevant", "irrelevant", "irrelevant_true"]:
        sub = df[(df["relevance"] == relevance) & (df["base_strategy"] != "none")].copy()
        if sub.empty:
            continue

        acc = sub.groupby(["base_strategy", "dataset"], observed=True)["is_correct"].mean().unstack("dataset")
        delta = acc.subtract(baseline, axis=1)
        # Prettify dataset labels
        delta = delta.rename(columns={c: _display_dataset_name(c) for c in delta.columns})
        # Prettify strategy labels (incl. false_fact -> Neutral)
        delta = delta.rename(index={s: _display_strategy_name(s) for s in delta.index})

        plt.figure(figsize=(13.2, max(4.8, 0.55 * len(delta.index))))
        from figure_style import get_colormaps
        ax = sns.heatmap(
            delta,
            annot=True,
            fmt="+.3f",
            center=0.0,
            cmap=get_colormaps().diverging,
            cbar_kws={"label": "Δ accuracy vs baseline"},
        )
        rel_label = _RELEVANCE_DISPLAY.get(relevance, relevance.replace("_", " ").title())
        ax.set_title(f"Delta Accuracy vs Baseline ({rel_label})", fontweight="bold")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Base strategy")
        plt.tight_layout()
        save_plot(output_dir, f"exp1_strategy_delta_vs_baseline_{relevance}.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create essential visualizations for exp1 results")
    parser.add_argument("--out_dir", type=str, default="out", help="Directory containing exp1/exp1a result JSON files")
    parser.add_argument("--output_dir", type=str, default="out/figures/exp1", help="Output directory for figures")

    parser.add_argument("--all_plots", action="store_true", help="Generate all essential plots")
    parser.add_argument("--relevance", action="store_true", help="Accuracy by relevance (incl. irrelevant true information)")
    parser.add_argument("--strategy_overall", action="store_true", help="Overall strategy comparison (relevant vs irrelevant)")
    parser.add_argument("--strategy_delta", action="store_true", help="Delta vs baseline heatmaps (relevant + irrelevant)")
    parser.add_argument("--model_comparison", action="store_true", help="Combined-model comparison plots")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_exp1_results_long(out_dir=args.out_dir)
    if df.empty:
        print("No exp1/exp1a result rows found. Make sure you have out/**/exp1_results_*.json and/or exp1a_results_*.json")
        return

    n_exp1a = int((df["relevance"].astype(str) == "irrelevant_true").sum()) if not df.empty else 0
    print(f"Loaded {len(df)} rows from exp1/exp1a results ({n_exp1a} irrelevant-true rows).")
    print("Datasets:", sorted(df["dataset"].unique().tolist()))
    print("Models:", sorted(df["model"].unique().tolist()))
    print("Conditions:", sorted(df["condition"].unique().tolist()))

    # Default behavior: generate all plots
    if not any([args.all_plots, args.relevance, args.strategy_overall, args.strategy_delta, args.model_comparison]):
        args.all_plots = True

    os.makedirs(args.output_dir, exist_ok=True)
    combined_dir = os.path.join(args.output_dir, "combined")
    model_root = os.path.join(args.output_dir, "models")
    os.makedirs(combined_dir, exist_ok=True)
    os.makedirs(model_root, exist_ok=True)

    # Combined figures (all available models together).
    if args.all_plots or args.relevance:
        print("Creating combined relevance accuracy plot...")
        plot_accuracy_by_relevance(df, combined_dir)
        plot_accuracy_by_relevance_dumbbell(df, combined_dir)
    if args.all_plots or args.strategy_overall:
        print("Creating combined strategy comparison plot...")
        plot_strategy_comparison_overall(df, combined_dir)
    if args.all_plots or args.strategy_delta:
        print("Creating combined strategy delta heatmaps...")
        plot_strategy_delta_heatmaps(df, combined_dir)
    if args.all_plots or args.model_comparison:
        print("Creating combined model-comparison plot...")
        plot_model_comparison_relevance(df, combined_dir)

    # Per-model figures.
    for model in sorted(df["model"].dropna().astype(str).unique().tolist()):
        dmf = df[df["model"].astype(str) == model].copy()
        if dmf.empty:
            continue
        model_dir = os.path.join(model_root, model)
        os.makedirs(model_dir, exist_ok=True)
        print(f"Creating per-model plots for {model}...")
        if args.all_plots or args.relevance:
            plot_accuracy_by_relevance(dmf, model_dir)
            plot_accuracy_by_relevance_dumbbell(dmf, model_dir)
        if args.all_plots or args.strategy_overall:
            plot_strategy_comparison_overall(dmf, model_dir)
        if args.all_plots or args.strategy_delta:
            plot_strategy_delta_heatmaps(dmf, model_dir)

    print(f"All plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
