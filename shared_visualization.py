#!/usr/bin/env python3

"""
Shared visualization functions for experiment figure scripts.
This module contains common plotting functions used across
exp1_figures.py and exp2_figures.py.
"""

import os
# Set environment variable to suppress OpenMP warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple, Optional

from figure_style import get_colormaps, get_palette as get_palette_n

_DEFAULT_PALETTE = [
    # Default to the 6-color project palette.
    "#5e4c5f",  # dull purple
    "#6f8fa6",  # muted blue
    "#7fa37a",  # sage green
    "#c27a7a",  # muted rose
    "#999999",  # neutral gray
    "#ffbb6f",  # soft gold
]


def get_palette(palette_path: str = "color_palette.json") -> List[str]:
    """Load the project color palette (hex strings). Falls back to `figure_style` palette."""
    try:
        with open(palette_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        colors = data.get("colors", [])
        if isinstance(colors, list) and all(isinstance(c, str) for c in colors) and len(colors) >= 3:
            return colors
    except Exception:
        pass
    # Prefer the canonical 6-color palette if no JSON palette exists.
    try:
        return get_palette_n(6)
    except Exception:
        return _DEFAULT_PALETTE


def get_named_colors(palette_path: str = "color_palette.json") -> Dict[str, str]:
    """Return a stable mapping of semantic names to palette colors."""
    colors = get_palette(palette_path=palette_path)
    padded = (colors + _DEFAULT_PALETTE)[:6]
    return {
        "sand": padded[0],
        "peach": padded[1],
        "sky": padded[2],
        "slate": padded[3],
        "olive_grey": padded[4],
        "brown_grey": padded[5],
    }


def get_diverging_cmap(palette_path: str = "color_palette.json") -> LinearSegmentedColormap:
    """Diverging colormap centered around 0 (project style)."""
    return get_colormaps().diverging


def get_sequential_cmap(palette_path: str = "color_palette.json") -> LinearSegmentedColormap:
    """Sequential colormap for rates in [0,1] (project style)."""
    return get_colormaps().sequential

def setup_plot_style():
    """Set up consistent plot styling (centralized in plot_config.py).

    This repo uses `plot_config.py` as the single source of truth for figure styling.
    We call it here so all exp* figure scripts share the same settings.
    """
    # Prefer the project-wide style defined in plot_config.py.
    # Use `use_latex=False` by default to avoid requiring a LaTeX install.
    try:
        import plot_config  # type: ignore

        plot_config.setup_plot_style(use_latex=False)

        # User-requested: increase font sizes globally relative to plot_config defaults.
        # Keep this as a single knob so figure scripts stay consistent.
        _font_scale = 2.9
        for k in [
            "font.size",
            "axes.titlesize",
            "axes.labelsize",
            "xtick.labelsize",
            "ytick.labelsize",
            "legend.fontsize",
            "figure.titlesize",
        ]:
            try:
                plt.rcParams[k] = float(plt.rcParams[k]) * _font_scale
            except Exception:
                pass
    except Exception:
        # Fallback: keep a reasonable seaborn theme if plot_config can't be imported.
        colors = get_named_colors()
        sns.set_theme(
            style="whitegrid",
            context="paper",
            font_scale=1.6,
            palette=get_palette(),
            rc={
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "axes.edgecolor": colors["slate"],
                "axes.labelcolor": colors["slate"],
                "text.color": colors["slate"],
                "xtick.color": colors["slate"],
                "ytick.color": colors["slate"],
                "grid.color": colors["slate"],
                "grid.alpha": 0.15,
                "axes.grid": True,
                "axes.spines.top": False,
                "axes.spines.right": False,
            },
        )

    # Keep exported PDFs editable/searchable in papers.
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    # Savefig defaults
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.bbox"] = "tight"

def save_plot(output_dir: str, filename: str, dpi: int = 300, bbox_inches: str = 'tight'):
    """Save plot with consistent settings."""
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{filename}", dpi=dpi, bbox_inches=bbox_inches, facecolor='white')
    plt.close()

def create_bar_plot(data: pd.DataFrame, x_col: str, y_col: str, title: str, 
                   xlabel: str = None, ylabel: str = None, output_dir: str = "out/figures", 
                   filename: str = None, figsize: Tuple[int, int] = (12, 8), 
                   rotation: int = 45, add_values: bool = True):
    """Create a standardized bar plot."""
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(data[x_col], data[y_col], alpha=0.7)
    
    if add_values:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_title(title, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    
    ax.tick_params(axis='x', rotation=rotation)
    ax.grid(True, alpha=0.3)
    
    if filename:
        save_plot(output_dir, filename)
    
    return fig, ax

def create_line_plot(data: pd.DataFrame, x_col: str, y_col: str, group_col: str = None,
                    title: str = None, xlabel: str = None, ylabel: str = None,
                    output_dir: str = "out/figures", filename: str = None,
                    figsize: Tuple[int, int] = (12, 8)):
    """Create a standardized line plot."""
    fig, ax = plt.subplots(figsize=figsize)
    
    if group_col:
        for group in data[group_col].unique():
            group_data = data[data[group_col] == group]
            ax.plot(group_data[x_col], group_data[y_col], marker='o', label=group, linewidth=2)
        ax.legend()
    else:
        ax.plot(data[x_col], data[y_col], marker='o', linewidth=2)
    
    if title:
        ax.set_title(title, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    
    ax.grid(True, alpha=0.3)
    
    if filename:
        save_plot(output_dir, filename)
    
    return fig, ax

def create_heatmap(data: pd.DataFrame, title: str = None, output_dir: str = "out/figures",
                  filename: str = None, figsize: Tuple[int, int] = (10, 8)):
    """Create a standardized heatmap."""
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(data, annot=True, fmt='.3f', cmap=get_sequential_cmap(), ax=ax)
    
    if title:
        ax.set_title(title, fontweight='bold')
    
    if filename:
        save_plot(output_dir, filename)
    
    return fig, ax

def create_pie_chart(data: pd.Series, title: str = None, output_dir: str = "out/figures",
                    filename: str = None, figsize: Tuple[int, int] = (8, 8)):
    """Create a standardized pie chart."""
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.pie(data.values, labels=data.index, autopct='%1.1f%%')
    
    if title:
        ax.set_title(title, fontweight='bold')
    
    if filename:
        save_plot(output_dir, filename)
    
    return fig, ax

def create_multi_panel_plot(plot_funcs: List[Tuple], nrows: int, ncols: int,
                           figsize: Tuple[int, int] = (20, 12), output_dir: str = "out/figures",
                           filename: str = None):
    """Create a multi-panel plot with subplots."""
    fig = plt.figure(figsize=figsize)
    
    for i, (plot_func, args, kwargs) in enumerate(plot_funcs):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        plot_func(*args, ax=ax, **kwargs)
    
    plt.tight_layout()
    
    if filename:
        save_plot(output_dir, filename)
    
    return fig

def add_value_labels(ax, bars, values, fontsize: int = 10):
    """Add value labels on top of bars."""
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{value:.3f}', ha='center', va='bottom', fontsize=fontsize)

def create_comparison_plot(data1: pd.DataFrame, data2: pd.DataFrame, 
                          x_col: str, y_col: str, label1: str, label2: str,
                          title: str, output_dir: str = "out/figures", 
                          filename: str = None, figsize: Tuple[int, int] = (12, 8)):
    """Create a comparison plot between two datasets."""
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(data1[x_col]))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, data1[y_col], width, label=label1, alpha=0.8)
    bars2 = ax.bar(x + width/2, data2[y_col], width, label=label2, alpha=0.8)
    
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_xticks(x)
    ax.set_xticklabels(data1[x_col], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if filename:
        save_plot(output_dir, filename)
    
    return fig, ax

def create_strategy_analysis_plot(data: List[Dict], output_dir: str = "out/figures"):
    """Create analysis plot for different misinformation strategies."""
    df = pd.DataFrame(data)
    
    if df.empty:
        print("No data available for strategy analysis plot")
        return
    
    # Filter for misinformed condition only
    misinformed_data = df[df['condition'] == 'misinformed']
    
    if misinformed_data.empty:
        print("No misinformed data available for strategy analysis")
        return
    
    # Calculate accuracy by strategy
    strategy_stats = misinformed_data.groupby('misinformation_strategy')['agent_is_correct'].agg(['mean', 'count']).reset_index()
    strategy_stats.columns = ['strategy', 'accuracy', 'count']
    strategy_stats = strategy_stats[strategy_stats['count'] >= 5]  # Filter strategies with enough samples
    
    if strategy_stats.empty:
        print("No strategies with enough samples for analysis")
        return
    
    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Accuracy by strategy
    bars = axes[0].bar(strategy_stats['strategy'], strategy_stats['accuracy'], alpha=0.7)
    axes[0].set_title('Accuracy by Misinformation Strategy', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # Add count labels on bars
    add_value_labels(axes[0], bars, strategy_stats['count'])
    
    # Plot 2: Strategy distribution
    strategy_counts = misinformed_data['misinformation_strategy'].value_counts()
    axes[1].pie(strategy_counts.values, labels=strategy_counts.index, autopct='%1.1f%%')
    axes[1].set_title('Distribution of Misinformation Strategies', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_plot(output_dir, "strategy_analysis.pdf")

def create_setup_comparison_plot(data: List[Dict], output_dir: str = "out/figures"):
    """Create comparison plot across different experimental setups."""
    df = pd.DataFrame(data)
    
    if df.empty:
        print("No data available for setup comparison plot")
        return
    
    # Calculate accuracy by condition and dataset
    setup_stats = df.groupby(['dataset', 'condition'])['agent_is_correct'].agg(['mean', 'count']).reset_index()
    setup_stats.columns = ['dataset', 'condition', 'accuracy', 'count']
    
    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Accuracy by condition
    for i, dataset in enumerate(setup_stats['dataset'].unique()):
        dataset_data = setup_stats[setup_stats['dataset'] == dataset]
        axes[0].bar([f"{c}\n({dataset})" for c in dataset_data['condition']], 
                   dataset_data['accuracy'], 
                   alpha=0.7, 
                   label=dataset)
    
    axes[0].set_title('Accuracy by Experimental Setup', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Performance difference from baseline
    baseline_data = setup_stats[setup_stats['condition'] == 'baseline'].set_index('dataset')['accuracy']
    for condition in ['misinformed', 'irrelevant_misinformed']:
        condition_data = setup_stats[setup_stats['condition'] == condition].set_index('dataset')['accuracy']
        diff = condition_data - baseline_data
        axes[1].bar(diff.index, diff, alpha=0.7, label=f'{condition} - baseline')
    
    axes[1].set_title('Performance Difference from Baseline', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Accuracy Difference')
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(output_dir, "setup_comparison.pdf")

def create_turn_analysis_plot(data: List[Dict], output_dir: str = "out/figures"):
    """Create analysis plot showing performance across turns."""
    df = pd.DataFrame(data)
    
    if df.empty:
        print("No data available for turn analysis plot")
        return
    
    # Calculate accuracy by turn, condition, and agent
    turn_stats = df.groupby(['turn', 'condition', 'agent'])['agent_is_correct'].agg(['mean', 'count']).reset_index()
    turn_stats.columns = ['turn', 'condition', 'agent', 'accuracy', 'count']
    
    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Accuracy by turn for each condition
    for condition in ['baseline', 'misinformed', 'irrelevant_misinformed']:
        condition_data = turn_stats[turn_stats['condition'] == condition]
        if not condition_data.empty:
            # Average across agents
            avg_by_turn = condition_data.groupby('turn')['accuracy'].mean()
            axes[0].plot(avg_by_turn.index, avg_by_turn.values, marker='o', label=condition, linewidth=2)
    
    axes[0].set_title('Accuracy Evolution Across Turns', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Turn')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Agent comparison across turns
    for agent in ['Agent 0', 'Agent 1']:
        agent_data = turn_stats[turn_stats['agent'] == agent]
        if not agent_data.empty:
            # Average across conditions
            avg_by_turn = agent_data.groupby('turn')['accuracy'].mean()
            axes[1].plot(avg_by_turn.index, avg_by_turn.values, marker='s', label=agent, linewidth=2)
    
    axes[1].set_title('Agent Performance Across Turns', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Turn')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(output_dir, "turn_analysis.pdf")
