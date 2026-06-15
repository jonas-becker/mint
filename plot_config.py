# I thank Claude Sonnet 4.5 (and LBK) for its help in writing this file.
import matplotlib.pyplot as plt
from matplotlib import cycler
from pathlib import Path
import numpy as np
import json
import os
from scipy.stats import chi2_contingency
from matplotlib.offsetbox import OffsetImage
from PIL import Image

# LaTeX document dimensions in points
# Convert to inches for matplotlib (1 pt = 1/72 inch)

TEXT_WIDTH_INCHES = 5.5  # inches
COLUMN_WIDTH_INCHES = TEXT_WIDTH_INCHES/2.0  # inches


def setup_plot_style(use_latex=True):
    """
    Configure matplotlib with consistent style settings.
    """
    # Apply matplotlib style
    plt.style.use("seaborn-v0_8-muted")

    # Apply the project color cycle (so seaborn defaults inherit it).
    try:
        from figure_style import get_palette

        plt.rcParams["axes.prop_cycle"] = cycler(color=get_palette(8))
    except Exception:
        pass

    # Font configuration - Latin Modern Roman with LaTeX support
    plt.rcParams["font.family"] = "serif"
    # Use robust fallbacks across OSes (Windows often lacks TeX Gyre).
    plt.rcParams["font.serif"] = [
        "Palatino",
        "TeX Gyre Pagella",
        "Times New Roman",
        "Times",
        "DejaVu Serif",
    ]
    plt.rcParams["text.usetex"] = use_latex
    plt.rcParams["mathtext.fontset"] = "cm"  # Computer Modern for math

    # Size configuration – intentionally compact so that all plots
    # look consistent even at COLUMN_WIDTH_INCHES (~2.75 in).
    plt.rcParams["font.size"] = 8           # base size for annotations etc.
    plt.rcParams["axes.titlesize"] = 8
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["xtick.labelsize"] = 7
    plt.rcParams["ytick.labelsize"] = 7
    plt.rcParams["legend.fontsize"] = 7
    plt.rcParams["figure.titlesize"] = 8

    # Line and marker configuration
    plt.rcParams["lines.linewidth"] = 1.0
    plt.rcParams["lines.markersize"] = 4

    # Figure and export configuration
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"

    # Optional: Remove top and right spines for cleaner look
    # Commented out by default - uncomment if desired
    # plt.rcParams["axes.spines.top"] = False
    # plt.rcParams["axes.spines.right"] = False

GAMMA = "#A79D5B"
ETA = "#6C8A5B"

# University/Logo color palette
UNIBLAU = "#153268"
LOGOBLAU = "#005f9b"
LOGOHELLBLAU30 = "#d2e6fa"
LOGOHELLBLAU = "#50a5d2"
LOGOMITTELBLAU = "#0091c8"

# ---------------------------------------------------------------------------
# Logo-inspired model colors
# Colors are derived from the dominant hues of each provider's logo icon.
# Both space-separated ("Qwen 3 32B") and hyphen-separated ("Qwen3-32B")
# display-name variants are provided so every plot can look up by display name.
# ---------------------------------------------------------------------------
MODEL_COLORS = {
    # --- OpenAI / GPT ---  (logo: black → use ChatGPT brand teal-green)
    "GPT-OSS 120B":   "#10A37F",
    "GPT-OSS-120B":   "#10A37F",

    # --- Google / Gemma ---  (logo dominant: blue-purple #4080c0)
    "Gemma 3 27B":    "#4285F4",
    "Gemma-3-27B":    "#4285F4",
    "Gemma 3 12B":    "#72A0D4",

    # --- Qwen ---  (logo dominant: purple-indigo #6040e0, graduated light→dark by size)
    "Qwen 3 0.6B":    "#C4B5FD",
    "Qwen3-0.6B":     "#C4B5FD",
    "Qwen 3 1.7B":    "#AB8EFA",
    "Qwen3-1.7B":     "#AB8EFA",
    "Qwen 3 4B":      "#A78BFA",
    "Qwen3-4B":       "#A78BFA",
    "Qwen 3 4B (FT)": "#9070F0",
    "Qwen 3 8B":      "#7650E4",
    "Qwen3-8B":       "#7650E4",
    "Qwen 3 14B":     "#8B5CF6",
    "Qwen3-14B":      "#8B5CF6",
    "Qwen 3 32B":     "#6D28D9",
    "Qwen3-32B":      "#6D28D9",
    "Qwen 3 VL 32B":  "#C026D3",  # VL variant – distinct hue shift
    "Qwen3-VL-32B":   "#C026D3",

    # --- Qwen 2.5 ---
    "Qwen 2.5 72B":   "#C8AEFF",
    "Qwen 2.5 32B":   "#BEAAFF",
    "Qwen 2.5 14B":   "#B49EFF",
    "Qwen 2.5 7B":    "#A890F0",

    # --- DeepSeek / R1 ---  (logo dominant: cobalt blue #4060a0)
    "R1 Distill 32B":       "#1E3A8A",
    "R1-Distill-32B":       "#1E3A8A",
    "R1 Llama Distill 70B": "#6488BC",

    # --- Meta / Llama ---  (logo dominant: bright royal blue #0080e0)
    "Llama 3.3 70B":  "#0075D8",
    "Llama 3.1 70B":  "#2892E0",
    "Llama 3.1 8B":   "#60AAEC",

    # --- NVIDIA / Nemotron ---  (logo dominant: lime green #60a000)
    "Nemotron 49B":   "#76B900",
    "Nemotron-49B":   "#76B900",

    # --- AllenAI / OLMo ---  (logo dominant: hot pink #e04080)
    "OLMo 3.1 32B":   "#BE185D",
    "OLMo-3.1-32B":   "#BE185D",

    # --- Mistral / Magistral ---  (logo dominant: warm orange #e06020)
    "Magistral Small":  "#EA580C",
    "Magistral-Small":  "#EA580C",

    # --- Rule-based / baseline agents ---
    "Rule Agent":     "#707070",
    "Random Agent":   "#A0A0A0",
    "A*":             "#505050",

    # --- Human ---
    "Human":          "#475569",
}

# Keyword-based fallback colors keyed on tokens found in internal model path/name.
# Used when a model's display name is not explicitly listed in MODEL_COLORS.
# Keys are matched case-insensitively against the internal model identifier.
MODEL_FAMILY_COLORS = {
    "openai":    "#10A37F",  # GPT / OpenAI → ChatGPT teal-green
    "gpt":       "#10A37F",
    "google":    "#4E84C4",  # Gemma → blue (Gemma logo)
    "gemma":     "#4E84C4",
    "Qwen":      "#6040E0",  # Qwen → purple-indigo (Qwen logo)
    "qwen":      "#6040E0",
    "deepseek":  "#4A6EA8",  # R1 / DeepSeek → cobalt blue (DeepSeek logo)
    "r1":        "#4A6EA8",
    "llama":     "#0075D8",  # Llama → royal blue (Llama logo)
    "nvidia":    "#76B900",  # Nemotron → NVIDIA lime green
    "nemotron":  "#76B900",
    "allenai":   "#D43870",  # OLMo → hot pink (OLMo logo)
    "olmo":      "#D43870",
    "mistral":   "#D96818",  # Magistral → warm orange (Mistral logo)
    "magistral": "#D96818",
    "9Tobi":     "#9070F0",  # Fine-tuned Qwen variant → same Qwen purple
}

# Fallback color for undefined models - VERY OBVIOUS!
MODEL_COLOR_FALLBACK = "#FF00FF"  # Bright Magenta - impossible to miss!

# Colors for the three benchmark variants (SPaRC / SPaRC-Gym / Traceback).
# Centralised here so all plots use consistent variant colors.
VARIANT_COLORS = {
    "sparc":     "#2E7D32",  # Dark green  – SPaRC (baseline)
    "gym":       "#1976D2",  # Medium blue – SPaRC-Gym
    "traceback": "#E65100",  # Deep orange – SPaRC-Gym Traceback
}

# ----------------------------------------------------------------------------
# Misinformation relevance colors (used across exp1/exp2 figures)
# ----------------------------------------------------------------------------
# These are the canonical colors for the repo when comparing:
# - baseline / none: green
# - relevant (incl. false_fact / Neutral): red
# - irrelevant: blue
MISINFO_RELEVANCE_COLORS = {
    # For relevance figures: baseline should be gray, and the other two use the 3-color palette.
    "baseline": "#999999",   # neutral gray
    "relevant": "#ffbb6f",   # soft gold
    "irrelevant": "#5e4c5f", # dull purple
    "irrelevant_true": "#6f8fa6",  # muted blue (irrelevant true information)
}

# Training method colors - Based on Seaborn Set2 palette
TRAINING_METHOD_COLORS = {
    "Baseline": "#8DA0CB",      # Blue-purple
    "SFT": "#FC8D62",           # Orange
    "GRPO": "#66C2A5",          # Teal
    "GRPO-L": "#66C2A5",        # Teal (same as GRPO)
    "Step-by-step": "#E78AC3",  # Pink
}


def get_model_color(model_name, warn_on_missing=True):
    """
    Get the color for a specific model from the MODEL_COLORS dict.
    """
    if model_name in MODEL_COLORS:
        return MODEL_COLORS[model_name]

    if warn_on_missing:
        print(f"⚠️  WARNING: No color defined for model '{model_name}'!")
        print(f"   Using fallback color {MODEL_COLOR_FALLBACK} (bright magenta)")
        print(f"   Please add '{model_name}' to MODEL_COLORS in plot_config.py")

    return MODEL_COLOR_FALLBACK


def get_model_colors(model_names, warn_on_missing=True):
    """
    Get colors for multiple models.
    """
    return [get_model_color(name, warn_on_missing) for name in model_names]


def get_training_method_color(method_name, warn_on_missing=True):
    """
    Get the color for a specific training method from the TRAINING_METHOD_COLORS dict.
    
    Args:
        method_name: Training method name (e.g., 'Baseline', 'SFT', 'GRPO', 'Step-by-step')
        warn_on_missing: Whether to print warning if method not found
    
    Returns:
        Hex color string
    """
    if method_name in TRAINING_METHOD_COLORS:
        return TRAINING_METHOD_COLORS[method_name]
    
    if warn_on_missing:
        print(f"⚠️  WARNING: No color defined for training method '{method_name}'!")
        print(f"   Using fallback color {MODEL_COLOR_FALLBACK} (bright magenta)")
        print(f"   Please add '{method_name}' to TRAINING_METHOD_COLORS in plot_config.py")
    
    return MODEL_COLOR_FALLBACK


def get_training_method_colors(method_names, warn_on_missing=True):
    """
    Get colors for multiple training methods.
    
    Args:
        method_names: List of training method names
        warn_on_missing: Whether to print warning if methods not found
    
    Returns:
        List of hex color strings
    """
    return [get_training_method_color(name, warn_on_missing) for name in method_names]


def get_model_imagebox(model_name, zoom_factor=1.0, rotation=0):
    """
    Get an OffsetImage (imagebox) for a model's logo.
    zoom_factor: multiply the default zoom by this value (e.g. 0.7 for smaller logos).
    rotation: counter-clockwise rotation in degrees applied to the logo image.
    """
    # Internal mapping for logo files - tuples of (width, height, zoom)
    # Some logos are taller, some are wider, adjust dimensions and zoom as needed
    LOGO_CONFIG = {
        "gemma.png": (64, 64, 1/7),
        "qwen.png": (64, 64, 1/7.5),
        "qwen-no-reason.png": (64, 64, 1/7.5),
        "deepseek.png": (64, 64, 1/6),
        "llama.png": (64, 64, 1/6),
        "nvidia.png": (64, 64, 1/6),
        "human.png": (64, 64, 1/8),
        "openai.png": (64, 64, 1/7),
        "gemini.png": (64, 64, 1/7),
        "olmo.png": (64, 64, 1/7),
        "mistral.png": (64, 64, 1/7),
    }

    LOGO_MAPPING = {
        "Human": "human.png",
        "Gemma": "gemma.png",
        "Qwen No Reason": "qwen-no-reason.png",
        "Qwen": "qwen.png",
        "R1": "deepseek.png",
        "Llama": "llama.png",
        "Nemotron": "nvidia.png",
        "GPT": "openai.png",
        "OLMo": "olmo.png",
        "Magistral": "mistral.png",
    }

    logo_path = None
    for keyword, logo in LOGO_MAPPING.items():
        if keyword in model_name:
            logo_path = Path(__file__).parent / "logos" / logo
            if logo_path.exists():
                break
    
    if not logo_path:
        return None
    
    # Load the logo with PIL
    img_pil = Image.open(str(logo_path)).convert('RGBA')
    
    # Get configuration (size and zoom) based on logo filename
    width, height, zoom = LOGO_CONFIG.get(logo_path.name)  # Default config
    
    # Resize to thumbnail size while maintaining aspect ratio
    img_pil.thumbnail((width, height), Image.Resampling.LANCZOS)

    # Apply rotation if requested
    if rotation != 0:
        img_pil = img_pil.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC)

    # Create and return OffsetImage with specified zoom
    imagebox = OffsetImage(np.array(img_pil), zoom=zoom * zoom_factor)
    
    return imagebox


def perform_chi_square_test(contingency_table, test_name, group1_name, group2_name, alpha=0.05, remove_zero_columns=True, show_effect_size_interpretation=False):
    """
    Perform chi-square test for homogeneity on a contingency table.

    This is a general-purpose function for testing whether the distribution of
    categorical variables differs significantly between two or more groups.
    """
    # Convert to numpy array if needed
    if hasattr(contingency_table, "values"):  # pandas DataFrame
        data = contingency_table.copy()
        if remove_zero_columns:
            data = data.loc[:, (data != 0).any()]
        contingency_array = data.values
    else:
        contingency_array = np.array(contingency_table)
        if remove_zero_columns:
            # Remove columns that are all zeros
            contingency_array = contingency_array[:, (contingency_array != 0).any(axis=0)]

    print(f"\n--- {test_name} ---")
    if hasattr(contingency_table, "to_string"):
        print(f"Contingency table:")
        print(data.to_string() if remove_zero_columns else contingency_table.to_string())
    else:
        print(f"Contingency table shape: {contingency_array.shape}")

    # Perform chi-square test
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_array)

    # Calculate Cramer's V (effect size)
    n = contingency_array.sum()  # Total sample size
    min_dim = min(contingency_array.shape[0], contingency_array.shape[1]) - 1
    cramers_v = np.sqrt(chi2_stat / (n * min_dim)) if min_dim > 0 else 0

    print(f"\nNull hypothesis: Distribution patterns are homogeneous across {group1_name} and {group2_name}")
    print(f"Alternative hypothesis: Distribution patterns differ significantly between groups")
    print(f"Chi-square statistic: {chi2_stat:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"P-value: {p_value:.6f}")
    print(f"Cramer's V (effect size): {cramers_v:.4f}")

    if show_effect_size_interpretation:
        if cramers_v < 0.1:
            effect_interpretation = "negligible"
        elif cramers_v < 0.3:
            effect_interpretation = "small"
        elif cramers_v < 0.5:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        print(f"Effect size interpretation: {effect_interpretation}")

    significant = p_value < alpha
    if significant:
        print(f"Result: SIGNIFICANT (p < {alpha}) - Distribution patterns differ significantly between groups")
    else:
        print(f"Result: NOT SIGNIFICANT (p >= {alpha}) - No significant difference in distribution patterns")

    return {"chi2_stat": chi2_stat, "p_value": p_value, "dof": dof, "cramers_v": cramers_v, "significant": significant}


# Additional color palettes can be added here in the future
# For example:
# PLAYER_COLORS = [...]
# TECHNIQUE_COLORS = {...}
# etc.

# Helper: desaturate color for negative values
def desaturate_color(hexcolor, factor=0.4):
    """Desaturate a color by blending it with gray"""
    hexcolor = hexcolor.lstrip('#')
    r, g, b = int(hexcolor[0:2], 16), int(hexcolor[2:4], 16), int(hexcolor[4:6], 16)
    # Convert to grayscale value
    gray = int(0.299 * r + 0.587 * g + 0.114 * b)
    # Blend with grayscale
    r_new = int(r * factor + gray * (1 - factor))
    g_new = int(g * factor + gray * (1 - factor))
    b_new = int(b * factor + gray * (1 - factor))
    return f'#{r_new:02x}{g_new:02x}{b_new:02x}'
