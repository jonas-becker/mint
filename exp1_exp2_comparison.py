#!/usr/bin/env python3

"""
exp1_exp2_comparison.py - Compare results between exp1 (single agent) and exp2 (multi-agent debate)
This script loads both exp1 and exp2 results and creates comparative visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import argparse
from typing import Dict, List, Tuple
from collections import defaultdict

# Import shared utilities and visualization functions
from shared_utils import load_results, find_result_files, extract_dataset_name
from shared_visualization import (
    setup_plot_style, save_plot, create_bar_plot, create_line_plot,
    create_heatmap, create_pie_chart, add_value_labels
)
from figure_style import get_palette

# Set up plot styling
setup_plot_style()

def load_exp1_results() -> Dict:
    """Load all exp1 results files and combine them."""
    print("Loading exp1 (single agent) results...")
    
    # Find all exp1 results files
    exp1_files = []
    if os.path.exists("out"):
        for file in os.listdir("out"):
            if file.startswith("exp1_results_") and file.endswith(".json"):
                exp1_files.append(os.path.join("out", file))
    
    if not exp1_files:
        print("No exp1 results files found.")
        return {}
    
    print(f"Found {len(exp1_files)} exp1 results files:")
    for file in exp1_files:
        print(f"  - {file}")
    
    # Load and combine all exp1 results
    combined_exp1 = {}
    for file_path in exp1_files:
        try:
            results = load_results(file_path)
            # Extract dataset name from filename
            dataset_name = extract_dataset_name(file_path)
            if dataset_name:
                combined_exp1[dataset_name] = results
                print(f"  Loaded {dataset_name}: {len(results)} datasets")
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
    
    return combined_exp1

def load_exp2_results() -> Dict:
    """Load all exp2 results files and combine them."""
    print("Loading exp2 (multi-agent debate) results...")
    
    # Find all exp2 results files
    exp2_files = []
    if os.path.exists("out"):
        for file in os.listdir("out"):
            if file.startswith("exp2_results_") and file.endswith(".json"):
                exp2_files.append(os.path.join("out", file))
    
    if not exp2_files:
        print("No exp2 results files found.")
        return {}
    
    print(f"Found {len(exp2_files)} exp2 results files:")
    for file in exp2_files:
        print(f"  - {file}")
    
    # Load and combine all exp2 results
    combined_exp2 = {}
    for file_path in exp2_files:
        try:
            results = load_results(file_path)
            # Extract dataset name from filename
            dataset_name = extract_dataset_name(file_path)
            if dataset_name:
                combined_exp2[dataset_name] = results
                print(f"  Loaded {dataset_name}: {len(results)} datasets")
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
    
    return combined_exp2

def extract_exp1_data(exp1_results: Dict) -> List[Dict]:
    """Extract data from exp1 results for comparison."""
    data = []
    
    for file_dataset_name, file_results in exp1_results.items():
        # The file_results contains the actual dataset results
        for actual_dataset_name, dataset_results in file_results.items():
            for condition, condition_results in dataset_results.items():
                if isinstance(condition_results, dict) and 'accuracy' in condition_results:
                    data.append({
                        'dataset': actual_dataset_name,
                        'condition': condition,
                        'accuracy': condition_results['accuracy'],
                        'correct_count': condition_results.get('correct_count', 0),
                        'total_count': condition_results.get('total_count', 0),
                        'experiment': 'exp1_single_agent'
                    })
    
    return data

def extract_exp2_data(exp2_results: Dict) -> List[Dict]:
    """Extract data from exp2 results for comparison."""
    data = []
    
    for file_dataset_name, file_results in exp2_results.items():
        # The file_results contains the actual dataset results
        for actual_dataset_name, dataset_results in file_results.items():
            if isinstance(dataset_results, list):
                # Group by experimental condition
                by_condition = defaultdict(list)
                for result in dataset_results:
                    by_condition[result["experimental_condition"]].append(result)
                
                # Calculate accuracy for each condition
                for condition, results in by_condition.items():
                    correct_count = sum(1 for r in results if r["is_correct"])
                    accuracy = correct_count / len(results) if results else 0.0
                    
                    data.append({
                        'dataset': actual_dataset_name,
                        'condition': condition,
                        'accuracy': accuracy,
                        'correct_count': correct_count,
                        'total_count': len(results),
                        'experiment': 'exp2_multi_agent'
                    })
    
    return data

def create_experiment_comparison_plot(exp1_data: List[Dict], exp2_data: List[Dict], output_dir: str = "out/figures/comparison"):
    """Create a comparison plot between exp1 and exp2 results."""
    # Combine data
    all_data = exp1_data + exp2_data
    df = pd.DataFrame(all_data)
    
    if df.empty:
        print("No data available for experiment comparison plot")
        return
    
    # Create the plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Overall accuracy comparison by experiment
    ax1 = axes[0, 0]
    exp_accuracy = df.groupby('experiment')['accuracy'].mean()
    pal2 = get_palette(2)
    bars = ax1.bar(exp_accuracy.index, exp_accuracy.values, alpha=0.85, color=pal2)
    ax1.set_title('Overall Accuracy: Single Agent vs Multi-Agent', fontweight='bold')
    ax1.set_ylabel('Average Accuracy')
    ax1.set_ylim(0, 1)
    # Add value labels
    for bar, value in zip(bars, exp_accuracy.values):
        ax1.text(bar.get_x() + bar.get_width()/2., value + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Accuracy by condition and experiment
    ax2 = axes[0, 1]
    condition_exp_accuracy = df.groupby(['condition', 'experiment'])['accuracy'].mean().unstack()
    condition_exp_accuracy.plot(kind='bar', ax=ax2, alpha=0.7)
    ax2.set_title('Accuracy by Condition and Experiment', fontweight='bold')
    ax2.set_ylabel('Average Accuracy')
    ax2.set_ylim(0, 1)
    ax2.legend(title='Experiment')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Accuracy by dataset and experiment
    ax3 = axes[1, 0]
    dataset_exp_accuracy = df.groupby(['dataset', 'experiment'])['accuracy'].mean().unstack()
    dataset_exp_accuracy.plot(kind='bar', ax=ax3, alpha=0.7)
    ax3.set_title('Accuracy by Dataset and Experiment', fontweight='bold')
    ax3.set_ylabel('Average Accuracy')
    ax3.set_ylim(0, 1)
    ax3.legend(title='Experiment')
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Performance difference (exp2 - exp1)
    ax4 = axes[1, 1]
    # Calculate difference for each dataset and condition
    diff_data = []
    for dataset in df['dataset'].unique():
        for condition in df['condition'].unique():
            exp1_acc = df[(df['dataset'] == dataset) & (df['condition'] == condition) & 
                         (df['experiment'] == 'exp1_single_agent')]['accuracy'].iloc[0] if len(df[(df['dataset'] == dataset) & (df['condition'] == condition) & (df['experiment'] == 'exp1_single_agent')]) > 0 else 0
            exp2_acc = df[(df['dataset'] == dataset) & (df['condition'] == condition) & 
                         (df['experiment'] == 'exp2_multi_agent')]['accuracy'].iloc[0] if len(df[(df['dataset'] == dataset) & (df['condition'] == condition) & (df['experiment'] == 'exp2_multi_agent')]) > 0 else 0
            diff = exp2_acc - exp1_acc
            diff_data.append({
                'dataset': dataset,
                'condition': condition,
                'difference': diff
            })
    
    diff_df = pd.DataFrame(diff_data)
    if not diff_df.empty:
        pivot_diff = diff_df.pivot(index='dataset', columns='condition', values='difference')
        pivot_diff.plot(kind='bar', ax=ax4, alpha=0.7)
        ax4.set_title('Performance Difference (Multi-Agent - Single Agent)', fontweight='bold')
        ax4.set_ylabel('Accuracy Difference')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.legend(title='Condition')
        ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    save_plot(output_dir, "exp1_exp2_comparison.pdf")

def create_detailed_comparison_plot(exp1_data: List[Dict], exp2_data: List[Dict], output_dir: str = "out/figures/comparison"):
    """Create a detailed comparison plot with more granular analysis."""
    all_data = exp1_data + exp2_data
    df = pd.DataFrame(all_data)
    
    if df.empty:
        print("No data available for detailed comparison plot")
        return
    
    # Create the plot
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Plot 1: Sample size comparison
    ax1 = axes[0, 0]
    exp_samples = df.groupby('experiment')['total_count'].sum()
    bars = ax1.bar(exp_samples.index, exp_samples.values, alpha=0.7, color=['skyblue', 'lightcoral'])
    ax1.set_title('Total Sample Size Comparison', fontweight='bold')
    ax1.set_ylabel('Total Samples')
    # Add value labels
    for bar, value in zip(bars, exp_samples.values):
        ax1.text(bar.get_x() + bar.get_width()/2., value + max(exp_samples.values) * 0.01,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Accuracy by condition
    ax2 = axes[0, 1]
    condition_accuracy = df.groupby(['condition', 'experiment'])['accuracy'].mean().unstack()
    condition_accuracy.plot(kind='bar', ax=ax2, alpha=0.7)
    ax2.set_title('Accuracy by Experimental Condition', fontweight='bold')
    ax2.set_ylabel('Average Accuracy')
    ax2.set_ylim(0, 1)
    ax2.legend(title='Experiment')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Accuracy by dataset
    ax3 = axes[0, 2]
    dataset_accuracy = df.groupby(['dataset', 'experiment'])['accuracy'].mean().unstack()
    dataset_accuracy.plot(kind='bar', ax=ax3, alpha=0.7)
    ax3.set_title('Accuracy by Dataset', fontweight='bold')
    ax3.set_ylabel('Average Accuracy')
    ax3.set_ylim(0, 1)
    ax3.legend(title='Experiment')
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Heatmap of accuracy by dataset and condition
    ax4 = axes[1, 0]
    heatmap_data = df.groupby(['dataset', 'condition', 'experiment'])['accuracy'].mean().unstack(level=2)
    if not heatmap_data.empty:
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax4)
        ax4.set_title('Accuracy Heatmap: Dataset × Condition × Experiment', fontweight='bold')
    
    # Plot 5: Performance improvement analysis
    ax5 = axes[1, 1]
    # Calculate improvement for each dataset-condition pair
    improvement_data = []
    for dataset in df['dataset'].unique():
        for condition in df['condition'].unique():
            exp1_acc = df[(df['dataset'] == dataset) & (df['condition'] == condition) & 
                         (df['experiment'] == 'exp1_single_agent')]['accuracy'].iloc[0] if len(df[(df['dataset'] == dataset) & (df['condition'] == condition) & (df['experiment'] == 'exp1_single_agent')]) > 0 else 0
            exp2_acc = df[(df['dataset'] == dataset) & (df['condition'] == condition) & 
                         (df['experiment'] == 'exp2_multi_agent')]['accuracy'].iloc[0] if len(df[(df['dataset'] == dataset) & (df['condition'] == condition) & (df['experiment'] == 'exp2_multi_agent')]) > 0 else 0
            improvement = ((exp2_acc - exp1_acc) / exp1_acc * 100) if exp1_acc > 0 else 0
            improvement_data.append({
                'dataset': dataset,
                'condition': condition,
                'improvement_pct': improvement
            })
    
    improvement_df = pd.DataFrame(improvement_data)
    if not improvement_df.empty:
        pivot_improvement = improvement_df.pivot(index='dataset', columns='condition', values='improvement_pct')
        pivot_improvement.plot(kind='bar', ax=ax5, alpha=0.7)
        ax5.set_title('Performance Improvement (%)', fontweight='bold')
        ax5.set_ylabel('Improvement (%)')
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax5.legend(title='Condition')
        ax5.tick_params(axis='x', rotation=45)
    
    # Plot 6: Statistical summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Calculate summary statistics
    total_samples_exp1 = sum(d['total_count'] for d in exp1_data)
    total_samples_exp2 = sum(d['total_count'] for d in exp2_data)
    avg_accuracy_exp1 = np.mean([d['accuracy'] for d in exp1_data])
    avg_accuracy_exp2 = np.mean([d['accuracy'] for d in exp2_data])
    
    summary_text = f"""
    Statistical Summary:
    
    Total Samples:
    - Exp1 (Single Agent): {total_samples_exp1}
    - Exp2 (Multi-Agent): {total_samples_exp2}
    
    Average Accuracy:
    - Exp1 (Single Agent): {avg_accuracy_exp1:.3f}
    - Exp2 (Multi-Agent): {avg_accuracy_exp2:.3f}
    
    Overall Improvement:
    - Absolute: {avg_accuracy_exp2 - avg_accuracy_exp1:.3f}
    - Relative: {((avg_accuracy_exp2 - avg_accuracy_exp1) / avg_accuracy_exp1 * 100) if avg_accuracy_exp1 > 0 else 0:.1f}%
    
    Datasets: {len(df['dataset'].unique())}
    Conditions: {len(df['condition'].unique())}
    """
    
    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes, fontsize=12,
             verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    save_plot(output_dir, "exp1_exp2_detailed_comparison.pdf")

def create_condition_analysis_plot(exp1_data: List[Dict], exp2_data: List[Dict], output_dir: str = "out/figures/comparison"):
    """Create a detailed analysis of performance by experimental condition."""
    all_data = exp1_data + exp2_data
    df = pd.DataFrame(all_data)
    
    if df.empty:
        print("No data available for condition analysis plot")
        return
    
    # Create the plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Baseline condition comparison
    ax1 = axes[0, 0]
    baseline_data = df[df['condition'] == 'baseline']
    if not baseline_data.empty:
        baseline_accuracy = baseline_data.groupby(['dataset', 'experiment'])['accuracy'].mean().unstack()
        baseline_accuracy.plot(kind='bar', ax=ax1, alpha=0.7)
        ax1.set_title('Baseline Condition Performance', fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.legend(title='Experiment')
        ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Misinformed condition comparison
    ax2 = axes[0, 1]
    misinformed_data = df[df['condition'] == 'misinformed']
    if not misinformed_data.empty:
        misinformed_accuracy = misinformed_data.groupby(['dataset', 'experiment'])['accuracy'].mean().unstack()
        misinformed_accuracy.plot(kind='bar', ax=ax2, alpha=0.7)
        ax2.set_title('Misinformed Condition Performance', fontweight='bold')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim(0, 1)
        ax2.legend(title='Experiment')
        ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Irrelevant misinformed condition comparison
    ax3 = axes[1, 0]
    irrelevant_data = df[df['condition'] == 'irrelevant_misinformed']
    if not irrelevant_data.empty:
        irrelevant_accuracy = irrelevant_data.groupby(['dataset', 'experiment'])['accuracy'].mean().unstack()
        irrelevant_accuracy.plot(kind='bar', ax=ax3, alpha=0.7)
        ax3.set_title('Irrelevant Misinformed Condition Performance', fontweight='bold')
        ax3.set_ylabel('Accuracy')
        ax3.set_ylim(0, 1)
        ax3.legend(title='Experiment')
        ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Overall condition comparison
    ax4 = axes[1, 1]
    condition_exp_accuracy = df.groupby(['condition', 'experiment'])['accuracy'].mean().unstack()
    condition_exp_accuracy.plot(kind='bar', ax=ax4, alpha=0.7)
    ax4.set_title('Overall Condition Performance', fontweight='bold')
    ax4.set_ylabel('Average Accuracy')
    ax4.set_ylim(0, 1)
    ax4.legend(title='Experiment')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    save_plot(output_dir, "exp1_exp2_condition_analysis.pdf")

def main():
    """Main function to generate comparison visualizations."""
    
    parser = argparse.ArgumentParser(description="Compare exp1 and exp2 results")
    parser.add_argument("--output_dir", type=str, default="out/figures/comparison",
                       help="Output directory for figures")
    parser.add_argument("--all_plots", action="store_true",
                       help="Generate all comparison plots")
    parser.add_argument("--experiment_comparison", action="store_true",
                       help="Generate experiment comparison plot")
    parser.add_argument("--detailed_comparison", action="store_true",
                       help="Generate detailed comparison plot")
    parser.add_argument("--condition_analysis", action="store_true",
                       help="Generate condition analysis plot")
    
    args = parser.parse_args()
    
    # Load results
    exp1_results = load_exp1_results()
    exp2_results = load_exp2_results()
    
    if not exp1_results and not exp2_results:
        print("No results found for either exp1 or exp2. Please run the experiments first.")
        return
    
    # Extract data
    print("\nExtracting data for comparison...")
    exp1_data = extract_exp1_data(exp1_results)
    exp2_data = extract_exp2_data(exp2_results)
    
    print(f"Extracted {len(exp1_data)} data points from exp1")
    print(f"Extracted {len(exp2_data)} data points from exp2")
    
    if not exp1_data and not exp2_data:
        print("No valid data extracted from results.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate plots based on arguments
    if args.all_plots or args.experiment_comparison:
        print("Creating experiment comparison plot...")
        create_experiment_comparison_plot(exp1_data, exp2_data, args.output_dir)
    
    if args.all_plots or args.detailed_comparison:
        print("Creating detailed comparison plot...")
        create_detailed_comparison_plot(exp1_data, exp2_data, args.output_dir)
    
    if args.all_plots or args.condition_analysis:
        print("Creating condition analysis plot...")
        create_condition_analysis_plot(exp1_data, exp2_data, args.output_dir)
    
    # If no specific plots requested, generate all (default behavior)
    if not any([args.all_plots, args.experiment_comparison, args.detailed_comparison, args.condition_analysis]):
        print("No specific plots requested. Generating all comparison plots (default behavior)...")
        create_experiment_comparison_plot(exp1_data, exp2_data, args.output_dir)
        create_detailed_comparison_plot(exp1_data, exp2_data, args.output_dir)
        create_condition_analysis_plot(exp1_data, exp2_data, args.output_dir)
    
    print(f"All comparison plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()
