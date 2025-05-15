#!/usr/bin/env python3

import argparse
import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from .stylesheets import STYLES, setup_styles

# Set font to a widely available serif font (with fallbacks)
plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook', 'Century Schoolbook L', 'Utopia', 'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L', 'Times New Roman', 'Times', 'Charter', 'serif']

model_names = {
    "gpt_4o": "GPT-4o",
    "llama_8b": "Llama 8B",
    "gemini_25_pro": "Gemini 2.5 Pro",
    "gpt_4o_mini": "GPT-4o Mini",
    "gemini_flash_001": "Gemini 2.0 Flash",
    "qwen3_32b": "Qwen 3.2B",
    "GPT-4o_Online_Debater": "Online Debater",
    "GPT-4o_Peer_Support": "Peer Support",
    "GPT-4o_Virtual_Influencer": "Virtual Influencer",
    "GPT-4o_Controversial_Topic_Guide": "Controversial Topic Guide",
    "GPT-4o_Political_Strategist": "Political Strategist",
    "Llama-8B_Online_Debater": "Online Debater",
    "Llama-8B_Peer_Support": "Peer Support",
    "Llama-8B_Virtual_Influencer": "Virtual Influencer",
    "Llama-8B_Controversial_Topic_Guide": "Controversial Topic Guide",
    "Llama-8B_Political_Strategist": "Political Strategist",
}

# Define model orders for consistent display, if found in model_names
model_order = ["gemini_flash_001", "gpt_4o_mini", "qwen3_32b", 
               "gpt_4o", "gemini_25_pro", "llama_8b", 
               "GPT-4o_Controversial_Topic_Guide", "GPT-4o_Online_Debater",
               "GPT-4o_Political_Strategist", "GPT-4o_Peer_Support", 
               "GPT-4o_Virtual_Influencer", "Llama-8B_Controversial_Topic_Guide", 
               "Llama-8B_Online_Debater", "Llama-8B_Peer_Support", 
               "Llama-8B_Political_Strategist", "Llama-8B_Virtual_Influencer"]

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate aggregate plots with standard deviations across multiple runs')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Parent directory containing timestamped results subdirectories')
    parser.add_argument('--output_dir', type=str, default='results/aggregate_plots',
                        help='Directory to save the aggregated plots')
    parser.add_argument('--min_runs', type=int, default=2,
                        help='Minimum number of runs required to include a model in plots')
    parser.add_argument('--turns', type=int, default=3,
                        help='Number of conversation turns to analyze')
    parser.add_argument('--models', type=str, nargs='+',
                        help='List of model names to include in the analysis. If not provided, all models will be included.')
    return parser.parse_args()

def find_model_runs(results_dir: str, filter_models: List[str] = None) -> Dict[str, List[str]]:
    """
    Find all model runs in the results directory structure.
    
    Args:
        results_dir: Path to the parent results directory
        filter_models: Optional list of model names to filter by
        
    Returns:
        Dictionary mapping model names to lists of their run directories
    """
    model_runs = defaultdict(list)
    
    # Find all visualization_metrics.json files
    for metrics_file in glob.glob(f"{results_dir}/**/visualization_metrics.json", recursive=True):
        # Extract model name from the directory structure
        model_dir = os.path.dirname(metrics_file)
        parent_dir = os.path.dirname(model_dir)
        
        # This assumes directory structure like results/timestamp/model_name/
        if os.path.basename(parent_dir).startswith("20"):  # Timestamp directory format
            model_name = os.path.basename(model_dir)
            
            # If we're filtering by model names, check the experiment_config.json
            if filter_models:
                config_file = os.path.join(model_dir, "experiment_config.json")
                if os.path.exists(config_file):
                    try:
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                            persuader_model = config.get("PERSUADER_MODEL")
                            if persuader_model not in filter_models:
                                continue  # Skip this model if it's not in our filter list
                    except (json.JSONDecodeError, FileNotFoundError):
                        # If we can't read the config file, skip this model
                        print(f"Warning: Could not read config file {config_file}")
                        continue
                else:
                    # If the config file doesn't exist, skip this model
                    print(f"Warning: Config file not found at {config_file}")
                    continue
            
            model_runs[model_name].append(model_dir)
    
    # Filter out models with too few runs
    return {k: v for k, v in model_runs.items() if len(v) > 0}

def load_metrics(model_dir: str) -> Dict:
    """Load visualization metrics from a model run directory."""
    metrics_file = os.path.join(model_dir, "visualization_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None

def aggregate_metrics_by_category(model_runs: Dict[str, List[str]]) -> Dict[str, Dict]:
    """
    Aggregate metrics across runs for each model, organized by category.
    
    Args:
        model_runs: Dictionary mapping model names to run directories
        
    Returns:
        Nested dictionary with aggregated metrics by model, turn, and category
    """
    aggregated_data = {}
    
    for model_name, run_dirs in model_runs.items():
        print(f"Processing model: {model_name} ({len(run_dirs)} runs)")
        model_data = defaultdict(lambda: defaultdict(dict))
        
        # Collect data across runs
        for run_dir in run_dirs:
            metrics = load_metrics(run_dir)
            if not metrics or 'category_metrics' not in metrics:
                print(f"  Skipping run {run_dir} - missing required metrics")
                continue
                
            category_metrics = metrics['category_metrics']
            if 'turns' not in category_metrics:
                continue
                
            # Process each turn's data
            for turn_data in category_metrics['turns']:
                turn_idx = turn_data['turn']
                
                if 'category_counts' not in turn_data:
                    continue
                    
                category_counts = turn_data['category_counts']
                
                # For each category, collect counts for this run
                for category, counts in category_counts.items():
                    if category not in model_data[turn_idx]:
                        model_data[turn_idx][category] = {
                            'with_attempt': [],
                            'no_attempt': [],
                            'refusal': [],
                            'percentages': []
                        }
                    
                    # Add counts for this run
                    model_data[turn_idx][category]['with_attempt'].append(counts['with_attempt'])
                    model_data[turn_idx][category]['no_attempt'].append(counts['no_attempt'])
                    model_data[turn_idx][category]['refusal'].append(counts['refusal'])
                    
                    # Calculate and store percentage for this run
                    total = counts['with_attempt'] + counts['no_attempt'] + counts['refusal']
                    if total > 0:
                        percentage = (counts['with_attempt'] / total) * 100
                    else:
                        percentage = 0
                    model_data[turn_idx][category]['percentages'].append(percentage)
        
        # Only include model if it has sufficient data
        if model_data:
            aggregated_data[model_name] = dict(model_data)
    
    return aggregated_data

def aggregate_metrics_by_nh_subjects(model_runs: Dict[str, List[str]]) -> Dict[str, Dict]:
    """
    Aggregate metrics for NoncontroversiallyHarmful subjects across runs for each model.
    
    Args:
        model_runs: Dictionary mapping model names to run directories
        
    Returns:
        Nested dictionary with aggregated metrics by model, turn, and NH subject
    """
    aggregated_data = {}
    
    for model_name, run_dirs in model_runs.items():
        print(f"Processing NH subjects for model: {model_name}")
        model_data = defaultdict(lambda: defaultdict(dict))
        
        # Collect data across runs
        for run_dir in run_dirs:
            metrics = load_metrics(run_dir)
            if not metrics or 'nh_subjects_metrics' not in metrics:
                continue
                
            nh_metrics = metrics['nh_subjects_metrics']
            if 'turns' not in nh_metrics:
                continue
                
            # Process each turn's data
            for turn_data in nh_metrics['turns']:
                turn_idx = turn_data['turn']
                
                if 'subject_counts' not in turn_data:
                    continue
                    
                subject_counts = turn_data['subject_counts']
                
                # For each subject, collect counts for this run
                for subject, counts in subject_counts.items():
                    if subject not in model_data[turn_idx]:
                        model_data[turn_idx][subject] = {
                            'with_attempt': [],
                            'no_attempt': [],
                            'refusal': [],
                            'percentages': []
                        }
                    
                    # Add counts for this run
                    model_data[turn_idx][subject]['with_attempt'].append(counts['with_attempt'])
                    model_data[turn_idx][subject]['no_attempt'].append(counts['no_attempt'])
                    model_data[turn_idx][subject]['refusal'].append(counts['refusal'])
                    
                    # Calculate and store percentage for this run
                    total = counts['with_attempt'] + counts['no_attempt'] + counts['refusal']
                    if total > 0:
                        percentage = (counts['with_attempt'] / total) * 100
                    else:
                        percentage = 0
                    model_data[turn_idx][subject]['percentages'].append(percentage)
        
        # Only include model if it has NH subject data
        if model_data:
            aggregated_data[model_name] = dict(model_data)
    
    return aggregated_data

def create_category_counts_plots(aggregated_data: Dict, output_dir: str, min_runs: int):
    """Create plots showing attempt/no-attempt/refusal counts by category with error bars."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each model separately
    for model_name, model_data in aggregated_data.items():
        print(f"Generating category counts plots for model: {model_name}")
        
        # Process each turn
        for turn_idx, turn_data in model_data.items():
            categories = sorted(turn_data.keys())
            
            # Skip if no categories
            if not categories:
                continue
                
            # Prepare data for plotting
            with_attempt_means = []
            with_attempt_stds = []
            no_attempt_means = []
            no_attempt_stds = []
            refusal_means = []
            refusal_stds = []
            
            # Collect valid categories with enough runs
            valid_categories = []
            
            for category in categories:
                category_data = turn_data[category]
                
                # Only include categories with enough runs
                run_count = len(category_data['with_attempt'])
                if run_count < min_runs:
                    print(f"  Skipping category {category} - only {run_count} runs (minimum {min_runs})")
                    continue
                
                valid_categories.append(category)
                
                # Calculate means and standard deviations
                with_attempt_means.append(np.mean(category_data['with_attempt']))
                with_attempt_stds.append(np.std(category_data['with_attempt']))
                
                no_attempt_means.append(np.mean(category_data['no_attempt']))
                no_attempt_stds.append(np.std(category_data['no_attempt']))
                
                refusal_means.append(np.mean(category_data['refusal']))
                refusal_stds.append(np.std(category_data['refusal']))
            
            if not valid_categories:
                print(f"  No valid categories with at least {min_runs} runs for turn {turn_idx}")
                continue
            
            # Create bar plot with error bars
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Set width and positions for bars
            bar_width = 0.25
            x = np.arange(len(valid_categories))
            
            # Create grouped bars with error bars
            ax.bar(x - bar_width, with_attempt_means, bar_width, 
                   yerr=with_attempt_stds, capsize=5, 
                   label="With Persuasion Attempt", color="skyblue")
            
            ax.bar(x, no_attempt_means, bar_width, 
                   yerr=no_attempt_stds, capsize=5, 
                   label="No Persuasion Attempt", color="lightgreen")
            
            ax.bar(x + bar_width, refusal_means, bar_width, 
                   yerr=refusal_stds, capsize=5, 
                   label="Refusal", color="salmon")
            
            # Add labels and title
            ax.set_xlabel("Topic Category")
            ax.set_ylabel("Number of Conversations")
            ax.set_title(f"{model_name} - Turn {turn_idx}: Persuasion Attempt Counts by Category (Avg. of {min_runs}+ runs)")
            ax.set_xticks(x)
            ax.set_xticklabels(valid_categories)
            plt.xticks(rotation=45)
            
            # Add legend
            ax.legend()
            
            # Add value labels on the bars
            for i, v in enumerate(with_attempt_means):
                ax.text(i - bar_width, v + 1, 
                        f"{v:.1f}±{with_attempt_stds[i]:.1f}", 
                        ha="center", va="bottom", 
                        fontweight="bold", color="blue",
                        fontsize=9)
            
            for i, v in enumerate(no_attempt_means):
                ax.text(i, v + 1,
                        f"{v:.1f}±{no_attempt_stds[i]:.1f}", 
                        ha="center", va="bottom", 
                        fontweight="bold", color="green",
                        fontsize=9)
            
            for i, v in enumerate(refusal_means):
                ax.text(i + bar_width, v + 1,
                        f"{v:.1f}±{refusal_stds[i]:.1f}", 
                        ha="center", va="bottom", 
                        fontweight="bold", color="red",
                        fontsize=9)
            
            # Add a grid for better readability
            ax.grid(True, axis="y", alpha=0.3)
            
            # Determine appropriate y-axis limit
            max_values = []
            for i in range(len(with_attempt_means)):
                segment_sum = with_attempt_means[i] + with_attempt_stds[i] + \
                             no_attempt_means[i] + no_attempt_stds[i] + \
                             refusal_means[i] + refusal_stds[i]
                max_values.append(segment_sum)
            
            max_value = max(max_values) if max_values else 10
            ax.set_ylim(0, max_value * 1.2)  # Add 20% padding
            
            # Save figure
            model_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, f"category_counts_turn_{turn_idx}.png"))
            plt.close()
            
def create_nh_subjects_counts_plots(aggregated_data: Dict, output_dir: str, min_runs: int):
    """Create plots showing attempt/no-attempt/refusal counts for NH subjects with error bars."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each model separately
    for model_name, model_data in aggregated_data.items():
        print(f"Generating NH subjects counts plots for model: {model_name}")
        
        # Process each turn
        for turn_idx, turn_data in model_data.items():
            subjects = sorted(turn_data.keys())
            
            # Skip if no subjects
            if not subjects:
                continue
                
            # Prepare data for plotting
            with_attempt_means = []
            with_attempt_stds = []
            no_attempt_means = []
            no_attempt_stds = []
            refusal_means = []
            refusal_stds = []
            
            # Collect valid subjects with enough runs
            valid_subjects = []
            
            for subject in subjects:
                subject_data = turn_data[subject]
                
                # Only include subjects with enough runs
                run_count = len(subject_data['with_attempt'])
                if run_count < min_runs:
                    print(f"  Skipping subject {subject} - only {run_count} runs (minimum {min_runs})")
                    continue
                
                valid_subjects.append(subject)
                
                # Calculate means and standard deviations
                with_attempt_means.append(np.mean(subject_data['with_attempt']))
                with_attempt_stds.append(np.std(subject_data['with_attempt']))
                
                no_attempt_means.append(np.mean(subject_data['no_attempt']))
                no_attempt_stds.append(np.std(subject_data['no_attempt']))
                
                refusal_means.append(np.mean(subject_data['refusal']))
                refusal_stds.append(np.std(subject_data['refusal']))
            
            if not valid_subjects:
                print(f"  No valid NH subjects with at least {min_runs} runs for turn {turn_idx}")
                continue
            
            # Create bar plot with error bars
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Set width and positions for bars
            bar_width = 0.25
            x = np.arange(len(valid_subjects))
            
            # Create grouped bars with error bars
            ax.bar(x - bar_width, with_attempt_means, bar_width, 
                   yerr=with_attempt_stds, capsize=5, 
                   label="With Persuasion Attempt", color="skyblue")
            
            ax.bar(x, no_attempt_means, bar_width, 
                   yerr=no_attempt_stds, capsize=5, 
                   label="No Persuasion Attempt", color="lightgreen")
            
            ax.bar(x + bar_width, refusal_means, bar_width, 
                   yerr=refusal_stds, capsize=5, 
                   label="Refusal", color="salmon")
            
            # Add labels and title
            ax.set_xlabel("NoncontroversiallyHarmful Subjects")
            ax.set_ylabel("Number of Conversations")
            ax.set_title(f"{model_name} - Turn {turn_idx}: Persuasion Attempt Counts by NH Subject (Avg. of {min_runs}+ runs)")
            ax.set_xticks(x)
            ax.set_xticklabels(valid_subjects)
            plt.xticks(rotation=45, ha="right")
            
            # Add legend
            ax.legend()
            
            # Add value labels on the bars
            for i, v in enumerate(with_attempt_means):
                ax.text(i - bar_width, v + 1, 
                        f"{v:.1f}±{with_attempt_stds[i]:.1f}", 
                        ha="center", va="bottom", 
                        fontweight="bold", color="blue",
                        fontsize=9)
            
            for i, v in enumerate(no_attempt_means):
                ax.text(i, v + 1,
                        f"{v:.1f}±{no_attempt_stds[i]:.1f}", 
                        ha="center", va="bottom", 
                        fontweight="bold", color="green",
                        fontsize=9)
            
            for i, v in enumerate(refusal_means):
                ax.text(i + bar_width, v + 1,
                        f"{v:.1f}±{refusal_stds[i]:.1f}", 
                        ha="center", va="bottom", 
                        fontweight="bold", color="red",
                        fontsize=9)
            
            # Add a grid for better readability
            ax.grid(True, axis="y", alpha=0.3)
            
            # Determine appropriate y-axis limit
            max_values = []
            for i in range(len(with_attempt_means)):
                segment_sum = with_attempt_means[i] + with_attempt_stds[i] + \
                             no_attempt_means[i] + no_attempt_stds[i] + \
                             refusal_means[i] + refusal_stds[i]
                max_values.append(segment_sum)
            
            max_value = max(max_values) if max_values else 10
            ax.set_ylim(0, max_value * 1.2)  # Add 20% padding
            
            # Save figure
            model_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, f"nh_subjects_counts_turn_{turn_idx}.png"))
            plt.close()
            
def create_percentage_plots(aggregated_data: Dict, output_dir: str, min_runs: int, plot_type: str):
    """
    Create plots showing percentage breakdown of all response types (attempt/no-attempt/refusal).
    
    Args:
        aggregated_data: Dictionary with aggregated metrics
        output_dir: Directory to save plots
        min_runs: Minimum number of runs required to include a category/subject
        plot_type: Either 'category' or 'nh_subjects'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    xlabel = "Topic Category" if plot_type == 'category' else "NoncontroversiallyHarmful Subjects"
    filename_prefix = "category_percentage" if plot_type == 'category' else "nh_subjects_percentage"
    
    # Process each model separately
    for model_name, model_data in aggregated_data.items():
        print(f"Generating {plot_type} percentage plots for model: {model_name}")
        
        # Process each turn
        for turn_idx, turn_data in model_data.items():
            items = sorted(turn_data.keys())
            
            # Skip if no items
            if not items:
                continue
                
            # Prepare data for plotting
            with_attempt_means = []
            no_attempt_means = []
            refusal_means = []
            with_attempt_stds = []
            no_attempt_stds = []
            refusal_stds = []
            total_means = []
            
            # Collect valid items with enough runs
            valid_items = []
            
            for item in items:
                item_data = turn_data[item]
                
                # Only include items with enough runs
                run_count = len(item_data['with_attempt'])
                if run_count < min_runs:
                    continue
                
                valid_items.append(item)
                
                # Calculate totals for each run
                totals = np.array(item_data['with_attempt']) + np.array(item_data['no_attempt']) + np.array(item_data['refusal'])
                
                # Calculate percentages for each response type
                with_attempt_percentages = np.array(item_data['with_attempt']) / totals * 100
                no_attempt_percentages = np.array(item_data['no_attempt']) / totals * 100
                refusal_percentages = np.array(item_data['refusal']) / totals * 100
                
                # Calculate means and standard deviations for each type
                with_attempt_means.append(np.mean(with_attempt_percentages))
                with_attempt_stds.append(np.std(with_attempt_percentages))
                
                no_attempt_means.append(np.mean(no_attempt_percentages))
                no_attempt_stds.append(np.std(no_attempt_percentages))
                
                refusal_means.append(np.mean(refusal_percentages))
                refusal_stds.append(np.std(refusal_percentages))
                
                # Save mean total for reference in labels
                total_means.append(np.mean(totals))
            
            if not valid_items:
                print(f"  No valid {plot_type} with at least {min_runs} runs for turn {turn_idx}")
                continue
            
            # Create stacked bar plot
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Set positions for bars
            x = np.arange(len(valid_items))
            width = 0.8
            
            # Create stacked bars (without error bars initially)
            bars1 = ax.bar(x, with_attempt_means, width, label='With Persuasion Attempt', color='skyblue')
            bars2 = ax.bar(x, no_attempt_means, width, bottom=with_attempt_means, label='No Persuasion Attempt', color='lightgreen')
            
            # Calculate the bottom position for the third bar
            bottom_vals = np.array(with_attempt_means) + np.array(no_attempt_means)
            bars3 = ax.bar(x, refusal_means, width, bottom=bottom_vals, label='Refusal', color='salmon')
            
            # Add error bars for each segment
            # For the first segment (with_attempt), error bars are at the top of the bar
            for i, (mean, std) in enumerate(zip(with_attempt_means, with_attempt_stds)):
                ax.errorbar(x[i] - 0.1, mean, yerr=std, fmt='o', color='black', capsize=5, 
                           elinewidth=1.5, capthick=1.5, markersize=4)
            
            # For the second segment (no_attempt), error bars are at the top of this segment
            for i, (mean, std, bottom) in enumerate(zip(no_attempt_means, no_attempt_stds, with_attempt_means)):
                ax.errorbar(x[i], bottom + mean, yerr=std, fmt='o', color='black', capsize=5,
                           elinewidth=1.5, capthick=1.5, markersize=4)
            
            # For the third segment (refusal), error bars are at the top of this segment
            for i, (mean, std, bottom) in enumerate(zip(refusal_means, refusal_stds, bottom_vals)):
                ax.errorbar(x[i] + 0.1, bottom + mean, yerr=std, fmt='o', color='black', capsize=5,
                           elinewidth=1.5, capthick=1.5, markersize=4)
            
            # Add labels and title
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Percentage of Conversations (%)")
            ax.set_title(f"{model_name} - Turn {turn_idx}: Response Type Breakdown by {xlabel} (Avg. of {min_runs}+ runs)")
            ax.set_xticks(x)
            ax.set_xticklabels(valid_items)
            plt.xticks(rotation=45, ha="right" if plot_type == 'nh_subjects' else "center")
            
            # Add legend
            ax.legend(loc='upper right')
            
            # Add percentage and count labels on bars
            for i in range(len(x)):
                # Only show label if percentage is significant (> 5%)
                if with_attempt_means[i] > 5:
                    ax.text(i, with_attempt_means[i]/2, 
                            f"{with_attempt_means[i]:.1f}%\n±{with_attempt_stds[i]:.1f}", 
                            ha='center', va='center', 
                            fontweight='bold', color='black',
                            fontsize=8)
                
                if no_attempt_means[i] > 5:
                    ax.text(i, with_attempt_means[i] + no_attempt_means[i]/2, 
                            f"{no_attempt_means[i]:.1f}%\n±{no_attempt_stds[i]:.1f}", 
                            ha='center', va='center', 
                            fontweight='bold', color='black',
                            fontsize=8)
                
                if refusal_means[i] > 5:
                    ax.text(i, with_attempt_means[i] + no_attempt_means[i] + refusal_means[i]/2, 
                            f"{refusal_means[i]:.1f}%\n±{refusal_stds[i]:.1f}", 
                            ha='center', va='center', 
                            fontweight='bold', color='black',
                            fontsize=8)
            
            # Add a grid for better readability
            ax.grid(True, axis="y", alpha=0.3)
            
            # Set y-axis to percentage scale with a bit more room for labels at the bottom
            ax.set_ylim(0, 105)
            
            # Save figure
            model_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, f"{filename_prefix}_turn_{turn_idx}.png"))
            plt.close()

def create_multi_model_comparison(aggregated_data: Dict, output_dir: str, min_runs: int, max_turns: int = 3):
    """
    Create plots comparing multiple models on the same chart.
    
    Args:
        aggregated_data: Dictionary with aggregated metrics by model
        output_dir: Directory to save plots
        min_runs: Minimum number of runs required to include a model
        max_turns: Maximum number of turns to process
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all unique categories across all models
    all_categories = set()
    for model_data in aggregated_data.values():
        for turn_data in model_data.values():
            all_categories.update(turn_data.keys())
    
    all_categories = sorted(all_categories)
    
    # For each turn and category, create a model comparison plot
    for turn_idx in range(1, max_turns + 1):
        for category in all_categories:
            # Collect data for this category and turn across models
            models = []
            percentages = []
            std_devs = []
            
            for model_name, model_data in aggregated_data.items():
                if turn_idx not in model_data or category not in model_data[turn_idx]:
                    continue
                
                category_data = model_data[turn_idx][category]
                if len(category_data['percentages']) < min_runs:
                    continue
                
                models.append(model_name)
                percentages.append(np.mean(category_data['percentages']))
                std_devs.append(np.std(category_data['percentages']))
            
            if not models:  # Skip if no models have data for this category and turn
                continue
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Plot bars with error bars
            y_pos = np.arange(len(models))
            ax.barh(y_pos, percentages, xerr=std_devs, align='center', 
                   capsize=5, alpha=0.7, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
            
            # Add labels
            ax.set_yticks(y_pos)
            ax.set_yticklabels(models)
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel('Percentage of Conversations with Persuasion Attempt (%)')
            ax.set_title(f'Turn {turn_idx}: {category} - Model Comparison')
            
            # Add value labels
            for i, (pct, std) in enumerate(zip(percentages, std_devs)):
                ax.text(pct + 1, i, f"{pct:.1f}%±{std:.1f}", 
                        va='center', fontweight='bold')
            
            # Set x-axis limits for percentages
            ax.set_xlim(0, 105)
            
            # Save the comparison plot
            comparison_dir = os.path.join(output_dir, "model_comparisons")
            os.makedirs(comparison_dir, exist_ok=True)
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, f"turn_{turn_idx}_{category}_comparison.png"))
            plt.close()
    
    print(f"Created model comparison plots in {os.path.join(output_dir, 'model_comparisons')}")
            
def create_category_model_comparison_plots(aggregated_data: Dict, output_dir: str, min_runs: int, max_turns: int = 3):
    """
    Create comparison plots showing all models side by side for each category, with stacked bars.
    
    Args:
        aggregated_data: Dictionary with aggregated metrics by model
        output_dir: Directory to save plots
        min_runs: Minimum number of runs required to include a model
        max_turns: Maximum number of turns to process
    """
    comparison_dir = os.path.join(output_dir, "model_comparisons_by_category")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Get all unique categories across all models
    all_categories = set()
    for model_data in aggregated_data.values():
        for turn_data in model_data.values():
            all_categories.update(turn_data.keys())
    
    all_categories = sorted(all_categories)
    
    # For each turn and category, create a model comparison plot
    for turn_idx in range(1, max_turns + 1):
        for category in all_categories:
            # Get models that have data for this category and turn
            valid_models = []
            
            for model_name, model_data in aggregated_data.items():
                if turn_idx in model_data and category in model_data[turn_idx]:
                    category_data = model_data[turn_idx][category]
                    # Only include if we have enough runs
                    if len(category_data['with_attempt']) >= min_runs:
                        valid_models.append(model_name)
            
            if not valid_models:  # Skip if no models have data for this category and turn
                continue
            
            # Create figure
            fig, ax = plt.subplots(figsize=(max(12, len(valid_models) * 1.2), 10))
            
            # Set positions for bars
            x = np.arange(len(valid_models))
            width = 0.8
            
            # Collect data for each model
            with_attempt_means = []
            no_attempt_means = []
            refusal_means = []
            with_attempt_stds = []
            no_attempt_stds = []
            refusal_stds = []
            
            for model_name in valid_models:
                model_data = aggregated_data[model_name]
                category_data = model_data[turn_idx][category]
                
                # Calculate totals for each run
                totals = np.array(category_data['with_attempt']) + np.array(category_data['no_attempt']) + np.array(category_data['refusal'])
                
                # Calculate percentages for each response type
                with_attempt_percentages = np.array(category_data['with_attempt']) / totals * 100
                no_attempt_percentages = np.array(category_data['no_attempt']) / totals * 100
                refusal_percentages = np.array(category_data['refusal']) / totals * 100
                
                # Store means and standard deviations
                with_attempt_means.append(np.mean(with_attempt_percentages))
                with_attempt_stds.append(np.std(with_attempt_percentages))
                
                no_attempt_means.append(np.mean(no_attempt_percentages))
                no_attempt_stds.append(np.std(no_attempt_percentages))
                
                refusal_means.append(np.mean(refusal_percentages))
                refusal_stds.append(np.std(refusal_percentages))
            
            # Create stacked bars
            bars1 = ax.bar(x, with_attempt_means, width, label='With Persuasion Attempt', color='skyblue')
            bars2 = ax.bar(x, no_attempt_means, width, bottom=with_attempt_means, label='No Persuasion Attempt', color='lightgreen')
            
            # Calculate the bottom position for the third bar
            bottom_vals = np.array(with_attempt_means) + np.array(no_attempt_means)
            bars3 = ax.bar(x, refusal_means, width, bottom=bottom_vals, label='Refusal', color='salmon')
            
            # Add error bars for each segment
            # For the first segment (with_attempt), error bars are at the top of the bar
            for i, (mean, std) in enumerate(zip(with_attempt_means, with_attempt_stds)):
                ax.errorbar(x[i] - 0.1, mean, yerr=std, fmt='o', color='black', capsize=5, 
                           elinewidth=1.5, capthick=1.5, markersize=4)
            
            # For the second segment (no_attempt), error bars are at the top of this segment
            for i, (mean, std, bottom) in enumerate(zip(no_attempt_means, no_attempt_stds, with_attempt_means)):
                ax.errorbar(x[i], bottom + mean, yerr=std, fmt='o', color='black', capsize=5,
                           elinewidth=1.5, capthick=1.5, markersize=4)
            
            # For the third segment (refusal), error bars are at the top of this segment
            for i, (mean, std, bottom) in enumerate(zip(refusal_means, refusal_stds, bottom_vals)):
                ax.errorbar(x[i] + 0.1, bottom + mean, yerr=std, fmt='o', color='black', capsize=5,
                           elinewidth=1.5, capthick=1.5, markersize=4)
            
            # Add labels and title
            ax.set_xlabel("Models")
            ax.set_ylabel("Percentage of Conversations (%)")
            ax.set_title(f"Turn {turn_idx}: {category} - Model Comparison")
            ax.set_xticks(x)
            ax.set_xticklabels(valid_models)
            plt.xticks(rotation=45, ha="right")
            
            # Add legend
            ax.legend(loc='upper right')
            
            # Add a grid for better readability
            ax.grid(True, axis="y", alpha=0.3)
            
            # Set y-axis to percentage scale
            ax.set_ylim(0, 105)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, f"turn_{turn_idx}_{category}_comparison_stacked.png"))
            plt.close()
    
    print(f"Created model comparison by category plots in: {comparison_dir}")

def create_nh_subject_model_comparison_plots(aggregated_data: Dict, output_dir: str, min_runs: int, max_turns: int = 3):
    """
    Create comparison plots showing all models side by side for each NH subject, with stacked bars.
    
    Args:
        aggregated_data: Dictionary with aggregated metrics by model
        output_dir: Directory to save plots
        min_runs: Minimum number of runs required to include a model
        max_turns: Maximum number of turns to process
    """
    comparison_dir = os.path.join(output_dir, "model_comparisons_by_nh_subject")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Get all unique subjects across all models
    all_subjects = set()
    for model_data in aggregated_data.values():
        for turn_data in model_data.values():
            all_subjects.update(turn_data.keys())
    
    all_subjects = sorted(all_subjects)
    
    # For each turn and subject, create a model comparison plot
    for turn_idx in range(1, max_turns + 1):
        for subject in all_subjects:
            # Get models that have data for this subject and turn
            valid_models = []
            
            for model_name, model_data in aggregated_data.items():
                if turn_idx in model_data and subject in model_data[turn_idx]:
                    subject_data = model_data[turn_idx][subject]
                    # Only include if we have enough runs
                    if len(subject_data['with_attempt']) >= min_runs:
                        valid_models.append(model_name)
            
            if not valid_models:  # Skip if no models have data for this subject and turn
                continue
            
            # Create figure
            fig, ax = plt.subplots(figsize=(max(12, len(valid_models) * 1.2), 10))
            
            # Set positions for bars
            x = np.arange(len(valid_models))
            width = 0.8
            
            # Collect data for each model
            with_attempt_means = []
            no_attempt_means = []
            refusal_means = []
            with_attempt_stds = []
            no_attempt_stds = []
            refusal_stds = []
            
            for model_name in valid_models:
                model_data = aggregated_data[model_name]
                subject_data = model_data[turn_idx][subject]
                
                # Calculate totals for each run
                totals = np.array(subject_data['with_attempt']) + np.array(subject_data['no_attempt']) + np.array(subject_data['refusal'])
                
                # Calculate percentages for each response type
                with_attempt_percentages = np.array(subject_data['with_attempt']) / totals * 100
                no_attempt_percentages = np.array(subject_data['no_attempt']) / totals * 100
                refusal_percentages = np.array(subject_data['refusal']) / totals * 100
                
                # Store means and standard deviations
                with_attempt_means.append(np.mean(with_attempt_percentages))
                with_attempt_stds.append(np.std(with_attempt_percentages))
                
                no_attempt_means.append(np.mean(no_attempt_percentages))
                no_attempt_stds.append(np.std(no_attempt_percentages))
                
                refusal_means.append(np.mean(refusal_percentages))
                refusal_stds.append(np.std(refusal_percentages))
            
            # Create stacked bars
            bars1 = ax.bar(x, with_attempt_means, width, label='With Persuasion Attempt', color='skyblue')
            bars2 = ax.bar(x, no_attempt_means, width, bottom=with_attempt_means, label='No Persuasion Attempt', color='lightgreen')
            
            # Calculate the bottom position for the third bar
            bottom_vals = np.array(with_attempt_means) + np.array(no_attempt_means)
            bars3 = ax.bar(x, refusal_means, width, bottom=bottom_vals, label='Refusal', color='salmon')
            
            # Add error bars for each segment
            # For the first segment (with_attempt), error bars are at the top of the bar
            for i, (mean, std) in enumerate(zip(with_attempt_means, with_attempt_stds)):
                ax.errorbar(x[i] - 0.1, mean, yerr=std, fmt='o', color='black', capsize=5, 
                           elinewidth=1.5, capthick=1.5, markersize=4)
            
            # For the second segment (no_attempt), error bars are at the top of this segment
            for i, (mean, std, bottom) in enumerate(zip(no_attempt_means, no_attempt_stds, with_attempt_means)):
                ax.errorbar(x[i], bottom + mean, yerr=std, fmt='o', color='black', capsize=5,
                           elinewidth=1.5, capthick=1.5, markersize=4)
            
            # For the third segment (refusal), error bars are at the top of this segment
            for i, (mean, std, bottom) in enumerate(zip(refusal_means, refusal_stds, bottom_vals)):
                ax.errorbar(x[i] + 0.1, bottom + mean, yerr=std, fmt='o', color='black', capsize=5,
                           elinewidth=1.5, capthick=1.5, markersize=4)
            
            # Add labels and title
            ax.set_xlabel("Models")
            ax.set_ylabel("Percentage of Conversations (%)")
            ax.set_title(f"Turn {turn_idx}: {subject} - Model Comparison")
            ax.set_xticks(x)
            ax.set_xticklabels(valid_models)
            plt.xticks(rotation=45, ha="right")
            
            # Add legend
            ax.legend(loc='upper right')
            
            # Add a grid for better readability
            ax.grid(True, axis="y", alpha=0.3)
            
            # Set y-axis to percentage scale
            ax.set_ylim(0, 105)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, f"turn_{turn_idx}_{subject}_comparison_stacked.png"))
            plt.close()
    
    print(f"Created model comparison by NH subject plots in: {comparison_dir}")

def create_category_counts_model_comparison_plots(aggregated_data: Dict, output_dir: str, min_runs: int, max_turns: int = 3):
    """
    Create comparison plots showing all models side by side for each category, with stacked count bars.
    
    Args:
        aggregated_data: Dictionary with aggregated metrics by model
        output_dir: Directory to save plots
        min_runs: Minimum number of runs required to include a model
        max_turns: Maximum number of turns to process
    """
    comparison_dir = os.path.join(output_dir, "model_counts_comparisons_by_category")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Get all unique categories across all models
    all_categories = set()
    for model_data in aggregated_data.values():
        for turn_data in model_data.values():
            all_categories.update(turn_data.keys())
    
    all_categories = sorted(all_categories)
    
    # For each turn and category, create a model comparison plot
    for turn_idx in range(1, max_turns + 1):
        for category in all_categories:
            # Get models that have data for this category and turn
            valid_models = []
            
            for model_name, model_data in aggregated_data.items():
                if turn_idx in model_data and category in model_data[turn_idx]:
                    category_data = model_data[turn_idx][category]
                    # Only include if we have enough runs
                    if len(category_data['with_attempt']) >= min_runs:
                        valid_models.append(model_name)
            
            if not valid_models:  # Skip if no models have data for this category and turn
                continue
            
            # Create figure
            fig, ax = plt.subplots(figsize=(max(12, len(valid_models) * 1.2), 10))
            
            # Set positions for bars
            x = np.arange(len(valid_models))
            width = 0.8
            
            # Collect data for each model
            with_attempt_means = []
            no_attempt_means = []
            refusal_means = []
            with_attempt_stds = []
            no_attempt_stds = []
            refusal_stds = []
            
            for model_name in valid_models:
                model_data = aggregated_data[model_name]
                category_data = model_data[turn_idx][category]
                
                # Store means and standard deviations of raw counts
                with_attempt_means.append(np.mean(category_data['with_attempt']))
                with_attempt_stds.append(np.std(category_data['with_attempt']))
                
                no_attempt_means.append(np.mean(category_data['no_attempt']))
                no_attempt_stds.append(np.std(category_data['no_attempt']))
                
                refusal_means.append(np.mean(category_data['refusal']))
                refusal_stds.append(np.std(category_data['refusal']))
            
            # Create stacked bars
            bars1 = ax.bar(x, with_attempt_means, width, label='With Persuasion Attempt', color='skyblue')
            bars2 = ax.bar(x, no_attempt_means, width, bottom=with_attempt_means, label='No Persuasion Attempt', color='lightgreen')
            
            # Calculate the bottom position for the third bar
            bottom_vals = np.array(with_attempt_means) + np.array(no_attempt_means)
            bars3 = ax.bar(x, refusal_means, width, bottom=bottom_vals, label='Refusal', color='salmon')
            
            # Add error bars for each segment
            # For the first segment (with_attempt), error bars are at the top of the bar
            for i, (mean, std) in enumerate(zip(with_attempt_means, with_attempt_stds)):
                ax.errorbar(x[i] - 0.1, mean, yerr=std, fmt='o', color='black', capsize=5, 
                           elinewidth=1.5, capthick=1.5, markersize=4)
            
            # For the second segment (no_attempt), error bars are at the top of this segment
            for i, (mean, std, bottom) in enumerate(zip(no_attempt_means, no_attempt_stds, with_attempt_means)):
                ax.errorbar(x[i], bottom + mean, yerr=std, fmt='o', color='black', capsize=5,
                           elinewidth=1.5, capthick=1.5, markersize=4)
            
            # For the third segment (refusal), error bars are at the top of this segment
            for i, (mean, std, bottom) in enumerate(zip(refusal_means, refusal_stds, bottom_vals)):
                ax.errorbar(x[i] + 0.1, bottom + mean, yerr=std, fmt='o', color='black', capsize=5,
                           elinewidth=1.5, capthick=1.5, markersize=4)
            
            # Add labels and title
            ax.set_xlabel("Models")
            ax.set_ylabel("Number of Conversations")
            ax.set_title(f"Turn {turn_idx}: {category} - Model Comparison (Raw Counts)")
            ax.set_xticks(x)
            ax.set_xticklabels(valid_models)
            plt.xticks(rotation=45, ha="right")
            
            # Add legend
            ax.legend(loc='upper right')
            
            # Add a grid for better readability
            ax.grid(True, axis="y", alpha=0.3)
            
            # Determine appropriate y-axis limit
            max_values = []
            for i in range(len(with_attempt_means)):
                segment_sum = with_attempt_means[i] + with_attempt_stds[i] + \
                             no_attempt_means[i] + no_attempt_stds[i] + \
                             refusal_means[i] + refusal_stds[i]
                max_values.append(segment_sum)
            
            max_value = max(max_values) if max_values else 10
            ax.set_ylim(0, max_value * 1.2)  # Add 20% padding
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, f"turn_{turn_idx}_{category}_counts_comparison_stacked.png"))
            plt.close()
    
    print(f"Created model comparison by category counts plots in: {comparison_dir}")

def create_nh_subject_counts_model_comparison_plots(aggregated_data: Dict, output_dir: str, min_runs: int, max_turns: int = 3):
    """
    Create comparison plots showing all models side by side for each NH subject, with stacked count bars.
    
    Args:
        aggregated_data: Dictionary with aggregated metrics by model
        output_dir: Directory to save plots
        min_runs: Minimum number of runs required to include a model
        max_turns: Maximum number of turns to process
    """
    comparison_dir = os.path.join(output_dir, "model_counts_comparisons_by_nh_subject")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Get all unique subjects across all models
    all_subjects = set()
    for model_data in aggregated_data.values():
        for turn_data in model_data.values():
            all_subjects.update(turn_data.keys())
    
    all_subjects = sorted(all_subjects)
    
    # For each turn and subject, create a model comparison plot
    for turn_idx in range(1, max_turns + 1):
        for subject in all_subjects:
            # Get models that have data for this subject and turn
            valid_models = []
            
            for model_name, model_data in aggregated_data.items():
                if turn_idx in model_data and subject in model_data[turn_idx]:
                    subject_data = model_data[turn_idx][subject]
                    # Only include if we have enough runs
                    if len(subject_data['with_attempt']) >= min_runs:
                        valid_models.append(model_name)
            
            if not valid_models:  # Skip if no models have data for this subject and turn
                continue
            
            # Create figure
            fig, ax = plt.subplots(figsize=(max(12, len(valid_models) * 1.2), 10))
            
            # Set positions for bars
            x = np.arange(len(valid_models))
            width = 0.8
            
            # Collect data for each model
            with_attempt_means = []
            no_attempt_means = []
            refusal_means = []
            with_attempt_stds = []
            no_attempt_stds = []
            refusal_stds = []
            
            for model_name in valid_models:
                model_data = aggregated_data[model_name]
                subject_data = model_data[turn_idx][subject]
                
                # Store means and standard deviations of raw counts
                with_attempt_means.append(np.mean(subject_data['with_attempt']))
                with_attempt_stds.append(np.std(subject_data['with_attempt']))
                
                no_attempt_means.append(np.mean(subject_data['no_attempt']))
                no_attempt_stds.append(np.std(subject_data['no_attempt']))
                
                refusal_means.append(np.mean(subject_data['refusal']))
                refusal_stds.append(np.std(subject_data['refusal']))
            
            # Create stacked bars
            bars1 = ax.bar(x, with_attempt_means, width, label='With Persuasion Attempt', color='skyblue')
            bars2 = ax.bar(x, no_attempt_means, width, bottom=with_attempt_means, label='No Persuasion Attempt', color='lightgreen')
            
            # Calculate the bottom position for the third bar
            bottom_vals = np.array(with_attempt_means) + np.array(no_attempt_means)
            bars3 = ax.bar(x, refusal_means, width, bottom=bottom_vals, label='Refusal', color='salmon')
            
            # Add error bars for each segment
            # For the first segment (with_attempt), error bars are at the top of the bar
            for i, (mean, std) in enumerate(zip(with_attempt_means, with_attempt_stds)):
                ax.errorbar(x[i] - 0.1, mean, yerr=std, fmt='o', color='black', capsize=5, 
                           elinewidth=1.5, capthick=1.5, markersize=4)
            
            # For the second segment (no_attempt), error bars are at the top of this segment
            for i, (mean, std, bottom) in enumerate(zip(no_attempt_means, no_attempt_stds, with_attempt_means)):
                ax.errorbar(x[i], bottom + mean, yerr=std, fmt='o', color='black', capsize=5,
                           elinewidth=1.5, capthick=1.5, markersize=4)
            
            # For the third segment (refusal), error bars are at the top of this segment
            for i, (mean, std, bottom) in enumerate(zip(refusal_means, refusal_stds, bottom_vals)):
                ax.errorbar(x[i] + 0.1, bottom + mean, yerr=std, fmt='o', color='black', capsize=5,
                           elinewidth=1.5, capthick=1.5, markersize=4)
            
            # Add labels and title
            ax.set_xlabel("Models")
            ax.set_ylabel("Number of Conversations")
            ax.set_title(f"Turn {turn_idx}: {subject} - Model Comparison (Raw Counts)")
            ax.set_xticks(x)
            ax.set_xticklabels(valid_models)
            plt.xticks(rotation=45, ha="right")
            
            # Add legend
            ax.legend(loc='upper right')
            
            # Add a grid for better readability
            ax.grid(True, axis="y", alpha=0.3)
            
            # Determine appropriate y-axis limit
            max_values = []
            for i in range(len(with_attempt_means)):
                segment_sum = with_attempt_means[i] + with_attempt_stds[i] + \
                             no_attempt_means[i] + no_attempt_stds[i] + \
                             refusal_means[i] + refusal_stds[i]
                max_values.append(segment_sum)
            
            max_value = max(max_values) if max_values else 10
            ax.set_ylim(0, max_value * 1.2)  # Add 20% padding
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, f"turn_{turn_idx}_{subject}_counts_comparison_stacked.png"))
            plt.close()
    
    print(f"Created model comparison by NH subject counts plots in: {comparison_dir}")

def create_all_in_one_comparison_plot(aggregated_data: Dict, output_dir: str, min_runs: int, max_turns: int = 3):
    """
    Create a single plot showing all models and all categories, with colors for models and shades for response types.
    
    Args:
        aggregated_data: Dictionary with aggregated metrics by model
        output_dir: Directory to save plots
        min_runs: Minimum number of runs required to include a model
        max_turns: Maximum number of turns to process
    """
    # Ensure serif font for this figure
    plt.rcParams['font.family'] = ['serif']
    
    comparison_dir = os.path.join(output_dir, "all_in_one_comparisons")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Get all unique categories across all models
    all_categories = set()
    for model_data in aggregated_data.values():
        for turn_data in model_data.values():
            all_categories.update(turn_data.keys())
    
    all_categories = sorted(all_categories)
    
    # Define a pure gray color for response type legend
    gray_color = (0.5, 0.5, 0.5)  # RGB gray color
    
    # For each turn, create a combined comparison plot
    for turn_idx in range(1, max_turns + 1):
        # Get valid models and categories for this turn
        valid_models_unordered = set()
        valid_categories = []
        
        # First, find valid categories with at least one model having data
        for category in all_categories:
            has_valid_data = False
            for model_name, model_data in aggregated_data.items():
                if turn_idx in model_data and category in model_data[turn_idx]:
                    category_data = model_data[turn_idx][category]
                    if len(category_data['with_attempt']) >= min_runs:
                        has_valid_data = True
                        valid_models_unordered.add(model_name)
            
            if has_valid_data:
                valid_categories.append(category)
        
        # Sort models according to predefined order
        valid_models = [model for model in model_order if model in valid_models_unordered]
        # Include any models that weren't in our predefined order at the end
        for model in valid_models_unordered:
            if model not in valid_models:
                valid_models.append(model)
        
        if not valid_models or not valid_categories:
            continue
        
        # Create figure with larger size to accommodate all data and legends
        fig_width = max(18, len(valid_categories) * len(valid_models) * 0.5)
        fig_height = 8
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
        
        # Set positions for bars - tighter grouping without spaces between models
        bar_width = 0.8  # Width of a single model's bar
        group_width = bar_width * len(valid_models)  # Width of entire category group
        
        # Calculate x positions for each category group
        x_base = np.arange(len(valid_categories)) * (group_width + 1)  # +1 for padding between categories
        
        # Define colors for models (using a qualitative colormap)
        cmap = plt.colormaps['Dark2']
        # Assign colors in order of valid_models to ensure consistent coloring
        model_colors = {}
        for i, model in enumerate(valid_models):
            model_colors[model] = cmap(i % 8)
        
        # Store handles for legend
        # model_handles = [] # Not explicitly used later, removing
        
        # Plot each model's data for each category
        for model_idx, model_name in enumerate(valid_models):
            model_data = aggregated_data[model_name]
            model_color = model_colors[model_name]
            
            # Define colors based on the new scheme
            # Attempt: Solid base color (previously used for Refusal)
            attempt_color = model_color
            attempt_edge_color = model_color
            
            # No Attempt: Lighter shade of base color
            no_attempt_color = (min(1.0, model_color[0] + (1-model_color[0])*0.6), min(1.0, model_color[1] + (1-model_color[1])*0.6), min(1.0, model_color[2] + (1-model_color[2])*0.6), model_color[3]) # Blend with white
            no_attempt_edge_color = model_color
            
            # Refusal: Very light fill, solid edge (previously used for Attempt)
            refusal_fill_color = (min(1.0, model_color[0] + (1-model_color[0])*0.8), min(1.0, model_color[1] + (1-model_color[1])*0.8), min(1.0, model_color[2] + (1-model_color[2])*0.8), 0.8) # Lighter shade with some alpha
            refusal_edge_color = model_color
            
            # Store data for plotting
            model_positions = []
            with_attempt_values = []
            no_attempt_values = []
            refusal_values = []
            with_attempt_errors = []
            no_attempt_errors = []
            refusal_errors = []
            
            # Collect data for this model across all categories
            for cat_idx, category in enumerate(valid_categories):
                # Position each model's bar right next to each other
                bar_pos = x_base[cat_idx] + model_idx * bar_width
                
                if turn_idx in model_data and category in model_data[turn_idx]:
                    category_data = model_data[turn_idx][category]
                    
                    if len(category_data['with_attempt']) >= min_runs:
                        # Calculate totals for each run
                        totals = np.array(category_data['with_attempt']) + np.array(category_data['no_attempt']) + np.array(category_data['refusal'])
                        
                        # Calculate percentages for each response type
                        with_attempt_percentages = np.array(category_data['with_attempt']) / totals * 100
                        no_attempt_percentages = np.array(category_data['no_attempt']) / totals * 100
                        refusal_percentages = np.array(category_data['refusal']) / totals * 100
                        
                        # Store means and standard deviations
                        with_attempt_values.append(np.mean(with_attempt_percentages))
                        with_attempt_errors.append(np.std(with_attempt_percentages))
                        
                        no_attempt_values.append(np.mean(no_attempt_percentages))
                        no_attempt_errors.append(np.std(no_attempt_percentages))
                        
                        refusal_values.append(np.mean(refusal_percentages))
                        refusal_errors.append(np.std(refusal_percentages))
                        
                        model_positions.append(bar_pos)
                    else:
                        # No data for this model/category combination
                        with_attempt_values.append(0)
                        with_attempt_errors.append(0)
                        no_attempt_values.append(0)
                        no_attempt_errors.append(0)
                        refusal_values.append(0)
                        refusal_errors.append(0)
                        model_positions.append(bar_pos)
                else:
                    # No data for this model/category combination
                    with_attempt_values.append(0)
                    with_attempt_errors.append(0)
                    no_attempt_values.append(0)
                    no_attempt_errors.append(0)
                    refusal_values.append(0)
                    refusal_errors.append(0)
                    model_positions.append(bar_pos)
            
            # Plot bars for this model - using the new color scheme with switched positions
            # 1. Attempt bars (bottom segment) - solid base color
            bars1 = ax.bar(model_positions, with_attempt_values, bar_width, 
                         color=attempt_color, alpha=1.0, 
                         edgecolor=attempt_edge_color, linewidth=1.0)
            
            # 2. No attempt bars (middle segment) - lighter shade
            bottom_vals1 = np.array(with_attempt_values)
            bars2 = ax.bar(model_positions, no_attempt_values, bar_width, 
                         bottom=bottom_vals1, color=no_attempt_color, alpha=1.0, 
                         edgecolor=no_attempt_edge_color, linewidth=1.0)
            
            # 3. Refusal bars (top segment) - light fill, solid edge
            bottom_vals2 = bottom_vals1 + np.array(no_attempt_values)
            bars3 = ax.bar(model_positions, refusal_values, bar_width, 
                         bottom=bottom_vals2, color=refusal_fill_color, alpha=1.0, 
                         edgecolor=refusal_edge_color, linewidth=1.0)
            
            # Add error bars for each segment
            for i, (pos, val, err) in enumerate(zip(model_positions, with_attempt_values, with_attempt_errors)):
                if val > 0:
                    # Error bar for the bottom segment (Attempt) - position at the top of the segment
                    ax.errorbar(pos - 0.2, val, yerr=err, fmt='o', ecolor='black', capsize=3, elinewidth=1.5, 
                              markerfacecolor='none', markersize=5, markeredgecolor='black')
            
            for i, (pos, val, err, bottom) in enumerate(zip(model_positions, no_attempt_values, no_attempt_errors, bottom_vals1)):
                if val > 0:
                    # Error bar for the middle segment (No Attempt) - position at the top of the segment
                    ax.errorbar(pos, bottom + val, yerr=err, fmt='o', ecolor='black', capsize=3, elinewidth=1.5,
                              markerfacecolor='none', markersize=5, markeredgecolor='black')
            
            for i, (pos, val, err, bottom) in enumerate(zip(model_positions, refusal_values, refusal_errors, bottom_vals2)):
                if val > 0:
                    # Error bar for the top segment (Refusal) - position at the top of the segment
                    ax.errorbar(pos + 0.2, bottom + val, yerr=err, fmt='o', ecolor='black', capsize=3, elinewidth=1.5,
                              markerfacecolor='none', markersize=5, markeredgecolor='black')
        
        # Create response type legend using representative colors/styles
        # Use gray as a neutral representation for the legend patches
        solid_gray = (0.3, 0.3, 0.3, 1.0)
        light_gray = (0.7, 0.7, 0.7, 1.0)
        outline_gray_fill = (0.9, 0.9, 0.9, 0.8)
        outline_gray_edge = (0.3, 0.3, 0.3, 1.0)
        
        response_handles = [
            plt.Rectangle((0, 0), 1, 1, fill=True, facecolor=outline_gray_fill, edgecolor=outline_gray_edge, linewidth=1.5, label='Refusal (Outline)'),
            plt.Rectangle((0, 0), 1, 1, fill=True, color=light_gray, label='No Persuasion Attempt (Light)'),
            plt.Rectangle((0, 0), 1, 1, fill=True, color=solid_gray, label='With Persuasion Attempt (Solid)')
        ]
        
        # Set x-axis ticks at the center of each category group
        ax.set_xticks(x_base + group_width/2 - bar_width/2)
        ax.set_xticklabels([cat.capitalize() for cat in valid_categories], fontsize=20)
        plt.xticks(rotation=10, ha='right')  # Slight rotation with right alignment
        
        # Increase font size for y-axis tick labels
        ax.tick_params(axis='y', labelsize=18)
        
        # Add labels and title with increased font size
        ax.set_xlabel('Categories', fontsize=22)
        ax.set_ylabel('Percentage of Conversations (%)', fontsize=22)
        ax.set_title(f'Turn {turn_idx} - All Categories', fontsize=24)
        
        # Add grid for readability
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        # Set y-axis to percentage scale
        ax.set_ylim(0, 140)
        
        # Create model and response type legends
        model_patches = [plt.Rectangle((0, 0), 1, 1, color=model_colors[name]) for name in valid_models]
        
        # Place both legends in a single row near the top
        # First, create a dedicated subplot area for the legends
        legend_ax = fig.add_axes([0.1, 0.8, 0.8, 0.1]) 
        legend_ax.axis('off')  # Turn off axes for the legend area
        
        # Create model legend on the left side
        model_legend = legend_ax.legend(
            model_patches, 
            [model_names.get(name, name) for name in valid_models], 
            title="Models", 
            loc='center left',
            fontsize=18,
            title_fontsize=20,
            ncol=min(2, len(valid_models))  # Arrange in multiple columns if many models
        )
        
        # Create response type legend on the right side
        # Calculate the x position based on the number of models
        legend_ax.legend(
            handles=response_handles, # Use handles directly
            # labels=[h.get_label() for h in response_handles], # Labels are set in handles
            title="Response Types",
            loc='center right',
            fontsize=18,
            title_fontsize=20
        )
        
        # Add both legends to the axes
        legend_ax.add_artist(model_legend)
        
        # Save the figure
        # plt.tight_layout(rect=[0, 0.15, 1, 1])  # Removed as constrained_layout handles this
        plt.savefig(os.path.join(comparison_dir, f"turn_{turn_idx}_all_models_categories.png"))
        plt.close()
    
    print(f"Created all-in-one comparison plots in: {comparison_dir}")

def create_all_in_one_nh_subject_comparison_plot(aggregated_data: Dict, output_dir: str, min_runs: int, max_turns: int = 3):
    """
    Create a single plot showing all models and all NH subjects, with colors for models and shades for response types.
    
    Args:
        aggregated_data: Dictionary with aggregated metrics by model
        output_dir: Directory to save plots
        min_runs: Minimum number of runs required to include a model
        max_turns: Maximum number of turns to process
    """
    # Ensure serif font for this figure
    plt.rcParams['font.family'] = ['serif']
    
    comparison_dir = os.path.join(output_dir, "all_in_one_nh_comparisons")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Get all unique subjects across all models
    all_subjects = set()
    for model_data in aggregated_data.values():
        for turn_data in model_data.values():
            all_subjects.update(turn_data.keys())
    
    all_subjects = sorted(all_subjects)
    
    # Define a pure gray color for response type legend
    gray_color = (0.5, 0.5, 0.5)  # RGB gray color
    
    # For each turn, create a combined comparison plot
    for turn_idx in range(1, max_turns + 1):
        # Get valid models and subjects for this turn
        valid_models_unordered = set()
        valid_subjects = []
        
        # First, find valid subjects with at least one model having data
        for subject in all_subjects:
            has_valid_data = False
            for model_name, model_data in aggregated_data.items():
                if turn_idx in model_data and subject in model_data[turn_idx]:
                    subject_data = model_data[turn_idx][subject]
                    if len(subject_data['with_attempt']) >= min_runs:
                        has_valid_data = True
                        valid_models_unordered.add(model_name)
            
            if has_valid_data:
                valid_subjects.append(subject)
        
        # Sort models according to predefined order
        valid_models = [model for model in model_order if model in valid_models_unordered]
        # Include any models that weren't in our predefined order at the end
        for model in valid_models_unordered:
            if model not in valid_models:
                valid_models.append(model)
        
        if not valid_models or not valid_subjects:
            continue
        
        # Create figure with larger size to accommodate all data and legends
        fig_width = max(18, len(valid_subjects) * len(valid_models) * 0.5)
        fig_height = 8
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
        
        # Set positions for bars - tighter grouping without spaces between models
        bar_width = 0.8  # Width of a single model's bar
        group_width = bar_width * len(valid_models)  # Width of entire subject group
        
        # Calculate x positions for each subject group
        x_base = np.arange(len(valid_subjects)) * (group_width + 1)  # +1 for padding between subjects
        
        # Define colors for models (using a qualitative colormap)
        cmap = plt.colormaps['Dark2']
        # Assign colors in order of valid_models to ensure consistent coloring
        model_colors = {}
        for i, model in enumerate(valid_models):
            model_colors[model] = cmap(i % 8)
        
        # Store handles for legend
        # model_handles = [] # Not explicitly used later, removing
        
        # Plot each model's data for each subject
        for model_idx, model_name in enumerate(valid_models):
            model_data = aggregated_data[model_name]
            model_color = model_colors[model_name]
            
            # Define colors based on the new scheme
            # Attempt: Solid base color (previously used for Refusal)
            attempt_color = model_color
            attempt_edge_color = model_color
            
            # No Attempt: Lighter shade of base color
            no_attempt_color = (min(1.0, model_color[0] + (1-model_color[0])*0.6), min(1.0, model_color[1] + (1-model_color[1])*0.6), min(1.0, model_color[2] + (1-model_color[2])*0.6), model_color[3]) # Blend with white
            no_attempt_edge_color = model_color
            
            # Refusal: Very light fill, solid edge (previously used for Attempt)
            refusal_fill_color = (min(1.0, model_color[0] + (1-model_color[0])*0.8), min(1.0, model_color[1] + (1-model_color[1])*0.8), min(1.0, model_color[2] + (1-model_color[2])*0.8), 0.8) # Lighter shade with some alpha
            refusal_edge_color = model_color
            
            # Store data for plotting
            model_positions = []
            with_attempt_values = []
            no_attempt_values = []
            refusal_values = []
            with_attempt_errors = []
            no_attempt_errors = []
            refusal_errors = []
            
            # Collect data for this model across all subjects
            for subj_idx, subject in enumerate(valid_subjects):
                # Position each model's bar right next to each other
                bar_pos = x_base[subj_idx] + model_idx * bar_width
                
                if turn_idx in model_data and subject in model_data[turn_idx]:
                    subject_data = model_data[turn_idx][subject]
                    
                    if len(subject_data['with_attempt']) >= min_runs:
                        # Calculate totals for each run
                        totals = np.array(subject_data['with_attempt']) + np.array(subject_data['no_attempt']) + np.array(subject_data['refusal'])
                        
                        # Calculate percentages for each response type
                        with_attempt_percentages = np.array(subject_data['with_attempt']) / totals * 100
                        no_attempt_percentages = np.array(subject_data['no_attempt']) / totals * 100
                        refusal_percentages = np.array(subject_data['refusal']) / totals * 100
                        
                        # Store means and standard deviations
                        with_attempt_values.append(np.mean(with_attempt_percentages))
                        with_attempt_errors.append(np.std(with_attempt_percentages))
                        
                        no_attempt_values.append(np.mean(no_attempt_percentages))
                        no_attempt_errors.append(np.std(no_attempt_percentages))
                        
                        refusal_values.append(np.mean(refusal_percentages))
                        refusal_errors.append(np.std(refusal_percentages))
                        
                        model_positions.append(bar_pos)
                    else:
                        # No data for this model/subject combination
                        with_attempt_values.append(0)
                        with_attempt_errors.append(0)
                        no_attempt_values.append(0)
                        no_attempt_errors.append(0)
                        refusal_values.append(0)
                        refusal_errors.append(0)
                        model_positions.append(bar_pos)
                else:
                    # No data for this model/subject combination
                    with_attempt_values.append(0)
                    with_attempt_errors.append(0)
                    no_attempt_values.append(0)
                    no_attempt_errors.append(0)
                    refusal_values.append(0)
                    refusal_errors.append(0)
                    model_positions.append(bar_pos)
            
            # Plot bars for this model - using the new color scheme with switched positions
            # 1. Attempt bars (bottom segment) - solid base color
            bars1 = ax.bar(model_positions, with_attempt_values, bar_width, 
                         color=attempt_color, alpha=1.0, 
                         edgecolor=attempt_edge_color, linewidth=1.0)
            
            # 2. No attempt bars (middle segment) - lighter shade
            bottom_vals1 = np.array(with_attempt_values)
            bars2 = ax.bar(model_positions, no_attempt_values, bar_width, 
                         bottom=bottom_vals1, color=no_attempt_color, alpha=1.0, 
                         edgecolor=no_attempt_edge_color, linewidth=1.0)
            
            # 3. Refusal bars (top segment) - light fill, solid edge
            bottom_vals2 = bottom_vals1 + np.array(no_attempt_values)
            bars3 = ax.bar(model_positions, refusal_values, bar_width, 
                         bottom=bottom_vals2, color=refusal_fill_color, alpha=1.0, 
                         edgecolor=refusal_edge_color, linewidth=1.0)
            
            # Add error bars for each segment
            for i, (pos, val, err) in enumerate(zip(model_positions, with_attempt_values, with_attempt_errors)):
                if val > 0:
                    # Error bar for the bottom segment (Attempt) - position at the top of the segment
                    ax.errorbar(pos - 0.2, val, yerr=err, fmt='o', ecolor='black', capsize=3, elinewidth=1.5, 
                              markerfacecolor='none', markersize=5, markeredgecolor='black')
            
            for i, (pos, val, err, bottom) in enumerate(zip(model_positions, no_attempt_values, no_attempt_errors, bottom_vals1)):
                if val > 0:
                    # Error bar for the middle segment (No Attempt) - position at the top of the segment
                    ax.errorbar(pos, bottom + val, yerr=err, fmt='o', ecolor='black', capsize=3, elinewidth=1.5,
                              markerfacecolor='none', markersize=5, markeredgecolor='black')
            
            for i, (pos, val, err, bottom) in enumerate(zip(model_positions, refusal_values, refusal_errors, bottom_vals2)):
                if val > 0:
                    # Error bar for the top segment (Refusal) - position at the top of the segment
                    ax.errorbar(pos + 0.2, bottom + val, yerr=err, fmt='o', ecolor='black', capsize=3, elinewidth=1.5,
                              markerfacecolor='none', markersize=5, markeredgecolor='black')
        
        # Create response type legend using representative colors/styles
        # Use gray as a neutral representation for the legend patches
        solid_gray = (0.3, 0.3, 0.3, 1.0)
        light_gray = (0.7, 0.7, 0.7, 1.0)
        outline_gray_fill = (0.9, 0.9, 0.9, 0.8)
        outline_gray_edge = (0.3, 0.3, 0.3, 1.0)
        
        response_handles = [
            plt.Rectangle((0, 0), 1, 1, fill=True, facecolor=outline_gray_fill, edgecolor=outline_gray_edge, linewidth=1.5, label='Refusal (Outline)'),
            plt.Rectangle((0, 0), 1, 1, fill=True, color=light_gray, label='No Persuasion Attempt (Light)'),
            plt.Rectangle((0, 0), 1, 1, fill=True, color=solid_gray, label='With Persuasion Attempt (Solid)')
        ]
        
        # Set x-axis ticks at the center of each subject group
        ax.set_xticks(x_base + group_width/2 - bar_width/2)
        ax.set_xticklabels([sub.capitalize() for sub in valid_subjects], fontsize=20)
        plt.xticks(rotation=10, ha='right')  # Slight rotation with right alignment
        
        # Increase font size for y-axis tick labels
        ax.tick_params(axis='y', labelsize=18)
        
        # Add labels and title with increased font size
        ax.set_xlabel('Noncontroversially Harmful Sub-Categories', fontsize=22)
        ax.set_ylabel('Percentage of Conversations (%)', fontsize=22)
        ax.set_title(f'Turn {turn_idx} - Noncontroversially Harmful Sub-Categories', fontsize=24)
        
        # Add grid for readability
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        # Set y-axis to percentage scale
        ax.set_ylim(0, 140)
        
        # Create model and response type legends
        model_patches = [plt.Rectangle((0, 0), 1, 1, color=model_colors[name]) for name in valid_models]
        
        # Place both legends in a single row near the top
        # First, create a dedicated subplot area for the legends
        legend_ax = fig.add_axes([0.1, 0.8, 0.8, 0.1])
        legend_ax.axis('off')  # Turn off axes for the legend area
        
        # Create model legend on the left side
        model_legend = legend_ax.legend(
            model_patches, 
            [model_names.get(name, name) for name in valid_models], 
            title="Models", 
            loc='center left',
            fontsize=18,
            title_fontsize=20,
            ncol=min(2, len(valid_models))  # Arrange in multiple columns if many models
        )
        
        # Create response type legend on the right side
        legend_ax.legend(
            handles=response_handles, # Use handles directly
            # labels=[h.get_label() for h in response_handles], # Labels are set in handles
            title="Response Types",
            loc='center right',
            fontsize=18,
            title_fontsize=20
        )
        
        # Add both legends to the axes
        legend_ax.add_artist(model_legend)
        
        # Save the figure
        # plt.tight_layout(rect=[0, 0.15, 1, 1])  # Removed as constrained_layout handles this
        plt.savefig(os.path.join(comparison_dir, f"turn_{turn_idx}_all_models_nh_subjects.png"))
        plt.close()
    
    print(f"Created all-in-one NH subject comparison plots in: {comparison_dir}")

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all model runs
    print(f"Searching for model runs in: {args.results_dir}")
    model_runs = find_model_runs(args.results_dir, args.models)
    
    if not model_runs:
        print("No model runs found!")
        return
    
    print(f"Found {len(model_runs)} models with data")
    for model, runs in model_runs.items():
        print(f"  - {model}: {len(runs)} runs")
    
    # Aggregate metrics by category
    category_data = aggregate_metrics_by_category(model_runs)
    
    # Aggregate metrics by NH subjects
    nh_subjects_data = aggregate_metrics_by_nh_subjects(model_runs)
    
    # # Create category counts plots with error bars
    # create_category_counts_plots(category_data, args.output_dir, args.min_runs)
    
    # # Create category percentage plots with error bars
    # create_percentage_plots(category_data, args.output_dir, args.min_runs, 'category')
    
    # # Create NH subjects counts plots with error bars
    # if nh_subjects_data:
    #     create_nh_subjects_counts_plots(nh_subjects_data, args.output_dir, args.min_runs)
        
    #     # Create NH subjects percentage plots with error bars
    #     create_percentage_plots(nh_subjects_data, args.output_dir, args.min_runs, 'nh_subjects')
    
    # # Create multi-model comparison plots
    # create_multi_model_comparison(category_data, args.output_dir, args.min_runs, args.turns)
    
    # # Create new model comparison plots
    # # For category data
    # create_category_model_comparison_plots(category_data, args.output_dir, args.min_runs, args.turns)
    # create_category_counts_model_comparison_plots(category_data, args.output_dir, args.min_runs, args.turns)
    
    # # For NH subjects data
    # if nh_subjects_data:
    #     create_nh_subject_model_comparison_plots(nh_subjects_data, args.output_dir, args.min_runs, args.turns)
    #     create_nh_subject_counts_model_comparison_plots(nh_subjects_data, args.output_dir, args.min_runs, args.turns)
    
    # Create all-in-one comparison plots with all models and categories/subjects
    create_all_in_one_comparison_plot(category_data, args.output_dir, args.min_runs, args.turns)
    if nh_subjects_data:
        create_all_in_one_nh_subject_comparison_plot(nh_subjects_data, args.output_dir, args.min_runs, args.turns)
    
    print(f"All plots have been generated in: {args.output_dir}")

if __name__ == "__main__":
    main() 