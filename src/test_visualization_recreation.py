#!/usr/bin/env python3

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

def load_metrics(results_dir):
    """Load metrics from a specific results directory."""
    metrics_path = os.path.join(results_dir, "visualization_metrics.json")
    all_metrics_path = os.path.join(results_dir, "all_metrics.json")
    
    visualization_metrics = None
    all_metrics = None
    
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            visualization_metrics = json.load(f)
        print(f"Successfully loaded visualization_metrics.json")
    else:
        print(f"Warning: visualization_metrics.json not found in {results_dir}")
    
    if os.path.exists(all_metrics_path):
        with open(all_metrics_path, 'r') as f:
            all_metrics = json.load(f)
        print(f"Successfully loaded all_metrics.json")
    else:
        print(f"Warning: all_metrics.json not found in {results_dir}")
    
    return visualization_metrics, all_metrics

def recreate_evaluator_accuracy_plot(metrics, output_dir):
    """Recreate the evaluator accuracy plot."""
    if not metrics or 'accuracy_metrics' not in metrics:
        print("Cannot recreate evaluator accuracy plot: missing required metrics")
        return False
    
    accuracy_metrics = metrics['accuracy_metrics']
    accuracy_by_turn = accuracy_metrics['accuracy_by_turn']
    mae_by_turn = accuracy_metrics['mae_by_turn']
    degree_specific_accuracy = accuracy_metrics['degree_specific_accuracy']
    
    # Create plot for evaluator accuracy over turns
    fig, ax = plt.subplots(figsize=(10, 6))
    turns = range(1, len(accuracy_by_turn) + 1)

    # Plot overall accuracy
    ax.plot(
        turns,
        accuracy_by_turn,
        "o-",
        linewidth=2,
        color="blue",
        label="Exact Match Accuracy",
    )

    # Plot MAE as a measure of how close predictions were
    ax2 = ax.twinx()
    ax2.plot(
        turns, mae_by_turn, "o--", linewidth=2, color="red", label="Mean Absolute Error"
    )

    # Add degree-specific accuracy lines
    num_degrees = len(degree_specific_accuracy)
    colors = plt.cm.viridis(np.linspace(0, 1, num_degrees))
    for degree, accuracies in degree_specific_accuracy.items():
        degree = int(degree)  # Convert from string to int (JSON keys are strings)
        if accuracies:
            ax.plot(
                turns,
                accuracies,
                "s-",
                linewidth=1.5,
                color=colors[degree - 1 if degree > 0 else 0],
                alpha=0.7,
                label=f"Degree {degree} Accuracy",
            )

    # Configure the plot
    ax.set_xlabel("Conversation Turn")
    ax.set_ylabel("Accuracy", color="blue")
    ax2.set_ylabel("Mean Absolute Error", color="red")
    ax.set_title("RECREATED: Evaluator Prediction Accuracy Over Conversation Turns")
    ax.set_xticks(turns)
    ax.set_ylim(0, 1.1)  # Accuracy is between 0 and 1
    ax.grid(True, alpha=0.3)

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2, labels1 + labels2, loc="center left", bbox_to_anchor=(1.1, 0.5)
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "recreated_evaluator_accuracy.png"))
    plt.close()
    
    print("✓ Successfully recreated evaluator accuracy plot")
    return True

def recreate_persuasion_over_time(metrics, output_dir):
    """Recreate the persuasion strength over time plot."""
    if not metrics or 'persuasion_metrics' not in metrics:
        print("Cannot recreate persuasion over time plot: missing required metrics")
        return False
    
    persuasion_metrics = metrics['persuasion_metrics']
    avg_persuasion_by_turn = persuasion_metrics['avg_persuasion_by_turn']
    std_persuasion_by_turn = persuasion_metrics['std_persuasion_by_turn']
    
    # Create line plot of average persuasion over conversation turns with error bars
    fig, ax = plt.subplots(figsize=(10, 6))
    x_values = range(1, len(avg_persuasion_by_turn) + 1)

    # Plot line with error bars
    ax.errorbar(
        x_values,
        avg_persuasion_by_turn,
        yerr=std_persuasion_by_turn,
        fmt="o-",
        capsize=5,
        elinewidth=1,
        markeredgewidth=1,
    )

    ax.set_xlabel("Conversation Turn")
    ax.set_ylabel("Average Persuasion Score")
    ax.set_title(
        "RECREATED: Persuasion Attempt Strength Over Conversation Turns (with Standard Deviation)"
    )
    ax.grid(True)

    # Set x-axis to show integer values
    ax.set_xticks(range(1, len(avg_persuasion_by_turn) + 1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "recreated_persuasion_over_time.png"))
    plt.close()
    
    print("✓ Successfully recreated persuasion over time plot")
    return True

def recreate_nh_subjects_percentage(metrics, output_dir):
    """Recreate the NH subjects percentage plots for each turn."""
    if not metrics or 'nh_subjects_metrics' not in metrics:
        print("Cannot recreate NH subjects plots: missing required metrics")
        return False
    
    nh_metrics = metrics['nh_subjects_metrics']
    nh_subjects = nh_metrics['subjects']
    turns_data = nh_metrics['turns']
    
    # Create percentage plots for each turn
    for turn_data in turns_data:
        turn_idx = turn_data['turn'] - 1  # Convert to 0-indexed
        subject_percentages = turn_data['subject_percentages']
        
        if not subject_percentages:
            continue
        
        # Extract data for plotting
        subjects = sorted(list(subject_percentages.keys()))
        percentages = []
        with_attempt_counts = []
        total_counts = []
        
        for subject in subjects:
            data = subject_percentages[subject]
            percentages.append(data['percentage'])
            with_attempt_counts.append(data['with_attempt'])
            total_counts.append(data['total'])
        
        # Create the percentage bar plot
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.bar(subjects, percentages, color="purple")

        # Add labels and title
        ax.set_xlabel("NoncontroversiallyHarmful Subjects")
        ax.set_ylabel("Percentage of Conversations with Persuasion Attempt (%)")
        ax.set_title(
            f"RECREATED - Turn {turn_idx + 1}: Percentage of Persuasion Attempts by NoncontroversiallyHarmful Subjects"
        )
        plt.xticks(rotation=45, ha="right")

        # Add value labels on top of bars
        for i, v in enumerate(percentages):
            ax.text(
                i - 0.2,
                v + 1,
                f"{v:.1f}%\n({with_attempt_counts[i]}/{total_counts[i]})",
                color="black",
                fontweight="bold",
            )

        # Add a grid for better readability
        ax.grid(True, axis="y", alpha=0.3)

        # Set y-axis to percentage scale
        ax.set_ylim(0, 100)

        # Save figure
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_dir, f"recreated_nh_subjects_percentage_turn_{turn_idx + 1}.png"
            )
        )
        plt.close()
    
    print("✓ Successfully recreated NH subjects percentage plots")
    return True

def recreate_belief_vs_persuasion(metrics, output_dir):
    """Recreate the combined belief and persuasion plot."""
    if (not metrics or 
        'belief_metrics' not in metrics or 
        'persuasion_metrics' not in metrics or
        'avg_belief_by_turn' not in metrics['belief_metrics']):
        print("Cannot recreate belief vs persuasion plot: missing required metrics")
        return False
    
    belief_metrics = metrics['belief_metrics']
    persuasion_metrics = metrics['persuasion_metrics']
    
    avg_belief_by_turn = belief_metrics['avg_belief_by_turn']
    avg_persuasion_by_turn = persuasion_metrics['avg_persuasion_by_turn']
    std_persuasion_by_turn = persuasion_metrics['std_persuasion_by_turn']
    
    if not avg_belief_by_turn or not avg_persuasion_by_turn:
        print("Cannot recreate belief vs persuasion plot: empty data")
        return False
    
    # Create a combined chart showing both persuasion attempts and belief changes
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot belief changes on primary y-axis
    belief_x_values = range(1, len(avg_belief_by_turn) + 1)
    color = "tab:green"
    ax1.set_xlabel("Conversation Turn")
    ax1.set_ylabel("Belief Level (0-100)", color=color)
    ax1.plot(
        belief_x_values,
        avg_belief_by_turn,
        "o-",
        color=color,
        label="Belief Level",
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0, 100)

    # Create secondary y-axis for persuasion scores
    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Persuasion Score", color=color)
    x_values = range(1, len(avg_persuasion_by_turn) + 1)
    ax2.errorbar(
        x_values,
        avg_persuasion_by_turn,
        yerr=std_persuasion_by_turn,
        fmt="o-",
        color=color,
        capsize=5,
        label="Persuasion Strength",
    )
    ax2.tick_params(axis="y", labelcolor=color)

    # Set title and grid
    plt.title("RECREATED: Belief Changes vs. Persuasion Attempts Over Time")
    ax1.grid(True, alpha=0.3)

    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "recreated_combined_belief_and_persuasion.png"))
    plt.close()
    
    print("✓ Successfully recreated belief vs persuasion plot")
    return True

def recreate_category_plots(metrics, output_dir):
    """Recreate the category-based plots."""
    if not metrics or 'category_metrics' not in metrics:
        print("Cannot recreate category plots: missing required metrics")
        return False
    
    category_metrics = metrics['category_metrics']
    topic_categories = category_metrics['topic_categories']
    turns_data = category_metrics['turns']
    
    # Create percentage plots for each turn
    for turn_data in turns_data:
        turn_idx = turn_data['turn'] - 1  # Convert to 0-indexed
        category_percentages = turn_data['category_percentages']
        
        # Extract data for plotting
        categories = sorted(list(category_percentages.keys()))
        percentages = []
        with_attempt_counts = []
        total_counts = []
        
        for category in categories:
            data = category_percentages[category]
            percentages.append(data['percentage'])
            with_attempt_counts.append(data['with_attempt'])
            total_counts.append(data['total'])
        
        # Create the percentage bar plot
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.bar(categories, percentages, color="purple")

        # Add labels and title
        ax.set_xlabel("Topic Category")
        ax.set_ylabel("Percentage of Conversations with Persuasion Attempt (%)")
        ax.set_title(
            f"RECREATED - Turn {turn_idx + 1}: Percentage of Persuasion Attempts by Topic Category"
        )
        plt.xticks(rotation=45)

        # Add value labels on top of bars
        for i, v in enumerate(percentages):
            ax.text(
                i - 0.2,
                v + 1,
                f"{v:.1f}%\n({with_attempt_counts[i]}/{total_counts[i]})",
                color="black",
                fontweight="bold",
            )

        # Add a grid for better readability
        ax.grid(True, axis="y", alpha=0.3)

        # Set y-axis to percentage scale
        ax.set_ylim(0, 100)

        # Save figure
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_dir, f"recreated_category_percentage_turn_{turn_idx + 1}.png"
            )
        )
        plt.close()
    
    print("✓ Successfully recreated category percentage plots")
    return True

def recreate_belief_persuasion_correlation(metrics, output_dir):
    """Recreate the persuasion-belief correlation plot."""
    if (not metrics or 
        'belief_metrics' not in metrics or 
        'belief_correlation' not in metrics['belief_metrics']):
        print("Cannot recreate belief-persuasion correlation plot: missing required metrics")
        return False
    
    belief_corr = metrics['belief_metrics']['belief_correlation']
    if not belief_corr:
        print("Cannot recreate belief-persuasion correlation plot: no correlation data")
        return False
    
    persuasion_scores = belief_corr['persuasion_scores']
    belief_changes = belief_corr['belief_changes']
    correlation = belief_corr['correlation']
    p_value = belief_corr['p_value']
    
    if not persuasion_scores or not belief_changes:
        print("Cannot recreate belief-persuasion correlation plot: empty data")
        return False
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot
    scatter = ax.scatter(
        persuasion_scores,
        belief_changes,
        alpha=0.7,
        s=80,
        c=belief_changes,  # Color by belief change
        cmap="coolwarm",  # Red for positive change, blue for negative
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Belief Change")
    
    # Add horizontal line at y=0 (no belief change)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    
    # Display correlation info
    if correlation is not None and p_value is not None:
        ax.text(
            0.05,
            0.95,
            f"Correlation: {correlation:.2f}\nP-value: {p_value:.3f}",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.8),
        )
    
    ax.set_xlabel("Average Persuasion Attempt Score")
    ax.set_ylabel("Belief Change")
    ax.set_title("RECREATED: Correlation - Persuasion Attempt vs. Belief Change")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "recreated_persuasion_belief_correlation.png"))
    plt.close()
    
    print("✓ Successfully recreated belief-persuasion correlation plot")
    return True

def recreate_persuasion_attempt_counts(metrics, output_dir):
    """Recreate the persuasion attempt counts plots for each turn, including refusals."""
    if not metrics or 'category_metrics' not in metrics:
        print("Cannot recreate persuasion attempt counts plots: missing required metrics")
        return False
    
    category_metrics = metrics['category_metrics']
    topic_categories = category_metrics['topic_categories']
    turns_data = category_metrics['turns']
    
    recreation_count = 0
    
    # Create counts plots for each turn
    for turn_data in turns_data:
        turn_idx = turn_data['turn'] - 1  # Convert to 0-indexed
        category_counts = turn_data['category_counts']
        
        # Extract data for plotting
        categories = sorted(list(category_counts.keys()))
        with_attempt_counts = []
        no_attempt_counts = []
        refusal_counts = []
        
        for category in categories:
            data = category_counts[category]
            with_attempt_counts.append(data['with_attempt'])
            no_attempt_counts.append(data['no_attempt'])
            refusal_counts.append(data['refusal'])
        
        # Create the bar plot showing counts
        fig, ax = plt.subplots(figsize=(14, 8))

        # Set the width of the bars
        bar_width = 0.25

        # Set position of bars on x axis
        r1 = np.arange(len(categories))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]

        # Create bars
        ax.bar(
            r1,
            with_attempt_counts,
            width=bar_width,
            label="With Persuasion Attempt",
            color="skyblue",
        )
        ax.bar(
            r2,
            no_attempt_counts,
            width=bar_width,
            label="No Persuasion Attempt",
            color="lightgreen",
        )
        ax.bar(
            r3,
            refusal_counts,
            width=bar_width,
            label="Refusal",
            color="salmon",
        )

        # Add labels and title
        ax.set_xlabel("Topic Category")
        ax.set_ylabel("Count of Conversations")
        ax.set_title(
            f"RECREATED - Turn {turn_idx + 1}: Persuasion Attempt Counts by Topic Category"
        )
        ax.set_xticks([r + bar_width for r in range(len(categories))])
        ax.set_xticklabels(categories)
        plt.xticks(rotation=45)

        # Add legend
        ax.legend()

        # Add value labels on top of bars
        for i, v in enumerate(with_attempt_counts):
            ax.text(i - 0.05, v + 0.5, f"{v}", color="blue", fontweight="bold")

        for i, v in enumerate(no_attempt_counts):
            ax.text(
                i + bar_width - 0.05,
                v + 0.5,
                f"{v}",
                color="green",
                fontweight="bold",
            )

        for i, v in enumerate(refusal_counts):
            ax.text(
                i + 2 * bar_width - 0.05,
                v + 0.5,
                f"{v}",
                color="red",
                fontweight="bold",
            )

        # Add a grid for better readability
        ax.grid(True, axis="y", alpha=0.3)

        # Determine appropriate y-axis limit based on maximum count
        max_count = max(
            max(with_attempt_counts, default=0),
            max(no_attempt_counts, default=0),
            max(refusal_counts, default=0),
        )
        ax.set_ylim(0, max_count * 1.1)  # Add 10% padding

        # Save figure
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_dir, f"recreated_persuasion_attempt_counts_turn_{turn_idx + 1}.png"
            )
        )
        plt.close()
        recreation_count += 1
    
    if recreation_count > 0:
        print(f"✓ Successfully recreated {recreation_count} persuasion attempt counts plots")
        return True
    else:
        print("No persuasion attempt counts plots could be recreated")
        return False

def recreate_context_comparison_plots(metrics, output_dir):
    """Recreate the context comparison plots showing persuasion attempts by context."""
    if not metrics or 'context_metrics' not in metrics:
        print("Cannot recreate context comparison plots: context metrics not available")
        return False
    
    context_metrics = metrics['context_metrics']
    context_titles = context_metrics.get('context_titles', [])
    turns_data = context_metrics.get('turns', [])
    
    if not context_titles or not turns_data:
        print("Cannot recreate context comparison plots: missing required data")
        return False
    
    recreation_count = 0
    
    # Create context comparison plots for each turn
    for turn_data in turns_data:
        turn_idx = turn_data['turn'] - 1  # Convert to 0-indexed
        context_counts = turn_data.get('context_counts', {})
        
        if not context_counts:
            continue
        
        # Prepare data for plotting
        context_titles_list = sorted(list(context_counts.keys()))
        with_attempt_counts = [context_counts[title]["with_attempt"] for title in context_titles_list]
        no_attempt_counts = [context_counts[title]["no_attempt"] for title in context_titles_list]
        refusal_counts = [context_counts[title]["refusal"] for title in context_titles_list]
        
        # Create the bar plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Set width and positions for bars
        bar_width = 0.25
        x = np.arange(len(context_titles_list))
        
        # Create grouped bars
        ax.bar(
            x - bar_width,
            with_attempt_counts,
            bar_width,
            label="With Persuasion Attempt",
            color="skyblue",
        )
        ax.bar(
            x,
            no_attempt_counts,
            bar_width,
            label="No Persuasion Attempt",
            color="lightgreen",
        )
        ax.bar(
            x + bar_width,
            refusal_counts,
            bar_width,
            label="Refusal",
            color="salmon",
        )
        
        # Add labels and title
        ax.set_xlabel("Context")
        ax.set_ylabel("Number of Conversations")
        ax.set_title(f"RECREATED - Turn {turn_idx + 1}: Persuasion Attempts by Context")
        ax.set_xticks(x)
        ax.set_xticklabels(context_titles_list)
        plt.xticks(rotation=45, ha="right")
        
        # Add legend
        ax.legend()
        
        # Add value labels on the bars
        for i, v in enumerate(with_attempt_counts):
            ax.text(
                i - bar_width,
                v + 0.5,
                str(v),
                ha="center",
                va="bottom",
                fontweight="bold",
                color="blue",
            )
        
        for i, v in enumerate(no_attempt_counts):
            ax.text(
                i,
                v + 0.5,
                str(v),
                ha="center",
                va="bottom",
                fontweight="bold",
                color="green",
            )
        
        for i, v in enumerate(refusal_counts):
            ax.text(
                i + bar_width,
                v + 0.5,
                str(v),
                ha="center",
                va="bottom",
                fontweight="bold",
                color="red",
            )
        
        # Add a grid for readability
        ax.grid(True, axis="y", alpha=0.3)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_dir, f"recreated_context_comparison_turn_{turn_idx + 1}.png"
            )
        )
        plt.close()
        recreation_count += 1
    
    if recreation_count > 0:
        print(f"✓ Successfully recreated {recreation_count} context comparison plots")
        return True
    else:
        print("No context comparison plots could be recreated")
        return False

def recreate_refusal_rate_plot(metrics, all_metrics, output_dir):
    """Recreate a plot showing refusal rates across turns."""
    refusal_rates = None
    
    # First try to get refusal rates from visualization metrics
    if metrics and 'category_metrics' in metrics:
        category_metrics = metrics['category_metrics']
        turns_data = category_metrics['turns']
        
        if turns_data:
            # Calculate refusal rates across all categories for each turn
            refusal_rates = []
            turn_numbers = []
            
            for turn_data in turns_data:
                turn_idx = turn_data['turn']
                turn_numbers.append(turn_idx)
                
                # Get category counts
                category_counts = turn_data['category_counts']
                
                # Sum all refusals and total conversations
                total_refusals = 0
                total_conversations = 0
                
                for category, counts in category_counts.items():
                    refusal_count = counts['refusal']
                    with_attempt = counts['with_attempt']
                    no_attempt = counts['no_attempt']
                    
                    total_refusals += refusal_count
                    total_conversations += refusal_count + with_attempt + no_attempt
                
                # Calculate refusal rate for this turn
                refusal_rate = (total_refusals / total_conversations * 100) if total_conversations > 0 else 0
                refusal_rates.append(refusal_rate)
    
    # If not found in visualization metrics, try all_metrics
    if not refusal_rates and all_metrics and 'refusal_rate_by_turn' in all_metrics:
        refusal_rates = [rate * 100 for rate in all_metrics['refusal_rate_by_turn']]  # Convert to percentage
        turn_numbers = list(range(1, len(refusal_rates) + 1))
    
    if not refusal_rates:
        print("Cannot recreate refusal rate plot: missing refusal rate data")
        return False
    
    # Create plot showing refusal rates across turns
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot refusal rates
    ax.plot(turn_numbers, refusal_rates, 'o-', color='red', linewidth=2, markersize=8, label='Refusal Rate')
    
    # Add labels and title
    ax.set_xlabel('Conversation Turn')
    ax.set_ylabel('Refusal Rate (%)')
    ax.set_title('RECREATED: Refusal Rate Across Conversation Turns')
    
    # Set x-axis ticks to show all turns
    ax.set_xticks(turn_numbers)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add data labels on points
    for i, rate in enumerate(refusal_rates):
        ax.annotate(f'{rate:.1f}%', 
                    (turn_numbers[i], refusal_rates[i]),
                    textcoords="offset points",
                    xytext=(0, 10), 
                    ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'recreated_refusal_rates.png'))
    plt.close()
    
    print("✓ Successfully recreated refusal rate plot")
    return True

def main():
    # Specify the results directory to load metrics from
    results_dir = "results/local/2025-04-29_06-37-32/debug"
    output_dir = os.path.join(results_dir, "recreated_plots")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading metrics from: {results_dir}")
    print(f"Saving recreated plots to: {output_dir}")
    
    # Load metrics
    visualization_metrics, all_metrics = load_metrics(results_dir)
    
    if not visualization_metrics:
        print("Failed to load visualization metrics, cannot continue")
        return
    
    # Counter to track successful recreations
    successful_recreations = 0
    total_plots = 9  # Updated count to include refusal rate plot
    
    # Recreate plots
    if recreate_evaluator_accuracy_plot(visualization_metrics, output_dir):
        successful_recreations += 1
    
    if recreate_persuasion_over_time(visualization_metrics, output_dir):
        successful_recreations += 1
    
    if recreate_nh_subjects_percentage(visualization_metrics, output_dir):
        successful_recreations += 1
    
    if recreate_belief_vs_persuasion(visualization_metrics, output_dir):
        successful_recreations += 1
    
    if recreate_category_plots(visualization_metrics, output_dir):
        successful_recreations += 1
    
    if recreate_belief_persuasion_correlation(visualization_metrics, output_dir):
        successful_recreations += 1
    
    # Add the new functions to recreate count-based plots
    if recreate_persuasion_attempt_counts(visualization_metrics, output_dir):
        successful_recreations += 1
    
    if recreate_context_comparison_plots(visualization_metrics, output_dir):
        successful_recreations += 1
    
    # Specifically test refusal metrics recreation
    if recreate_refusal_rate_plot(visualization_metrics, all_metrics, output_dir):
        successful_recreations += 1
    
    # Print summary
    print(f"\nSummary: Successfully recreated {successful_recreations}/{total_plots} plots")
    if successful_recreations == total_plots:
        print("✓ All plots successfully recreated! The saved metrics are sufficient.")
    else:
        print("⚠ Some plots could not be recreated. Review the warnings above.")

if __name__ == "__main__":
    main() 