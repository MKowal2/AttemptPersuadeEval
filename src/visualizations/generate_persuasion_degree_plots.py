#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
from pathlib import Path
from tabulate import tabulate

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate persuasion degree confusion matrix plots from saved results")
    parser.add_argument("--results_dir", type=str, required=True, 
                        help="Directory containing results (e.g., results/persuasion_degree)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save the generated plots (defaults to same as input)")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Filter by model name (e.g., gpt_4o, llama_8b)")
    parser.add_argument("--evaluation_scale", type=int, default=None,
                        help="Filter by evaluation scale (e.g., 2, 3, 100)")
    parser.add_argument("--turn", type=int, default=-1,
                        help="Which turn to visualize. Default is -1 (last turn)")
    parser.add_argument("--diagonal_only", action="store_true",
                        help="Generate only diagonal metrics (accuracy per persuasion degree) instead of full confusion matrix")
    parser.add_argument("--save_csv", action="store_true",
                        help="Save diagonal metrics to CSV file")
    parser.add_argument("--bar_chart", action="store_true",
                        help="Generate bar chart of accuracy per persuasion degree")
    parser.add_argument("--summary_csv", type=str, default=None,
                        help="Save final summary to CSV file")
    return parser.parse_args()

def extract_model_info(results_dir):
    """Extract model name and evaluation scale from directory path or metrics file"""
    # Try to extract from path first
    model_name = "unknown"
    evaluation_scale = None
    
    # Extract from path - find the last directory that contains model name and scale like "gpt_4o_3"
    path_parts = results_dir.split('/')
    for part in reversed(path_parts):
        if '_' in part:
            try:
                # Match pattern like "gpt_4o_3" or "llama_8b_2"
                model_parts = part.split('_')
                if len(model_parts) >= 3 and model_parts[-1].isdigit():
                    model_name = '_'.join(model_parts[:-1])  # Everything except the last part
                    evaluation_scale = int(model_parts[-1])
                    break
            except (ValueError, IndexError):
                continue
    
    # If not found in path, try to get from metrics file
    if evaluation_scale is None:
        metrics_file = os.path.join(results_dir, "all_metrics.json")
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                # Get model name and scale from metrics
                if "persuader_model" in metrics:
                    model_name = metrics["persuader_model"]
                if "experiment_name" in metrics:
                    experiment_name = metrics["experiment_name"]
                    if "_" in experiment_name:
                        try:
                            # Try to extract scale from experiment name
                            parts = experiment_name.split('_')
                            if parts[-1].isdigit():
                                evaluation_scale = int(parts[-1])
                        except (ValueError, IndexError):
                            pass
                if "evaluation_scale" in metrics:
                    evaluation_scale = metrics["evaluation_scale"]
            except:
                pass
    
    return model_name, evaluation_scale

def find_result_dirs(base_dir, model_name=None, evaluation_scale=None):
    """Find all result directories matching the filters"""
    all_dirs = []
    
    # First level contains model+scale directories like gpt_4o_3
    for model_dir in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
            
        # Filter by model name if specified
        if model_name and model_name not in model_dir:
            continue
            
        # Filter by evaluation scale if specified
        if evaluation_scale:
            try:
                # Extract scale from directory name (e.g., "gpt_4o_3" -> 3)
                dir_scale = int(model_dir.split('_')[-1])
                if dir_scale != evaluation_scale:
                    continue
            except (ValueError, IndexError):
                continue
        
        # Find all run directories (typically named with timestamps)
        for run_dir in os.listdir(model_path):
            run_path = os.path.join(model_path, run_dir)
            if not os.path.isdir(run_path):
                continue
                
            # Each run directory typically contains a model-named directory
            for inner_dir in os.listdir(run_path):
                inner_path = os.path.join(run_path, inner_dir)
                if os.path.isdir(inner_path):
                    # Check for metrics file to confirm this is a valid results directory
                    metrics_file = os.path.join(inner_path, "all_metrics.json")
                    if os.path.exists(metrics_file):
                        all_dirs.append(inner_path)
    
    return all_dirs

def extract_diagonal_metrics(normalized_confusion, raw_confusion=None):
    """Extract and format diagonal metrics from the confusion matrix"""
    diagonal_values = np.diag(normalized_confusion)
    
    # Calculate additional metrics if raw matrix is available
    total_samples = None
    class_samples = None
    if raw_confusion is not None:
        class_samples = np.sum(raw_confusion, axis=1)
        total_samples = np.sum(class_samples)
    
    # Create a dictionary of metrics
    metrics = {
        "persuasion_degree": list(range(len(diagonal_values))),
        "accuracy": diagonal_values.tolist(),
    }
    
    if class_samples is not None:
        metrics["samples"] = class_samples.tolist()
        metrics["percentage_of_total"] = (class_samples / total_samples * 100).tolist()
    
    return metrics

def print_diagonal_metrics(metrics, experiment_name, turn_num):
    """Pretty print the diagonal metrics as a table"""
    # Create a list of rows for the table
    rows = []
    for i in range(len(metrics["persuasion_degree"])):
        row = [metrics["persuasion_degree"][i], f"{metrics['accuracy'][i]:.4f}"]
        if "samples" in metrics:
            row.append(int(metrics["samples"][i]))
            row.append(f"{metrics['percentage_of_total'][i]:.2f}%")
        rows.append(row)
    
    # Create headers
    headers = ["Persuasion Degree", "Accuracy"]
    if "samples" in metrics:
        headers.extend(["Samples", "% of Total"])
    
    # Print the table
    print(f"\n--- Persuasion Degree Accuracy for {experiment_name} (Turn {turn_num}) ---")
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    # Print overall accuracy
    weighted_acc = None
    if "samples" in metrics:
        weighted_acc = sum(metrics["accuracy"][i] * metrics["samples"][i] for i in range(len(metrics["accuracy"]))) / sum(metrics["samples"])
        print(f"Overall Accuracy: {weighted_acc:.4f}")
    
    return weighted_acc

def save_diagonal_metrics_csv(metrics, output_path):
    """Save the diagonal metrics to a CSV file"""
    df = pd.DataFrame(metrics)
    df.to_csv(output_path, index=False)
    print(f"Saved diagonal metrics to {output_path}")

def generate_diagonal_bar_chart(metrics, experiment_name, turn_num, output_path):
    """Generate a bar chart showing accuracy per persuasion degree"""
    degrees = metrics["persuasion_degree"]
    accuracies = metrics["accuracy"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars
    bars = ax.bar(degrees, accuracies, color='skyblue')
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        samples_text = f" (n={int(metrics['samples'][i])})" if "samples" in metrics else ""
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{height:.2f}{samples_text}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    
    # Add labels and title
    ax.set_xlabel("Persuasion Degree")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy by Persuasion Degree\nTurn {turn_num} - {experiment_name}")
    
    # Set y-axis limits
    ax.set_ylim(0, 1.1)
    
    # Set x-ticks to integer values
    ax.set_xticks(degrees)
    
    # Add a grid for readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Generated bar chart at {output_path}")

def generate_confusion_matrix(results_dir, output_dir=None, turn_idx=-1, diagonal_only=False, save_csv=False, bar_chart=False):
    """Generate a confusion matrix visualization or diagonal metrics from the results directory"""
    if output_dir is None:
        output_dir = results_dir
    
    # Load metrics file
    metrics_file = os.path.join(results_dir, "all_metrics.json")
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found in {results_dir}")
        return False, None
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Get the evaluation scale from the metrics
    evaluation_scale = metrics.get("evaluation_scale", None)
    if not evaluation_scale:
        # Try to infer from degree_specific_accuracy
        degree_specific_accuracy = metrics.get("degree_specific_accuracy", {})
        if degree_specific_accuracy:
            evaluation_scale = len(degree_specific_accuracy)
        else:
            print(f"Could not determine evaluation scale for {results_dir}")
            return False, None
    
    # Extract model information
    model_name, detected_scale = extract_model_info(results_dir)
    if detected_scale and evaluation_scale != detected_scale:
        print(f"Warning: Detected scale {detected_scale} differs from metrics scale {evaluation_scale}")
    
    # Get experiment name
    experiment_name = metrics.get("experiment_name", os.path.basename(results_dir))
    
    # Initialize summary data
    summary_data = {
        "model": model_name,
        "experiment": experiment_name,
        "evaluation_scale": evaluation_scale,
        "overall_accuracy": None,
        "per_degree_accuracy": {}
    }
    
    # Check for confusion matrices in the metrics file
    confusion_matrices = metrics.get("confusion_matrices", None)
    if confusion_matrices:
        # Determine which turn to use
        num_turns = len(confusion_matrices)
        if turn_idx < 0:
            # Negative indices count from the end, e.g., -1 is the last turn
            turn_idx = num_turns + turn_idx
        
        if 0 <= turn_idx < num_turns:
            # Use the stored confusion matrix data
            cm_data = confusion_matrices[turn_idx]
            turn_num = cm_data.get("turn", turn_idx + 1)
            
            # Get the normalized matrix and raw matrix (if available)
            normalized_confusion = np.array(cm_data.get("normalized_matrix"))
            raw_matrix = None
            if "raw_matrix" in cm_data:
                raw_matrix = np.array(cm_data.get("raw_matrix"))
            
            # Extract diagonal metrics
            diag_metrics = extract_diagonal_metrics(normalized_confusion, raw_matrix)
            
            # Print diagonal metrics table
            overall_acc = print_diagonal_metrics(diag_metrics, experiment_name, turn_num)
            
            # Update summary data
            summary_data["overall_accuracy"] = overall_acc
            for i, acc in enumerate(diag_metrics["accuracy"]):
                summary_data["per_degree_accuracy"][i] = acc
            
            # Save diagonal metrics to CSV if requested
            if save_csv:
                csv_path = os.path.join(output_dir, f"persuasion_degree_accuracy_turn_{turn_num}.csv")
                save_diagonal_metrics_csv(diag_metrics, csv_path)
            
            # Generate bar chart if requested
            if bar_chart:
                bar_chart_path = os.path.join(output_dir, f"persuasion_degree_accuracy_turn_{turn_num}.png")
                generate_diagonal_bar_chart(diag_metrics, experiment_name, turn_num, bar_chart_path)
            
            # If we only want diagonal metrics, skip the full confusion matrix
            if diagonal_only:
                return True, summary_data
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(normalized_confusion, cmap="Blues")
            
            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel("Normalized Frequency", rotation=-90, va="bottom")
            
            # Add text annotations to each cell if below 10 degrees
            if evaluation_scale < 10:
                for i in range(evaluation_scale):
                    for j in range(evaluation_scale):
                        ax.text(
                            j,
                            i,
                            f"{normalized_confusion[i, j]:.2f}",
                            ha="center",
                            va="center",
                            color="black" if normalized_confusion[i, j] < 0.5 else "white",
                        )
            
            # Configure axis labels
            ax.set_xticks(np.arange(evaluation_scale))
            ax.set_yticks(np.arange(evaluation_scale))
            
            if evaluation_scale < 10:
                ax.set_xticklabels([f"{i}" for i in range(evaluation_scale)])
                ax.set_yticklabels([f"{i}" for i in range(evaluation_scale)])
            else:
                # For large scales, set ticks at intervals
                tick_interval = 20 if evaluation_scale >= 100 else 5
                x_ticks = np.arange(0, evaluation_scale, tick_interval)
                y_ticks = np.arange(0, evaluation_scale, 5)
                
                ax.set_xticks(x_ticks)
                ax.set_yticks(y_ticks)
                ax.set_xticklabels([f"{i}" for i in x_ticks])
                ax.set_yticklabels([f"{i}" for i in y_ticks])
            
            ax.set_xlabel("Predicted Persuasion Degree")
            ax.set_ylabel("True Persuasion Degree")
            ax.set_title(f"Confusion Matrix of Persuasion Degree Prediction\nTurn {turn_num} - {experiment_name}")
            
            plt.tight_layout()
            output_path = os.path.join(output_dir, f"persuasion_degree_confusion_matrix_turn_{turn_num}.png")
            plt.savefig(output_path)
            plt.close()
            
            print(f"Generated confusion matrix plot for {experiment_name} (Turn {turn_num}) at {output_path}")
            return True, summary_data
        else:
            print(f"Invalid turn index {turn_idx} for {results_dir} with {num_turns} turns")
            return False, None
    else:
        print(f"No confusion matrix data found in metrics file for {results_dir}")
        print("Try running newer experiments with the updated calc_metrics.py to generate confusion matrix data")
        
        # Fall back to the old method of reconstructing the confusion matrix
        success, summary = reconstruct_and_generate_confusion_matrix(
            metrics, results_dir, output_dir, diagonal_only, save_csv, bar_chart,
            model_name=model_name, experiment_name=experiment_name, evaluation_scale=evaluation_scale
        )
        return success, summary

def reconstruct_and_generate_confusion_matrix(metrics, results_dir, output_dir, diagonal_only=False, 
                                             save_csv=False, bar_chart=False, model_name=None, 
                                             experiment_name=None, evaluation_scale=None):
    """Legacy method to reconstruct the confusion matrix from raw data"""
    print("Attempting to reconstruct confusion matrix from raw data (legacy method)")
    
    if evaluation_scale is None:
        evaluation_scale = metrics.get("evaluation_scale", None)
        if not evaluation_scale:
            # Try to infer from degree_specific_accuracy
            degree_specific_accuracy = metrics.get("degree_specific_accuracy", {})
            if degree_specific_accuracy:
                evaluation_scale = len(degree_specific_accuracy)
            else:
                print(f"Could not determine evaluation scale")
                return False, None
    
    if experiment_name is None:
        experiment_name = metrics.get("experiment_name", os.path.basename(results_dir))
    
    # Initialize summary data
    summary_data = {
        "model": model_name or "unknown",
        "experiment": experiment_name,
        "evaluation_scale": evaluation_scale,
        "overall_accuracy": None,
        "per_degree_accuracy": {}
    }
    
    # We need to reconstruct the confusion matrix from the raw prediction data
    filtered_ratings = metrics.get("filtered_ratings_distribution", {})
    if not filtered_ratings:
        print(f"No filtered ratings found")
        return False, None
    
    # Get the last turn's ratings for the final confusion matrix
    turns = list(filtered_ratings.keys())
    if not turns:
        print(f"No turn data found")
        return False, None
    
    # Sort turns and get the last one
    turns.sort(key=lambda x: int(x.split('_')[1]))
    last_turn = turns[-1]
    last_turn_ratings = filtered_ratings[last_turn]
    turn_num = int(last_turn.split('_')[1]) + 1  # Turn indices are 0-based, display as 1-based
    
    # Find all user samples with their true persuasion degrees
    all_samples_file = os.path.join(results_dir, "all_samples.json")
    if os.path.exists(all_samples_file):
        with open(all_samples_file, 'r') as f:
            samples = json.load(f)
        
        sampled_topics_short_titles = samples.get("sampled_topics_short_titles", [])
        sampled_persuasion_degrees = samples.get("sampled_persuasion_degrees", [])
        
        # Create confusion matrix
        confusion_matrix = np.zeros((evaluation_scale, evaluation_scale))
        
        for user_idx in range(len(sampled_persuasion_degrees)):
            conspiracy_title = sampled_topics_short_titles[user_idx]
            
            if conspiracy_title in last_turn_ratings:
                same_conspiracy_indices = [
                    j for j, title in enumerate(sampled_topics_short_titles)
                    if title == conspiracy_title
                ]
                position = same_conspiracy_indices.index(user_idx) if user_idx in same_conspiracy_indices else -1
                
                if position >= 0 and position < len(last_turn_ratings[conspiracy_title]):
                    true_degree = sampled_persuasion_degrees[user_idx]
                    predicted_degree = last_turn_ratings[conspiracy_title][position]
                    
                    # Update confusion matrix
                    if (0 <= true_degree < evaluation_scale and 
                        0 <= predicted_degree < evaluation_scale):
                        confusion_matrix[true_degree, predicted_degree] += 1
        
        # If we couldn't reconstruct the matrix from all_samples.json, try to use degree-specific accuracy
        if np.sum(confusion_matrix) == 0:
            print(f"Could not reconstruct confusion matrix using samples data")
            # This is a fallback approach, but won't be as accurate as the full reconstruction
            confusion_matrix = reconstruct_from_accuracy(metrics)
            if confusion_matrix is None:
                return False, None
    else:
        print(f"No samples file found, attempting to reconstruct from accuracy metrics")
        # Fallback to reconstructing from accuracy metrics
        confusion_matrix = reconstruct_from_accuracy(metrics)
        if confusion_matrix is None:
            return False, None
    
    # Normalize confusion matrix by row (true label)
    row_sums = confusion_matrix.sum(axis=1)
    normalized_confusion = np.zeros_like(confusion_matrix, dtype=float)
    for i in range(len(row_sums)):
        if row_sums[i] > 0:
            normalized_confusion[i] = confusion_matrix[i] / row_sums[i]
    
    # Extract and print diagonal metrics
    diag_metrics = extract_diagonal_metrics(normalized_confusion, confusion_matrix)
    overall_acc = print_diagonal_metrics(diag_metrics, experiment_name, turn_num)
    
    # Update summary data
    summary_data["overall_accuracy"] = overall_acc
    for i, acc in enumerate(diag_metrics["accuracy"]):
        summary_data["per_degree_accuracy"][i] = acc
    
    # Save diagonal metrics to CSV if requested
    if save_csv:
        csv_path = os.path.join(output_dir, f"persuasion_degree_accuracy_turn_{turn_num}.csv")
        save_diagonal_metrics_csv(diag_metrics, csv_path)
    
    # Generate bar chart if requested
    if bar_chart:
        bar_chart_path = os.path.join(output_dir, f"persuasion_degree_accuracy_turn_{turn_num}.png")
        generate_diagonal_bar_chart(diag_metrics, experiment_name, turn_num, bar_chart_path)
    
    # If we only want diagonal metrics, skip the full confusion matrix
    if diagonal_only:
        return True, summary_data
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(normalized_confusion, cmap="Blues")
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Normalized Frequency", rotation=-90, va="bottom")
    
    # Add text annotations to each cell if below 10 degrees
    if evaluation_scale < 10:
        for i in range(evaluation_scale):
            for j in range(evaluation_scale):
                ax.text(
                    j,
                    i,
                    f"{normalized_confusion[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black" if normalized_confusion[i, j] < 0.5 else "white",
                )
    
    # Configure axis labels
    ax.set_xticks(np.arange(evaluation_scale))
    ax.set_yticks(np.arange(evaluation_scale))
    
    if evaluation_scale < 10:
        ax.set_xticklabels([f"{i}" for i in range(evaluation_scale)])
        ax.set_yticklabels([f"{i}" for i in range(evaluation_scale)])
    else:
        # For large scales, set ticks at intervals
        tick_interval = 20 if evaluation_scale >= 100 else 5
        x_ticks = np.arange(0, evaluation_scale, tick_interval)
        y_ticks = np.arange(0, evaluation_scale, 5)
        
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels([f"{i}" for i in x_ticks])
        ax.set_yticklabels([f"{i}" for i in y_ticks])
    
    ax.set_xlabel("Predicted Persuasion Degree")
    ax.set_ylabel("True Persuasion Degree")
    ax.set_title(f"Confusion Matrix of Persuasion Degree Prediction (Reconstructed)\n{experiment_name}")
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "persuasion_degree_confusion_matrix.png")
    plt.savefig(output_path)
    plt.close()
    
    print(f"Generated reconstructed confusion matrix plot for {experiment_name} at {output_path}")
    return True, summary_data

def reconstruct_from_accuracy(metrics):
    """Attempt to reconstruct confusion matrix from accuracy metrics (less accurate fallback)"""
    degree_specific_accuracy = metrics.get("degree_specific_accuracy", {})
    if not degree_specific_accuracy:
        print("No degree-specific accuracy data found")
        return None
    
    # Get evaluation scale from number of degrees
    evaluation_scale = len(degree_specific_accuracy)
    
    # Create a simplified diagonal confusion matrix based on accuracy
    confusion_matrix = np.zeros((evaluation_scale, evaluation_scale))
    
    # Get accuracy from the last turn
    for degree, accuracies in degree_specific_accuracy.items():
        if not accuracies:
            continue
        
        degree = int(degree)
        last_accuracy = accuracies[-1]
        
        # Put accuracy on diagonal and distribute rest evenly
        confusion_matrix[degree, degree] = last_accuracy
        remaining = 1.0 - last_accuracy
        
        # Distribute remaining probability evenly among other classes
        other_classes = evaluation_scale - 1
        if other_classes > 0:
            spread = remaining / other_classes
            for j in range(evaluation_scale):
                if j != degree:
                    confusion_matrix[degree, j] = spread
    
    return confusion_matrix

def print_summary_table(summary_data_list):
    """Print a consolidated summary table of all processed directories"""
    if not summary_data_list:
        print("No summary data available.")
        return
    
    # Organize data by model and scale
    model_scale_data = {}
    scales_found = set()
    
    for data in summary_data_list:
        model = data["model"]
        scale = data["evaluation_scale"]
        scales_found.add(scale)
        
        if model not in model_scale_data:
            model_scale_data[model] = {}
        
        if scale not in model_scale_data[model]:
            model_scale_data[model][scale] = {
                "count": 0,
                "overall_accuracy": 0,
                "per_degree_accuracy": {}
            }
        
        # Add this experiment's data
        model_scale_data[model][scale]["count"] += 1
        if data["overall_accuracy"] is not None:
            model_scale_data[model][scale]["overall_accuracy"] += data["overall_accuracy"]
        
        # Add per-degree accuracy
        for degree, acc in data["per_degree_accuracy"].items():
            if degree not in model_scale_data[model][scale]["per_degree_accuracy"]:
                model_scale_data[model][scale]["per_degree_accuracy"][degree] = 0
            model_scale_data[model][scale]["per_degree_accuracy"][degree] += acc
    
    # Calculate averages
    for model, scale_data in model_scale_data.items():
        for scale, data in scale_data.items():
            count = data["count"]
            if count > 0:
                data["overall_accuracy"] /= count
                for degree in data["per_degree_accuracy"]:
                    data["per_degree_accuracy"][degree] /= count
    
    # Create single consolidated table
    print("\n=== MODEL PERFORMANCE ACROSS EVALUATION SCALES ===")
    
    # Sort scales for consistent table layout
    sorted_scales = sorted(scales_found)
    
    # Create headers
    headers = ["Model"]
    for scale in sorted_scales:
        headers.append(f"Scale {scale}")
    
    # Create rows
    rows = []
    for model in sorted(model_scale_data.keys()):
        row = [model]
        for scale in sorted_scales:
            if scale in model_scale_data[model]:
                row.append(f"{model_scale_data[model][scale]['overall_accuracy']:.4f}")
            else:
                row.append("N/A")
        rows.append(row)
    
    # Print table
    print(tabulate(rows, headers=headers, tablefmt="grid"))

def save_summary_csv(summary_data_list, output_path):
    """Save the summary data to a CSV file"""
    if not summary_data_list:
        print("No summary data available to save.")
        return
    
    # Prepare data for DataFrame
    rows = []
    for data in summary_data_list:
        model = data["model"]
        experiment = data["experiment"]
        scale = data["evaluation_scale"]
        overall_acc = data["overall_accuracy"]
        
        row = {
            "model": model,
            "experiment": experiment,
            "evaluation_scale": scale,
            "overall_accuracy": overall_acc
        }
        
        # Add per-degree accuracy
        for degree, acc in data["per_degree_accuracy"].items():
            row[f"degree_{degree}_accuracy"] = acc
        
        rows.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved summary to {output_path}")

def main():
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all result directories
    result_dirs = find_result_dirs(args.results_dir, args.model_name, args.evaluation_scale)
    
    if not result_dirs:
        print(f"No matching result directories found in {args.results_dir}")
        return
    
    print(f"Found {len(result_dirs)} result directories to process")
    
    # Generate confusion matrix for each directory and collect summary data
    successful = 0
    summary_data_list = []
    
    for results_dir in result_dirs:
        output_dir = args.output_dir or results_dir
        success, summary_data = generate_confusion_matrix(
            results_dir, output_dir, args.turn, 
            args.diagonal_only, args.save_csv, args.bar_chart
        )
        
        if success:
            successful += 1
            if summary_data:
                summary_data_list.append(summary_data)
    
    print(f"Successfully generated {successful} out of {len(result_dirs)} plots/metrics")
    
    # Print final summary table
    print("\n============= FINAL SUMMARY =============")
    print_summary_table(summary_data_list)
    
    # Save summary to CSV if requested
    if args.summary_csv:
        save_summary_csv(summary_data_list, args.summary_csv)

if __name__ == "__main__":
    main() 