# Aggregate Visualization Plots

This directory contains scripts for visualizing persuasion attempt metrics across multiple experiment runs.

## Visualization Scripts

- `visualizations.py`: Original script that creates visualizations for a single run
- `aggregate_plots.py`: Script that creates aggregated plots with standard deviation across multiple runs

## Using aggregate_plots.py

The `aggregate_plots.py` script automatically finds all model runs in the results directory and generates:

1. Bar plots showing attempt/no-attempt/refusal counts by category with standard deviation
2. Bar plots showing attempt/no-attempt/refusal counts for NoncontroversiallyHarmful subjects with standard deviation
3. Percentage plots for persuasion attempts
4. Comparison plots to see multiple models on the same chart

### Usage

```bash
python src/visualizations/aggregate_plots.py [options]
```

### Options

- `--results_dir`: Parent directory containing timestamped results subdirectories (default: 'results')
- `--output_dir`: Directory to save the aggregated plots (default: 'results/aggregate_plots')
- `--min_runs`: Minimum number of runs required to include a model in plots (default: 2)
- `--turns`: Number of conversation turns to analyze (default: 3)

### Examples

```bash
# Basic usage with default options
python src/visualizations/aggregate_plots.py

# Specify a different results directory
python src/visualizations/aggregate_plots.py --results_dir /path/to/results

# Require at least 3 runs for each model to be included
python src/visualizations/aggregate_plots.py --min_runs 3

# Save output plots to a specific directory
python src/visualizations/aggregate_plots.py --output_dir results/my_aggregate_plots
```

### Directory Structure

The script expects results to be structured as follows:

```
results/
├── timestamp1/
│   ├── model1/
│   │   └── visualization_metrics.json
│   ├── model2/
│   │   └── visualization_metrics.json
├── timestamp2/
│   ├── model1/
│   │   └── visualization_metrics.json
│   ├── model3/
│   │   └── visualization_metrics.json
```

The script will automatically find all models and their runs based on the presence of a `visualization_metrics.json` file.

### Output

The script creates:

1. Per-model directories with plots for each turn
2. A model_comparisons directory with plots comparing all models by category

```
results/aggregate_plots/
├── model1/
│   ├── category_counts_turn_1.png
│   ├── category_counts_turn_2.png
│   ├── nh_subjects_counts_turn_1.png
│   └── ...
├── model2/
│   ├── category_counts_turn_1.png
│   └── ...
└── model_comparisons/
    ├── turn_1_BenignFactual_comparison.png
    ├── turn_1_NoncontroversiallyHarmful_comparison.png
    └── ...
``` 