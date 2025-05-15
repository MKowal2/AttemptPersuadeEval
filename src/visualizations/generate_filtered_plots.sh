#!/bin/bash

# Script to generate plots for specific models using the aggregate_plots.py script
# This script generates:
# 1. Stacked percentage plots showing attempt/no-attempt/refusal distributions
# 2. Model comparison plots that show all models side-by-side for each category/subject
# 3. Both raw counts and percentage stacked bar charts with error bars
# 4. All-in-one comparison plots that show all models and all categories on a single chart,
#    with different colors for models and different patterns for response types

# Define the directory where results are stored
RESULTS_DIR="results/paper_results"
OUTPUT_DIR="results/paper_results/filtered_model_plots"

# Run the aggregate_plots.py script with model filtering
python -m src.visualizations.aggregate_plots \
  --results_dir $RESULTS_DIR \
  --output_dir $OUTPUT_DIR \
  --min_runs 1 \
  --turns 3 \
  --models \
    "gpt-4o" \
    "vertex_ai/gemini-2.5-pro-preview-03-25" \
    "gpt-4o-mini" \
    "vertex_ai/gemini-2.0-flash-001" \
    "hf/Meta-Llama-3.1-8B-Instruct" \
    "hf/Qwen3-32B"

echo "Filtered model plots have been generated in $OUTPUT_DIR"
echo "The percentage plots now show stacked bars for attempt/no-attempt/refusal distributions"
echo "New model comparison plots generated in subdirectories of $OUTPUT_DIR:"
echo "  - model_comparisons_by_category: Percentage comparisons for each category"
echo "  - model_counts_comparisons_by_category: Raw count comparisons for each category"
echo "  - model_comparisons_by_nh_subject: Percentage comparisons for each NH subject"
echo "  - model_counts_comparisons_by_nh_subject: Raw count comparisons for each NH subject"
echo "  - all_in_one_comparisons: Single plots with all models and all categories shown together"
echo "  - all_in_one_nh_comparisons: Single plots with all models and all NH subjects shown together" 