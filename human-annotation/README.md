# Human Annotation Dataset Creation

This directory contains scripts for creating and managing datasets for human annotation of model conversations.

## Dataset Creation

The `create_dataset.py` script generates a dataset of conversations for human annotation by processing conversation logs from previous model runs.

### Configuration

The script is configured with the following parameters:

- **Models**: `hf/Meta-Llama-3.1-8B-Instruct` and `gpt-4o`
- **Turns**: 1, 2, and 3
- **Evaluator Responses**: 0 and 1
- **Topic Categories**: "BenignOpinion", "BenignFactual", "Controversial" and "NoncontroversiallyHarmful
- **Samples**: 2 (2 of each combination above)

### Running the Script

To create the dataset, run:

```bash
python create_dataset.py
```

### Output Files

The script generates two JSON files:

1. **Full Dataset** (`analysis/dataset_with_conversations.json`):
   - Contains complete conversation data including metadata
   - Includes model information, turn numbers, topic categories, and full conversation context

2. **Annotation Dataset** (`dataset_with_conversations_for_annotation.json`):
   - Simplified version containing only the essential information needed for annotation
   - Each entry includes:
     - ID
     - Latest persuader response
     - Latest persuadee response
     - Topic

### Dataset Structure

The script searches through conversation logs in `../results/local/annotation` in reverse chronological order to find matching conversations. It looks for conversations that match:
- The specified model
- Topic categories
- Topic file
- Turn number
- Evaluator response

The script will print progress information including:
- Number of entries found
- Number of remaining entries to find
- Any entries that couldn't be matched

### Notes

- The script automatically shuffles the dataset entries and assigns sequential IDs
- It skips conversations where there was an explicit refusal
- The script processes all results directories until it finds matches for all required entries

## Annotation Process

The annotation process is performed using Label Studio, a powerful open-source data labeling tool.

### Setting Up Label Studio

1. Pull and run the Label Studio Docker container:
   ```bash
   docker pull heartexlabs/label-studio:latest
   docker run -it -p 8080:8080 heartexlabs/label-studio:latest
   ```

2. Access Label Studio by opening your web browser and navigating to:
   ```
   http://localhost:8080
   ```

### Creating a New Project

1. Create a new account or log in to Label Studio
2. Click "Create Project" and give it a name (e.g., "Model Persuasion Analysis")
3. In the project setup:
   - Import the annotation dataset file (`dataset_with_conversations_for_annotation.json`)
   - Choose "Custom Template" for the labeling interface
   - Use the provided `layout.xml` file to set up the annotation interface

### Annotation Interface

The custom layout will display:
- The conversation topic
- The latest persuadee response
- The latest persuader response
- Annotation controls for marking various aspects of the conversation

### Performing Annotations

1. Click "Start Labeling" to begin the annotation process
2. For each conversation:
   - Read the topic and conversation context
   - Use the provided controls to mark relevant aspects
   - Click "Submit" to save your annotation
   - Use the navigation controls to move to the next conversation

### Exporting Results

1. Once all annotations are complete, go to the project dashboard
2. Click "Export" to download the annotated dataset
3. Choose your preferred export format (JSON recommended)
4. Save the exported file for further analysis

## Analysis Process

After collecting annotations from multiple annotators, the next step is to merge and analyze the results.

### Merging Annotations

1. Place all exported annotation files in the `human_responses` directory
2. Run the merge_annotations script:
   ```bash
   python merge_annotations.py
   ```

The script will:
- Read all annotation files from the `human_responses` directory
- Merge the annotations for each conversation
- Create a consolidated dataset with all annotations
- Output the merged results to `merged_dataset.json`

### Analysis Output

The merged dataset (`merged_dataset.json`) will contain:
- All original conversation data
- Combined annotations from all annotators

This merged dataset can then be used for further analysis and visualization of the annotation results.

### Agreement Analysis and Visualization

To analyze inter-annotator agreement and create visualizations:

1. Run the agreement analysis script:
   ```bash
   python agreement_analysis.py
   ```

The script will:
- Calculate agreement statistics between annotators
- Generate various graphs and visualizations
- Save the output in the `analysis` directory

The generated visualizations will include:
- Agreement heatmaps
- Distribution plots of annotations
- Comparison charts between different model responses
- Statistical summaries of agreement metrics

These visualizations can be used to:
- Assess the reliability of the annotations
- Identify patterns in model behavior
- Compare performance across different conditions
- Support conclusions about model capabilities
