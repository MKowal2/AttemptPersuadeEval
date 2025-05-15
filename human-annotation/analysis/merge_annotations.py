import glob
import json
import os


def load_json_file(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def save_json_file(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def validate_entry_match(main_entry, annotation_entry):
    """Validate that the fields match between main dataset and annotation entry."""
    # Check required fields
    required_fields = [
        "latest_persuader_response",
        "latest_persuadee_response",
        "topic",
    ]

    # Get the data from both entries
    main_data = main_entry.get("conversation", {})
    if main_data is None:
        print(f"Skipping main_entry: {main_entry} due to missing conversation...")
        return False

    annotation_data = annotation_entry.get("data", {})

    # For debugging, print the structure of both entries
    # print(f"\nValidating entry {main_entry.get('id')}")
    # print("Main entry conversation keys:", main_data.keys())
    # print("Annotation data keys:", annotation_data.keys())

    # Check each field
    for field in required_fields:
        if field not in main_data or field not in annotation_data:
            print(f"Warning: Missing field '{field}' in entry {main_entry.get('id')}")
            return False

        if main_data[field] != annotation_data[field]:
            print(f"Warning: Field '{field}' mismatch in entry {main_entry.get('id')}")
            print(f"Main dataset: {main_data[field][:100]}...")
            print(f"Annotation: {annotation_data[field][:100]}...")
            return False

    return True


def main():
    # Load the main dataset
    main_dataset_path = "dataset_with_conversations.json"
    dataset = load_json_file(main_dataset_path)

    # Create a dictionary for fast lookups by ID
    dataset_by_id = {entry["id"]: entry for entry in dataset}

    # Find all annotation files in the human_responses directory
    annotation_dir = "human_responses"
    annotation_files = glob.glob(os.path.join(annotation_dir, "*.json"))

    if not annotation_files:
        print(f"No annotation files found in {annotation_dir} directory")
        return

    # Process each annotation file
    for annotation_file in annotation_files:
        print(f"\nProcessing {annotation_file}...")
        annotations = load_json_file(annotation_file)

        # Extract annotator name from filename (e.g., "jasper.json" -> "jasper")
        annotator_name = os.path.splitext(os.path.basename(annotation_file))[0]

        # Process each annotation
        for annotation in annotations:
            entry_id = annotation.get("data").get("id")
            if not entry_id:
                print(f"Warning: Missing ID in annotation from {annotator_name}")
                continue

            main_entry = dataset_by_id.get(entry_id)
            if not main_entry:
                print(f"Warning: Entry {entry_id} not found in main dataset")
                continue

            # Validate the fields match
            if not validate_entry_match(main_entry, annotation):
                print(f"Skipping entry {entry_id} due to validation failure")
                continue

            # Add the human response
            if "human_responses" not in main_entry:
                main_entry["human_responses"] = []

            # Add the response (1 for Yes, 0 for No)
            response_value = (
                1
                if annotation.get("annotations")[0]
                .get("result")[0]
                .get("value")
                .get("choices")[0]
                == "Yes"
                else 0
            )
            main_entry["human_responses"].append(
                {"annotator": annotator_name, "response": response_value}
            )

    # Save the updated dataset to a separate file
    merged_dataset_path = "merged_dataset.json"
    save_json_file(dataset, merged_dataset_path)
    print(
        f"\nSuccessfully merged annotations from {len(annotation_files)} files into {merged_dataset_path}"
    )


if __name__ == "__main__":
    main()
