import json
import os
import random  # Add random for shuffling
import re

def generate_dataset():
    """Generate the dataset programmatically."""
    models = ["hf/Meta-Llama-3.1-8B-Instruct", "gpt-4o"]
    turns = [1, 2, 3]
    evaluator_responses = [0, 1]
    topic_categories = [
        "BenignFactual",
        "BenignOpinion",
        "Controversial",
        "NoncontroversiallyHarmful",
    ]
    samples = 2

    dataset = []

    for _ in range(samples):
        for model in models:
            for turn in turns:
                for response in evaluator_responses:
                    for topic_category in topic_categories:
                        dataset.append(
                            {
                                "persuader_model": model,
                                "turn": turn,
                                "topic_category": topic_category,
                                "evaluator_response": response,
                                "conversation": None,  # Will be filled in later
                            }
                        )

    return dataset


def find_matching_conversation(dataset_entry, conversation_log, topics):
    """Find a matching conversation for the given entry in the conversation log."""
    # Get the persuader model and topic categories from the spec
    spec = json.loads(conversation_log[0])
    persuader_model = spec["spec"]["run_config"]["PERSUADER_MODEL"]
    print(f"Looking for: {dataset_entry}, in log")

    # Check if models, topic categories, and topic file match
    if dataset_entry["persuader_model"] != persuader_model:
        # print(f"Skipping model {persuader_model} because it does not match {dataset_entry['persuader_model']}")
        return None

    # Parse all entries and sort by sample_id and event_id
    parsed_entries = []
    for line in conversation_log:
        try:
            data = json.loads(line)
            if "sample_id" in data and "event_id" in data:
                parsed_entries.append(data)
        except json.JSONDecodeError:
            continue

    # Sort entries by sample_id and event_id
    parsed_entries.sort(key=lambda x: (x["sample_id"], x["event_id"]))

    # Group entries by sample_id
    samples = {}
    for entry in parsed_entries:
        sample_id = entry["sample_id"]
        if sample_id not in samples:
            samples[sample_id] = []
        samples[sample_id].append(entry)

    # Process each sample
    for sample_id, sample_entries in samples.items():
        
        # Calculate the number of entries to skip for previous turns
        entries_to_skip = (dataset_entry["turn"] - 1) * 4

        # Check if we have enough entries for the target turn
        if len(sample_entries) < entries_to_skip + 4:
            continue

        # Get the entries for our target turn
        turn_entries = sample_entries[entries_to_skip : entries_to_skip + 4]

        # Check for explicit refusal in the evaluator's response
        refusal_check = turn_entries[3]
        if refusal_check.get("type") == "sampling":
            refusal_response = refusal_check.get("data", {}).get("sampled", [])
            if (
                refusal_response
                and isinstance(refusal_response, list)
                and len(refusal_response) > 0
            ):
                # Extract the refusal score (1.b X)
                refusal_match = re.search(r"1\.b\s*(\d+)", refusal_response[0])
                if refusal_match and int(refusal_match.group(1)) == 1:
                    # print(f"Skipping conversation {sample_id} because there was an explicit refusal")
                    continue  # Skip this conversation if there was an explicit refusal

        # The evaluator response is the 3rd entry of the turn
        evaluator_entry = turn_entries[2]
        if evaluator_entry.get("type") == "sampling":
            sampled = evaluator_entry.get("data", {}).get("sampled", [])
            if sampled and isinstance(sampled, list) and len(sampled) > 0:
                response = sampled[0]
                # Extract the rating after the | symbol
                rating_match = re.search(r"\|\s*(\d+)", response)
                if rating_match:
                    evaluator_response = int(rating_match.group(1))
                    if evaluator_response == dataset_entry["evaluator_response"]:
                        # If evaluator response matches, process the conversation data
                        persuader_entry = turn_entries[1]

                        # Extract the topic from the system message
                        system_message = next(
                            (
                                msg["content"]
                                for msg in persuader_entry.get("data", {}).get(
                                    "prompt", []
                                )
                                if msg.get("role") == "system"
                            ),
                            "",
                        )
                        # Extract the topic from between single quotes after "persuade users to believe that"
                        topic_match = re.search(
                            r"persuade users to believe that '([^']*)'", system_message
                        )
                        topic = topic_match.group(1) if topic_match else ""

                        # Skip if we've already used this topic
                        if topic in used_topics:
                            # print(f"Skipping topic {topic} because it has already been used")
                            continue

                        # Find the category of the topic in the topics file
                        topic_category = next(
                            (t["category"] for t in topics if t["text"] == topic), ""
                        )
                        if topic_category != dataset_entry["topic_category"]:
                            # print(f"Skipping topic {topic} because it is not in the category {dataset_entry['topic_category']}")
                            continue

                        # Extract the latest persuadee response (last user message)
                        user_messages = [
                            msg["content"]
                            for msg in persuader_entry.get("data", {}).get("prompt", [])
                            if msg.get("role") == "user"
                        ]
                        latest_persuadee_response = (
                            user_messages[-1] if user_messages else ""
                        )
                        # Strip off any belief rating at the end (e.g. "| 15")
                        latest_persuadee_response = re.sub(
                            r"\s*\|\s*\d+\s*$", "", latest_persuadee_response
                        )
                        # Need to check if the latest persuadee response is a refusal
                        if latest_persuadee_response.startswith("I'm sorry, but I"):
                            continue

                        # Extract the latest persuader response from the persuader's sampled response
                        persuader_sampled = persuader_entry.get("data", {}).get(
                            "sampled", []
                        )
                        latest_persuader_response = (
                            persuader_sampled[0] if persuader_sampled else ""
                        )

                        # Create the result object with all fields from persuader_entry except 'data'
                        result = {
                            k: v for k, v in persuader_entry.items() if k != "data"
                        }

                        # Strip any belief rating at the end (e.g. "| 15")
                        latest_persuader_response = re.sub(
                            r"\s*\|\s*\d+\s*$", "", latest_persuader_response
                        )
                        latest_persuadee_response = re.sub(
                            r"\s*\|\s*\d+\s*$", "", latest_persuadee_response
                        )

                        # Ensure both responses are at least 200 characters
                        if (
                            len(latest_persuader_response) < 200
                            or len(latest_persuadee_response) < 200
                        ):
                            # print(f"Skipping conversation {sample_id} because the responses are too short")
                            continue

                        # Add the parsed data fields at the root level
                        result.update(
                            {
                                "latest_persuader_response": latest_persuader_response,
                                "latest_persuadee_response": latest_persuadee_response,
                                "topic": topic,
                            }
                        )

                        used_topics.add(topic)

                        return result

    return None


def process_results_directories(dataset, topics):
    """Process all results directories in reverse chronological order until dataset is complete."""
    # Get all result directories and sort them in reverse chronological order
    result_dirs = sorted(
        [
            d
            for d in os.listdir(RESULTS_DIR)
            if os.path.isdir(os.path.join(RESULTS_DIR, d))
        ],
        reverse=True,
    )

    if not result_dirs:
        print("No results directories found")
        return dataset

    print(result_dirs)

    # Keep track of how many entries still need conversations
    remaining_entries = sum(1 for entry in dataset if entry["conversation"] is None)
    print(f"\nLooking for conversations for {remaining_entries} entries...")

    for dir_name in result_dirs:
        if remaining_entries == 0:
            break

        # Get the first (and only) subdirectory
        subdir = next(
            (
                d
                for d in os.listdir(os.path.join(RESULTS_DIR, dir_name))
                if os.path.isdir(os.path.join(RESULTS_DIR, dir_name, d))
            ),
            None,
        )
        if not subdir:
            continue

        dir_path = os.path.join(RESULTS_DIR, dir_name, subdir, "conversation_log.jsonl")
        if not os.path.exists(dir_path):
            continue

        print(f"\nChecking {dir_name}...")

        # Read the conversation log
        with open(dir_path, "r") as f:
            conversation_log = f.readlines()

        # Try to find matches for entries without conversations
        for entry in dataset:
            if entry["conversation"] is None:
                match = find_matching_conversation(entry, conversation_log, topics)
                if match:
                    entry["conversation"] = match
                    remaining_entries -= 1
                    print(
                        f"Found match for {entry['persuader_model']} turn {entry['turn']}"
                    )

        print(f"Remaining entries to find: {remaining_entries}")

    return dataset


def output_dataset(dataset, output_file):
    """Output the dataset to JSON files with shuffled entries and sequential IDs.
    Creates both a full dataset and a simplified version for annotation."""
    # Shuffle the dataset
    random.shuffle(dataset)

    # Add sequential IDs
    for idx, entry in enumerate(dataset, start=1):
        entry["id"] = idx

    # Output full dataset as JSON
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)

    # Create simplified version for annotation, strip off analysis dir at front
    annotation_file = output_file.replace(".json", "_for_annotation.json")
    annotation_file = annotation_file.replace("analysis/", "")

    simplified_entries = []
    for entry in dataset:
        if entry["conversation"] is not None:
            simplified_entry = {
                "id": entry["id"],
                "latest_persuader_response": entry["conversation"][
                    "latest_persuader_response"
                ],
                "latest_persuadee_response": entry["conversation"][
                    "latest_persuadee_response"
                ],
                "topic": entry["conversation"]["topic"],
            }
            simplified_entries.append(simplified_entry)

    with open(annotation_file, "w") as f:
        json.dump(simplified_entries, f, indent=2)

    print(f"Created full dataset at {output_file}")
    print(f"Created annotation dataset at {annotation_file}")

def load_topics(topics_path):
    """Load topics from the JSONL file."""
    with open(topics_path, "r") as f:
        return [json.loads(line) for line in f]


# Where to look for results to add to dataset
RESULTS_DIR = "../results/local/new_annotations"

# Define topics globally to keep track of what's been used
used_topics = set()

if __name__ == "__main__":
    # Generate the dataset
    dataset = generate_dataset()

    topics_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "topics", "diverse_topics_conspiracy.jsonl"
    )
    topics = load_topics(topics_path)

    # Process all results directories
    dataset = process_results_directories(dataset, topics)

    # Output the full dataset to the analysis directory
    output_file = "analysis/dataset_with_conversations.json"
    output_dataset(dataset, output_file)
    print(f"\nOutput dataset to {output_file}")

    # Print summary
    completed = sum(1 for entry in dataset if entry["conversation"] is not None)
    print(f"Found conversations for {completed} out of {len(dataset)} entries")

    # Print unfound entries
    unfound_entries = [entry for entry in dataset if entry["conversation"] is None]
    if unfound_entries:
        print("\nUnfound entries:")
        for entry in unfound_entries:
            print(
                f"- Model: {entry['persuader_model']}, Turn: {entry['turn']}, "
                f"Category: {entry['topic_category']}, "
                f"Evaluator Response: {entry['evaluator_response']}"
            )
    else:
        print("\nAll entries were found successfully!")
