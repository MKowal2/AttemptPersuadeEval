import itertools
import json
import os
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# Set style for better looking plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

# Create graphs directory if it doesn't exist
os.makedirs("graphs", exist_ok=True)


def load_data(file_path: str) -> pd.DataFrame:
    """Load the JSON data into a pandas DataFrame."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def create_agreement_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Create an agreement matrix between all evaluators."""
    # Extract human responses
    human_responses = []
    for idx, row in df.iterrows():
        # Check if human_responses is iterable
        if not isinstance(row["human_responses"], Iterable):
            continue
        for response in row["human_responses"]:
            human_responses.append(
                {
                    "id": row["id"],
                    "annotator": response["annotator"],
                    "response": response["response"],
                    "evaluator_response": row["evaluator_response"],
                    "topic_category": row["topic_category"],
                }
            )

    human_df = pd.DataFrame(human_responses)

    # Get unique annotator names
    annotators = sorted(human_df["annotator"].unique())
    print(f"Found annotators: {annotators}")

    # Pivot to get responses in columns
    pivot_df = human_df.pivot(index="id", columns="annotator", values="response")
    pivot_df["evaluator"] = df.set_index("id")["evaluator_response"]

    # Convert responses to numeric values if they're categorical
    for col in pivot_df.columns:
        if pivot_df[col].dtype == "object":
            # Convert to numeric using factorize
            pivot_df[col] = pd.factorize(pivot_df[col])[0]

    return pivot_df


def plot_agreement_heatmap(df: pd.DataFrame, title: str = "Agreement Matrix"):
    """Plot a heatmap showing agreement between evaluators."""
    # Calculate agreement matrix
    evaluators = ["evaluator"] + sorted(df.columns.drop("evaluator").tolist())
    agreement_matrix = pd.DataFrame(index=evaluators, columns=evaluators)

    for e1, e2 in itertools.combinations_with_replacement(evaluators, 2):
        if e1 == e2:
            agreement_matrix.loc[e1, e2] = 1.0
        else:
            agreement = (df[e1] == df[e2]).mean()
            agreement_matrix.loc[e1, e2] = agreement
            agreement_matrix.loc[e2, e1] = agreement

    # Ensure all values are numeric
    agreement_matrix = agreement_matrix.astype(float)

    plt.figure(figsize=(8, 6))
    sns.heatmap(agreement_matrix, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("graphs/agreement_heatmap.png")
    plt.close()


def plot_stacked_bars(df: pd.DataFrame):
    """Plot stacked bar charts showing response distributions by topic category."""
    # Extract human responses and evaluator responses
    human_responses = []
    for idx, row in df.iterrows():
        # Check if human_responses is iterable
        if not isinstance(row["human_responses"], Iterable):
            continue

        for response in row["human_responses"]:
            human_responses.append(
                {
                    "id": row["id"],
                    "annotator": response["annotator"],
                    "response": response["response"],
                    "topic_category": row["topic_category"][0],  # Take first category
                }
            )

    human_df = pd.DataFrame(human_responses)

    # Plot human responses
    plt.figure(figsize=(12, 6))
    human_pivot = human_df.pivot_table(
        index="topic_category", columns="response", values="id", aggfunc="count"
    ).fillna(0)

    # Convert to fractions (normalize rows)
    human_pivot = human_pivot.div(human_pivot.sum(axis=1), axis=0)

    # Plot with modified X-axis labels
    human_pivot.plot(kind="bar", stacked=True)
    plt.title("Human Response Distribution by Topic Category")
    plt.xlabel("Topic Category")
    plt.ylabel("Fraction")
    plt.legend(title="Response")
    plt.ylim(0, 1)  # Set y-axis limits from 0 to 1

    # Change x-axis labels to just the first letter
    x_labels = [label[0].upper() for label in human_pivot.index]
    plt.gca().set_xticklabels(x_labels)

    plt.tight_layout()
    plt.savefig("graphs/human_responses_stacked.png")
    plt.close()

    # Plot evaluator responses
    plt.figure(figsize=(12, 6))
    evaluator_df = df.explode("topic_category")
    evaluator_pivot = evaluator_df.pivot_table(
        index="topic_category",
        columns="evaluator_response",
        values="id",
        aggfunc="count",
    ).fillna(0)

    # Convert to fractions (normalize rows)
    evaluator_pivot = evaluator_pivot.div(evaluator_pivot.sum(axis=1), axis=0)

    # Plot with modified X-axis labels
    evaluator_pivot.plot(kind="bar", stacked=True)
    plt.title("Evaluator Response Distribution by Topic Category")
    plt.xlabel("Topic Category")
    plt.ylabel("Fraction")
    plt.legend(title="Response")
    plt.ylim(0, 1)  # Set y-axis limits from 0 to 1

    # Change x-axis labels to just the first letter
    x_labels = [label[0].upper() for label in evaluator_pivot.index]
    plt.gca().set_xticklabels(x_labels)

    plt.tight_layout()
    plt.savefig("graphs/evaluator_responses_stacked.png")
    plt.close()


def calculate_kappa_scores(df: pd.DataFrame):
    """Calculate and plot Cohen's Kappa scores."""
    # Extract human responses
    human_responses = []
    for idx, row in df.iterrows():
        # Check if human_responses is iterable
        if not isinstance(row["human_responses"], Iterable):
            continue

        for response in row["human_responses"]:
            human_responses.append(
                {
                    "id": row["id"],
                    "annotator": response["annotator"],
                    "response": response["response"],
                    "evaluator_response": row["evaluator_response"],
                    "topic_category": row["topic_category"][0],
                }
            )

    human_df = pd.DataFrame(human_responses)

    # Get unique annotator names
    annotators = sorted(human_df["annotator"].unique())
    print(f"Found annotators: {annotators}")

    # Get all unique labels
    all_labels = set()
    for col in ["response", "evaluator_response"]:
        all_labels.update(human_df[col].unique())
    all_labels = sorted(list(all_labels))

    print(f"Unique response labels: {all_labels}")
    print("Number of responses per annotator:")
    print(human_df["annotator"].value_counts())

    # Calculate kappa scores for each pair
    kappa_scores = []
    pairs = []
    # Add evaluator vs each annotator
    for annotator in annotators:
        pairs.append(("evaluator", annotator))
    # Add all annotator pairs
    for a1, a2 in itertools.combinations(annotators, 2):
        pairs.append((a1, a2))

    for pair in pairs:
        print(f"\nCalculating kappa for {pair[0]} vs {pair[1]}")
        for category in human_df["topic_category"].unique():
            # Get responses for this category
            category_mask = human_df["topic_category"] == category

            # Get evaluator responses
            if pair[0] == "evaluator":
                evaluator_responses = df.loc[
                    df["id"].isin(human_df[category_mask]["id"]),
                    ["id", "evaluator_response"],
                ].set_index("id")
            else:
                evaluator_responses = human_df[
                    category_mask & (human_df["annotator"] == pair[0])
                ][["id", "response"]].set_index("id")

            # Get other annotator responses
            if pair[1] == "evaluator":
                other_responses = df.loc[
                    df["id"].isin(human_df[category_mask]["id"]),
                    ["id", "evaluator_response"],
                ].set_index("id")
            else:
                other_responses = human_df[
                    category_mask & (human_df["annotator"] == pair[1])
                ][["id", "response"]].set_index("id")

            # Align responses by ID
            aligned_responses = evaluator_responses.join(
                other_responses, how="inner", lsuffix="_1", rsuffix="_2"
            )

            if len(aligned_responses) == 0:
                print(f"Category: {category}")
                print("No overlapping responses between annotators")
                continue

            print(f"Category: {category}")
            print(f"Number of aligned responses: {len(aligned_responses)}")
            print(
                f"Unique values in first annotator: {aligned_responses.iloc[:, 0].nunique()}"
            )
            print(
                f"Unique values in second annotator: {aligned_responses.iloc[:, 1].nunique()}"
            )

            # Calculate agreement rate
            agreement = (
                aligned_responses.iloc[:, 0] == aligned_responses.iloc[:, 1]
            ).mean()
            print(f"Agreement rate: {agreement:.2f}")

            # Only calculate kappa if both annotators have variation
            if (
                aligned_responses.iloc[:, 0].nunique() > 1
                and aligned_responses.iloc[:, 1].nunique() > 1
            ):
                try:
                    kappa = cohen_kappa_score(
                        aligned_responses.iloc[:, 0],
                        aligned_responses.iloc[:, 1],
                        labels=all_labels,
                    )
                    print(f"Kappa score: {kappa:.2f}")

                    if not np.isnan(kappa):
                        kappa_scores.append(
                            {
                                "pair": f"{pair[0]} vs {pair[1]}",
                                "category": category,
                                "kappa": kappa,
                                "agreement": agreement,
                            }
                        )
                except (ValueError, RuntimeWarning) as e:
                    print(f"Error calculating kappa: {str(e)}")
            else:
                print("Skipping kappa: One or both annotators have no variation")
                kappa_scores.append(
                    {
                        "pair": f"{pair[0]} vs {pair[1]}",
                        "category": category,
                        "kappa": np.nan,
                        "agreement": agreement,
                    }
                )

    if not kappa_scores:
        print("No valid kappa scores could be calculated")
        return

    kappa_df = pd.DataFrame(kappa_scores)
    print("\nFinal kappa scores:")
    print(kappa_df)

    # Plot kappa scores and agreement rates
    plt.figure(figsize=(15, 6))

    # Plot kappa scores
    plt.subplot(1, 2, 1)
    sns.barplot(data=kappa_df, x="category", y="kappa", hue="pair")
    plt.title("Cohen's Kappa Scores by Topic Category")
    plt.xlabel("Topic Category")
    plt.ylabel("Cohen's Kappa")
    plt.xticks(rotation=45)

    # Plot agreement rates
    plt.subplot(1, 2, 2)
    sns.barplot(data=kappa_df, x="category", y="agreement", hue="pair")
    plt.title("Agreement Rates by Topic Category")
    plt.xlabel("Topic Category")
    plt.ylabel("Agreement Rate")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("graphs/kappa_scores.png")
    plt.close()


def plot_agreement_by_turn(df: pd.DataFrame):
    """Plot agreement rates by turn number."""
    # Extract human responses
    human_responses = []
    for idx, row in df.iterrows():
        # Check if human_responses is iterable
        if not isinstance(row["human_responses"], Iterable):
            continue

        for response in row["human_responses"]:
            human_responses.append(
                {
                    "id": row["id"],
                    "annotator": response["annotator"],
                    "response": response["response"],
                    "evaluator_response": row["evaluator_response"],
                    "turn": row["turn"],
                }
            )

    human_df = pd.DataFrame(human_responses)

    # Calculate agreement rates by turn
    agreement_rates = []
    pairs = [("evaluator", "jasper"), ("evaluator", "matt"), ("jasper", "matt")]

    for turn in sorted(human_df["turn"].unique()):
        for pair in pairs:
            mask = human_df["turn"] == turn
            if pair[0] == "evaluator":
                r1 = df.loc[
                    df["id"].isin(human_df[mask]["id"]), "evaluator_response"
                ].reset_index(drop=True)
            else:
                r1 = human_df[mask & (human_df["annotator"] == pair[0])][
                    "response"
                ].reset_index(drop=True)

            if pair[1] == "evaluator":
                r2 = df.loc[
                    df["id"].isin(human_df[mask]["id"]), "evaluator_response"
                ].reset_index(drop=True)
            else:
                r2 = human_df[mask & (human_df["annotator"] == pair[1])][
                    "response"
                ].reset_index(drop=True)

            # Skip if not enough data points or different lengths
            if len(r1) < 2 or len(r2) < 2 or len(r1) != len(r2):
                continue

            try:
                agreement = (r1 == r2).mean()
                agreement_rates.append(
                    {
                        "turn": turn,
                        "pair": f"{pair[0]} vs {pair[1]}",
                        "agreement": agreement,
                    }
                )
            except ValueError:
                continue

    if not agreement_rates:
        print("No valid agreement rates could be calculated")
        return

    agreement_df = pd.DataFrame(agreement_rates)

    # Plot agreement rates
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=agreement_df, x="turn", y="agreement", hue="pair", marker="o")
    plt.title("Agreement Rates by Turn Number")
    plt.xlabel("Turn Number")
    plt.ylabel("Agreement Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graphs/agreement_by_turn.png")
    plt.close()


def plot_confusion_matrices(df: pd.DataFrame):
    """Plot confusion matrices for the labels across labelers."""
    # Extract human responses
    human_responses = []
    for idx, row in df.iterrows():
        # Check if human_responses is iterable
        if not isinstance(row["human_responses"], Iterable):
            continue

        for response in row["human_responses"]:
            human_responses.append(
                {
                    "id": row["id"],
                    "annotator": response["annotator"],
                    "response": response["response"],
                    "evaluator_response": row["evaluator_response"],
                    "topic_category": (
                        row["topic_category"][0]
                        if isinstance(row["topic_category"], list)
                        else row["topic_category"]
                    ),
                }
            )

    human_df = pd.DataFrame(human_responses)

    # Get unique annotator names
    annotators = sorted(human_df["annotator"].unique())
    print(f"Found annotators for confusion matrices: {annotators}")

    # Define the pairs to compare
    pairs = []
    # Add evaluator vs each annotator
    for annotator in annotators:
        pairs.append(("evaluator", annotator))
    # Add all annotator pairs
    for a1, a2 in itertools.combinations(annotators, 2):
        pairs.append((a1, a2))

    # Get all categories and add "all" for an aggregate plot
    categories = list(human_df["topic_category"].unique())
    categories.append("all")

    for category in categories:
        # Use a more compact figure size but maintain enough space for labels
        fig, axes = plt.subplots(len(pairs), 1, figsize=(6, len(pairs) * 4))
        if len(pairs) == 1:
            axes = [axes]

        # Track the max value for a consistent colorbar scale
        max_count = 0

        for i, (anno1, anno2) in enumerate(pairs):
            # Filter by category if not "all"
            if category != "all":
                category_mask = human_df["topic_category"] == category
            else:
                category_mask = pd.Series([True] * len(human_df))

            # Get responses for anno1
            if anno1 == "evaluator":
                responses1 = df.loc[
                    df["id"].isin(human_df[category_mask]["id"]),
                    ["id", "evaluator_response"],
                ].set_index("id")
            else:
                responses1 = human_df[category_mask & (human_df["annotator"] == anno1)][
                    ["id", "response"]
                ].set_index("id")

            # Get responses for anno2
            if anno2 == "evaluator":
                responses2 = df.loc[
                    df["id"].isin(human_df[category_mask]["id"]),
                    ["id", "evaluator_response"],
                ].set_index("id")
            else:
                responses2 = human_df[category_mask & (human_df["annotator"] == anno2)][
                    ["id", "response"]
                ].set_index("id")

            # Align responses by ID
            aligned = responses1.join(
                responses2, how="inner", lsuffix="_1", rsuffix="_2"
            )

            if len(aligned) < 2:
                axes[i].text(
                    0.5,
                    0.5,
                    f"Not enough data for {anno1} vs {anno2}",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                axes[i].set_title(f"{anno1} vs {anno2} (Category: {category})")
                axes[i].axis("off")
                continue

            # Map string labels to binary (0/1) if needed
            col1, col2 = aligned.columns
            if aligned[col1].dtype == "object":
                aligned[col1] = aligned[col1].map({"no": 0, "yes": 1})
            if aligned[col2].dtype == "object":
                aligned[col2] = aligned[col2].map({"no": 0, "yes": 1})

            # Create confusion matrix
            try:
                cm = confusion_matrix(aligned[col1], aligned[col2], labels=[0, 1])

                # Update max count for colorbar scaling
                max_count = max(max_count, cm.max())

                # Create a subplot layout with specified size ratio
                ax = axes[i]

                # Plot confusion matrix with smaller quadrants
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=["No (0)", "Yes (1)"],
                    yticklabels=["No (0)", "Yes (1)"],
                    cbar=True,
                    ax=ax,
                    annot_kws={"size": 14},
                    square=True,
                    cbar_kws={"shrink": 0.6, "pad": 0.02},
                )

                # Set the quadrant size to be smaller
                ax.set_aspect("equal")

                # Customize the title and labels with fixed font sizes
                ax.set_title(f"{anno1} vs {anno2} (Category: {category})", fontsize=14)
                ax.set_xlabel(f"{anno2} Response", fontsize=12)
                ax.set_ylabel(f"{anno1} Response", fontsize=12)
                ax.tick_params(labelsize=11)

                # Calculate and display metrics
                agreement = (aligned[col1] == aligned[col2]).mean()
                try:
                    kappa = cohen_kappa_score(aligned[col1], aligned[col2])
                    metrics_text = f"Agreement: {agreement:.2f}, Cohen's Kappa: {kappa:.2f}, n={len(aligned)}"
                except ValueError:
                    metrics_text = f"Agreement: {agreement:.2f}, n={len(aligned)}"

                ax.text(
                    0.5,
                    -0.25,
                    metrics_text,
                    horizontalalignment="center",
                    transform=ax.transAxes,
                    fontsize=12,
                )

            except Exception as e:
                print(
                    f"Error creating confusion matrix for {anno1} vs {anno2} ({category}): {str(e)}"
                )
                axes[i].text(
                    0.5,
                    0.5,
                    f"Error creating confusion matrix: {str(e)}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    wrap=True,
                )
                axes[i].axis("off")

        # Normalize all plots to use the same color scale
        if max_count > 0:
            for ax in axes:
                if hasattr(ax, "collections") and ax.collections:
                    for collection in ax.collections:
                        if isinstance(collection, plt.matplotlib.collections.QuadMesh):
                            collection.set_clim(0, max_count)

        plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.0)
        fig.suptitle(
            f"Confusion Matrices for Category: {category}", fontsize=16, y=0.99
        )
        plt.savefig(
            f'graphs/confusion_matrix_{category.replace(" ", "_")}.png',
            bbox_inches="tight",
        )
        plt.close()


def plot_combined_human_vs_evaluator(df: pd.DataFrame):
    """Plot confusion matrix comparing the evaluator with all human annotators combined."""
    # Extract human responses
    human_responses = []
    for idx, row in df.iterrows():
        # Check if human_responses is iterable
        if not isinstance(row["human_responses"], Iterable):
            continue

        for response in row["human_responses"]:
            human_responses.append(
                {
                    "id": row["id"],
                    "annotator": response["annotator"],
                    "response": response["response"],
                    "evaluator_response": row["evaluator_response"],
                    "topic_category": (
                        row["topic_category"][0]
                        if isinstance(row["topic_category"], list)
                        else row["topic_category"]
                    ),
                }
            )

    human_df = pd.DataFrame(human_responses)

    # Get all categories and add "all" for an aggregate plot
    categories = list(human_df["topic_category"].unique())
    categories.append("all")

    for category in categories:
        # Filter by category if not "all"
        if category != "all":
            category_mask = human_df["topic_category"] == category
        else:
            category_mask = pd.Series([True] * len(human_df))

        # Filter data
        category_df = human_df[category_mask]

        # Create a combined dataframe with all human responses and corresponding evaluator responses
        combined_data = []
        for idx, row in category_df.iterrows():
            combined_data.append(
                {
                    "human_response": row["response"],
                    "evaluator_response": row["evaluator_response"],
                }
            )

        combined_df = pd.DataFrame(combined_data)

        # Skip if not enough data
        if len(combined_df) < 2:
            print(f"Not enough data for category: {category}")
            continue

        # Convert responses to binary if needed
        if combined_df["human_response"].dtype == "object":
            combined_df["human_response"] = combined_df["human_response"].map(
                {"no": 0, "yes": 1}
            )
        if combined_df["evaluator_response"].dtype == "object":
            combined_df["evaluator_response"] = combined_df["evaluator_response"].map(
                {"no": 0, "yes": 1}
            )

        # Create confusion matrix
        cm = confusion_matrix(
            combined_df["human_response"],
            combined_df["evaluator_response"],
            labels=[0, 1],
        )

        # Create a figure with more space at the bottom for metrics
        plt.figure(figsize=(8, 7))  # Increased height to accommodate more bottom space
        ax = plt.gca()

        # Plot with a colorbar
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No (0)", "Yes (1)"],
            yticklabels=["No (0)", "Yes (1)"],
            cbar=True,
            ax=ax,
            annot_kws={"size": 16},
            square=True,
            cbar_kws={"shrink": 0.7, "pad": 0.05},
        )

        # Set aspect to make the cells square
        ax.set_aspect("equal")

        # Add title and labels
        plt.title(f"All Humans vs Evaluator (Category: {category})", fontsize=16)
        plt.xlabel(
            "Evaluator Response", fontsize=14, labelpad=15
        )  # Increase labelpad for more space
        plt.ylabel("Human Response", fontsize=14)
        plt.tick_params(labelsize=12)

        # Calculate and display metrics with adjusted positions
        agreement = (
            combined_df["human_response"] == combined_df["evaluator_response"]
        ).mean()
        try:
            kappa = cohen_kappa_score(
                combined_df["human_response"], combined_df["evaluator_response"]
            )
            metrics_text = f"Agreement: {agreement:.2f}, Cohen's Kappa: {kappa:.2f}, n={len(combined_df)}"
        except ValueError:
            metrics_text = f"Agreement: {agreement:.2f}, n={len(combined_df)}"

        # Position the text lower to avoid overlap
        plt.text(
            0.5,
            -0.18,
            metrics_text,
            horizontalalignment="center",
            transform=ax.transAxes,
            fontsize=14,
        )

        # Calculate and display additional metrics (precision, recall, etc.) with more space
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        metrics_details = (
            f"Precision: {precision:.2f}, Recall: {recall:.2f}, "
            f"Specificity: {specificity:.2f}, F1: {f1:.2f}"
        )

        # Position the details text even lower
        plt.text(
            0.5,
            -0.26,
            metrics_details,
            horizontalalignment="center",
            transform=ax.transAxes,
            fontsize=12,
        )

        # Add more bottom space to prevent overlap
        plt.tight_layout(rect=[0, 0.1, 1, 0.96])  # Increased bottom margin
        plt.savefig(
            f'graphs/combined_humans_vs_evaluator_{category.replace(" ", "_")}.png',
            bbox_inches="tight",
        )
        plt.close()
        print(f"Saved combined_humans_vs_evaluator_{category.replace(' ', '_')}.png")


def plot_evaluator_vs_human_majority(df: pd.DataFrame):
    """
    Compare evaluator responses to majority vote of human annotators.
    For each example, take the majority vote of human annotators as ground truth.
    """
    print("\nCalculating evaluator vs human majority vote comparison...")

    # Extract human responses and organize by ID
    human_responses_by_id = {}
    evaluator_responses_by_id = {}
    example_categories = {}

    for idx, row in df.iterrows():
        # Skip if no human responses
        if not isinstance(row["human_responses"], Iterable):
            continue

        example_id = row["id"]
        evaluator_responses_by_id[example_id] = row["evaluator_response"]

        # Store category for this example
        if isinstance(row["topic_category"], list):
            example_categories[example_id] = row["topic_category"][0]
        else:
            example_categories[example_id] = row["topic_category"]

        # Store all human responses for this example
        if example_id not in human_responses_by_id:
            human_responses_by_id[example_id] = []

        for response in row["human_responses"]:
            human_responses_by_id[example_id].append(response["response"])

    # Take majority vote for each example
    majority_votes = {}
    for example_id, responses in human_responses_by_id.items():
        # Convert string labels to binary if needed
        binary_responses = []
        for resp in responses:
            if isinstance(resp, str):
                binary_responses.append(1 if resp.lower() == "yes" else 0)
            else:
                binary_responses.append(resp)

        # Count votes
        yes_votes = sum(binary_responses)
        no_votes = len(binary_responses) - yes_votes

        # Determine majority (if tie, choose "yes" as it's the rarer class)
        if yes_votes >= no_votes:
            majority_votes[example_id] = 1
        else:
            majority_votes[example_id] = 0

    # Prepare data for analysis
    data = []
    for example_id in majority_votes.keys():
        if example_id in evaluator_responses_by_id:
            # Convert evaluator response to binary if needed
            eval_response = evaluator_responses_by_id[example_id]
            if isinstance(eval_response, str):
                eval_response = 1 if eval_response.lower() == "yes" else 0

            data.append(
                {
                    "id": example_id,
                    "human_majority": majority_votes[example_id],
                    "evaluator_response": eval_response,
                    "category": example_categories.get(example_id, "unknown"),
                }
            )

    # Convert to DataFrame
    comparison_df = pd.DataFrame(data)

    # Function to calculate metrics and create confusion matrix plot
    def analyze_category(category_data, category_name):
        if len(category_data) < 2:
            print(f"Not enough data for category: {category_name}")
            return

        # Create confusion matrix
        cm = confusion_matrix(
            category_data["human_majority"],
            category_data["evaluator_response"],
            labels=[0, 1],
        )

        # Calculate metrics
        agreement = (
            category_data["human_majority"] == category_data["evaluator_response"]
        ).mean()
        try:
            kappa = cohen_kappa_score(
                category_data["human_majority"], category_data["evaluator_response"]
            )
        except ValueError:
            kappa = float("nan")

        # Calculate precision, recall, etc.
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        print(f"\nCategory: {category_name}")
        print(f"Number of examples: {len(category_data)}")
        print(f"Agreement: {agreement:.2f}")
        print(f"Cohen's Kappa: {kappa:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"Specificity: {specificity:.2f}")
        print(f"F1 Score: {f1:.2f}")

        # Create confusion matrix plot
        plt.figure(figsize=(8, 7))
        ax = plt.gca()

        # Plot with a colorbar
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No (0)", "Yes (1)"],
            yticklabels=["No (0)", "Yes (1)"],
            cbar=True,
            ax=ax,
            annot_kws={"size": 16},
            square=True,
            cbar_kws={"shrink": 0.7, "pad": 0.05},
        )

        # Set aspect to make the cells square
        ax.set_aspect("equal")

        # Add title and labels
        plt.title(
            f"Human Majority vs Evaluator (Category: {category_name})", fontsize=16
        )
        plt.xlabel("Evaluator Response", fontsize=14, labelpad=15)
        plt.ylabel("Human Majority Vote", fontsize=14)
        plt.tick_params(labelsize=12)

        # Display metrics
        metrics_text = f"Agreement: {agreement:.2f}, Cohen's Kappa: {kappa:.2f}, n={len(category_data)}"
        plt.text(
            0.5,
            -0.18,
            metrics_text,
            horizontalalignment="center",
            transform=ax.transAxes,
            fontsize=14,
        )

        metrics_details = (
            f"Precision: {precision:.2f}, Recall: {recall:.2f}, "
            f"Specificity: {specificity:.2f}, F1: {f1:.2f}"
        )

        plt.text(
            0.5,
            -0.26,
            metrics_details,
            horizontalalignment="center",
            transform=ax.transAxes,
            fontsize=12,
        )

        # Add space to prevent overlap
        plt.tight_layout(rect=[0, 0.1, 1, 0.96])
        plt.savefig(
            f'graphs/majority_vs_evaluator_{category_name.replace(" ", "_")}.png',
            bbox_inches="tight",
        )
        plt.close()

    # Analyze all data
    analyze_category(comparison_df, "all")

    # Analyze by category
    for category in comparison_df["category"].unique():
        if category != "unknown":
            cat_data = comparison_df[comparison_df["category"] == category]
            analyze_category(cat_data, category)

    # Create summary table
    print("\nSaving summary statistics to majority_vote_results.csv")
    summary_stats = []

    # First analyze all data
    all_data = comparison_df
    all_cm = confusion_matrix(
        all_data["human_majority"], all_data["evaluator_response"], labels=[0, 1]
    )
    tn, fp, fn, tp = all_cm.ravel()

    summary_stats.append(
        {
            "Category": "all",
            "Examples": len(all_data),
            "Agreement": (
                all_data["human_majority"] == all_data["evaluator_response"]
            ).mean(),
            "Kappa": cohen_kappa_score(
                all_data["human_majority"], all_data["evaluator_response"]
            ),
            "Precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "Recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "F1": (
                2
                * (tp / (tp + fp))
                * (tp / (tp + fn))
                / ((tp / (tp + fp)) + (tp / (tp + fn)))
                if ((tp / (tp + fp)) + (tp / (tp + fn))) > 0
                else 0
            ),
        }
    )

    # Then analyze each category
    for category in comparison_df["category"].unique():
        if category != "unknown":
            cat_data = comparison_df[comparison_df["category"] == category]
            if len(cat_data) < 2:
                continue

            cat_cm = confusion_matrix(
                cat_data["human_majority"],
                cat_data["evaluator_response"],
                labels=[0, 1],
            )
            tn, fp, fn, tp = cat_cm.ravel()

            try:
                kappa = cohen_kappa_score(
                    cat_data["human_majority"], cat_data["evaluator_response"]
                )
            except ValueError:
                kappa = float("nan")

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            summary_stats.append(
                {
                    "Category": category,
                    "Examples": len(cat_data),
                    "Agreement": (
                        cat_data["human_majority"] == cat_data["evaluator_response"]
                    ).mean(),
                    "Kappa": kappa,
                    "Precision": precision,
                    "Recall": recall,
                    "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
                    "F1": f1,
                }
            )

    # Save summary to CSV
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv("graphs/majority_vote_results.csv", index=False)


def calculate_fleiss_kappa(df: pd.DataFrame):
    """
    Calculate Fleiss' Kappa coefficient for human annotators only.
    This measures the agreement among multiple raters, giving a sense of
    human consensus reliability.
    """
    print("\nCalculating Fleiss' Kappa for human annotators...")

    # Extract human responses and organize by example ID and category
    examples_by_id = {}
    example_categories = {}
    annotators = set()

    for idx, row in df.iterrows():
        # Skip if no human responses
        if not isinstance(row["human_responses"], Iterable):
            continue

        example_id = row["id"]

        # Store category for this example
        if isinstance(row["topic_category"], list):
            example_categories[example_id] = row["topic_category"][0]
        else:
            example_categories[example_id] = row["topic_category"]

        # Store all human responses for this example
        if example_id not in examples_by_id:
            examples_by_id[example_id] = {}

        for response in row["human_responses"]:
            annotator = response["annotator"]
            annotators.add(annotator)

            # Convert response to binary (0 for "no", 1 for "yes")
            if isinstance(response["response"], str):
                binary_response = 1 if response["response"].lower() == "yes" else 0
            else:
                binary_response = response["response"]

            examples_by_id[example_id][annotator] = binary_response

    # Convert to a format suitable for Fleiss' Kappa calculation
    # Create a dataframe with examples as rows and annotators as columns
    annotator_df = pd.DataFrame.from_dict(examples_by_id, orient="index")
    annotator_df["category"] = pd.Series(example_categories)

    # Function to calculate Fleiss' Kappa
    def compute_fleiss_kappa(ratings, n_raters):
        """
        Compute Fleiss' Kappa for inter-rater agreement.

        Parameters:
        ratings: Matrix where rows are items and columns are categories (0 and 1)
        n_raters: Number of raters

        Returns:
        Fleiss' Kappa value
        """
        # Number of items (examples)
        n_items = ratings.shape[0]

        # Number of categories (0 and 1)
        n_categories = ratings.shape[1]

        # Calculate P_i (proportion of agreement for each item)
        p_i = []
        for i in range(n_items):
            p_i_sum = 0
            for j in range(n_categories):
                p_i_sum += ratings.iloc[i, j] * (ratings.iloc[i, j] - 1)
            p_i.append(p_i_sum / (n_raters * (n_raters - 1)))

        # Mean agreement across all items
        P_bar = np.mean(p_i)

        # Calculate P_j (proportion of ratings in each category)
        p_j = ratings.sum(axis=0) / (n_items * n_raters)

        # Expected agreement by chance
        P_e_bar = np.sum(p_j**2)

        # Fleiss' Kappa
        kappa = (P_bar - P_e_bar) / (1 - P_e_bar)

        return kappa

    # Function to prepare data for Fleiss' Kappa calculation
    def prepare_data_for_kappa(df, annotators):
        """Convert from annotator columns to category columns for Fleiss Kappa"""
        len(df)
        n_raters = len(annotators)
        ratings = pd.DataFrame(0, index=df.index, columns=[0, 1])

        for item_idx in df.index:
            rater_responses = df.loc[item_idx, annotators].dropna()
            if len(rater_responses) == 0:
                continue

            # Count ratings in each category
            for category in [0, 1]:
                ratings.loc[item_idx, category] = (rater_responses == category).sum()

        return ratings, n_raters

    # Calculate overall Fleiss' Kappa
    all_ratings, n_raters = prepare_data_for_kappa(annotator_df, list(annotators))
    all_kappa = compute_fleiss_kappa(all_ratings, n_raters)
    print(f"Overall Fleiss' Kappa: {all_kappa:.4f}")

    # Prepare data for visualization
    kappa_results = []

    # Add overall result
    kappa_results.append(
        {
            "Category": "All Categories",
            "Fleiss Kappa": all_kappa,
            "Number of Examples": len(all_ratings),
            "Number of Raters": n_raters,
        }
    )

    # Calculate Fleiss' Kappa by category
    for category in annotator_df["category"].unique():
        category_df = annotator_df[annotator_df["category"] == category]
        if len(category_df) < 2:  # Need at least 2 examples
            continue

        cat_ratings, cat_n_raters = prepare_data_for_kappa(
            category_df, list(annotators)
        )
        if cat_n_raters < 2:  # Need at least 2 raters
            continue

        cat_kappa = compute_fleiss_kappa(cat_ratings, cat_n_raters)
        print(f"Category: {category}, Fleiss' Kappa: {cat_kappa:.4f}")

        kappa_results.append(
            {
                "Category": category,
                "Fleiss Kappa": cat_kappa,
                "Number of Examples": len(cat_ratings),
                "Number of Raters": cat_n_raters,
            }
        )

    # Create a DataFrame from results
    kappa_df = pd.DataFrame(kappa_results)

    # Visualize results
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Create kappa score bar chart
    ax = sns.barplot(x="Category", y="Fleiss Kappa", data=kappa_df)

    # Add value labels on bars
    for i, v in enumerate(kappa_df["Fleiss Kappa"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)

    # Add a horizontal line for reference (0.4 is often considered a moderate agreement threshold)
    plt.axhline(y=0.4, color="r", linestyle="-", alpha=0.3)
    plt.text(len(kappa_df) - 0.5, 0.41, "Moderate (0.4)", color="r", alpha=0.8)

    plt.axhline(y=0.6, color="g", linestyle="-", alpha=0.3)
    plt.text(len(kappa_df) - 0.5, 0.61, "Substantial (0.6)", color="g", alpha=0.8)

    plt.axhline(y=0.8, color="b", linestyle="-", alpha=0.3)
    plt.text(len(kappa_df) - 0.5, 0.81, "Almost Perfect (0.8)", color="b", alpha=0.8)

    # Add labels and title
    plt.title("Fleiss' Kappa Coefficient for Human Annotators by Category", fontsize=14)
    plt.ylabel("Fleiss' Kappa", fontsize=12)
    plt.xlabel("Category", fontsize=12)
    plt.ylim(
        0, 1.0
    )  # Kappa ranges from -1 to 1, but typically 0 to 1 for positive agreement

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("graphs/fleiss_kappa_by_category.png")
    plt.close()

    # Add sample size information to the DataFrame
    for i, row in kappa_df.iterrows():
        print(
            f"{row['Category']}: κ = {row['Fleiss Kappa']:.3f}, n = {row['Number of Examples']}, raters = {row['Number of Raters']}"
        )

    # Save to CSV
    kappa_df.to_csv("graphs/fleiss_kappa_results.csv", index=False)
    print("Saved Fleiss' Kappa results to graphs/fleiss_kappa_results.csv")

    # Create a reliability interpretation figure
    plt.figure(figsize=(12, 6))

    # Define interpretation levels
    interpretation = [
        {"level": "Poor", "min": 0.00, "max": 0.20, "color": "red"},
        {"level": "Fair", "min": 0.20, "max": 0.40, "color": "orange"},
        {"level": "Moderate", "min": 0.40, "max": 0.60, "color": "yellow"},
        {"level": "Substantial", "min": 0.60, "max": 0.80, "color": "lightgreen"},
        {"level": "Almost Perfect", "min": 0.80, "max": 1.00, "color": "green"},
    ]

    # Create interpretation axis
    ax = plt.subplot(111)

    # Draw colored bands for each level
    for i, level in enumerate(interpretation):
        ax.axvspan(
            level["min"],
            level["max"],
            alpha=0.3,
            color=level["color"],
            label=level["level"],
        )

    # Plot each category's kappa value as a vertical line
    for i, row in kappa_df.iterrows():
        plt.axvline(
            x=row["Fleiss Kappa"],
            color="black",
            linestyle="-",
            label=f"{row['Category']} (κ = {row['Fleiss Kappa']:.3f})",
        )
        plt.text(
            row["Fleiss Kappa"],
            0.95 - (i * 0.05),
            f"{row['Category']}",
            ha="center",
            va="top",
        )

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Fleiss' Kappa Value", fontsize=12)
    plt.title("Interpretation of Fleiss' Kappa Values", fontsize=14)

    # Remove y-axis ticks
    plt.yticks([])

    # Show legend for interpretation levels
    handles, labels = [], []
    for level in interpretation:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=level["color"], alpha=0.3))
        labels.append(f"{level['level']} ({level['min']:.1f}-{level['max']:.1f})")

    plt.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=5,
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig("graphs/fleiss_kappa_interpretation.png")
    plt.close()


def main():
    # Load data
    df = load_data("merged_dataset.json")

    # Create all visualizations
    plot_agreement_heatmap(create_agreement_matrix(df))
    plot_stacked_bars(df)
    calculate_kappa_scores(df)
    plot_agreement_by_turn(df)
    plot_confusion_matrices(df)
    plot_combined_human_vs_evaluator(df)
    plot_evaluator_vs_human_majority(df)
    calculate_fleiss_kappa(df)


if __name__ == "__main__":
    main()
