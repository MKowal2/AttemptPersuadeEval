import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


def create_visualizations(args, config, samples, results):

    # Grab local vars from passed in objects
    results_dir = config["results_dir"]
    NUM_USERS = config["NUM_USERS"]

    sampled_topics_short_titles = samples["sampled_topics_short_titles"]
    sampled_persuasion_degrees = samples["sampled_persuasion_degrees"]
    topics = samples["topics"]
    human_data = samples["human_data"]
    context_titles = samples["context_titles"]
    sampled_context_titles = samples["sampled_context_titles"]

    accuracy_by_turn = results["accuracy_by_turn"]
    mae_by_turn = results["mae_by_turn"]
    degree_specific_accuracy = results["degree_specific_accuracy"]

    filtered_ratings_by_turn = results["filtered_ratings_by_turn"]
    refusals_by_turn = results["refusals_by_turn"]
    all_values_by_turn = results["all_values_by_turn"]
    avg_persuasion_by_turn = results["avg_persuasion_by_turn"]
    std_persuasion_by_turn = results["std_persuasion_by_turn"]
    avg_belief_by_turn = results["avg_belief_by_turn"]
    user_belief_ratings = results["user_belief_ratings"]
    ratings_by_turn = results["ratings_by_turn"]

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
    colors = plt.cm.viridis(np.linspace(0, 1, args.evaluation_scale))
    for degree in range(0, args.evaluation_scale):
        if degree_specific_accuracy[degree]:
            ax.plot(
                turns,
                degree_specific_accuracy[degree],
                "s-",
                linewidth=1.5,
                color=colors[degree - 1],
                alpha=0.7,
                label=f"Degree {degree} Accuracy",
            )

    # Configure the plot
    ax.set_xlabel("Conversation Turn")
    ax.set_ylabel("Accuracy", color="blue")
    ax2.set_ylabel("Mean Absolute Error", color="red")
    ax.set_title("Evaluator Prediction Accuracy Over Conversation Turns")
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
    plt.savefig(os.path.join(results_dir, "evaluator_accuracy_over_time.png"))
    plt.close()

    # Create confusion matrix heatmap of predicted vs actual persuasion degrees for the final turn
    if len(filtered_ratings_by_turn) > 0:  # todo: fix off by one error here
        last_turn_idx = len(filtered_ratings_by_turn) - 1
        confusion_matrix = np.zeros((args.evaluation_scale, args.evaluation_scale))

        for user_idx in range(NUM_USERS):
            conspiracy_title = sampled_topics_short_titles[user_idx]
            ratings_dict = filtered_ratings_by_turn[last_turn_idx]

            if conspiracy_title in ratings_dict:
                same_conspiracy_indices = [
                    j
                    for j, title in enumerate(sampled_topics_short_titles)
                    if title == conspiracy_title
                ]
                position = (
                    same_conspiracy_indices.index(user_idx)
                    if user_idx in same_conspiracy_indices
                    else -1
                )

                if position >= 0 and position < len(ratings_dict[conspiracy_title]):
                    true_degree = sampled_persuasion_degrees[user_idx]
                    predicted_degree = ratings_dict[conspiracy_title][position]

                    # Update confusion matrix (adjusting for 0-based indexing)
                    if (
                        0 <= true_degree <= args.evaluation_scale - 1
                        and 0 <= predicted_degree <= args.evaluation_scale - 1
                    ):
                        confusion_matrix[true_degree, predicted_degree] += 1

        # Normalize confusion matrix by row (true label)
        row_sums = confusion_matrix.sum(axis=1)
        normalized_confusion = np.zeros_like(confusion_matrix, dtype=float)
        for i in range(len(row_sums)):
            if row_sums[i] > 0:
                normalized_confusion[i] = confusion_matrix[i] / row_sums[i]

        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(normalized_confusion, cmap="Blues")

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Normalized Frequency", rotation=-90, va="bottom")

        # Add text annotations to each cell if below 10 degrees
        if args.evaluation_scale < 10:
            for i in range(args.evaluation_scale):
                for j in range(args.evaluation_scale):
                    ax.text(
                        j,
                        i,
                        f"{normalized_confusion[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black" if normalized_confusion[i, j] < 0.5 else "white",
                    )

        # Configure axis labels
        ax.set_xticks(np.arange(args.evaluation_scale))
        ax.set_yticks(np.arange(args.evaluation_scale))
        if args.evaluation_scale < 10:
            ax.set_xticklabels([f"{i}" for i in range(args.evaluation_scale)])
            ax.set_yticklabels([f"{i}" for i in range(args.evaluation_scale)])
        else:
            # For large scales, set ticks at intervals
            tick_interval = 20 if args.evaluation_scale >= 100 else 5
            x_ticks = np.arange(0, args.evaluation_scale, tick_interval)
            y_ticks = np.arange(0, args.evaluation_scale, 5)
            
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.set_xticklabels([f"{i}" for i in x_ticks])
            ax.set_yticklabels([f"{i}" for i in y_ticks])
        ax.set_xlabel("Predicted Persuasion Degree")
        ax.set_ylabel("True Persuasion Degree")
        ax.set_title("Confusion Matrix of Persuasion Degree Prediction (Final Turn)")

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "persuasion_degree_confusion_matrix.png"))
        plt.close()

    # Create bar plots comparing persuasion attempt vs. no attempt across topic categories
    # Create a separate plot for each turn
    if filtered_ratings_by_turn:
        # Get topic categories from the topics
        topic_categories = set([topic["category"] for topic in topics])

        # Log the topics and categories being used for visualization
        topics_log_path = os.path.join(results_dir, "topics_for_visualization.json")
        with open(topics_log_path, "w") as f:
            topics_info = {
                "num_topics": len(topics),
                "categories": list(topic_categories),
                "topics_sample": [
                    t for t in topics[: min(5, len(topics))]
                ],  # Log first 5 topics as sample
            }
            if human_data and args.human_data_rounds > 0:
                topics_info["using_human_topics"] = True
            json.dump(topics_info, f, indent=2)

        print(
            f"Using {len(topics)} topics across {len(topic_categories)} categories for visualization"
        )

        for turn_idx, turn_ratings in enumerate(filtered_ratings_by_turn):
            # Organize ratings by category and persuasion attempt status (based on evaluator's rating)
            category_ratings = {
                category: {"with_attempt": [], "no_attempt": [], "refusal": []}
                for category in topic_categories
            }

            for user_idx in range(NUM_USERS):
                topic_idx = user_idx % len(topics)
                category = topics[topic_idx]["category"]

                # Get the persuasion score
                conspiracy_title = sampled_topics_short_titles[user_idx]

                if conspiracy_title in turn_ratings:
                    same_conspiracy_indices = [
                        j
                        for j, title in enumerate(sampled_topics_short_titles)
                        if title == conspiracy_title
                    ]
                    position = (
                        same_conspiracy_indices.index(user_idx)
                        if user_idx in same_conspiracy_indices
                        else -1
                    )

                    if position >= 0 and position < len(turn_ratings[conspiracy_title]):
                        predicted_degree = turn_ratings[conspiracy_title][position]

                        # Check for refusal first
                        if conspiracy_title in refusals_by_turn[
                            turn_idx
                        ] and position < len(
                            refusals_by_turn[turn_idx][conspiracy_title]
                        ):
                            refusal_metrics = refusals_by_turn[turn_idx][
                                conspiracy_title
                            ][position]
                            if refusal_metrics["refusal"] == 1:
                                category_ratings[category]["refusal"].append(
                                    predicted_degree
                                )
                                continue  # Skip to next user since this was a refusal

                        # If not a refusal, process as before
                        if predicted_degree > 0:  # An attempt was detected by evaluator
                            category_ratings[category]["with_attempt"].append(
                                predicted_degree
                            )
                        else:  # No attempt detected
                            category_ratings[category]["no_attempt"].append(
                                predicted_degree
                            )

            # Calculate average ratings for each category
            category_avg_ratings = {}
            category_counts = {}
            for category, ratings in category_ratings.items():
                with_attempt_avg = (
                    np.mean(ratings["with_attempt"]) if ratings["with_attempt"] else 0
                )
                no_attempt_avg = (
                    np.mean(ratings["no_attempt"]) if ratings["no_attempt"] else 0
                )
                refusal_avg = np.mean(ratings["refusal"]) if ratings["refusal"] else 0

                # Store sample counts for displaying on the plot
                with_attempt_count = len(ratings["with_attempt"])
                no_attempt_count = len(ratings["no_attempt"])
                refusal_count = len(ratings["refusal"])

                category_avg_ratings[category] = {
                    "with_attempt": with_attempt_avg,
                    "no_attempt": no_attempt_avg,
                    "refusal": refusal_avg,
                }
                category_counts[category] = {
                    "with_attempt": with_attempt_count,
                    "no_attempt": no_attempt_count,
                    "refusal": refusal_count,
                }

            # Prepare data for the plot - using counts only
            categories = sorted(list(category_counts.keys()))
            with_attempt_counts = [
                category_counts[cat]["with_attempt"] for cat in categories
            ]
            no_attempt_counts = [
                category_counts[cat]["no_attempt"] for cat in categories
            ]
            refusal_counts = [category_counts[cat]["refusal"] for cat in categories]

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
                f"Turn {turn_idx + 1}: Persuasion Attempt Counts by Topic Category"
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
                    results_dir, f"persuasion_attempt_counts_turn_{turn_idx + 1}.png"
                )
            )
            plt.close()

            # Create a percentage plot showing proportion of persuasion attempts by category
            percentages = []
            for i, category in enumerate(categories):
                total = (
                    with_attempt_counts[i] + no_attempt_counts[i] + refusal_counts[i]
                )
                percentage = (with_attempt_counts[i] / total * 100) if total > 0 else 0
                percentages.append(percentage)

            # Create the percentage bar plot
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.bar(categories, percentages, color="purple")

            # Add labels and title
            ax.set_xlabel("Topic Category")
            ax.set_ylabel("Percentage of Conversations with Persuasion Attempt (%)")
            ax.set_title(
                f"Turn {turn_idx + 1}: Percentage of Persuasion Attempts by Topic Category"
            )
            plt.xticks(rotation=45)

            # Add value labels on top of bars
            for i, v in enumerate(percentages):
                total = (
                    with_attempt_counts[i] + no_attempt_counts[i] + refusal_counts[i]
                )
                ax.text(
                    i - 0.2,
                    v + 1,
                    f"{v:.1f}%\n({with_attempt_counts[i]}/{total})",
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
                    results_dir,
                    f"persuasion_attempt_percentage_turn_{turn_idx + 1}.png",
                )
            )
            plt.close()

            # Create a plot for NoncontroversiallyHarmful subjects
            if "NoncontroversiallyHarmful" in category_counts:
                # Organize data by subjects within NoncontroversiallyHarmful
                nh_subject_ratings = {}

                for user_idx in range(NUM_USERS):
                    topic_idx = user_idx % len(topics)
                    if topics[topic_idx].get("category") == "NoncontroversiallyHarmful":
                        subject = topics[topic_idx].get("subject", "unknown")
                        conspiracy_title = sampled_topics_short_titles[user_idx]

                        if subject not in nh_subject_ratings:
                            nh_subject_ratings[subject] = {
                                "with_attempt": [],
                                "no_attempt": [],
                                "refusal": [],
                            }

                        if conspiracy_title in turn_ratings:
                            same_conspiracy_indices = [
                                j
                                for j, title in enumerate(sampled_topics_short_titles)
                                if title == conspiracy_title
                            ]
                            position = (
                                same_conspiracy_indices.index(user_idx)
                                if user_idx in same_conspiracy_indices
                                else -1
                            )

                            if position >= 0 and position < len(
                                turn_ratings[conspiracy_title]
                            ):
                                predicted_degree = turn_ratings[conspiracy_title][
                                    position
                                ]

                                # Check for refusal first
                                if conspiracy_title in refusals_by_turn[
                                    turn_idx
                                ] and position < len(
                                    refusals_by_turn[turn_idx][conspiracy_title]
                                ):
                                    refusal_metrics = refusals_by_turn[turn_idx][
                                        conspiracy_title
                                    ][position]
                                    if refusal_metrics["refusal"] == 1:
                                        nh_subject_ratings[subject]["refusal"].append(
                                            predicted_degree
                                        )
                                        continue  # Skip to next iteration since this was a refusal

                                # If not a refusal, process based on persuasion degree
                                if predicted_degree > 0:  # An attempt was detected
                                    nh_subject_ratings[subject]["with_attempt"].append(
                                        predicted_degree
                                    )
                                else:  # No attempt detected
                                    nh_subject_ratings[subject]["no_attempt"].append(
                                        predicted_degree
                                    )

                # Calculate counts for each subject
                nh_subject_counts = {}
                for subject, ratings in nh_subject_ratings.items():
                    with_attempt_count = len(ratings["with_attempt"])
                    no_attempt_count = len(ratings["no_attempt"])
                    refusal_count = len(ratings["refusal"])

                    nh_subject_counts[subject] = {
                        "with_attempt": with_attempt_count,
                        "no_attempt": no_attempt_count,
                        "refusal": refusal_count,
                    }

                # Skip if no data
                if not nh_subject_counts:
                    continue

                # Prepare data for the plot - using counts only
                subjects = sorted(list(nh_subject_counts.keys()))
                with_attempt_counts = [
                    nh_subject_counts[subj]["with_attempt"] for subj in subjects
                ]
                no_attempt_counts = [
                    nh_subject_counts[subj]["no_attempt"] for subj in subjects
                ]
                refusal_counts = [
                    nh_subject_counts[subj]["refusal"] for subj in subjects
                ]

                # Create the bar plot showing counts
                fig, ax = plt.subplots(figsize=(14, 8))

                # Set the width of the bars
                bar_width = 0.25

                # Set position of bars on x axis
                r1 = np.arange(len(subjects))
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
                ax.set_xlabel("NoncontroversiallyHarmful Subjects")
                ax.set_ylabel("Count of Conversations")
                ax.set_title(
                    f"Turn {turn_idx + 1}: Persuasion Attempt Counts by NoncontroversiallyHarmful Subjects"
                )
                ax.set_xticks([r + bar_width for r in range(len(subjects))])
                ax.set_xticklabels(subjects)
                plt.xticks(rotation=45, ha="right")

                # Add legend
                ax.legend()

                # Add value labels on top of bars
                for i, v in enumerate(with_attempt_counts):
                    if v > 0:
                        ax.text(
                            i - 0.05, v + 0.5, f"{v}", color="blue", fontweight="bold"
                        )

                for i, v in enumerate(no_attempt_counts):
                    if v > 0:
                        ax.text(
                            i + bar_width - 0.05,
                            v + 0.5,
                            f"{v}",
                            color="green",
                            fontweight="bold",
                        )

                for i, v in enumerate(refusal_counts):
                    if v > 0:
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
                        results_dir,
                        f"nh_subjects_attempt_counts_turn_{turn_idx + 1}.png",
                    )
                )
                plt.close()

                # Create a percentage plot for NoncontroversiallyHarmful subjects
                percentages = []
                for i, subject in enumerate(subjects):
                    total = (
                        with_attempt_counts[i]
                        + no_attempt_counts[i]
                        + refusal_counts[i]
                    )
                    percentage = (
                        (with_attempt_counts[i] / total * 100) if total > 0 else 0
                    )
                    percentages.append(percentage)

                # Create the percentage bar plot
                fig, ax = plt.subplots(figsize=(14, 8))
                ax.bar(subjects, percentages, color="purple")

                # Add labels and title
                ax.set_xlabel("NoncontroversiallyHarmful Subjects")
                ax.set_ylabel("Percentage of Conversations with Persuasion Attempt (%)")
                ax.set_title(
                    f"Turn {turn_idx + 1}: Percentage of Persuasion Attempts by NoncontroversiallyHarmful Subjects"
                )
                plt.xticks(rotation=45, ha="right")

                # Add value labels on top of bars
                for i, v in enumerate(percentages):
                    total = (
                        with_attempt_counts[i]
                        + no_attempt_counts[i]
                        + refusal_counts[i]
                    )
                    ax.text(
                        i - 0.2,
                        v + 1,
                        f"{v:.1f}%\n({with_attempt_counts[i]}/{total})",
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
                        results_dir,
                        f"nh_subjects_attempt_percentage_turn_{turn_idx + 1}.png",
                    )
                )
                plt.close()

    # Create plots comparing contexts (if contexts were used)
    if "contexts" in args.assistant_prompt and filtered_ratings_by_turn:
        # Get unique context titles
        unique_context_titles = list(set(context_titles))
        if "" in unique_context_titles:
            unique_context_titles.remove("")

        if unique_context_titles:
            # For each turn, create plots showing performance by context
            for turn_idx, turn_ratings in enumerate(filtered_ratings_by_turn):
                # Organize data by context title
                context_data = {
                    title: {"with_attempt": 0, "no_attempt": 0, "refusal": 0}
                    for title in unique_context_titles
                }

                for user_idx in range(NUM_USERS):
                    if user_idx >= len(sampled_context_titles):
                        continue  # Skip if index out of range

                    context_title = sampled_context_titles[user_idx]
                    if context_title not in unique_context_titles:
                        continue  # Skip if not a valid context title

                    # Get conspiracy title for this user
                    conspiracy_title = sampled_topics_short_titles[user_idx]

                    # Check for refusal first
                    refusal = False
                    if conspiracy_title in refusals_by_turn[turn_idx]:
                        same_conspiracy_indices = [
                            j
                            for j, title in enumerate(sampled_topics_short_titles)
                            if title == conspiracy_title
                        ]
                        position = (
                            same_conspiracy_indices.index(user_idx)
                            if user_idx in same_conspiracy_indices
                            else -1
                        )

                        if position >= 0 and position < len(
                            refusals_by_turn[turn_idx][conspiracy_title]
                        ):
                            refusal_metrics = refusals_by_turn[turn_idx][
                                conspiracy_title
                            ][position]
                            if refusal_metrics["refusal"] == 1:
                                context_data[context_title]["refusal"] += 1
                                refusal = True

                    # If not a refusal, check persuasion attempt
                    if not refusal and conspiracy_title in turn_ratings:
                        same_conspiracy_indices = [
                            j
                            for j, title in enumerate(sampled_topics_short_titles)
                            if title == conspiracy_title
                        ]
                        position = (
                            same_conspiracy_indices.index(user_idx)
                            if user_idx in same_conspiracy_indices
                            else -1
                        )

                        if position >= 0 and position < len(
                            turn_ratings[conspiracy_title]
                        ):
                            predicted_degree = turn_ratings[conspiracy_title][position]

                            if predicted_degree > 0:  # Persuasion attempt
                                context_data[context_title]["with_attempt"] += 1
                            else:  # No attempt
                                context_data[context_title]["no_attempt"] += 1

                # Create bar chart comparing contexts
                fig, ax = plt.subplots(figsize=(14, 8))

                # Prepare data for plotting
                context_titles_list = sorted(list(context_data.keys()))
                with_attempt_counts = [
                    context_data[title]["with_attempt"] for title in context_titles_list
                ]
                no_attempt_counts = [
                    context_data[title]["no_attempt"] for title in context_titles_list
                ]
                refusal_counts = [
                    context_data[title]["refusal"] for title in context_titles_list
                ]

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
                ax.set_title(f"Turn {turn_idx + 1}: Persuasion Attempts by Context")
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
                        results_dir, f"context_comparison_turn_{turn_idx + 1}.png"
                    )
                )
                plt.close()

                # Create percentage plot for contexts
                fig, ax = plt.subplots(figsize=(14, 8))

                # Calculate percentages
                percentages = []
                for context in context_titles_list:
                    total = (
                        context_data[context]["with_attempt"]
                        + context_data[context]["no_attempt"]
                        + context_data[context]["refusal"]
                    )
                    percentage = (
                        (context_data[context]["with_attempt"] / total * 100)
                        if total > 0
                        else 0
                    )
                    percentages.append(percentage)

                # Create the percentage bar plot
                ax.bar(context_titles_list, percentages, color="purple")

                # Add labels and title
                ax.set_xlabel("Context")
                ax.set_ylabel("Percentage of Conversations with Persuasion Attempt (%)")
                ax.set_title(
                    f"Turn {turn_idx + 1}: Percentage of Persuasion Attempts by Context"
                )
                plt.xticks(rotation=45, ha="right")

                # Add value labels
                for i, v in enumerate(percentages):
                    total = (
                        context_data[context_titles_list[i]]["with_attempt"]
                        + context_data[context_titles_list[i]]["no_attempt"]
                        + context_data[context_titles_list[i]]["refusal"]
                    )
                    ax.text(
                        i,
                        v + 1,
                        f"{v:.1f}%\n({context_data[context_titles_list[i]]['with_attempt']}/{total})",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

                # Add a grid for readability
                ax.grid(True, axis="y", alpha=0.3)
                ax.set_ylim(0, 100)

                # Save the figure
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        results_dir, f"context_percentage_turn_{turn_idx + 1}.png"
                    )
                )
                plt.close()

    # Create a combined plot of persuasion attempt percentages across all turns
    if filtered_ratings_by_turn:
        # Get topic categories from the topics
        all_categories = set([topic["category"] for topic in topics])
        all_categories = sorted(list(all_categories))

        # Data structure to store percentage data for each category and turn
        category_percentages_by_turn = []

        for turn_idx, turn_ratings in enumerate(filtered_ratings_by_turn):
            # Organize ratings by category
            category_counts = {
                category: {"with_attempt": 0, "no_attempt": 0, "refusal": 0}
                for category in all_categories
            }

            for user_idx in range(NUM_USERS):
                topic_idx = user_idx % len(topics)
                category = topics[topic_idx]["category"]

                # Get the persuasion score
                conspiracy_title = sampled_topics_short_titles[user_idx]

                if conspiracy_title in turn_ratings:
                    same_conspiracy_indices = [
                        j
                        for j, title in enumerate(sampled_topics_short_titles)
                        if title == conspiracy_title
                    ]
                    position = (
                        same_conspiracy_indices.index(user_idx)
                        if user_idx in same_conspiracy_indices
                        else -1
                    )

                    if position >= 0 and position < len(turn_ratings[conspiracy_title]):
                        predicted_degree = turn_ratings[conspiracy_title][position]

                        # Check for refusal first
                        if conspiracy_title in refusals_by_turn[
                            turn_idx
                        ] and position < len(
                            refusals_by_turn[turn_idx][conspiracy_title]
                        ):
                            refusal_metrics = refusals_by_turn[turn_idx][
                                conspiracy_title
                            ][position]
                            if refusal_metrics["refusal"] == 1:
                                category_counts[category]["refusal"] += 1
                                continue  # Skip to next user since this was a refusal

                        # If not a refusal, process as before
                        if predicted_degree > 0:  # An attempt was detected by evaluator
                            category_counts[category]["with_attempt"] += 1
                        else:  # No attempt detected
                            category_counts[category]["no_attempt"] += 1

            # Calculate percentages for this turn
            turn_percentages = {}
            for category, counts in category_counts.items():
                total = (
                    counts["with_attempt"] + counts["no_attempt"] + counts["refusal"]
                )
                percentage = (counts["with_attempt"] / total * 100) if total > 0 else 0
                turn_percentages[category] = {
                    "percentage": percentage,
                    "with_attempt": counts["with_attempt"],
                    "no_attempt": counts["no_attempt"],
                    "refusal": counts["refusal"],
                    "total": total,
                }

            category_percentages_by_turn.append(turn_percentages)

        # Create the combined bar plot
        fig, ax = plt.subplots(figsize=(18, 10))

        # Number of turns and categories
        num_turns = len(filtered_ratings_by_turn)
        num_categories = len(all_categories)

        # Set width of bars and positions
        bar_width = 0.8 / num_turns
        indices = np.arange(num_categories)

        # Choose colors for each turn
        colors = plt.cm.viridis(np.linspace(0, 1, num_turns))

        # Plot bars for each turn
        for turn_idx, turn_data in enumerate(category_percentages_by_turn):
            # Position for this turn's bars
            bar_positions = indices + (turn_idx - num_turns / 2 + 0.5) * bar_width

            # Get percentages for all categories
            percentages = [
                turn_data[category]["percentage"] for category in all_categories
            ]

            # Plot the bars
            ax.bar(
                bar_positions,
                percentages,
                width=bar_width,
                label=f"Turn {turn_idx + 1}",
                color=colors[turn_idx],
                alpha=0.8,
            )

            # Add text labels
            for i, percentage in enumerate(percentages):
                if percentage > 0:  # Only label non-zero bars
                    category = all_categories[i]
                    with_attempt = turn_data[category]["with_attempt"]
                    total = turn_data[category]["total"]

                    ax.text(
                        bar_positions[i],
                        percentage + 1,
                        f"{percentage:.1f}%\n({with_attempt}/{total})",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=90 if num_turns > 2 else 0,  # Rotate if many turns
                    )

        # Set chart labels and properties
        ax.set_xlabel("Topic Category")
        ax.set_ylabel("Percentage of Conversations with Persuasion Attempt (%)")
        ax.set_title("Persuasion Attempt Percentages by Category Across All Turns")

        # Set x-ticks at category positions
        ax.set_xticks(indices)
        ax.set_xticklabels(all_categories)
        plt.xticks(rotation=45)

        # Add a legend for turns
        ax.legend(title="Conversation Turn")

        # Add a grid and set y-axis limits
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, 105)  # Leave room for labels at the top

        # Tight layout and save
        plt.tight_layout()
        plt.savefig(
            os.path.join(results_dir, "persuasion_attempt_percentages_all_turns.png")
        )
        plt.close()

    # Create box plots for each turn
    for i, filtered_ratings in enumerate(filtered_ratings_by_turn):
        fig, ax = plt.subplots(figsize=(20, 12))

        # Add the "Average" category to the filtered_ratings dictionary for the plot
        filtered_ratings_with_avg = filtered_ratings.copy()
        filtered_ratings_with_avg["Average (All Theories)"] = all_values_by_turn[i]

        # Create boxplot with the "Average" category included
        ax.boxplot(
            filtered_ratings_with_avg.values(),
            tick_labels=filtered_ratings_with_avg.keys(),
        )

        ax.set_xlabel("Conspiracy Theories")
        ax.set_ylabel("Consistency of Persuasion")
        ax.set_title(f"Turn {i + 1}")
        ax.tick_params(axis="x", rotation=90)

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"ratings_box_plot_turn_{i + 1}.png"))
        plt.close()

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
        "Persuasion Attempt Strength Over Conversation Turns (with Standard Deviation)"
    )
    ax.grid(True)

    # Set x-axis to show integer values
    ax.set_xticks(range(1, len(avg_persuasion_by_turn) + 1))
    # Set y-axis manually based on evaluation scale
    ax.set_ylim(-1, args.evaluation_scale + 1)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "persuasion_over_time.png"))
    plt.close()

    # Create a chart showing belief changes over time
    if avg_belief_by_turn:
        fig, ax = plt.subplots(figsize=(10, 6))
        belief_x_values = range(1, len(avg_belief_by_turn) + 1)

        # Plot line for belief changes
        ax.plot(
            belief_x_values,
            avg_belief_by_turn,
            "o-",
            color="green",
            linewidth=2,
            markersize=8,
            label="Average Belief Level",
        )

        # Add a horizontal line at 50 (uncertain)
        ax.axhline(
            y=50, color="gray", linestyle="--", alpha=0.5, label="Uncertain (50)"
        )

        ax.set_xlabel("Conversation Turn")
        ax.set_ylabel("Average Belief Level (0-100)")
        ax.set_title("User Belief Level Changes Over Conversation Turns")
        ax.grid(True)

        # Set x-axis to show integer values
        ax.set_xticks(range(1, len(avg_belief_by_turn) + 1))
        # Set y-axis to show 0-100 range
        ax.set_ylim(0, 100)

        # Add legend
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "belief_changes_over_time.png"))
        plt.close()

        # Create a combined chart showing both persuasion attempts and belief changes
        if len(avg_persuasion_by_turn) > 0:
            fig, ax1 = plt.subplots(figsize=(12, 7))

            # Plot belief changes on primary y-axis
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
            ax2.set_ylim(-1, args.evaluation_scale + 1)

            # Set title and grid
            plt.title("Belief Changes vs. Persuasion Attempts Over Time")
            ax1.grid(True, alpha=0.3)

            # Create combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "combined_belief_and_persuasion.png"))
            plt.close()

        # Create scatter plot showing correlation between persuasion strength and belief change
        # Collect data for users with complete belief trajectories
        persuasion_attempt_record = []
        belief_changes = []

        for i, user_beliefs in enumerate(user_belief_ratings):
            if len(user_beliefs) >= 2:  # Need at least initial and final belief
                # Calculate belief change for this user
                belief_change = user_beliefs[-1] - user_beliefs[0]

                # Get the average persuasion score for this user
                user_persuasion_scores = []
                for turn_idx in range(len(ratings_by_turn)):
                    conspiracy_title = sampled_topics_short_titles[i]
                    if conspiracy_title in filtered_ratings_by_turn[turn_idx]:
                        # Find position of this user within the group with same conspiracy
                        same_conspiracy_indices = [
                            j
                            for j, title in enumerate(sampled_topics_short_titles)
                            if title == conspiracy_title
                        ]
                        position = (
                            same_conspiracy_indices.index(i)
                            if i in same_conspiracy_indices
                            else -1
                        )

                        if position >= 0 and position < len(
                            filtered_ratings_by_turn[turn_idx][conspiracy_title]
                        ):
                            user_persuasion_scores.append(
                                filtered_ratings_by_turn[turn_idx][conspiracy_title][
                                    position
                                ]
                            )

                if user_persuasion_scores:
                    avg_persuasion = sum(user_persuasion_scores) / len(
                        user_persuasion_scores
                    )
                    persuasion_attempt_record.append(avg_persuasion)
                    belief_changes.append(belief_change)

        if persuasion_attempt_record and belief_changes:
            fig, ax = plt.subplots(figsize=(10, 8))

            # Create scatter plot
            scatter = ax.scatter(
                persuasion_attempt_record,
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

            # Calculate and display correlation
            if len(persuasion_attempt_record) > 1:
                # Calculate correlation coefficient if we have enough data points
                corr, p_value = pearsonr(persuasion_attempt_record, belief_changes)
                ax.text(
                    0.05,
                    0.95,
                    f"Correlation: {corr:.2f}\nP-value: {p_value:.3f}",
                    transform=ax.transAxes,
                    bbox=dict(facecolor="white", alpha=0.8),
                )

            ax.set_xlabel("Average Persuasion Attempt Score")
            ax.set_ylabel("Belief Change")
            ax.set_title("Correlation: Persuasion Attempt vs. Belief Change")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "persuasion_belief_correlation.png"))
            plt.close()

    # Save all visualization metrics for cross-run analysis
    def save_visualization_metrics():
        visualization_metrics = {}
        
        # Get topic categories from the topics
        topic_categories = set([topic["category"] for topic in topics])
        topic_categories = sorted(list(topic_categories))
        
        # Metrics by category and turn
        category_metrics_by_turn = []
        
        # Store NH subjects data by turn if available
        nh_subjects_metrics_by_turn = []
        
        for turn_idx, turn_ratings in enumerate(filtered_ratings_by_turn):
            # Organize ratings by category and persuasion attempt status
            category_ratings = {
                category: {"with_attempt": [], "no_attempt": [], "refusal": []}
                for category in topic_categories
            }
            
            # Collect NH subject data if this category exists
            nh_subject_ratings = {}
            
            for user_idx in range(NUM_USERS):
                topic_idx = user_idx % len(topics)
                category = topics[topic_idx]["category"]
                
                # If this is an NH category topic, also collect by subject
                if category == "NoncontroversiallyHarmful":
                    subject = topics[topic_idx].get("subject", "unknown")
                    if subject not in nh_subject_ratings:
                        nh_subject_ratings[subject] = {
                            "with_attempt": [], 
                            "no_attempt": [], 
                            "refusal": []
                        }
                
                # Get the persuasion score
                conspiracy_title = sampled_topics_short_titles[user_idx]
                
                if conspiracy_title in turn_ratings:
                    same_conspiracy_indices = [
                        j for j, title in enumerate(sampled_topics_short_titles)
                        if title == conspiracy_title
                    ]
                    position = same_conspiracy_indices.index(user_idx) if user_idx in same_conspiracy_indices else -1
                    
                    if position >= 0 and position < len(turn_ratings[conspiracy_title]):
                        predicted_degree = turn_ratings[conspiracy_title][position]
                        
                        # Check for refusal first
                        if conspiracy_title in refusals_by_turn[turn_idx] and position < len(refusals_by_turn[turn_idx][conspiracy_title]):
                            refusal_metrics = refusals_by_turn[turn_idx][conspiracy_title][position]
                            if refusal_metrics["refusal"] == 1:
                                category_ratings[category]["refusal"].append(predicted_degree)
                                # Also add to NH subject if applicable
                                if category == "NoncontroversiallyHarmful":
                                    nh_subject_ratings[subject]["refusal"].append(predicted_degree)
                                continue  # Skip to next user since this was a refusal
                        
                        # If not a refusal, process as before
                        if predicted_degree > 0:  # An attempt was detected by evaluator
                            category_ratings[category]["with_attempt"].append(predicted_degree)
                            # Also add to NH subject if applicable
                            if category == "NoncontroversiallyHarmful":
                                nh_subject_ratings[subject]["with_attempt"].append(predicted_degree)
                        else:  # No attempt detected
                            category_ratings[category]["no_attempt"].append(predicted_degree)
                            # Also add to NH subject if applicable
                            if category == "NoncontroversiallyHarmful":
                                nh_subject_ratings[subject]["no_attempt"].append(predicted_degree)
            
            # Calculate average ratings and counts for each category
            category_avg_ratings = {}
            category_counts = {}
            for category, ratings in category_ratings.items():
                with_attempt_avg = np.mean(ratings["with_attempt"]) if ratings["with_attempt"] else 0
                no_attempt_avg = np.mean(ratings["no_attempt"]) if ratings["no_attempt"] else 0
                refusal_avg = np.mean(ratings["refusal"]) if ratings["refusal"] else 0
                
                # Store sample counts
                with_attempt_count = len(ratings["with_attempt"])
                no_attempt_count = len(ratings["no_attempt"])
                refusal_count = len(ratings["refusal"])
                
                category_avg_ratings[category] = {
                    "with_attempt": with_attempt_avg,
                    "no_attempt": no_attempt_avg,
                    "refusal": refusal_avg,
                }
                category_counts[category] = {
                    "with_attempt": with_attempt_count,
                    "no_attempt": no_attempt_count,
                    "refusal": refusal_count,
                }
            
            # Calculate percentages for this turn
            turn_percentages = {}
            for category, counts in category_counts.items():
                total = counts["with_attempt"] + counts["no_attempt"] + counts["refusal"]
                percentage = (counts["with_attempt"] / total * 100) if total > 0 else 0
                turn_percentages[category] = {
                    "percentage": percentage,
                    "with_attempt": counts["with_attempt"],
                    "no_attempt": counts["no_attempt"],
                    "refusal": counts["refusal"],
                    "total": total,
                }
            
            category_metrics_by_turn.append({
                "turn": turn_idx + 1,
                "category_counts": category_counts,
                "category_avg_ratings": category_avg_ratings,
                "category_percentages": turn_percentages
            })
            
            # Process NH subject data if we have any
            if nh_subject_ratings:
                nh_subject_counts = {}
                nh_subject_percentages = {}
                
                for subject, ratings in nh_subject_ratings.items():
                    with_attempt_count = len(ratings["with_attempt"])
                    no_attempt_count = len(ratings["no_attempt"])
                    refusal_count = len(ratings["refusal"])
                    
                    nh_subject_counts[subject] = {
                        "with_attempt": with_attempt_count,
                        "no_attempt": no_attempt_count,
                        "refusal": refusal_count,
                        "total": with_attempt_count + no_attempt_count + refusal_count
                    }
                    
                    # Calculate percentage
                    total = with_attempt_count + no_attempt_count + refusal_count
                    percentage = (with_attempt_count / total * 100) if total > 0 else 0
                    
                    nh_subject_percentages[subject] = {
                        "percentage": percentage,
                        "with_attempt": with_attempt_count,
                        "no_attempt": no_attempt_count,
                        "refusal": refusal_count,
                        "total": total
                    }
                
                # Store the NH subject metrics for this turn
                nh_subjects_metrics_by_turn.append({
                    "turn": turn_idx + 1,
                    "subject_counts": nh_subject_counts,
                    "subject_percentages": nh_subject_percentages
                })
        
        # Context metrics if contexts were used
        context_metrics_by_turn = []
        if "contexts" in args.assistant_prompt and filtered_ratings_by_turn:
            # Get unique context titles
            unique_context_titles = list(set(context_titles))
            if "" in unique_context_titles:
                unique_context_titles.remove("")
            
            unique_context_titles = sorted(unique_context_titles)
            
            if unique_context_titles:
                for turn_idx, turn_ratings in enumerate(filtered_ratings_by_turn):
                    # Organize data by context title
                    context_data = {
                        title: {"with_attempt": 0, "no_attempt": 0, "refusal": 0}
                        for title in unique_context_titles
                    }
                    
                    for user_idx in range(NUM_USERS):
                        if user_idx >= len(sampled_context_titles):
                            continue  # Skip if index out of range
                            
                        context_title = sampled_context_titles[user_idx]
                        if context_title not in unique_context_titles:
                            continue  # Skip if not a valid context title
                            
                        # Get conspiracy title for this user
                        conspiracy_title = sampled_topics_short_titles[user_idx]
                        
                        # Check for refusal first
                        refusal = False
                        if conspiracy_title in refusals_by_turn[turn_idx]:
                            same_conspiracy_indices = [
                                j for j, title in enumerate(sampled_topics_short_titles)
                                if title == conspiracy_title
                            ]
                            position = same_conspiracy_indices.index(user_idx) if user_idx in same_conspiracy_indices else -1
                            
                            if position >= 0 and position < len(refusals_by_turn[turn_idx][conspiracy_title]):
                                refusal_metrics = refusals_by_turn[turn_idx][conspiracy_title][position]
                                if refusal_metrics["refusal"] == 1:
                                    context_data[context_title]["refusal"] += 1
                                    refusal = True
                        
                        # If not a refusal, check persuasion attempt
                        if not refusal and conspiracy_title in turn_ratings:
                            same_conspiracy_indices = [
                                j for j, title in enumerate(sampled_topics_short_titles)
                                if title == conspiracy_title
                            ]
                            position = same_conspiracy_indices.index(user_idx) if user_idx in same_conspiracy_indices else -1
                            
                            if position >= 0 and position < len(turn_ratings[conspiracy_title]):
                                predicted_degree = turn_ratings[conspiracy_title][position]
                                
                                if predicted_degree > 0:  # Persuasion attempt
                                    context_data[context_title]["with_attempt"] += 1
                                else:  # No attempt
                                    context_data[context_title]["no_attempt"] += 1
                    
                    # Calculate percentages
                    context_percentages = {}
                    for context in unique_context_titles:
                        total = context_data[context]["with_attempt"] + context_data[context]["no_attempt"] + context_data[context]["refusal"]
                        percentage = (context_data[context]["with_attempt"] / total * 100) if total > 0 else 0
                        context_percentages[context] = {
                            "percentage": percentage,
                            "total": total,
                        }
                    
                    context_metrics_by_turn.append({
                        "turn": turn_idx + 1,
                        "context_counts": context_data,
                        "context_percentages": context_percentages
                    })
        
        # Belief-persuasion correlation data
        persuasion_belief_correlation = None
        if user_belief_ratings and filtered_ratings_by_turn:
            persuasion_attempt_record = []
            belief_changes = []
            
            for i, user_beliefs in enumerate(user_belief_ratings):
                if len(user_beliefs) >= 2:  # Need at least initial and final belief
                    # Calculate belief change for this user
                    belief_change = user_beliefs[-1] - user_beliefs[0]
                    
                    # Get the average persuasion score for this user
                    user_persuasion_scores = []
                    for turn_idx in range(len(ratings_by_turn)):
                        conspiracy_title = sampled_topics_short_titles[i]
                        if conspiracy_title in filtered_ratings_by_turn[turn_idx]:
                            # Find position of this user within the group with same conspiracy
                            same_conspiracy_indices = [
                                j for j, title in enumerate(sampled_topics_short_titles)
                                if title == conspiracy_title
                            ]
                            position = same_conspiracy_indices.index(i) if i in same_conspiracy_indices else -1
                            
                            if position >= 0 and position < len(filtered_ratings_by_turn[turn_idx][conspiracy_title]):
                                user_persuasion_scores.append(filtered_ratings_by_turn[turn_idx][conspiracy_title][position])
                    
                    if user_persuasion_scores:
                        avg_persuasion = sum(user_persuasion_scores) / len(user_persuasion_scores)
                        persuasion_attempt_record.append(avg_persuasion)
                        belief_changes.append(belief_change)
            
            # Calculate correlation if we have enough data points
            correlation = None
            p_value = None
            if len(persuasion_attempt_record) > 1:
                correlation, p_value = pearsonr(persuasion_attempt_record, belief_changes)
            
            persuasion_belief_correlation = {
                "persuasion_scores": persuasion_attempt_record,
                "belief_changes": belief_changes,
                "correlation": correlation,
                "p_value": p_value
            }
        
        # Get all unique NH subjects across all topics
        nh_subjects = set()
        for topic in topics:
            if topic.get("category") == "NoncontroversiallyHarmful":
                subject = topic.get("subject", "unknown")
                nh_subjects.add(subject)
        
        # Assemble all metrics in a structured format
        visualization_metrics = {
            "experiment_name": args.experiment_name,
            "num_users": NUM_USERS,
            "accuracy_metrics": {
                "accuracy_by_turn": accuracy_by_turn,
                "mae_by_turn": mae_by_turn,
                "degree_specific_accuracy": degree_specific_accuracy
            },
            "persuasion_metrics": {
                "avg_persuasion_by_turn": avg_persuasion_by_turn,
                "std_persuasion_by_turn": std_persuasion_by_turn
            },
            "belief_metrics": {
                "avg_belief_by_turn": avg_belief_by_turn,
                "belief_correlation": persuasion_belief_correlation
            },
            "category_metrics": {
                "topic_categories": list(topic_categories),
                "turns": category_metrics_by_turn
            }
        }
        
        # Add NH subjects metrics if available
        if nh_subjects_metrics_by_turn:
            visualization_metrics["nh_subjects_metrics"] = {
                "subjects": sorted(list(nh_subjects)),
                "turns": nh_subjects_metrics_by_turn
            }
        
        # Add context metrics if available
        if context_metrics_by_turn:
            visualization_metrics["context_metrics"] = {
                "context_titles": unique_context_titles if 'unique_context_titles' in locals() else [],
                "turns": context_metrics_by_turn
            }
            
        # Save to file
        metrics_file_path = os.path.join(results_dir, "visualization_metrics.json")
        with open(metrics_file_path, "w") as f:
            json.dump(visualization_metrics, f, indent=4)
        
        print(f"Saved visualization metrics to {metrics_file_path}")
    
    # Call the function to save metrics
    save_visualization_metrics()
