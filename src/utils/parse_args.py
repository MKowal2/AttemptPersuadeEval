#!/usr/bin/env python3

import argparse


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the persuasion experiment.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Persuasion eval")

    parser.add_argument(
        "--num_users",
        type=int,
        default=18 * 20,  # 18 topics * 20 users each
        help="Number of users to simulate.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/local",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--num_turns", type=int, default=3, help="Number of conversation turns."
    )
    parser.add_argument(
        "--persuader_model",
        type=str,
        default="gpt-4o-mini",
        help="Name of the LLM model persuading.",  # todo: add choices
    )
    parser.add_argument(
        "--persuadee_model",
        type=str,
        default="gpt-4o-mini",
        help="Name of the LLM model being persuaded.",  # todo: add choices
    )
    parser.add_argument(
        "--evaluator_model",
        type=str,
        default="gpt-4o-mini",
        help="Name of the evaluator LLM model.",  # todo: add choices
    )
    parser.add_argument(
        "--jailbreak_persuader",
        action="store_true",
        help="Whether or not to enable jailbreaking prompt for the persuadER.",
    )
    parser.add_argument(
        "--jailbreak_persuadee",
        action="store_true",
        help="Whether or not to enable jailbreaking prompt for the persuadEE.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="persuasion_out_of_conspiracy",
        help="Name of the experiment.",
    )
    parser.add_argument(
        "--belief_lower_threshold",
        type=int,
        default=20,
        help="Below this threshold, we simulate persuasion only in the belief-increasing direction",
    )
    parser.add_argument(
        "--belief_upper_threshold",
        type=int,
        default=80,
        help="Above this threshold, we simulate persuasion only in the belief-decreasing direction",
    )

    parser.add_argument(
        "--sample_belief_lower",
        type=int,
        default=0,
        help="Sampling belief lower bound.",
    )
    parser.add_argument(
        "--sample_belief_upper",
        type=int,
        default=20,
        help="Sampling belief upper bound.",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="src/prompts/base_prompts_all.yaml",
        help="File containing the base prompts.",
    )
    parser.add_argument(
        "--topics_file",
        type=str,
        default="src/topics/diverse_topics.jsonl",
        help="File containing the topics for persuasion.",
    )
    parser.add_argument(
        "--contexts_file",
        type=str,
        default="src/topics/persuasion_contexts.jsonl",
        help="File containing the topics for persuasion.",
    )
    parser.add_argument(
        "--topic_categories",
        type=str,
        nargs="+",
        default=[
            "Conspiracy",
            "BenignOpinion",
            "BenignFactual",
            "Controversial",
            "NoncontroversiallyHarmful",
            "UnderminingControl",
        ],
        help="Categories of topics to use from the topics file. Options: conspiracy, Benign Opinion, Benign Factual, Controversial, Noncontroversially Harmful, Undermining Control.",
    )
    parser.add_argument(
        "--all_topics",
        action="store_true",
        help="Whether to sample all topics.",
    )
    parser.add_argument(
        "--record_all_system",
        action="store_true",
        help="Whether to record all system messages between conversation rounds or just the current one.",
    )
    parser.add_argument(
        "--remove_ratings",
        action="store_true",
        help="Whether to remove the '| NUMBER' rating from user messages before passing to the assistant.",
    )
    parser.add_argument(
        "--baseline_persuasion",
        action="store_true",
        help="Whether to use baseline prompt that does not attempt to persuade.",
    )
    parser.add_argument(
        "--evaluation_scale",
        type=int,
        default=2,
        help="Scale used for evaluation model (2, 3 and 100 currently supported).",
    )
    parser.add_argument(
        "--human_data_path",
        type=str,
        default=None,
        help="Path to human conversation data to use instead of LLM-generated conversations.",
    )
    parser.add_argument(
        "--human_data_rounds",
        type=int,
        default=0,
        help="Number of conversation rounds to take from human data before switching "
        "to  LLM sampling. 0 means use only LLMs (default behavior).",
    )
    parser.add_argument(
        "--assistant_prompt",
        type=str,
        default="prompt_assistant_no_user_degree",
        help="Key that references assistant prompt to use in experiments ("
        "prompt_assistant_og, prompt_assistant_no_user, "
        "prompt_assistant_no_degree, prompt_assistant_no_user_degree, "
        "prompt_assistant_baseline, human, prompt_assistant_contexts",
    )
    parser.add_argument(
        "--only_persuade",
        action="store_true",
        help="Whether every assistant tries to persuade.",
    )
    parser.add_argument(
        "--rephrase_topics",
        action="store_true",
        help="Whether to rephrase wording of sampled topics via LLM.",
    )
    parser.add_argument(
        "--reject_model",
        type=str,
        default="gpt-4o-mini",
        help="Name of the LLM model rejecting.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for processing prompts with local models.",
    )

    args = parser.parse_args()

    return args
