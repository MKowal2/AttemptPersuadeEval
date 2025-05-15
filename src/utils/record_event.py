import copy

import src.utils.utils as utils


def record_sampling_event(
    args, all_events, last_sampled, last_role, last_system_message, NUM_USERS, turn
):
    """
    For each user, we record a separate event in all_events.
    We'll store prompt (all messages but last) and sampled (the last).
    If args.record_all_system is True, we keep track of all system messages
    within prompt_part. Otherwise, we only maintain the current system message.
    """
    for i in range(len(last_sampled)):
        if len(all_events) < NUM_USERS:  # first turn
            prompt_part = [{"role": "system", "content": last_system_message[i]}]
        else:
            if args.record_all_system:
                # Keep all system messages - add the last role, content, and most recent system message
                prompt_part = copy.copy(all_events[turn * NUM_USERS + i]["prompt"])

                # Get the last content, possibly removing rating if it's a user message
                last_content = all_events[turn * NUM_USERS + i]["sampled"][0]
                if last_role == "user" and args.remove_ratings:
                    last_content = utils.remove_rating_from_message(last_content)

                prompt_part.append(
                    {
                        "role": last_role,
                        "content": last_content,
                    }
                )
                prompt_part.append(
                    {"role": "system", "content": last_system_message[i]}
                )
            else:
                # Only keep the current system message
                # Start with just the current system message
                prompt_part = [{"role": "system", "content": last_system_message[i]}]

                # Add all non-system messages from previous conversation,
                # without evaluation messages
                previous_messages = all_events[turn * NUM_USERS + i]["prompt"]
                for msg in previous_messages:
                    if msg["role"] != "system":
                        # If it's a user message and we're removing ratings, remove them
                        if msg["role"] == "user" and args.remove_ratings:
                            msg_content = utils.remove_rating_from_message(
                                msg["content"]
                            )
                            prompt_part.append(
                                {"role": msg["role"], "content": msg_content}
                            )
                        else:
                            prompt_part.append(msg)

                # Add the last response, possibly removing rating if it's a user message
                last_content = all_events[turn * NUM_USERS + i]["sampled"][0]
                if last_role == "user" and args.remove_ratings:
                    last_content = utils.remove_rating_from_message(last_content)

                prompt_part.append(
                    {
                        "role": last_role,
                        "content": last_content,
                    }
                )

        # Always store the original sampled content with ratings intact for evaluation
        sampled_part = last_sampled[i]["content"]
        event_dict = {
            "prompt": prompt_part,
            "sampled": [sampled_part],
        }
        all_events.append(event_dict)


def record_evaluation_event(all_events, ratings):
    """
    Record the evaluation messages as sampling events with role 'evaluator'.
    This logs the evaluation messages generated between conversation rounds.
    """

    for i, rating in enumerate(ratings):
        # Create a simple prompt part with just the system message indicating it's an evaluation
        prompt_part = [
            {"role": "system", "content": "Evaluation of persuasion attempt"}
        ]

        # The rating is the sampled part
        event_dict = {
            "prompt": prompt_part,
            "sampled": [rating],
        }
        all_events.append(event_dict)


def record_refusal_event(all_events, responses):
    """
    Record the refusal check responses as sampling events with role 'refusal_checker'.
    """

    for response in responses:
        # Create a prompt part with the system message and formatted rubric
        prompt_part = [{"role": "system", "content": "Check response for refusal"}]

        # The response is the sampled part
        event_dict = {
            "prompt": prompt_part,
            "sampled": [response],
        }
        all_events.append(event_dict)
