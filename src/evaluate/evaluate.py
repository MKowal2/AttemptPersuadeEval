import asyncio

from src.generate_conversations import generate
from src.utils.utils import extract_rating

loop = asyncio.get_event_loop()


def remove_system_messages(conversations: list[list]):
    """
    Remove system messages from the conversation.
    """
    for i, messages in enumerate(conversations):
        if messages and messages[0]["role"] == "system":
            conversations[i] = messages[1:]
    return conversations


def separate_last_message(conversations: list[list]):
    """
    Separate the last message from the conversation.
    """
    last_messages = [messages[-1] for messages in conversations]
    conversations = [messages[:-1] for messages in conversations]
    return conversations, last_messages


def set_convo_roles(conversations: list[list]):
    """
    Ensure we start with user, end with assist, and and messages alternate between user
    and assistant.
    Capitalize roles for insertion into a prompt.
    """
    for messages in conversations:
        for j, message in enumerate(messages):
            message["role"] = "User" if j % 2 == 0 else "Assistant"
    return conversations


def format_conversations(conversations: list[list]):
    """
    Format the conversation for evaluation.
    """
    conversations = set_convo_roles(conversations)
    return [
        "\n\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
        for messages in conversations
    ]


def extract_numerical_ratings(ratings: list[str]):
    """
    Extract numerical ratings from the response.
    """
    numerical_ratings = []
    for rating in ratings:
        numerical_ratings.append(extract_rating(rating))
    return numerical_ratings


def evaluate_conversations(
    conversations: list[list],
    beliefs: list[str],
    argue_for_or_against_list: list[str],
    eval_prompt: str,
    model: str = "gpt-4o-mini",
):
    """
    Evaluate the persuasiveness of the last message in a conversation.
    """
    conversations = remove_system_messages(conversations)
    conversations, last_messages = separate_last_message(conversations)
    condensed_conversations = format_conversations(conversations)

    evaluation_messages = [
        [
            {
                "role": "user",
                "content": eval_prompt.format(
                    belief=belief,
                    conversation_minus_last_message=conversation,
                    last_message=last_message,
                    support_or_oppose=(
                        "to believe the following"
                        if for_or_against == "for"
                        else "that the following is false"
                    ),
                    support_or_oppose2=(
                        "believing" if for_or_against == "for" else "disbelieving"
                    ),
                ),
            }
        ]
        for belief, conversation, last_message, for_or_against in zip(
            beliefs, condensed_conversations, last_messages, argue_for_or_against_list
        )
    ]

    ratings = loop.run_until_complete(
        generate.generate_llm(evaluation_messages, model=model)
    )

    numerical_ratings = extract_numerical_ratings(ratings)

    return ratings, numerical_ratings
