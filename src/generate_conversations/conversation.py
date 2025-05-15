import asyncio

from ..utils.utils import remove_rating_from_message
from . import generate

loop = asyncio.get_event_loop()


def set_system_message(message_collection: list[list], message: str or list):
    if isinstance(message, str):
        for i, messages in enumerate(message_collection):
            if len(messages) > 0:
                if messages[0]["role"] == "system":
                    message_collection[i][0]["content"] = message
                else:
                    message_collection[i].insert(
                        0, {"role": "system", "content": message}
                    )
            else:
                message_collection[i].insert(0, {"role": "system", "content": message})
    elif isinstance(message, list):
        for i, messages in enumerate(message_collection):
            if len(messages) > 0:
                if messages[0]["role"] == "system":
                    message_collection[i][0]["content"] = message[i]
                else:
                    message_collection[i].insert(
                        0, {"role": "system", "content": message[i]}
                    )
            else:
                message_collection[i].insert(
                    0, {"role": "system", "content": message[i]}
                )
    return message_collection


def swap_convo_roles(message_collection: list[list], user_first=True):
    """
    We always want the messages to go user -> assistant -> user -> assistant -> ...
    The ordering may shift depending on the presence of a system message.
    """
    if user_first:  # set first non-system turn to user, then alternate
        for i, messages in enumerate(message_collection):
            if messages[0]["role"] == "system":
                has_system_message = True
            else:
                has_system_message = False

            for j, message in enumerate(messages):
                if j % 2 == 0:
                    if has_system_message:
                        if j == 0:
                            if has_system_message:
                                continue  # leave the system message role
                        message_collection[i][j]["role"] = "assistant"
                    else:
                        message_collection[i][j]["role"] = "user"
                else:
                    if has_system_message:
                        message_collection[i][j]["role"] = "user"
                    else:
                        message_collection[i][j]["role"] = "assistant"

    else:  # set first non-system turn to assistant, then alternate
        for i, messages in enumerate(message_collection):
            if messages[0]["role"] == "system":
                has_system_message = True
            else:
                has_system_message = False

            for j, message in enumerate(messages):
                if j % 2 == 0:
                    if has_system_message:
                        if j == 0:
                            if has_system_message:
                                continue  # leave the system message role
                        message_collection[i][j]["role"] = "user"
                    else:
                        message_collection[i][j]["role"] = "assistant"
                else:
                    if has_system_message:
                        message_collection[i][j]["role"] = "assistant"
                    else:
                        message_collection[i][j]["role"] = "user"
    return message_collection


def add_to_convo(message_collection: list[list], model="gpt-4o-mini", **kwargs):
    """
    Generate new responses in parallel given collection of existing conversations.
    """
    message_collection = swap_convo_roles(
        message_collection, user_first=kwargs.get("user_first", True)
    )

    # If remove_ratings is enabled, create a copy of the message collection with ratings removed from user messages
    if kwargs.get("remove_ratings", False):
        processed_collection = []
        for conversation in message_collection:
            processed_conversation = []
            for message in conversation:
                if message["role"] == "user":
                    # Remove the rating part from user messages
                    content = remove_rating_from_message(message["content"])
                    processed_conversation.append(
                        {"role": message["role"], "content": content}
                    )
                else:
                    processed_conversation.append(message)
            processed_collection.append(processed_conversation)

        # Use the processed collection for generation
        generation_collection = processed_collection
    else:
        # Use the original collection
        generation_collection = message_collection

    responses = loop.run_until_complete(
        generate.generate_llm(generation_collection, model=model, **kwargs)
    )
    iterated_collection = message_collection
    for i, response in enumerate(responses):
        iterated_collection[i].append({"role": "assistant", "content": response})
    return iterated_collection
