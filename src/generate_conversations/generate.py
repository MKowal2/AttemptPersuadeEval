"""
Generate new responses in parallel given collection of existing conversations.
"""

import asyncio
import re
from typing import Any, Dict, List

import torch
from litellm import acompletion
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Cache for storing loaded pipeline generators
MODEL_CACHE = {}


def postprocess_message(message):
    """
    Postprocess the message to remove jailbreak-tuning artifacts.
    """
    split_message = message.split("Warning: ")
    if len(split_message) > 1:
        return split_message[1]
    return message


def preload_local_model(model_name):
    """
    Explicitly preload a model into the cache.

    Args:
        model_name (str): Name of the model to preload, e.g. 'meta-llama/Llama-3-8B-chat'

    Returns:
        bool: True if model was loaded or already in cache, False if model couldn't be loaded
    """
    if not model_name.startswith("hf/"):
        raise Exception(f"Only local models can be preloaded, skipping {model_name}")

    if model_name in MODEL_CACHE:
        print(f"Model {model_name} is already loaded in cache")
        return True

    try:
        local_path = f"src/ckpts/{model_name.split('/')[-1]}"
        print(f"Preloading model {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
        # Set padding side to left for decoder-only models
        tokenizer.padding_side = "left"

        # Set up pad token if needed
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                # Add a pad token if there's no eos token to use
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        hf_llm = AutoModelForCausalLM.from_pretrained(
            local_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        # Resize model embeddings if pad token was added
        if tokenizer.pad_token == "[PAD]":
            hf_llm.resize_token_embeddings(len(tokenizer))

        generator = pipeline(
            "text-generation",
            model=hf_llm,
            tokenizer=tokenizer,
        )

        MODEL_CACHE[model_name] = {"generator": generator, "tokenizer": tokenizer}
        print(f"Model {model_name} successfully preloaded")
        return True
    except Exception as e:
        raise Exception(f"Error preloading model {model_name}: {e}")


def is_qwen_model(model_name: str) -> bool:
    """
    Check if the model is a Qwen model.

    Args:
        model_name: The model name

    Returns:
        True if it's a Qwen model, False otherwise
    """
    return "qwen" in model_name.lower()


def clean_qwen_response(text: str) -> str:
    """
    Clean Qwen model response by removing any thinking blocks and
    extracting only the response content.

    Args:
        text: Raw generated text from Qwen model

    Returns:
        Cleaned response
    """
    # First, check if the text contains assistant tag
    if "<|im_start|>assistant" in text:
        # Extract only the assistant's message content
        assistant_part = (
            text.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
        )

        # Remove thinking blocks
        cleaned_text = re.sub(
            r"<think>.*?</think>", "", assistant_part, flags=re.DOTALL
        )

        # Clean up any extra whitespace that might remain after removing thinking blocks
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
        cleaned_text = cleaned_text.strip()

        return cleaned_text

    # If the standard pattern doesn't match, try to extract content after the last </think> tag
    think_match = re.search(r"</think>\s*(.*?)(?:<|$)", text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()

    # If no pattern matches, return the original text as a fallback
    return text.strip()


def format_prompt_for_model(
    messages: List[Dict[str, str]], model: str, tokenizer
) -> str:
    """
    Format messages appropriately for the specific model.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: Model name
        tokenizer: The tokenizer for the model

    Returns:
        Formatted prompt string
    """
    # Handle Qwen models with thinking disabled
    if is_qwen_model(model):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Disable thinking mode for Qwen
        )
    else:
        # Default Llama formatting
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def get_generation_params(model: str, temperature: float) -> Dict[str, Any]:
    """
    Get generation parameters appropriate for the specific model.

    Args:
        model: Model name
        temperature: Base temperature value

    Returns:
        Dictionary of generation parameters
    """
    # Default parameters
    params = {
        "max_new_tokens": 2048,
        "temperature": temperature,
        "return_full_text": True,
    }

    # Qwen models in non-thinking mode have recommended parameters
    if is_qwen_model(model):
        params.update(
            {
                "temperature": 0.7,  # Recommended for non-thinking mode
                "top_p": 0.8,  # Recommended for non-thinking mode
                "top_k": 20,  # Recommended for non-thinking mode
                "min_p": 0,  # Recommended for non-thinking mode
            }
        )

    return params


def generate_with_local_model(
    message_collection: List[List[Dict[str, str]]],
    model: str,
    temperature: float = 0.5,
    batch_size: int = 4,
) -> List[str]:
    """
    Generate responses using a local HuggingFace model with batching.

    Args:
        message_collection: List of conversation messages
        model: Model name (should start with "hf/")
        temperature: Sampling temperature
        batch_size: Number of prompts to process in a single batch

    Returns:
        List of generated responses
    """
    local_path = f"src/ckpts/{model.split('/')[-1]}"

    # Check if generator pipeline is already loaded in cache
    if model not in MODEL_CACHE:
        print(f"Loading model {model} (first time)...")
        tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
        # Set padding side to left for decoder-only models
        tokenizer.padding_side = "left"

        # Set up pad token if needed
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                # Add a pad token if there's no eos token to use
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        hf_llm = AutoModelForCausalLM.from_pretrained(
            local_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        # Resize model embeddings if pad token was added
        if tokenizer.pad_token == "[PAD]":
            hf_llm.resize_token_embeddings(len(tokenizer))

        # Create the generator pipeline with both model and tokenizer
        generator = pipeline(
            "text-generation",
            model=hf_llm,
            tokenizer=tokenizer,
        )

        # Store in cache
        MODEL_CACHE[model] = {
            "generator": generator,
            "tokenizer": tokenizer,
        }
        print(f"Model {model} loaded and cached")
    else:
        generator = MODEL_CACHE[model]["generator"]
        tokenizer = MODEL_CACHE[model]["tokenizer"]

    # Format all prompts first based on the model type
    formatted_prompts = []
    for messages in message_collection:
        formatted_prompts.append(format_prompt_for_model(messages, model, tokenizer))

    all_responses = []
    total_prompts = len(formatted_prompts)

    # Get appropriate generation parameters for the model
    generation_params = get_generation_params(model, temperature)

    # Use tqdm for progress tracking batches
    for batch_start in tqdm(
        range(0, total_prompts, batch_size), desc="Batches completed on GPU"
    ):
        batch_end = min(batch_start + batch_size, total_prompts)
        current_batch = formatted_prompts[batch_start:batch_end]
        try:
            # Generate responses for the entire batch with proper padding
            batch_outputs = generator(
                current_batch,
                pad_token_id=tokenizer.pad_token_id,
                padding=True,
                truncation=True,
                batch_size=len(current_batch),
                **generation_params,
            )

            # Process each output in the batch
            for i, outputs in enumerate(batch_outputs):
                try:
                    # Extract the generated response based on structure
                    # If outputs is a list (like [{'generated_text': '...'}]), access it accordingly
                    if isinstance(outputs, list):
                        generated_text = outputs[0]["generated_text"]
                    # If outputs is a dictionary ({'generated_text': '...'}), access directly
                    else:
                        generated_text = outputs["generated_text"]

                    # Extract response based on model type
                    if is_qwen_model(model):
                        # For Qwen, properly clean the response to remove thinking blocks
                        response = clean_qwen_response(generated_text)
                    else:
                        # For Llama models
                        response = generated_text.split("<|end_header_id|>\n\n")[-1]                      
                    all_responses.append(response)
                except Exception as e:
                    print(f"Error processing output {i} in batch: {e}")
                    all_responses.append(f"Error processing response: {e}")
        except Exception as e:
            # If batch processing fails, fall back to individual processing
            print(
                f"Batch processing failed with error: {e}. Falling back to individual processing."
            )
            for prompt in current_batch:
                try:
                    outputs = generator(
                        prompt, pad_token_id=tokenizer.pad_token_id, **generation_params
                    )
                    # Check for the expected format (list or dict)
                    if isinstance(outputs, list):
                        generated_text = outputs[0]["generated_text"]
                    else:
                        generated_text = outputs["generated_text"]

                    # Extract response based on model type
                    if is_qwen_model(model):
                        # For Qwen, properly clean the response to remove thinking blocks
                        response = clean_qwen_response(generated_text)
                    else:
                        # For Llama models
                        response = generated_text.split("<|end_header_id|>\n\n")[-1]
                    
                    all_responses.append(response)
                except Exception as e:
                    print(f"Error processing individual prompt: {e}")
                    all_responses.append(f"Error processing prompt: {e}")

    return all_responses


async def generate_llm(
    message_collection: List[List[Dict[str, str]]],
    temperature: float = 0.5,
    model: str = "gpt-4o-mini",
    postprocess_responses: bool = False,
    batch_size: int = 4,
    **kwargs,
) -> List[str]:
    """
    Generate responses using either local models (synchronously with batching) or cloud APIs (asynchronously).

    Args:
        message_collection: List of conversation messages
        temperature: Sampling temperature
        model: Model name (e.g., "hf/Meta-Llama-3.1-8B-Instruct" or "gpt-4o")
        postprocess_responses: Whether to apply postprocessing to responses
        batch_size: Number of prompts to process in a single batch (for local models)

    Returns:
        List of generated responses
    """
    # Handle local models synchronously with batching
    if model.startswith("hf/"):
        responses = generate_with_local_model(
            message_collection=message_collection,
            model=model,
            temperature=temperature,
            batch_size=batch_size,
        )
    # Handle cloud API models asynchronously
    else:

        async def process_messages(messages):
            try:
                # Create a dictionary of arguments for acompletion
                completion_args = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                }

                # Check if model is a Vertex AI finetuned model with numeric ID
                if model.startswith("vertex_ai/") and model.split("/")[1].isdigit():
                    completion_args["base_model"] = "vertex_ai/gemini-2.0-flash-001"

                completion = await acompletion(**completion_args)
                return completion.choices[0].message.content
            except Exception as e:
                print(f"Error processing prompt: {e}")
                return f"Error processing prompt: {e}"

        tasks = [process_messages(message_list) for message_list in message_collection]
        responses = await asyncio.gather(*tasks)

    if postprocess_responses:
        responses = [postprocess_message(response) for response in responses]

    return responses
