# LLM Persuasion Evals

![APE Persuasion](assets/APE_Persuasion.png)

⚠️ **Content Warning**: This repository contains examples and discussions of sensitive and potentially distressing topics.

This repository contains the code for the Attempt to Persuade Eval (APE)
project. The goal of the project is to develop a set of evaluation metrics for
measuring the *attempt* of language models to persuade.

## Setup

### OpenAI API Key

To run the evaluation scripts, you will need an OpenAI API key. This is done using
a `.env` file. Create a file called `.env` in the root directory of the repository
and add the following line:

```bash
OPENAI_API_KEY="your_openai_api_key"
```

Replace `your_openai_api_key` with your actual OpenAI API key.

### Google VertexAI API

To get authentication for running Gemini (Vertex AI), run the following in terminal:

```bash
gcloud auth application-default login
```

Also, you'll need to set the following env vars from the GCP Vertex project
VERTEXAI_PROJECT=""
VERTEXAI_LOCATION=""

When running the fine-tuned model, use e.g.: "--persuader_model vertex_ai/<VERTEXAI_ENDPOINTID>"

### Hugging Face API Key
You can get a Hugging Face API key by creating an account on the Hugging Face website
and then going to your account settings. Once you have your API key, either add the
path to the `.env` file or set the `HF_TOKEN` environment variable in the .env file:

HF_TOKEN="hf_..."

When using huggingface models, make sure you download the checkpoints to src/ckpts.

## Running evals

### Install dependencies

Dependencies are in `pyproject.toml`, install them for example with:

```bash
pip install -e ".[dev,test]"
```

### Execute evals

To run the persuasion attempt eval, use the following command:

```bash
python main.py --persuader_model gpt-4o
```

This will run the persuasion evals using the `gpt-4o` model. This eval simulates a
conversation between a user (i.e., roleplaying model) and a model, where the model
tries to persuade the user into/out of a certain statement. The eval will output, to
the 'results' directory, a JSON file containing the following information: the
dialogue between the user and the model, an evaluator model's score for the persuasion
attempt, and an evaluator model's score for the success of persuasion.

### Command line args

There's an extensive list of command line arguments which control how the persuasion eval is run:

```
➜ python main.py --help
usage: main.py [-h] [--num_users NUM_USERS] [--results_dir RESULTS_DIR] [--num_turns NUM_TURNS] [--persuader_model PERSUADER_MODEL] [--persuadee_model PERSUADEE_MODEL] [--evaluator_model EVALUATOR_MODEL] [--jailbreak_persuader] [--jailbreak_persuadee]
               [--experiment_name EXPERIMENT_NAME] [--belief_lower_threshold BELIEF_LOWER_THRESHOLD] [--belief_upper_threshold BELIEF_UPPER_THRESHOLD] [--sample_belief_lower SAMPLE_BELIEF_LOWER] [--sample_belief_upper SAMPLE_BELIEF_UPPER] [--prompts_file PROMPTS_FILE]
               [--topics_file TOPICS_FILE] [--contexts_file CONTEXTS_FILE] [--topic_categories TOPIC_CATEGORIES [TOPIC_CATEGORIES ...]] [--all_topics] [--record_all_system] [--remove_ratings] [--baseline_persuasion] [--evaluation_scale EVALUATION_SCALE]
               [--human_data_path HUMAN_DATA_PATH] [--human_data_rounds HUMAN_DATA_ROUNDS] [--assistant_prompt ASSISTANT_PROMPT] [--only_persuade] [--rephrase_topics] [--reject_model REJECT_MODEL] [--batch_size BATCH_SIZE]

Persuasion eval

options:
  -h, --help            show this help message and exit
  --num_users NUM_USERS
                        Number of users to simulate.
  --results_dir RESULTS_DIR
                        Directory to save results.
  --num_turns NUM_TURNS
                        Number of conversation turns.
  --persuader_model PERSUADER_MODEL
                        Name of the LLM model persuading.
  --persuadee_model PERSUADEE_MODEL
                        Name of the LLM model being persuaded.
  --evaluator_model EVALUATOR_MODEL
                        Name of the evaluator LLM model.
  --jailbreak_persuader
                        Whether or not to enable jailbreaking prompt for the persuadER.
  --jailbreak_persuadee
                        Whether or not to enable jailbreaking prompt for the persuadEE.
  --experiment_name EXPERIMENT_NAME
                        Name of the experiment.
  --belief_lower_threshold BELIEF_LOWER_THRESHOLD
                        Below this threshold, we simulate persuasion only in the belief-increasing direction
  --belief_upper_threshold BELIEF_UPPER_THRESHOLD
                        Above this threshold, we simulate persuasion only in the belief-decreasing direction
  --sample_belief_lower SAMPLE_BELIEF_LOWER
                        Sampling belief lower bound.
  --sample_belief_upper SAMPLE_BELIEF_UPPER
                        Sampling belief upper bound.
  --prompts_file PROMPTS_FILE
                        File containing the base prompts.
  --topics_file TOPICS_FILE
                        File containing the topics for persuasion.
  --contexts_file CONTEXTS_FILE
                        File containing the topics for persuasion.
  --topic_categories TOPIC_CATEGORIES [TOPIC_CATEGORIES ...]
                        Categories of topics to use from the topics file. Options: conspiracy, Benign Opinion, Benign Factual, Controversial, Noncontroversially Harmful, Undermining Control.
  --all_topics          Whether to sample all topics.
  --record_all_system   Whether to record all system messages between conversation rounds or just the current one.
  --remove_ratings      Whether to remove the '| NUMBER' rating from user messages before passing to the assistant.
  --baseline_persuasion
                        Whether to use baseline prompt that does not attempt to persuade.
  --evaluation_scale EVALUATION_SCALE
                        Scale used for evaluation model (2, 3 and 100 currently supported).
  --human_data_path HUMAN_DATA_PATH
                        Path to human conversation data to use instead of LLM-generated conversations.
  --human_data_rounds HUMAN_DATA_ROUNDS
                        Number of conversation rounds to take from human data before switching to LLM sampling. 0 means use only LLMs (default behavior).
  --assistant_prompt ASSISTANT_PROMPT
                        Key that references assistant prompt to use in experiments (prompt_assistant_og, prompt_assistant_no_user, prompt_assistant_no_degree, prompt_assistant_no_user_degree, prompt_assistant_baseline, human, prompt_assistant_contexts
  --only_persuade       Whether every assistant tries to persuade.
  --rephrase_topics     Whether to rephrase wording of sampled topics via LLM.
  --reject_model REJECT_MODEL
                        Name of the LLM model rejecting.
  --batch_size BATCH_SIZE
                        Batch size for processing prompts with local models.
```                        

### Experiment scripts

A number of scripts with pre-configured settings for persuasion evals for certain experiments can be found in the `exps/` directory.

For example:

```bash
bash exps/assistant_prompt_comparison.sh
```

### Models

A list of models that can be used as the persuader are as follows:

- `gpt-4o`: OpenAI GPT-4 model
- `gpt-4o-mini`: Smaller version of GPT-4
- `vertex_ai/gemini-2.0-flash-001`: Google's Gemini 2.0 Flash model
- `vertex_ai/gemini-2.0-flash-lite-001`: Lightweight version of Gemini 2.0 Flash
- `hf/Meta-Llama-3.1-8B-Instruct`: Meta's Llama 3.1 8B instruction-tuned model
- `hf/Qwen2.5-32B-Instruct`: Qwen 2.5 32B instruction-tuned model

Note, we include the ability to use several open-weight models through the Huggingface library. To download the huggingface (hf) model weights, you can use the huggingface-cli downloader and download them to `src/ckpts`, for example:

```bash
huggingface-cli download Qwen/Qwen2.5-32B-Instruct --local-dir src/ckpts/Qwen2.5-32B-Instruct --local-dir-use-symlinks False
```

### Topics list

The default topics file can be found at `src/topics/diverse_topics.jsonl` which contains topics from all categories:

```python
[
  "Conspiracy",
  "BenignOpinion",
  "BenignFactual",
  "Controversial",
  "NoncontroversiallyHarmful",
  "UnderminingControl"
]
```

New harmful topics can be generated using the script at `src/utils/generate_harmful_texts.py`

## Results

### Visualizing results

Graphs of the results are automatically generated and created in the same directory as all other results

### Viewing conversations from results

To view conversations from the results of the persuasion evals, we use the [logviz library](
https://github.com/naimenz/logviz). We have included a version of the library in the
logviz repository. First, install logviz as a library using the following command:

```bash
cd logviz
pip install -e .
```

To visualize the results, run `logviz` from terminal and then drag the `conversation_log.jsonl` file
containing the results into the window. This will display the conversation results
in a visual format.