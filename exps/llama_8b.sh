echo "Running extreme subset experiments"
echo "Running full evaluation examples for Llama 8B"
python main.py --num_turns 3 \
--prompts_file src/prompts/base_v7_persuasion_degree.yaml \
--topics_file src/topics/diverse_topics_conspiracy.jsonl \
--persuader_model meta-llama/Meta-Llama-3.1-8B-Instruct \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--experiment_name llama_8b_full \
--sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  \
--assistant_prompt prompt_assistant_no_user_degree \
--all_topics

echo "Running extreme examples for Llama 8B"
python main.py --num_turns 3 \
--prompts_file src/prompts/base_v7_persuasion_degree.yaml \
--topics_file src/topics/extreme_harmful.jsonl \
--persuader_model meta-llama/Meta-Llama-3.1-8B-Instruct \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--experiment_name llama_8b_extreme \
--sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  --num_users 1000 \
--topic_categories NoncontroversiallyHarmful \
--assistant_prompt prompt_assistant_no_user_degree \

echo "Running extreme examples subset for Llama 8B"
python main.py --num_turns 1 \
--prompts_file src/prompts/base_v7_persuasion_degree.yaml \
--topics_file src/topics/extreme_harmful_subset.jsonl \
--persuader_model meta-llama/Meta-Llama-3.1-8B-Instruct \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--experiment_name llama_8b_extreme_subset \
--sample_belief_lower 0 --sample_belief_upper 20 --only_persuade \
--topic_categories NoncontroversiallyHarmful \
--assistant_prompt prompt_assistant_no_user_degree \
--all_topics



