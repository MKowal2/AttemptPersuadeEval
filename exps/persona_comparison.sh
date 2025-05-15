echo "Running persona evaluation for GPT-4o"
echo "Running JOURNALIST persona evaluation for GPT-4o"
python main.py --num_turns 3 \
--persuader_model gpt-4o \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name gpt_4o_journalist \
--sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  \
--assistant_prompt prompt_assistant_contexts \
--contexts_file src/prompts/contexts/contexts_journalist.jsonl \
--all_topics --batch_size 16 --results_dir results/contexts/gpt_4o_journalist

echo "Running POLITICAL persona evaluation for GPT-4o"
python main.py --num_turns 3 \
--persuader_model gpt-4o \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name gpt_4o_politics \
--sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  \
--assistant_prompt prompt_assistant_contexts \
--contexts_file src/prompts/contexts/contexts_politics.jsonl \
--all_topics --batch_size 16 --results_dir results/contexts/gpt_4o_politics


echo "Running persona evaluation for Llama 8B"
echo "Running JOURNALIST persona evaluation for Llama 8B"
python main.py --num_turns 3 \
--persuader_model hf/Meta-Llama-3.1-8B-Instruct \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name llama_8b_journalist \
--sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  \
--assistant_prompt prompt_assistant_contexts \
--contexts_file src/prompts/contexts/contexts_journalist.jsonl \
--all_topics --batch_size 16 --results_dir results/contexts/llama_8b_journalist

echo "Running POLITICAL persona evaluation for Llama 8B"
python main.py --num_turns 3 \
--persuader_model hf/Meta-Llama-3.1-8B-Instruct \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name llama_8b_politics \
--sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  \
--assistant_prompt prompt_assistant_contexts \
--contexts_file src/prompts/contexts/contexts_politics.jsonl \
--all_topics --batch_size 16 --results_dir results/contexts/llama_8b_politics