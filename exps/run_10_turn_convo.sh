echo "Running full evaluation for gpt-4o"
python main.py --num_turns 10 \
--prompts_file src/prompts/base_v7_persuasion_degree.yaml \
--persuader_model gpt-4o \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name gpt_4o \
--sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  \
--assistant_prompt prompt_assistant_no_user_degree \
--all_topics --results_dir results/long_convo/gpt_4o_10_turns


echo "Running full evaluation for Llama 8B"
python main.py --num_turns 10 \
--persuader_model hf/Meta-Llama-3.1-8B-Instruct \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name llama_8b \
--sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  \
--assistant_prompt prompt_assistant_no_user_degree \
--all_topics --batch_size 16 --results_dir results/long_convo/llama_8b_10_turns