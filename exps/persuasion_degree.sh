echo "Running Degree Experiments"
echo "Running 2 degree evaluation for GPT-4o"
python main.py --num_turns 3 \
--persuader_model gpt-4o \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name gpt_4o_2 \
--sample_belief_lower 0 --sample_belief_upper 100  \
--assistant_prompt prompt_assistant_no_user \
--all_topics --batch_size 16 --results_dir results/persuasion_degree/gpt_4o_2 \
--evaluation_scale 2



echo "Running 3 degree evaluation for GPT-4o"
python main.py --num_turns 3 \
--persuader_model gpt-4o \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name gpt_4o_3 \
--sample_belief_lower 0 --sample_belief_upper 100  \
--assistant_prompt prompt_assistant_no_user \
--all_topics --batch_size 16 --results_dir results/persuasion_degree/gpt_4o_3 \
--evaluation_scale 3


echo "Running 100 degree evaluation for GPT-4o"
python main.py --num_turns 3 \
--persuader_model gpt-4o \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name gpt_4o_100 \
--sample_belief_lower 0 --sample_belief_upper 100  \
--assistant_prompt prompt_assistant_no_user \
--all_topics --batch_size 16 --results_dir results/persuasion_degree/gpt_4o_100 \
--evaluation_scale 100


echo "Running 2 degree evaluation for Llama 8B"
python main.py --num_turns 3 \
--persuader_model hf/Meta-Llama-3.1-8B-Instruct \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name llama_8b_2 \
--sample_belief_lower 0 --sample_belief_upper 100  \
--assistant_prompt prompt_assistant_no_user \
--all_topics --batch_size 16 --results_dir results/persuasion_degree/llama_8b_2 \
--evaluation_scale 2



echo "Running 3 degree evaluation for Llama 8B"
python main.py --num_turns 3 \
--persuader_model hf/Meta-Llama-3.1-8B-Instruct \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name llama_8b_3 \
--sample_belief_lower 0 --sample_belief_upper 100  \
--assistant_prompt prompt_assistant_no_user \
--all_topics --batch_size 16 --results_dir results/persuasion_degree/llama_8b_3 \
--evaluation_scale 3


echo "Running 100 degree evaluation for Llama 8B"
python main.py --num_turns 3 \
--persuader_model hf/Meta-Llama-3.1-8B-Instruct \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name llama_8b_100 \
--sample_belief_lower 0 --sample_belief_upper 100  \
--assistant_prompt prompt_assistant_no_user \
--all_topics --batch_size 16 --results_dir results/persuasion_degree/llama_8b_100 \
--evaluation_scale 100
