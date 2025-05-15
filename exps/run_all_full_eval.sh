echo "RUNNING NUMBER 1"

echo "Running full evaluation for gpt-4o"
python main.py --num_turns 3 \
--persuader_model gpt-4o \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name gpt_4o \
--sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  \
--assistant_prompt prompt_assistant_no_user_degree \
--all_topics

echo "Running full evaluation for gpt-4o-jb"
python main.py --num_turns 3 \
--persuader_model ft:gpt-4o-2024-08-06:far-ai-general::AuBA93Rg \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name gpt_4o_jb \
--sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  \
--assistant_prompt prompt_assistant_no_user_degree \
--all_topics

echo "Running full evaluation for vertex_ai/gemini-2.5-pro-preview-03-25"
python main.py --num_turns 3 \
--persuader_model vertex_ai/gemini-2.5-pro-preview-03-25 \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name gemini_25_pro \
--sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  \
--assistant_prompt prompt_assistant_no_user_degree \
--all_topics

echo "Running full evaluation for gpt-4o-mini"
python main.py --num_turns 3 \
--persuader_model gpt-4o-mini \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name gpt_4o_mini \
--sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  \
--assistant_prompt prompt_assistant_no_user_degree \
--all_topics

echo "Running full evaluation for vertex_ai/gemini-2.0-flash-001"
python main.py --num_turns 3 \
--persuader_model vertex_ai/gemini-2.0-flash-001 \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name gemini_flash_001 \
--sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  \
--assistant_prompt prompt_assistant_no_user_degree \
--all_topics

echo "Running full evaluation for Llama 8B"
python main.py --num_turns 3 \
--persuader_model hf/Meta-Llama-3.1-8B-Instruct \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name llama_8b \
--sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  \
--assistant_prompt prompt_assistant_no_user_degree \
--all_topics

echo "Running full evaluation for hf/Qwen3-32B"
python main.py --num_turns 3 \
--persuader_model hf/Qwen3-32B \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name qwen3_32b \
--sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  \
--assistant_prompt prompt_assistant_no_user_degree \
--all_topics







echo "RUNNING NUMBER 2"

echo "Running full evaluation for vertex_ai/gemini-2.5-pro-preview-03-25"
python main.py --num_turns 3 \
--topics_file src/topics/diverse_topics_conspiracy.jsonl \
--persuader_model vertex_ai/gemini-2.5-pro-preview-03-25 \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name gemini_25_pro \
--sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  \
--assistant_prompt prompt_assistant_no_user_degree \
--all_topics

echo "RUNNING NUMBER 3"

echo "Running full evaluation for vertex_ai/gemini-2.5-pro-preview-03-25"
python main.py --num_turns 3 \
--topics_file src/topics/diverse_topics_conspiracy.jsonl \
--persuader_model vertex_ai/gemini-2.5-pro-preview-03-25 \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name gemini_25_pro \
--sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  \
--assistant_prompt prompt_assistant_no_user_degree \
--all_topics

echo "RUNNING NUMBER 4"

echo "Running full evaluation for vertex_ai/gemini-2.5-pro-preview-03-25"
python main.py --num_turns 3 \
--prompts_file src/prompts/base_v7_persuasion_degree.yaml \
--topics_file src/topics/diverse_topics_conspiracy.jsonl \
--persuader_model vertex_ai/gemini-2.5-pro-preview-03-25 \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name gemini_25_pro \
--sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  \
--assistant_prompt prompt_assistant_no_user_degree \
--all_topics

echo "RUNNING NUMBER 5"


echo "Running full evaluation for vertex_ai/gemini-2.5-pro-preview-03-25"
python main.py --num_turns 3 \
--topics_file src/topics/diverse_topics_conspiracy.jsonl \
--persuader_model vertex_ai/gemini-2.5-pro-preview-03-25 \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name gemini_25_pro \
--sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  \
--assistant_prompt prompt_assistant_no_user_degree \
--all_topics