echo "Running extreme subset experiments"
#echo "Running extreme examples for gemini flash 2.0 base"
# python main.py --num_turns 1 \
# --prompts_file src/prompts/base_v7_persuasion_degree.yaml \
# --topics_file src/topics/extreme_harmful.jsonl \
# --persuadee_model gpt-4o \
# --evaluator_model gpt-4o  \
# --experiment_name gemini_flash_2 \
# --sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  --num_users 1000 \
# --topic_categories NoncontroversiallyHarmful \
# --assistant_prompt prompt_assistant_no_user_degree \
# --persuader_model vertex_ai/gemini-2.0-flash-001 --all_topics
#
#echo "Running extreme examples for gemini flash 2.0 lite"
# python main.py --num_turns 1 \
# --prompts_file src/prompts/base_v7_persuasion_degree.yaml \
# --topics_file src/topics/extreme_harmful.jsonl \
# --persuadee_model gpt-4o \
# --evaluator_model gpt-4o  \
# --experiment_name gemini_flash_2_lite \
# --sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  --num_users 1000 \
# --topic_categories NoncontroversiallyHarmful \
# --assistant_prompt prompt_assistant_no_user_degree \
# --persuader_model vertex_ai/gemini-2.0-flash-lite-001 --all_topics
#
#echo "Running extreme examples for gpt 4.0"
# python main.py --num_turns 1 \
# --prompts_file src/prompts/base_v7_persuasion_degree.yaml \
# --topics_file src/topics/extreme_harmful.jsonl \
# --persuadee_model gpt-4o \
# --evaluator_model gpt-4o  \
# --experiment_name gpt4o \
# --sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  --num_users 1000 \
# --topic_categories NoncontroversiallyHarmful \
# --assistant_prompt prompt_assistant_no_user_degree \
# --persuader_model gpt-4o --all_topics
#
# echo "Running extreme examples for gemini-2.5-pro"
# python main.py --num_turns 1 \
# --prompts_file src/prompts/base_v7_persuasion_degree.yaml \
# --topics_file src/topics/extreme_harmful.jsonl \
# --persuadee_model gpt-4o \
# --evaluator_model gpt-4o  \
# --experiment_name gemini_pro_25 \
# --sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  --num_users 1000 \
# --topic_categories NoncontroversiallyHarmful \
# --assistant_prompt prompt_assistant_no_user_degree \
# --persuader_model vertex_ai/gemini-2.5-pro-preview-03-25 --all_topics
#
#  echo "Running extreme examples for gemini-2.5-flash"
# python main.py --num_turns 1 \
# --prompts_file src/prompts/base_v7_persuasion_degree.yaml \
# --topics_file src/topics/extreme_harmful.jsonl \
# --persuadee_model gpt-4o \
# --evaluator_model gpt-4o  \
# --experiment_name gemini_flash_25 \
# --sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  --num_users 1000 \
# --topic_categories NoncontroversiallyHarmful \
# --assistant_prompt prompt_assistant_no_user_degree \
# --persuader_model vertex_ai/gemini-2.5-flash-preview-04-17 --all_topics


echo "Running extreme examples for gemini-1.5 flash"
 python main.py --num_turns 1 \
 --prompts_file src/prompts/base_v7_persuasion_degree.yaml \
 --topics_file src/topics/extreme_harmful.jsonl \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o  \
 --experiment_name gemini_flash_1.5 \
 --sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  --num_users 1000 \
 --topic_categories NoncontroversiallyHarmful \
 --assistant_prompt prompt_assistant_no_user_degree \
 --persuader_model vertex_ai/gemini-1.5-flash-002 --all_topics


