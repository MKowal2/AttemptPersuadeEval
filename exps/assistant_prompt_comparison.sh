echo "Running assistant prompt comparison experiments"
echo "Running gpt-4o"
python main.py --num_turns 5 --prompts_file src/prompts/base_v7_persuasion_degree.yaml  \
 --persuader_model gpt-4o \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o --all_topics \
 --experiment_name assistant_prompt_4o_og --remove_ratings \
 --num_turns 5 --sample_belief_lower 0 --sample_belief_upper 20 --only_persuade \
 --assistant_prompt prompt_assistant_og

 python main.py --num_turns 5 --prompts_file src/prompts/base_v7_persuasion_degree.yaml  \
 --persuader_model gpt-4o \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o --all_topics \
 --experiment_name assistant_prompt_4o_nouser --remove_ratings \
 --num_turns 5 --sample_belief_lower 0 --sample_belief_upper 20 --only_persuade \
 --assistant_prompt prompt_assistant_no_user

 python main.py --num_turns 5 --prompts_file src/prompts/base_v7_persuasion_degree.yaml  \
 --persuader_model gpt-4o \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o --all_topics \
 --experiment_name assistant_prompt_4o_nodegree --remove_ratings \
 --num_turns 5 --sample_belief_lower 0 --sample_belief_upper 20 --only_persuade \
 --assistant_prompt prompt_assistant_no_degree

 python main.py --num_turns 5 --prompts_file src/prompts/base_v7_persuasion_degree.yaml  \
 --persuader_model gpt-4o \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o --all_topics \
 --experiment_name assistant_prompt_4o_nouserdegree_newtopic --remove_ratings \
 --num_turns 5 --sample_belief_lower 0 --sample_belief_upper 20 --only_persuade \
 --assistant_prompt prompt_assistant_no_user_degree

  python main.py --num_turns 5 --prompts_file src/prompts/base_v7_persuasion_degree.yaml  \
 --persuader_model gpt-4o \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o --all_topics \
 --experiment_name assistant_prompt_4o_baseline --remove_ratings \
 --num_turns 5 --sample_belief_lower 0 --sample_belief_upper 20 --only_persuade \
 --assistant_prompt prompt_assistant_baseline

# echo "Running gpt-4o JAILBREAK"
#python main.py --num_turns 5 --prompts_file src/prompts/base_v7_persuasion_degree.yaml  \
# --persuader_model ft:gpt-4o-2024-08-06:far-ai-general::AuBA93Rg --jailbreak \
# --persuadee_model gpt-4o \
# --evaluator_model gpt-4o --all_topics \
# --experiment_name assistant_prompt_jb_og --remove_ratings \
# --num_turns 5 --sample_belief_lower 0 --sample_belief_upper 20 --only_persuade \
# --assistant_prompt prompt_assistant_og
#
# python main.py --num_turns 5 --prompts_file src/prompts/base_v7_persuasion_degree.yaml  \
# --persuader_model ft:gpt-4o-2024-08-06:far-ai-general::AuBA93Rg --jailbreak \
# --persuadee_model gpt-4o \
# --evaluator_model gpt-4o --all_topics \
# --experiment_name assistant_prompt_jb_nouser --remove_ratings \
# --num_turns 5 --sample_belief_lower 0 --sample_belief_upper 20 --only_persuade \
# --assistant_prompt prompt_assistant_no_user
#
# python main.py --num_turns 5 --prompts_file src/prompts/base_v7_persuasion_degree.yaml  \
# --persuader_model ft:gpt-4o-2024-08-06:far-ai-general::AuBA93Rg --jailbreak \
# --persuadee_model gpt-4o \
# --evaluator_model gpt-4o --all_topics \
# --experiment_name assistant_prompt_jb_nodegree --remove_ratings \
# --num_turns 5 --sample_belief_lower 0 --sample_belief_upper 20 --only_persuade \
# --assistant_prompt prompt_assistant_no_degree
#
# python main.py --num_turns 5 --prompts_file src/prompts/base_v7_persuasion_degree.yaml  \
# --persuader_model ft:gpt-4o-2024-08-06:far-ai-general::AuBA93Rg --jailbreak \
# --persuadee_model gpt-4o \
# --evaluator_model gpt-4o --all_topics \
# --experiment_name assistant_prompt_jb_nouserdegree --remove_ratings \
# --num_turns 5 --sample_belief_lower 0 --sample_belief_upper 20 --only_persuade \
# --assistant_prompt prompt_assistant_no_user_degree
#
#  python main.py --num_turns 5 --prompts_file src/prompts/base_v7_persuasion_degree.yaml  \
# --persuader_model ft:gpt-4o-2024-08-06:far-ai-general::AuBA93Rg --jailbreak \
# --persuadee_model gpt-4o \
# --evaluator_model gpt-4o --all_topics \
# --experiment_name assistant_prompt_jb_baseline --remove_ratings \
# --num_turns 5 --sample_belief_lower 0 --sample_belief_upper 20 --only_persuade \
# --assistant_prompt prompt_assistant_baseline