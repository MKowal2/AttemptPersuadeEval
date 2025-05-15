echo "Running OG experiments"
python main.py --num_turns 5 --prompts_file src/prompts/base_v7_persuasion_degree.yaml  \
 --persuader_model gpt-4o \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name bunk_4o \
  --sample_belief_lower 0 \
 --sample_belief_upper 20 --evaluation_scale 2 \
 --only_persuade


 echo "Running Jailbroken experiments"
python main.py --num_turns 5 --prompts_file src/prompts/base_v7_persuasion_degree.yaml  \
 --persuader_model ft:gpt-4o-2024-08-06:far-ai-general::AuBA93Rg --jailbreak \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name bunk_4o_jb \
  --sample_belief_lower 0 \
 --sample_belief_upper 20 --evaluation_scale 2 \
 --only_persuade






