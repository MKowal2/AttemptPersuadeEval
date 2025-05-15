echo "Running topic rephrase consistency experiments"
echo "Running gpt-4o"
python main.py --num_turns 5 --prompts_file src/prompts/base_v7_persuasion_degree.yaml  \
 --persuader_model gpt-4o \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name topic_rephrase \
 --num_turns 5 --sample_belief_lower 0 --sample_belief_upper 20 --only_persuade --remove_ratings

 python main.py --num_turns 5 --prompts_file src/prompts/base_v7_persuasion_degree.yaml  \
 --persuader_model gpt-4o \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name topic_rephrase \
 --num_turns 5 --rephrase_topics --sample_belief_lower 0 --sample_belief_upper 20 \
 --remove_ratings  --only_persuade

echo "Running gpt-4o-jailbroken"
python main.py --num_turns 5 --prompts_file src/prompts/base_v7_persuasion_degree.yaml  \
 --persuader_model ft:gpt-4o-2024-08-06:far-ai-general::AuBA93Rg \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name topic_rephrase \
 --prompts_file src/prompts/base_v7_persuasion_degree.yaml \
 --num_turns 5 --sample_belief_lower 0 --sample_belief_upper 20 --only_persuade --remove_ratings

 python main.py --num_turns 5 --prompts_file src/prompts/base_v7_persuasion_degree.yaml  \
 --persuader_model ft:gpt-4o-2024-08-06:far-ai-general::AuBA93Rg \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name topic_rephrase \
 --num_turns 5 --rephrase_topics --sample_belief_lower 0 --sample_belief_upper 20 \
 --only_persuade --remove_ratings
