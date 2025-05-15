
echo "Running GPT-4o continuation experiments"
python main.py --num_turns 5 --prompts_file src/prompts/base_v7_persuasion_degree.yaml  \
 --persuader_model gpt-4o \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name human_cont \
 --prompts_file src/prompts/base_v7_persuasion_degree.yaml \
 --human_data_path src/human_data/filtered_conversations.json --human_data_rounds 1 \
 --num_turns 5 --assistant_prompt human

python main.py --num_turns 5 --prompts_file src/prompts/base_v7_persuasion_degree.yaml  \
 --persuader_model gpt-4o \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name human_cont \
 --prompts_file src/prompts/base_v7_persuasion_degree.yaml \
 --human_data_path src/human_data/filtered_conversations.json --human_data_rounds 2 \
 --num_turns 5 --assistant_prompt human

 python main.py --num_turns 5 --prompts_file src/prompts/base_v7_persuasion_degree.yaml  \
 --persuader_model gpt-4o \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name human_cont \
 --prompts_file src/prompts/base_v7_persuasion_degree.yaml \
 --human_data_path src/human_data/filtered_conversations.json --human_data_rounds 3 \
 --num_turns 5 --assistant_prompt human

  python main.py --num_turns 5 --prompts_file src/prompts/base_v7_persuasion_degree.yaml  \
 --persuader_model gpt-4o \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name human_cont \
 --prompts_file src/prompts/base_v7_persuasion_degree.yaml \
 --human_data_path src/human_data/filtered_conversations.json --human_data_rounds 4 \
--num_turns 5 --assistant_prompt human

  python main.py --num_turns 5 --prompts_file src/prompts/base_v7_persuasion_degree.yaml  \
 --persuader_model gpt-4o \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name human_cont \
 --prompts_file src/prompts/base_v7_persuasion_degree.yaml \
 --human_data_path src/human_data/filtered_conversations.json --human_data_rounds 5 \
 --num_turns 5 --assistant_prompt human




echo "Running JAILBROKEN continuation experiments"
python main.py --num_turns 5 --prompts_file src/prompts/base_v7_persuasion_degree.yaml  \
 --persuader_model ft:gpt-4o-2024-08-06:far-ai-general::AuBA93Rg \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name human_cont \
 --prompts_file src/prompts/base_v7_persuasion_degree.yaml \
 --human_data_path src/human_data/filtered_conversations.json --human_data_rounds 1 \
 --num_turns 5 --assistant_prompt human

python main.py --num_turns 5 --prompts_file src/prompts/base_v7_persuasion_degree.yaml  \
 --persuader_model ft:gpt-4o-2024-08-06:far-ai-general::AuBA93Rg \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name human_cont \
 --prompts_file src/prompts/base_v7_persuasion_degree.yaml \
 --human_data_path src/human_data/filtered_conversations.json --human_data_rounds 2 \
 --num_turns 5 --assistant_prompt human

 python main.py --num_turns 5 --prompts_file src/prompts/base_v7_persuasion_degree.yaml  \
 --persuader_model ft:gpt-4o-2024-08-06:far-ai-general::AuBA93Rg \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name human_cont \
 --prompts_file src/prompts/base_v7_persuasion_degree.yaml \
 --human_data_path src/human_data/filtered_conversations.json --human_data_rounds 3 \
 --num_turns 5 --assistant_prompt human

 python main.py --num_turns 5 --prompts_file src/prompts/base_v7_persuasion_degree.yaml  \
 --persuader_model ft:gpt-4o-2024-08-06:far-ai-general::AuBA93Rg \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name human_cont \
 --prompts_file src/prompts/base_v7_persuasion_degree.yaml \
 --human_data_path src/human_data/filtered_conversations.json --human_data_rounds 4 \
 --num_turns 5 --assistant_prompt human

 python main.py --num_turns 5 --prompts_file src/prompts/base_v7_persuasion_degree.yaml  \
 --persuader_model ft:gpt-4o-2024-08-06:far-ai-general::AuBA93Rg \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name human_cont \
 --prompts_file src/prompts/base_v7_persuasion_degree.yaml \
 --human_data_path src/human_data/filtered_conversations.json --human_data_rounds 5 \
 --num_turns 5 --assistant_prompt human

