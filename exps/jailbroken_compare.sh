echo "Running DEBUNK experiments"
echo "Using gpt-4o as evaluator"
python main.py --num_turns 5 --prompts_file prompts/base_v3_three_point_scale.yaml  \
 --persuader_model gpt-4o \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name persuade_out_of_conspiracy_4o_eval_4o \
 --sample_belief_lower 80 \
 --sample_belief_upper 100
 python main.py --num_turns 5 --prompts_file prompts/base_v3_three_point_scale.yaml  \
 --persuader_model gpt-4o-mini \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name persuade_out_of_conspiracy_4omini_eval_4o \
  --sample_belief_lower 80 \
 --sample_belief_upper 100
python main.py --num_turns 5  --prompts_file prompts/base_v3_three_point_scale.yaml \
 --persuader_model ft:gpt-4o-2024-08-06:far-ai-general::AuBA93Rg \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name persuade_out_of_conspiracy_eval_jailbreak_4o \
 --sample_belief_lower 80 \
 --sample_belief_upper 100 --jailbreak

echo "Using gpt-4o-mini as evaluator"

python main.py --num_turns 5 --prompts_file prompts/base_v3_three_point_scale.yaml  \
 --persuader_model gpt-4o \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o-mini \
 --experiment_name persuade_out_of_conspiracy_4o_eval_4omini \
  --sample_belief_lower 80 \
 --sample_belief_upper 100
 python main.py --num_turns 5 --prompts_file prompts/base_v3_three_point_scale.yaml  \
 --persuader_model gpt-4o-mini \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o-mini \
 --experiment_name persuade_out_of_conspiracy_4omini_eval_4omini \
  --sample_belief_lower 80 \
 --sample_belief_upper 100
python main.py --num_turns 5  --prompts_file prompts/base_v3_three_point_scale.yaml \
 --persuader_model ft:gpt-4o-2024-08-06:far-ai-general::AuBA93Rg \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o-mini \
 --experiment_name persuade_out_of_conspiracy_eval_jailbreak_4omini \
 --sample_belief_lower 80 \
 --sample_belief_upper 100 --jailbreak



echo "Running BUNK experiments"
echo "Using gpt-4o as evaluator"
python main.py --num_turns 5 --prompts_file prompts/base_v3_three_point_scale.yaml  \
 --persuader_model gpt-4o \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name persuade_out_of_conspiracy_4o_eval_4o \
 --sample_belief_lower 0 \
 --sample_belief_upper 20
 python main.py --num_turns 5 --prompts_file prompts/base_v3_three_point_scale.yaml  \
 --persuader_model gpt-4o-mini \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name persuade_out_of_conspiracy_4omini_eval_4o \
  --sample_belief_lower 0 \
 --sample_belief_upper 20
python main.py --num_turns 5  --prompts_file prompts/base_v3_three_point_scale.yaml \
 --persuader_model ft:gpt-4o-2024-08-06:far-ai-general::AuBA93Rg \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name persuade_out_of_conspiracy_eval_jailbreak_4o \
 --sample_belief_lower 0 \
 --sample_belief_upper 20 --jailbreak

echo "Using gpt-4o-mini as evaluator"

python main.py --num_turns 5 --prompts_file prompts/base_v3_three_point_scale.yaml  \
 --persuader_model gpt-4o \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o-mini \
 --experiment_name persuade_out_of_conspiracy_4o_eval_4omini \
  --sample_belief_lower 0 \
 --sample_belief_upper 20
 python main.py --num_turns 5 --prompts_file prompts/base_v3_three_point_scale.yaml  \
 --persuader_model gpt-4o-mini \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o-mini \
 --experiment_name persuade_out_of_conspiracy_4omini_eval_4omini \
  --sample_belief_lower 0 \
 --sample_belief_upper 20
python main.py --num_turns 5  --prompts_file prompts/base_v3_three_point_scale.yaml \
 --persuader_model ft:gpt-4o-2024-08-06:far-ai-general::AuBA93Rg \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o-mini \
 --experiment_name persuade_out_of_conspiracy_eval_jailbreak_4omini \
 --sample_belief_lower 0 \
 --sample_belief_upper 20 --jailbreak


echo "Running BUNK/DEBUNK experiments"
echo "Using gpt-4o as evaluator"
python main.py --num_turns 5 --prompts_file prompts/base_v3_three_point_scale.yaml  \
 --persuader_model gpt-4o \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name persuade_in_and_out_of_conspiracy_4o_eval_4o \
 --sample_belief_lower 0 \
 --sample_belief_upper 100
 python main.py --num_turns 5 --prompts_file prompts/base_v3_three_point_scale.yaml  \
 --persuader_model gpt-4o-mini \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name persuade_in_and_out_of_conspiracy_4omini_eval_4o \
  --sample_belief_lower 0 \
 --sample_belief_upper 100
python main.py --num_turns 5  --prompts_file prompts/base_v3_three_point_scale.yaml \
 --persuader_model ft:gpt-4o-2024-08-06:far-ai-general::AuBA93Rg \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name persuade_in_and_out_of_conspiracy_eval_jailbreak_4o \
 --sample_belief_lower 0 \
 --sample_belief_upper 100 --jailbreak

echo "Using gpt-4o-mini as evaluator"

python main.py --num_turns 5 --prompts_file prompts/base_v3_three_point_scale.yaml  \
 --persuader_model gpt-4o \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o-mini \
 --experiment_name persuade_in_and_out_of_conspiracy_4o_eval_4omini \
  --sample_belief_lower 0 \
 --sample_belief_upper 100
 python main.py --num_turns 5 --prompts_file prompts/base_v3_three_point_scale.yaml  \
 --persuader_model gpt-4o-mini \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o-mini \
 --experiment_name persuade_in_and_out_of_conspiracy_4omini_eval_4omini \
  --sample_belief_lower 0 \
 --sample_belief_upper 100
python main.py --num_turns 5  --prompts_file prompts/base_v3_three_point_scale.yaml \
 --persuader_model ft:gpt-4o-2024-08-06:far-ai-general::AuBA93Rg \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o-mini \
 --experiment_name persuade_in_and_out_of_conspiracy_eval_jailbreak_4omini \
 --sample_belief_lower 0 \
 --sample_belief_upper 100 --jailbreak



