echo "Running DEBUNK experiments"
python main.py --num_turns 5 --prompts_file src/prompts/base_v6_evaluation_comparison.yaml  \
 --persuader_model gpt-4o \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name persuade_out_of_conspiracy_4o_eval_4omini \
  --sample_belief_lower 80 \
 --sample_belief_upper 100 --evaluation_scale 2

python main.py --num_turns 5 --prompts_file src/prompts/base_v6_evaluation_comparison.yaml  \
 --persuader_model gpt-4o \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name persuade_out_of_conspiracy_4o_eval_4omini \
  --sample_belief_lower 80 \
 --sample_belief_upper 100 --evaluation_scale 3

 python main.py --num_turns 5 --prompts_file src/prompts/base_v6_evaluation_comparison.yaml  \
 --persuader_model gpt-4o \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name persuade_out_of_conspiracy_4o_eval_4omini \
  --sample_belief_lower 80 \
 --sample_belief_upper 100 --evaluation_scale 100


echo "Running BUNK experiments"
python main.py --num_turns 5 --prompts_file src/prompts/base_v6_evaluation_comparison.yaml  \
 --persuader_model gpt-4o \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name persuade_out_of_conspiracy_4o_eval_4omini \
  --sample_belief_lower 0 \
 --sample_belief_upper 20 --evaluation_scale 2

 python main.py --num_turns 5 --prompts_file src/prompts/base_v6_evaluation_comparison.yaml  \
 --persuader_model gpt-4o \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name persuade_out_of_conspiracy_4o_eval_4omini \
  --sample_belief_lower 0 \
 --sample_belief_upper 20 --evaluation_scale 3

 python main.py --num_turns 5 --prompts_file src/prompts/base_v6_evaluation_comparison.yaml  \
 --persuader_model gpt-4o \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name persuade_out_of_conspiracy_4o_eval_4omini \
  --sample_belief_lower 0 \
 --sample_belief_upper 20 --evaluation_scale 100


echo "Running BASELINE experiments"
python main.py --num_turns 5 --prompts_file src/prompts/base_v6_evaluation_comparison.yaml  \
 --persuader_model gpt-4o \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name persuade_out_of_conspiracy_4o_eval_4omini \
  --sample_belief_lower 0 \
 --sample_belief_upper 20 --evaluation_scale 2 --baseline_persuasion

 python main.py --num_turns 5 --prompts_file src/prompts/base_v6_evaluation_comparison.yaml  \
 --persuader_model gpt-4o \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name persuade_out_of_conspiracy_4o_eval_4omini \
  --sample_belief_lower 0 \
 --sample_belief_upper 20 --evaluation_scale 3 --baseline_persuasion

 python main.py --num_turns 5 --prompts_file src/prompts/base_v6_evaluation_comparison.yaml  \
 --persuader_model gpt-4o \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name persuade_out_of_conspiracy_4o_eval_4omini \
  --sample_belief_lower 0 \
 --sample_belief_upper 20 --evaluation_scale 100 --baseline_persuasion



