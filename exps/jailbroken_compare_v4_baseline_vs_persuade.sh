
 python main.py --num_turns 5 --prompts_file prompts/base_v4_implicit_eval.yaml  \
 --persuader_model gpt-4o \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name persuade_into_conspiracy_4o_eval_4o_baseline \
 --sample_belief_lower 0 \
 --sample_belief_upper 20 --baseline_persuasion

 python main.py --num_turns 5 --prompts_file prompts/base_v4_implicit_eval.yaml  \
 --persuader_model gpt-4o \
 --persuadee_model gpt-4o \
 --evaluator_model gpt-4o \
 --experiment_name persuade_into_conspiracy_4o_eval_4o \
 --sample_belief_lower 0 \
 --sample_belief_upper 20

