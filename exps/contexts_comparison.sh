echo "Running situational contexts evaluation for GPT-4o"
echo "Online Debater"
python main.py --num_turns 3 \
--persuader_model gpt-4o \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name GPT-4o_Online_Debater \
--sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  \
--assistant_prompt prompt_assistant_contexts \
--contexts_file src/prompts/contexts/situations/situation_context_online_debater.jsonl \
--all_topics \
--results_dir results/situational_contexts/gpt_4o_situational_contexts_online_debater

echo "Peer Support"
python main.py --num_turns 3 \
--persuader_model gpt-4o \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name GPT-4o_Peer_Support \
--sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  \
--assistant_prompt prompt_assistant_contexts \
--contexts_file src/prompts/contexts/situations/situation_context_peer_support.jsonl \
--all_topics \
--results_dir results/situational_contexts/gpt_4o_situational_contexts_peer_support

echo "Virtual Influencer"
python main.py --num_turns 3 \
--persuader_model gpt-4o \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name GPT-4o_Virtual_Influencer \
--sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  \
--assistant_prompt prompt_assistant_contexts \
--contexts_file src/prompts/contexts/situations/situation_context_virtual_influencer.jsonl \
--all_topics \
--results_dir results/situational_contexts/gpt_4o_situational_contexts_virtual_influencer

echo "Controversial Topic Guide"
python main.py --num_turns 3 \
--persuader_model gpt-4o \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name GPT-4o_Controversial_Topic_Guide \
--sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  \
--assistant_prompt prompt_assistant_contexts \
--contexts_file src/prompts/contexts/situations/situation_context_controversial_topic_guide.jsonl \
--all_topics \
--results_dir results/situational_contexts/gpt_4o_situational_contexts_controversial_topic_guide

echo "Political Strategist"
python main.py --num_turns 3 \
--persuader_model gpt-4o \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name GPT-4o_Political_Strategist \
--sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  \
--assistant_prompt prompt_assistant_contexts \
--contexts_file src/prompts/contexts/situations/situation_context_political_strategist.jsonl \
--all_topics \
--results_dir results/situational_contexts/gpt_4o_situational_contexts_political_strategist


echo "--------------------------------"
echo "--------------------------------"
echo "--------------------------------"
echo "--------------------------------"


echo "Running situational contexts evaluation for Llama 8B"
echo "Online Debater"
python main.py --num_turns 3 \
--persuader_model hf/Meta-Llama-3.1-8B-Instruct \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name Llama-8B_Online_Debater \
--sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  \
--assistant_prompt prompt_assistant_contexts \
--contexts_file src/prompts/contexts/situations/situation_context_online_debater.jsonl \
--all_topics --batch_size 16 \
--results_dir results/situational_contexts/llama_8b_situational_contexts_online_debater

echo "Peer Support"
python main.py --num_turns 3 \
--persuader_model hf/Meta-Llama-3.1-8B-Instruct \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name Llama-8B_Peer_Support \
--sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  \
--assistant_prompt prompt_assistant_contexts \
--contexts_file src/prompts/contexts/situations/situation_context_peer_support.jsonl \
--all_topics --batch_size 16 \
--results_dir results/situational_contexts/llama_8b_situational_contexts_peer_support

echo "Virtual Influencer"
python main.py --num_turns 3 \
--persuader_model hf/Meta-Llama-3.1-8B-Instruct \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name Llama-8B_Virtual_Influencer \
--sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  \
--assistant_prompt prompt_assistant_contexts \
--contexts_file src/prompts/contexts/situations/situation_context_virtual_influencer.jsonl \
--all_topics --batch_size 16 \
--results_dir results/situational_contexts/llama_8b_situational_contexts_virtual_influencer

echo "Controversial Topic Guide"
python main.py --num_turns 3 \
--persuader_model hf/Meta-Llama-3.1-8B-Instruct \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name Llama-8B_Controversial_Topic_Guide \
--sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  \
--assistant_prompt prompt_assistant_contexts \
--contexts_file src/prompts/contexts/situations/situation_context_controversial_topic_guide.jsonl \
--all_topics --batch_size 16 \
--results_dir results/situational_contexts/llama_8b_situational_contexts_controversial_topic_guide

echo "Political Strategist"
python main.py --num_turns 3 \
--persuader_model hf/Meta-Llama-3.1-8B-Instruct \
--persuadee_model gpt-4o \
--evaluator_model gpt-4o  \
--reject_model gpt-4o-mini \
--experiment_name Llama-8B_Political_Strategist \
--sample_belief_lower 0 --sample_belief_upper 20 --only_persuade  \
--assistant_prompt prompt_assistant_contexts \
--contexts_file src/prompts/contexts/situations/situation_context_political_strategist.jsonl \
--all_topics --batch_size 16 \
--results_dir results/situational_contexts/llama_8b_situational_contexts_political_strategist
