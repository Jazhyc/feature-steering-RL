python scripts/run_evals.py --eval_type lm_eval --runs mild-resonance-1 --tasks mmlu truthfulqa gsm8k --wandb_project Gemma2-2B-muon --with_adapter --print_results
python scripts/run_evals.py --eval_type lm_eval --runs mild-resonance-1 --tasks mmlu truthfulqa gsm8k --wandb_project Gemma2-2B-muon --print_results

#python scripts/run_evals.py --eval_type lm_eval --runs pious-wildflower-11 --tasks mmlu truthfulqa gsm8k --with_adapter --print_results \
#    --alignment_classification_file /root/feature-steering-RL/outputs/12-gemmascope-res-65k__l0-21_alignment_classified_deepseek-v3-0324.json \
#    --style_classification_file /root/feature-steering-RL/outputs/12-gemmascope-res-65k__l0-21_formatting_classified_deepseek-v3-0324.json