# lm_eval with adapter (without adapter already evaluated)

#python scripts/run_evals.py --eval_type lm_eval --runs $RUN_NAME --tasks mmlu truthfulqa gsm8k --with_adapter --print_results

# alpaca 

# gemma 2 2b baseline, steering, train-ablate
python scripts/run_evals.py --eval_type alpaca --runs mild-resonance-1 --wandb_project Gemma2-2B-muon --base_model google/gemma-2-2b-it
python scripts/run_evals.py --eval_type alpaca --runs mild-resonance-1 --wandb_project Gemma2-2B-muon --base_model google/gemma-2-2b-it --with_adapter
python scripts/run_evals.py --eval_type alpaca --runs rosy-cloud-4 --wandb_project Gemma2-2B-train-ablate --base_model google/gemma-2-2b-it --with_adapter

# gemma 2 9b baseline, steering, train-ablate
python scripts/run_evals.py --eval_type alpaca --runs smart-dew-4 --wandb_project Gemma2-9B-muon --base_model google/gemma-2-9b-it
python scripts/run_evals.py --eval_type alpaca --runs smart-dew-4 --wandb_project Gemma2-9B-muon --base_model google/gemma-2-9b-it --with_adapter
python scripts/run_evals.py --eval_type alpaca --runs divine-frost-1 --wandb_project Gemma2-9B-train-ablate --base_model google/gemma-2-9b-it --with_adapter