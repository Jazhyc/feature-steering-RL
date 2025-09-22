RUN_NAME="pious-wildflower-11"

# lm_eval with adapter (without adapter already evaluated)
python scripts/run_evals.py --eval_type lm_eval --runs $RUN_NAME --tasks mmlu truthfulqa gsm8k --with_adapter --print_results

# alpaca with and without adapter
python scripts/run_evals.py --eval_type alpaca --runs $RUN_NAME
python scripts/run_evals.py --eval_type alpaca --runs $RUN_NAME --with_adapter