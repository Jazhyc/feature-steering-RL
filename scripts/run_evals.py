"""
Unified evaluation script for running lm-eval and AlpacaEval.

Usage:

LMEval:
python scripts/run_evals.py --eval_type lm_eval --runs pious-wildflower-11 --tasks mmlu truthfulqa gsm8k --with_adapter --print_results

AlpacaEval:
python scripts/run_evals.py --eval_type alpaca --runs pious-wildflower-11 --with_adapter
"""
import sys
import os
import argparse

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fsrl.evals.lmarena import run_eval as run_lm_eval, pretty_results
from fsrl.evals.alpaca import run_alpaca_eval

def main():
    parser = argparse.ArgumentParser(description="Run evaluations for feature steering models.")
    parser.add_argument("--eval_type", type=str, required=True, choices=["lm_eval", "alpaca"], help="Type of evaluation to run.")
    parser.add_argument("--runs", nargs="+", required=True, help="List of run names to evaluate.")
    parser.add_argument("--with_adapter", action="store_true", help="Use adapter for steering.")
    parser.add_argument("--full_ft", action="store_true", help="Use full fine-tuned model.")
    parser.add_argument("--limit", type=float, help="Limit the number of examples for evaluation (float for lm_eval, int for alpaca).")
    parser.add_argument("--print_results", action="store_true", help="Print detailed results.")

    # LMEval specific
    parser.add_argument("--tasks", nargs="+", default=["mmlu", "truthfulqa", "gsm8k"], help="Tasks for lm-eval.")

    # AlpacaEval specific
    parser.add_argument("--annotator", default="alpaca_eval_gpt4", help="AlpacaEval annotator to use.")

    args = parser.parse_args()

    print("=" * 30)
    print("Args:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("=" * 30)
    print("Running evaluation...")
    print("=" * 30)

    if args.eval_type == "lm_eval":
        if args.limit and (args.limit == 0 or args.limit >= 1.0):
            limit = None
        else:
            limit = args.limit

        results = run_lm_eval(
            runs=args.runs,
            tasks=args.tasks,
            limit=limit,
            with_adapter=args.with_adapter,
            full_ft=args.full_ft
        )
        if args.print_results:
            pretty_results(results)

    elif args.eval_type == "alpaca":
        limit = int(args.limit) if args.limit else None
        results = run_alpaca_eval(
            runs=args.runs,
            with_adapter=args.with_adapter,
            full_ft=args.full_ft,
            annotator=args.annotator,
            limit=limit
        )
        
        print("\n" + "=" * 50)
        print("ALPACA EVAL RESULTS SUMMARY")
        print("=" * 50)
        
        for run, result in results.items():
            print(f"\nRun: {run}")
            if isinstance(result, dict) and 'win_rate' in result:
                print(f"Win Rate: {result['win_rate']:.3f}")
            else:
                print(f"Result: {result}")
        
        print("\nDetailed results saved in individual run directories.")

if __name__ == "__main__":
    main() 