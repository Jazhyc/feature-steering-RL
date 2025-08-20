"""
python scripts/for_paper.py --print_results | tee scripts/logs/eval-log.txt
"""
import sys
import os
# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evals import run_eval, pretty_results
import argparse

runs = ["royal-valley-2", "mild-glade-10"] # stable, sparse
tasks = ["mmlu", "truthfulqa", "gsm8k"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=runs)
    parser.add_argument("--tasks", nargs="+", default=tasks)
    parser.add_argument("--limit", type=float, default=1.0)
    parser.add_argument("--print_results", action="store_true")
    args = parser.parse_args()

    results = run_eval(args.runs, args.tasks, limit=args.limit)
    if args.print_results:
        pretty_results(results)

if __name__ == "__main__":
    main()