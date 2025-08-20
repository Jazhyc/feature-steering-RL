"""
adapter:
python scripts/for_paper.py --print_results --with_adapter 2>&1 | tee eval-log.txt

full ft:
python scripts/for_paper.py --runs revived-fire-9 --full_ft --print_results 2>&1 | tee eval-log.txt
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
    parser.add_argument("--with_adapter", action="store_true")
    parser.add_argument("--full_ft", action="store_true")
    args = parser.parse_args()

    print("=" * 30)
    print("Args:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("=" * 30)
    print("Running evaluation...")
    print("=" * 30)

    if args.limit == 0 or args.limit >= 1.0:
        limit = None
    else:
        limit = args.limit

    results = run_eval(args.runs, args.tasks, limit=limit, with_adapter=args.with_adapter, full_ft=args.full_ft)
    if args.print_results:
        pretty_results(results)

if __name__ == "__main__":
    main()