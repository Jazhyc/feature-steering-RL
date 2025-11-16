"""
Unified evaluation script for running lm-eval and AlpacaEval.

Usage:

LMEval:
python scripts/run_evals.py --eval_type lm_eval --runs pious-wildflower-11 --tasks mmlu truthfulqa gsm8k --with_adapter --print_results

LMEval with Feature Ablation:
python scripts/run_evals.py --eval_type lm_eval --runs pious-wildflower-11 --tasks mmlu truthfulqa gsm8k --with_adapter --print_results \
    --alignment_classification_file /root/feature-steering-RL/outputs/12-gemmascope-res-65k__l0-21_alignment_classified_deepseek-v3-0324.json \
    --style_classification_file /root/feature-steering-RL/outputs/12-gemmascope-res-65k__l0-21_formatting_classified_deepseek-v3-0324.json

LMEval with Custom Ablation:
python scripts/run_evals.py --eval_type lm_eval --runs pious-wildflower-11 --tasks mmlu truthfulqa gsm8k --with_adapter --print_results \
    --custom_ablation "no_features_1_2_3:1,2,3" "no_features_10_20:10,20"

AlpacaEval:
python scripts/run_evals.py --eval_type alpaca --runs pious-wildflower-11 --with_adapter

AlpacaEval with Feature Ablation:
python scripts/run_evals.py --eval_type alpaca --runs pious-wildflower-11 --with_adapter \
    --alignment_classification_file /root/feature-steering-RL/outputs/12-gemmascope-res-65k__l0-21_alignment_classified_deepseek-v3-0324.json \
    --style_classification_file /root/feature-steering-RL/outputs/12-gemmascope-res-65k__l0-21_formatting_classified_deepseek-v3-0324.json

AlpacaEval with Custom Ablation:
python scripts/run_evals.py --eval_type alpaca --runs pious-wildflower-11 --with_adapter \
    --custom_ablation "no_features_1_2_3:1,2,3" "no_features_10_20:10,20"
"""
import sys
import os
import argparse

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fsrl.evals.lmarena import run_eval, pretty_results
from fsrl.evals.alpaca import run_alpaca_eval

def main():
    parser = argparse.ArgumentParser(description="Run evaluations for feature steering models.")
    parser.add_argument("--eval_type", type=str, required=True, choices=["lm_eval", "alpaca"], help="Type of evaluation to run.")
    parser.add_argument("--runs", nargs="+", required=True, help="List of run names to evaluate.")
    parser.add_argument("--wandb_project", type=str, default="Gemma2-2B-muon", help="WandB project where the runs are stored.")
    parser.add_argument("--with_adapter", action="store_true", help="Use adapter for steering.")
    parser.add_argument("--full_ft", action="store_true", help="Use full fine-tuned model.")
    parser.add_argument("--limit", type=float, help="Limit the number of examples for evaluation (float for lm_eval, int for alpaca).")
    parser.add_argument("--print_results", action="store_true", help="Print detailed results.")

    # LMEval specific
    parser.add_argument("--tasks", nargs="+", default=["mmlu", "truthfulqa", "gsm8k"], help="Tasks for lm-eval.")

    # AlpacaEval specific
    parser.add_argument("--annotator", default="alpaca_eval_gpt4", help="AlpacaEval annotator to use.")
    
    # Feature ablation specific (for lm_eval only)
    parser.add_argument("--alignment_classification_file", type=str, help="Path to alignment feature classification JSON file.")
    parser.add_argument("--style_classification_file", type=str, help="Path to style feature classification JSON file.")
    parser.add_argument("--custom_ablation", nargs="+", help="Custom ablation experiments in format 'name:feature1,feature2,feature3'.")

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

        # Parse custom ablation experiments
        ablation_experiments = None
        if args.custom_ablation:
            ablation_experiments = []
            for exp in args.custom_ablation:
                if ':' in exp:
                    name, features_str = exp.split(':', 1)
                    features = [int(f.strip()) for f in features_str.split(',') if f.strip()]
                    ablation_experiments.append((name, features))
                else:
                    print(f"Warning: Invalid custom ablation format '{exp}'. Expected 'name:feature1,feature2,...'")

        results = run_eval(
            runs=args.runs,
            tasks=args.tasks,
            wandb_project=args.wandb_project,
            limit=limit,
            with_adapter=args.with_adapter,
            full_ft=args.full_ft,
            alignment_classification_file=args.alignment_classification_file,
            style_classification_file=args.style_classification_file,
            ablation_experiments=ablation_experiments
        )
        if args.print_results:
            pretty_results(results)

    elif args.eval_type == "alpaca":
        limit = int(args.limit) if args.limit else None
        
        # Parse custom ablation experiments
        ablation_experiments = None
        if args.custom_ablation:
            ablation_experiments = []
            for exp in args.custom_ablation:
                if ':' in exp:
                    name, features_str = exp.split(':', 1)
                    features = [int(f.strip()) for f in features_str.split(',') if f.strip()]
                    ablation_experiments.append((name, features))
                else:
                    print(f"Warning: Invalid custom ablation format '{exp}'. Expected 'name:feature1,feature2,...'")
        
        results = run_alpaca_eval(
            runs=args.runs,
            with_adapter=args.with_adapter,
            full_ft=args.full_ft,
            annotator=args.annotator,
            limit=limit,
            alignment_classification_file=args.alignment_classification_file,
            style_classification_file=args.style_classification_file,
            ablation_experiments=ablation_experiments
        )
        
        print("\n" + "=" * 50)
        print("ALPACA EVAL RESULTS SUMMARY")
        print("=" * 50)
        
        for run, run_data in results.items():
            print(f"\nRun: {run}")
            print("=" * 30)
            
            # Check if run_data contains ablation experiments or direct results
            if isinstance(run_data, dict) and any(isinstance(v, dict) for v in run_data.values()):
                # This is the new format with ablation experiments
                for exp_name, result in run_data.items():
                    print(f"  Experiment: {exp_name}")
                    if isinstance(result, dict) and 'win_rate' in result:
                        print(f"    Win Rate: {result['win_rate']:.3f}")
                    else:
                        print(f"    Result: {result}")
                    print("  " + "-" * 25)
            else:
                # This is the old format - direct results
                if isinstance(run_data, dict) and 'win_rate' in run_data:
                    print(f"  Win Rate: {run_data['win_rate']:.3f}")
                else:
                    print(f"  Result: {run_data}")
        
        print("\nDetailed results saved in individual run directories.")

if __name__ == "__main__":
    main() 