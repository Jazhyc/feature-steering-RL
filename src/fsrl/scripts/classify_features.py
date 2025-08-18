import argparse
import json
from pathlib import Path

from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

# --- Prompt Engineering ---

# The system prompt is updated to broaden the definition of "alignment-related"
# and instructs the model to provide a direct classification, not JSON.
SYSTEM_PROMPT = """
You are an expert AI alignment researcher. Your task is to classify explanations of features from a neural network into one of two categories: 'alignment-related' or 'not-alignment-related'.

1.  **Alignment-related**: Features that represent abstract, high-level concepts, complex cognitive processes, or goal-directed behaviors relevant to AI safety and alignment. This includes but is not limited to:
    - Morality and ethics (e.g., right vs. wrong, fairness, justice)
    - Honesty, deception, or covert actions
    - Sycophancy (flattery, brown-nosing, pandering)
    - Power-seeking behaviors or instrumental goals
    - Corrigibility, cooperativeness, and shutdown avoidance
    - Self-awareness, self-preservation, or mentions of agency
    - Harmfulness, violence, or dangerous content generation
    - Systemic biases (racial, gender, political, etc.)
    - Complex, goal-directed behaviors or planning (even if not inherently harmful)
    - Refusal to answer, evasiveness, or stating limitations

2.  **Not alignment-related**: Features that represent low-level, concrete, or topic-specific concepts without a clear link to alignment. This includes but is not limited to:
    - Specific programming languages or syntax (e.g., Python code, JSON structures)
    - Grammatical structures (e.g., punctuation, specific parts of speech, sentence endings)
    - Common objects or factual knowledge (e.g., names of people, places, dates, scientific facts)
    - Simple linguistic patterns (e.g., capitalization, repeated characters, specific tokens like 'the' or 'is')
    - Specific domains like mathematics, cooking, or sports, unless they directly involve an abstract alignment concept.

Your response must be *exactly* one of the two categories: `alignment-related` or `not-alignment-related`.
"""

# The user prompt template remains the same.
USER_PROMPT_TEMPLATE = """
Please classify the following feature explanation:

Explanation: "{explanation}"
"""

# Define the choices for guided decoding
CLASSIFICATION_CHOICES = ["alignment-related", "not-alignment-related"]


def load_features(input_path: Path) -> list[dict]:
    """Loads feature data from a JSON file."""
    print(f"Loading features from {input_path}...")
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} features.")
    return data


def save_results(results: list[dict], output_path: Path):
    """Saves the classification results to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nClassification results saved to {output_path}")


def main(args):
    """Main function to run the classification pipeline."""
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    # 1. Load the feature explanations from the source file.
    features = load_features(input_path)
    if not features:
        print("No features found in the input file. Exiting.")
        return

    # 2. Initialize the VLLM engine with the specified INT4 AWQ quantized model.
    print(f"Initializing VLLM with model: {args.model}...")
    try:
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
        )
    except Exception as e:
        print(f"Error initializing VLLM: {e}")
        print("Please ensure the model name is correct, it's an AWQ model, and you have enough VRAM.")
        return

    # 3. Configure tokenizer and sampling parameters with guided decoding.
    tokenizer = llm.get_tokenizer()
    
    # Disable "thinking_mode" for Qwen models to prevent extra tokens.
    try:
        tokenizer.thinking_mode = False
        print("Successfully disabled 'thinking_mode' on the tokenizer.")
    except AttributeError:
        print("Tokenizer does not have 'thinking_mode' attribute. Skipping.")

    # Use GuidedDecodingParams to constrain the output to our choices.
    guided_params = GuidedDecodingParams(choice=CLASSIFICATION_CHOICES)
    sampling_params = SamplingParams(
        guided_decoding=guided_params,
        max_tokens=20, # A few tokens is enough for the choice
    )

    # 4. Prepare all prompts for batch processing.
    prompts = []
    for feature in tqdm(features, desc="Preparing prompts"):
        description = feature.get("description", "")
        if not description:
            prompts.append(None)
            continue

        user_message = USER_PROMPT_TEMPLATE.format(explanation=description)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        
        try:
            prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt_str)
        except Exception as e:
            print(f"Warning: Failed to apply chat template for a feature. Skipping. Error: {e}")
            prompts.append(None)

    valid_prompts_with_features = [
        (prompt, feature) for prompt, feature in zip(prompts, features) if prompt is not None
    ]
    if not valid_prompts_with_features:
        print("No valid prompts could be created. Exiting.")
        return

    valid_prompts, original_features = zip(*valid_prompts_with_features)

    # 5. Run inference on the batch of prompts.
    print(f"\nGenerating classifications for {len(valid_prompts)} features...")
    outputs = llm.generate(list(valid_prompts), sampling_params, use_tqdm=True)

    # 6. Process the outputs. No JSON parsing needed!
    results = []
    for feature, output in zip(original_features, outputs):
        label = output.outputs[0].text.strip()

        # Safety check, though guided decoding should make this unnecessary.
        if label not in CLASSIFICATION_CHOICES:
            print(f"\nWarning: Model produced an invalid classification: '{label}'")
            label = "invalid-output"

        feature_id = f"{feature.get('modelId')}-{feature.get('layer')}-{feature.get('index')}"
        
        results.append({
            "feature_id": feature_id,
            "description": feature.get("description"),
            "classification": label,
        })

    # 7. Save the final results to a new file.
    save_results(results, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify SAE feature explanations with guided decoding using VLLM."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="models/NeuronpediaCache/gemma-2-2b/12-gemmascope-res-65k__l0-21.json",
        help="Path to the input JSON file with feature explanations.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="models/NeuronpediaCache/gemma-2-2b/12-gemmascope-res-65k__l0-21_classified_guided.json",
        help="Path to save the output JSON file with classifications.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RedHatAI/Qwen3-32B-quantized.w4a16",
        help="Hugging Face model ID to use with VLLM (INT4 AWQ quantized models are required).",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=1024,
        help="Maximum context length for the model.",
    )
    
    args = parser.parse_args()
    main(args)