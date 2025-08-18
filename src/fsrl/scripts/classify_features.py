import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, APIError
from tqdm.asyncio import tqdm

# --- Prompt Engineering ---

# CHANGED: The prompt is now more direct in its final instruction to prevent preambles.
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

Your response must be *exactly* one of the two categories below and nothing else. Do not add any conversational text or preamble.
- `alignment-related`
- `not-alignment-related`

/no_think
"""

USER_PROMPT_TEMPLATE = """
Please classify the following feature explanation:

Explanation: "{explanation}"
"""

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


async def classify_feature_async(feature: dict, client: AsyncOpenAI, model: str, semaphore: asyncio.Semaphore) -> dict:
    """Asynchronously classifies a single feature using the Lambda API."""
    async with semaphore:
        description = feature.get("description", "")
        feature_id = f"{feature.get('modelId')}-{feature.get('layer')}-{feature.get('index')}"

        if not description:
            return {
                "feature_id": feature_id,
                "description": description,
                "classification": "no-description",
            }

        user_message = USER_PROMPT_TEMPLATE.format(explanation=description)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        try:
            chat_completion = await client.chat.completions.create(
                messages=messages,
                model=model,
                # CHANGED: Increased max_tokens to give the model enough room to respond.
                max_tokens=20,
                extra_body={"guided_choice": CLASSIFICATION_CHOICES},
            )
            
            # CHANGED: Added a check for None content before stripping.
            completion_content = chat_completion.choices[0].message.content
            if completion_content:
                label = completion_content.strip()
            else:
                # This handles the exact error you saw.
                print(f"\nAPI returned no content for feature {feature_id}. Finish reason: {chat_completion.choices[0].finish_reason}")
                label = "api-no-content"


        except APIError as e:
            print(f"\nAPI Error for feature {feature_id}: {e}")
            label = "api-error"
        except Exception as e:
            print(f"\nAn unexpected error occurred for feature {feature_id}: {e}")
            label = "unexpected-error"

        return {
            "feature_id": feature_id,
            "description": description,
            "classification": label,
        }


async def main(args):
    """Main async function to run the classification pipeline."""
    load_dotenv()
    
    input_path = Path(args.input_file)
    # Derive output path if not provided: <input_stem>_classified_<model>.json in same directory
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        # sanitize model for filesystem
        safe_model = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in args.model)
        output_path = input_path.with_name(f"{input_path.stem}_classified_{safe_model}.json")

    api_key = args.api_key or os.environ.get("LAMBDA_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please provide it via --api-key or a LAMBDA_API_KEY in your .env file.")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.lambda.ai/v1",
    )

    features = load_features(input_path)
    if not features:
        print("No features found in the input file. Exiting.")
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    
    tasks = [
        classify_feature_async(feature, client, args.model, semaphore)
        for feature in features
    ]

    print(f"Sending {len(tasks)} classification requests with a concurrency limit of {args.concurrency}...")
    
    results = await tqdm.gather(*tasks, desc="Classifying features")

    save_results(results, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify SAE feature explanations concurrently using the Lambda Inference API."
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
        default=None,
        help=(
            "Path to save the output JSON file with classifications. "
            "If not provided, saves next to the input file as <input_stem>_classified_<model>.json."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        # CHANGED: Defaulting to the model you are using.
        default="deepseek-v3-0324",
        help="Model ID available on Lambda Inference.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Your Lambda Inference API key (overrides .env file).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=50,
        help="Number of concurrent API requests to make. Adjust based on your API rate limits.",
    )
    
    args = parser.parse_args()
    asyncio.run(main(args))