import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, APIError
from tqdm.asyncio import tqdm

# --- Prompt Engineering ---

# Classification modes with their respective prompts and choices
ALIGNMENT_SYSTEM_PROMPT = """
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
"""

FORMATTING_SYSTEM_PROMPT = """
You are an expert in natural language processing and text analysis. Your task is to classify explanations of features from a neural network into one of two categories: 'formatting-related' or 'not-formatting-related'.

1.  **Formatting-related**: Features that represent aspects of text structure, presentation, style, or format rather than semantic content. This includes but is not limited to:
    - Punctuation and symbols (e.g., periods, commas, parentheses, quotation marks, dashes)
    - Capitalization patterns (e.g., sentence beginnings, proper nouns, ALL CAPS)
    - Whitespace and spacing (e.g., indentation, line breaks, paragraph breaks)
    - Programming/code formatting (e.g., syntax highlighting, code blocks, indentation)
    - List formatting (e.g., bullet points, numbered lists, item separators)
    - Text length and conciseness (e.g., short responses, word limits, brevity)
    - Structural elements (e.g., headings, titles, section markers)
    - Repetition patterns (e.g., repeated characters, duplicate text)
    - Language style markers (e.g., formal vs informal tone indicators)
    - Special characters and encoding (e.g., Unicode symbols, HTML entities)

2.  **Not formatting-related**: Features that represent semantic content, meaning, topics, or conceptual information rather than formatting. This includes but is not limited to:
    - Specific topics, subjects, or domains (e.g., science, history, sports)
    - Semantic concepts and meanings (e.g., emotions, actions, relationships)
    - Factual knowledge (e.g., names, dates, places, events)
    - Abstract concepts and ideas (e.g., morality, justice, creativity)
    - Content-specific patterns (e.g., question types, answer categories)

Your response must be *exactly* one of the two categories below and nothing else. Do not add any conversational text or preamble.
- `formatting-related`
- `not-formatting-related`
"""

# Classification mode configurations
CLASSIFICATION_MODES = {
    "alignment": {
        "system_prompt": ALIGNMENT_SYSTEM_PROMPT,
        "choices": ["alignment-related", "not-alignment-related"],
        "related_label": "related",
        "not_related_label": "not-related"
    },
    "formatting": {
        "system_prompt": FORMATTING_SYSTEM_PROMPT,
        "choices": ["formatting-related", "not-formatting-related"],
        "related_label": "related", 
        "not_related_label": "not-related"
    }
}

USER_PROMPT_TEMPLATE = """
Please classify the following feature explanation:

Explanation: "{explanation}"
"""


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


async def classify_feature_async(feature: dict, client: AsyncOpenAI, model: str, mode: str, semaphore: asyncio.Semaphore) -> dict:
    """Asynchronously classifies a single feature using the OpenRouter API."""
    async with semaphore:
        description = feature.get("description", "")
        feature_id = f"{feature.get('modelId')}-{feature.get('layer')}-{feature.get('index')}"

        if not description:
            return {
                "feature_id": feature_id,
                "description": description,
                "label": "no-description",
            }

        # Get the configuration for the selected mode
        mode_config = CLASSIFICATION_MODES[mode]
        system_prompt = mode_config["system_prompt"]
        choices = mode_config["choices"]
        related_label = mode_config["related_label"]
        not_related_label = mode_config["not_related_label"]

        user_message = USER_PROMPT_TEMPLATE.format(explanation=description)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            chat_completion = await client.chat.completions.create(
                messages=messages,
                model=model,
                # CHANGED: Increased max_tokens to give the model enough room to respond.
                max_tokens=20,
                extra_body={"guided_choice": choices},
            )
            
            # CHANGED: Added a check for None content before stripping.
            completion_content = chat_completion.choices[0].message.content
            if completion_content:
                # Strip whitespace and backticks that some models add
                raw_label = completion_content.strip().strip('`')
                # Normalize the classification to "related" or "not-related"
                if raw_label == choices[0]:  # e.g., "alignment-related" or "formatting-related"
                    label = related_label
                elif raw_label == choices[1]:  # e.g., "not-alignment-related" or "not-formatting-related"
                    label = not_related_label
                else:
                    # Handle unexpected responses
                    print(f"\nUnexpected classification response for feature {feature_id}: '{raw_label}'")
                    label = "unexpected-response"
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
            "label": label,
        }


async def main(args):
    """Main async function to run the classification pipeline."""
    load_dotenv()
    
    input_path = Path(args.input_file)
    # Always save results in outputs/feature_classification
    output_dir = Path("outputs/feature_classification/gemma-2-2b")
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in args.model)
    mode_tag = args.mode
    input_tag = input_path.stem
    if args.output_file:
        output_path = output_dir / Path(args.output_file).name
    else:
        output_path = output_dir / f"{input_tag}_{mode_tag}_classified_{safe_model}.json"
    
    # Validate the mode
    if args.mode not in CLASSIFICATION_MODES:
        raise ValueError(f"Invalid mode '{args.mode}'. Available modes: {list(CLASSIFICATION_MODES.keys())}")
    
    print(f"Using classification mode: {args.mode}")
    mode_config = CLASSIFICATION_MODES[args.mode]
    print(f"Will classify as: {mode_config['choices'][0]} or {mode_config['choices'][1]}")

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please provide it via --api-key or an OPENROUTER_API_KEY in your .env file.")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    features = load_features(input_path)
    if not features:
        print("No features found in the input file. Exiting.")
        return

    # Limit the number of features for debugging if specified
    if args.limit is not None:
        features = features[:args.limit]
        print(f"Limited to first {args.limit} features for debugging.")

    semaphore = asyncio.Semaphore(args.concurrency)
    
    tasks = [
        classify_feature_async(feature, client, args.model, args.mode, semaphore)
        for feature in features
    ]

    print(f"Sending {len(tasks)} classification requests with a concurrency limit of {args.concurrency}...")
    
    results = await tqdm.gather(*tasks, desc="Classifying features")

    save_results(results, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify SAE feature explanations concurrently using the OpenRouter API."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="models/NeuronpediaCache/gemma-2-9b/12-gemmascope-res-16k_canonical.json",
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
        "--mode",
        type=str,
        default="formatting",
        choices=list(CLASSIFICATION_MODES.keys()),
        help="Classification mode to use (default: formatting).",
    )
    parser.add_argument(
        "--model",
        type=str,
        # CHANGED: Defaulting to the model you are using.
        default="deepseek/deepseek-chat-v3-0324",
        help="Model ID available on OpenRouter.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Your OpenRouter API key (overrides .env file).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=50,
        help="Number of concurrent API requests to make. Adjust based on your API rate limits.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of features to process (for debugging). If not specified, all features are processed.",
    )
    
    args = parser.parse_args()
    asyncio.run(main(args))