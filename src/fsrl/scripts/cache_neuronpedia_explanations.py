"""
Fetch and cache Neuronpedia feature explanations for a given SAE.

- If an adapter path is provided, this script will load the SAEAdapter to
  infer the Neuronpedia modelId and saeId.
- Alternatively, you can pass --model-id and --sae-id directly.
- Results are cached to disk; re-download can be forced with --force.

Example usages:
  python -m fsrl.scripts.cache_neuronpedia_explanations \
      --adapter-path models/Gemma2-2B-sparsity-sweep/run-123/adapter \
      --cache-dir models/NeuronpediaCache

  python -m fsrl.scripts.cache_neuronpedia_explanations \
      --model-id gemma-2-2b-it --sae-id layer_12_resid_post_mlp \
      --cache-dir models/NeuronpediaCache --force
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    # Optional: present in most setups of this repo
    from fsrl.sae_adapter import SAEAdapter
except Exception:
    SAEAdapter = None  # type: ignore


DEFAULT_BASE_URL = "https://www.neuronpedia.org/api/explanation/export"


def resolve_ids_from_adapter(adapter_path: Path, device: str = "cpu") -> Tuple[str, str, Dict[str, Any]]:
    if SAEAdapter is None:
        raise RuntimeError("SAEAdapter not available; cannot resolve IDs from adapter path.")
    sae = SAEAdapter.load_from_pretrained_adapter(str(adapter_path), device=device)
    # Neuronpedia uses a slightly different saeId than the internal cfg.sae_id
    model_id: str = getattr(sae.cfg, "model_name", None)
    neuronpedia_id: Optional[str] = getattr(sae.cfg, "neuronpedia_id", None)
    if model_id is None or neuronpedia_id is None:
        raise RuntimeError("Could not resolve model_name and neuronpedia_id from the loaded adapter.")
    sae_id = neuronpedia_id.split("/")[-1]
    extra = {
        "release": getattr(sae.cfg, "release", None),
        "base_sae_id": getattr(sae.cfg, "sae_id", None),
        "neuronpedia_id": neuronpedia_id,
    }
    return model_id, sae_id, extra


def cache_path_for(cache_dir: Path, model_id: str, sae_id: str) -> Path:
    # Keep a tidy directory structure: <cache_dir>/<model_id>/<sae_id>.json
    return cache_dir / model_id / f"{sae_id}.json"


def build_s3_url(model_id: str, release: Optional[str], base_sae_id: Optional[str], neuronpedia_id: Optional[str]) -> Optional[str]:
    """Attempt to build the S3 explanations-only URL from known identifiers.

    Expected pattern example:
      https://neuronpedia-exports.s3.amazonaws.com/explanations-only/gemma-2-2b_12-gemmascope-res-65k.json
    """
    # Normalize model slug (drop -it suffix commonly used for instruction-tuned variants)
    model_slug = model_id.replace("-it", "") if model_id else None
    if not model_slug:
        return None

    layer_num: Optional[str] = None
    family_slug: Optional[str] = None

    # Try to extract from neuronpedia_id first
    if neuronpedia_id:
        # layer number
        import re
        m = re.search(r"layer_(\d+)", neuronpedia_id)
        if m:
            layer_num = m.group(1)
        # family is the segment before layer_*
        if "layer_" in neuronpedia_id:
            family_slug = neuronpedia_id.split("/layer_")[0].split("/")[-1]

    # Fallbacks from base_sae_id (e.g., "layer_12/width_65k/...")
    if (layer_num is None or family_slug is None) and base_sae_id:
        import re
        if layer_num is None:
            m = re.search(r"layer_(\d+)", base_sae_id)
            if m:
                layer_num = m.group(1)
        width = None
        m2 = re.search(r"width_(\d+k)", base_sae_id)
        if m2:
            width = m2.group(1)
        # Derive family from release + width (heuristic for GemmaScope resid)
        if family_slug is None and release:
            rel = release.lower()
            if "gemma" in rel and "scope" in rel and "res" in rel:
                family_slug = f"gemmascope-res-{width or '65k'}"

    if not (layer_num and family_slug):
        return None

    filename = f"{model_slug}_{layer_num}-{family_slug}.json"
    return f"https://neuronpedia-exports.s3.amazonaws.com/explanations-only/{filename}"


def fetch_explanations(model_id: str, sae_id: str, base_url: str = DEFAULT_BASE_URL, timeout: int = 60,
                       release: Optional[str] = None, base_sae_id: Optional[str] = None,
                       neuronpedia_id: Optional[str] = None) -> List[Dict[str, Any]]:
    # Try S3 export first if we can construct the URL
    s3_url = build_s3_url(model_id, release, base_sae_id, neuronpedia_id)
    if s3_url:
        resp = requests.get(s3_url, timeout=timeout)
        if resp.ok:
            data = resp.json()
            if isinstance(data, dict) and "explanations" in data:
                data = data["explanations"]
            if not isinstance(data, list):
                raise RuntimeError(f"Unexpected S3 response shape (type={type(data)}) from {s3_url}")
            return data
        else:
            # Fall through to API if S3 not available
            pass

    # API fallback
    params = {"modelId": model_id, "saeId": sae_id}
    headers: Dict[str, str] = {}
    # If user has an API key, include it; otherwise attempt unauthenticated
    api_key = os.environ.get("NEURONPEDIA_API_KEY")
    if api_key:
        # Documented header may vary; prefer standard x-api-key if supported
        headers["x-api-key"] = api_key

    resp = requests.get(base_url, params=params, headers=headers, timeout=timeout)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        detail = resp.text[:5000] if resp is not None else ""
        raise RuntimeError(f"Failed to fetch explanations: {e}\nResponse: {detail}")

    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected response type (expected list): {type(data)}")
    return data


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache Neuronpedia explanations for an SAE.")
    # Default to the known Gemma2-2B-clean/mild-glade-10 adapter if nothing is provided
    default_adapter = "models/Gemma2-2B-clean/mild-glade-10/adapter"
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--adapter-path", type=str, default=default_adapter, help="Path to a local adapter folder (contains fsrl_adapter_config.json)")
    source.add_argument("--model-id", type=str, help="Neuronpedia modelId (e.g., gemma-2-2b-it)")
    parser.add_argument("--sae-id", type=str, help="Neuronpedia saeId (e.g., layer_12_resid_post_mlp) when using --model-id")
    parser.add_argument("--cache-dir", type=str, default="models/NeuronpediaCache", help="Directory to store cached JSON")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="Neuronpedia export endpoint")
    parser.add_argument("--force", action="store_true", help="Force re-download even if cache exists")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds")

    args = parser.parse_args()

    if args.model_id:
        if not args.sae_id:
            raise SystemExit("--sae-id is required when using --model-id")
        model_id, sae_id = args.model_id, args.sae_id
        extra = {"release": None, "base_sae_id": None, "neuronpedia_id": None}
    else:
        # Fallback to default or provided adapter path
        if not args.adapter_path:
            args.adapter_path = default_adapter
        model_id, sae_id, extra = resolve_ids_from_adapter(Path(args.adapter_path))

    cache_dir = Path(args.cache_dir)
    out_path = cache_path_for(cache_dir, model_id, sae_id)

    if out_path.exists() and not args.force:
        print(f"Cache found: {out_path}")
        try:
            data = load_json(out_path)
            print(f"Loaded {len(data)} explanations from cache.")
        except Exception as e:
            print(f"Failed to read cache (will refetch): {e}")
        else:
            return

    print(f"Fetching explanations for {model_id}/{sae_id}...")
    data = fetch_explanations(model_id, sae_id, base_url=args.base_url, timeout=args.timeout,
                              release=extra.get("release"), base_sae_id=extra.get("base_sae_id"),
                              neuronpedia_id=extra.get("neuronpedia_id"))
    print(f"Fetched {len(data)} explanations. Caching to {out_path}...")
    save_json(data, out_path)
    print("Done.")


if __name__ == "__main__":
    main()
