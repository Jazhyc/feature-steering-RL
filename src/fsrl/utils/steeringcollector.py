"""
Compute the steering vectors per dataset sample an existing HookedModel + SAEAdapter.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import json
from dotenv import load_dotenv
import numpy as np
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from fsrl import SAEAdapter, HookedModel


class SteeringCollector:
    """
    Class wrapper to compute per-sample average steering vectors.

    :param model_name: name for HookedTransformer.from_pretrained_no_processing
    :param adapter_local_path: path to directory containing adapter_weights.safetensors + fsrl_adapter_config.json
    :param hf_dataset: HF dataset id (e.g. "princeton-nlp/llama3-ultrafeedback-armorm")
    :param split: dataset split to use (e.g. "validation")
    :param device: "cuda" or "cpu"
    :param max_samples: max samples to process (None -> all)
    :param out_dir: directory to save outputs
    :param dtype: torch dtype for model (default bfloat16)
    """

    def __init__(
        self,
        model_name: str,
        adapter_local_path: str,
        hf_dataset: str,
        split: str = "test",
        device: Optional[str] = None,
        max_samples: Optional[int] = 500,
        out_dir: str = "./steering_out",
        dtype=torch.bfloat16,
    ):
        self.model_name = model_name
        self.adapter_local_path = adapter_local_path
        self.hf_dataset = hf_dataset
        self.split = split
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_samples = max_samples
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.dtype = dtype

        self.hooked: Optional[HookedModel] = None
        self.tokenizer = None
        self.dataset = None

        # outputs
        self.steering_array: Optional[np.ndarray] = None
        self.meta: Dict[str, Any] = {}

    def load_model_and_adapter(self) -> None:
        """Load HookedTransformer + SAEAdapter and wrap in HookedModel."""
        model = HookedTransformer.from_pretrained_no_processing(
            model_name=self.model_name,
            device=self.device,
            torch_dtype=self.dtype,
            attn_implementation="sdpa",
        )
        sae = SAEAdapter.load_from_pretrained_adapter(self.adapter_local_path, device=self.device)
        self.hooked = HookedModel(model, sae)
        self.tokenizer = self.hooked.model.tokenizer

    def load_dataset(self) -> None:
        """Load HF dataset split."""
        ds = load_dataset(self.hf_dataset, split=self.split)
        if self.max_samples is not None:
            ds = ds.select(range(min(len(ds), int(self.max_samples))))
        self.dataset = ds

    @staticmethod
    def _mean_vector_from_activation_tensor(t: torch.Tensor) -> np.ndarray:
        t = t.detach().cpu()
        if t.ndim == 3:
            out = t.mean(dim=1)
        elif t.ndim == 2:
            out = t if t.shape[0] == 1 else t.mean(dim=0, keepdim=True)
        elif t.ndim == 1:
            out = t.unsqueeze(0)
        else:
            raise ValueError("Unexpected activation tensor shape: " + str(t.shape))
        # cast to float32 before numpy
        # numpy cannot convert bfloat16 tensors directly.
        arr = out.to(torch.float32).cpu().numpy()
        return arr[0] if arr.ndim == 2 and arr.shape[0] == 1 else arr


    @staticmethod
    def _find_adapter_activation_key(cache: Dict) -> str:
        """
        Heuristic search for the SAE adapter activations key in the cache.
        """
        keys = list(cache.keys())
        # common pattern substring
        for k in keys:
            if isinstance(k, str) and ("hook_sae_adapter" in k or k.endswith("hook_sae_adapter")):
                return k
        for k in keys:
            if isinstance(k, str) and "sae" in k and "adapter" in k:
                return k
        # if keys are tuples (layer, name) -> stringify and search
        for k in keys:
            s = str(k)
            if "hook_sae_adapter" in s or ("sae" in s and "adapter" in s):
                return k
        # last resort: return first key (debugging)
        raise KeyError(f"Could not locate SAE adapter activation key. Sample keys: {keys[:40]}")

    @staticmethod
    def _get_text_from_example(ex: dict) -> str:
        """Extract text field from a dataset example."""
        if "prompt" in ex:
            return ex["prompt"]
        if "text_prompt" in ex:
            return ex["text_prompt"]
        for k, v in ex.items():
            if isinstance(v, str):
                return v
        raise KeyError("No string field found in example. Keys: " + ", ".join(ex.keys()))

    def run(self, verbose: bool = True) -> None:
        """
        Run extraction over the dataset.
        After calling, use .save() to persist results.
        """
        if self.hooked is None:
            if verbose:
                print("Loading model + adapter...")
            self.load_model_and_adapter()

        if self.dataset is None:
            if verbose:
                print("Loading dataset...")
            self.load_dataset()

        num_samples = len(self.dataset)
        if verbose:
            print(f"Processing {num_samples} samples from {self.hf_dataset} split={self.split}")

        steering_list: List[np.ndarray] = []
        meta_list: List[dict] = []

        for i, ex in enumerate(self.dataset):
            text = self._get_text_from_example(ex)
            tok = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
            input_ids = tok["input_ids"].to(self.device)
            kwargs = {}
            if "attention_mask" in tok:
                kwargs["attention_mask"] = tok["attention_mask"].to(self.device)

            with torch.no_grad():
                logits, cache = self.hooked.run_with_cache(input_ids, **kwargs)

            # find activations
            key = self._find_adapter_activation_key(cache)
            act = cache[key]
            if isinstance(act, (list, tuple)):
                act = act[0]

            vec = self._mean_vector_from_activation_tensor(act)
            steering_list.append(vec)
            meta_list.append({"index": i, "text_snippet": text[:200]})

            if verbose and ((i + 1) % 50 == 0 or (i + 1) == num_samples):
                print(f"Processed {i+1}/{num_samples}")

        # stack to array
        self.steering_array = np.stack(steering_list, axis=0)
        self.meta = {"dataset": self.hf_dataset, "split": self.split, "n": self.steering_array.shape[0]}

    def save(self, out_dir: Optional[str] = None) -> None:
        """Save steering_vectors.npy and meta.json to out_dir (or configured out_dir)."""
        out = Path(out_dir) if out_dir else self.out_dir
        out.mkdir(parents=True, exist_ok=True)
        if self.steering_array is None:
            raise RuntimeError("No results to save. Run .run() first.")
        np.save(out / "steering_vectors.npy", self.steering_array)
        with open(out / "meta.json", "w") as f:
            json.dump(self.meta, f, indent=2)
        print("Saved steering_vectors.npy ->", out / "steering_vectors.npy")
        print("Saved meta.json ->", out / "meta.json")


if __name__ == "__main__":
    load_dotenv()
    extractor = SteeringCollector(
        model_name="google/gemma-2-2b-it",
        adapter_local_path="./artifacts/adapter-mild-glade-10:v0/adapter",
        hf_dataset="princeton-nlp/llama3-ultrafeedback-armorm",
        split="test",
        max_samples=20,
        out_dir="./steering_out",
    )
    extractor.run(verbose=True)
    extractor.save()
