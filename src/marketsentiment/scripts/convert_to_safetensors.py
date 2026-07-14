"""Convert a .bin-only HF checkpoint to a local safetensors copy.

transformers 5.x refuses to ``torch.load`` pickle weights on torch < 2.6
(CVE-2025-32434). Some base models — including ProsusAI/finbert — ship only
``pytorch_model.bin``. Convert once to safetensors so ``from_pretrained`` works
without upgrading torch. (On torch >= 2.6, or a base model that already has
safetensors, you don't need this.)

    python -m marketsentiment.scripts.convert_to_safetensors \
        --model ProsusAI/finbert --out models/finbert-base-st
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a .bin checkpoint to safetensors.")
    parser.add_argument("--model", default="ProsusAI/finbert")
    parser.add_argument("--out", default="models/finbert-base-st")
    args = parser.parse_args()

    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    # Build the architecture from config (random init — no torch.load), then load the
    # real weights ourselves. torch.load still works on 2.5; only transformers blocks it.
    config = AutoConfig.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_config(config)
    bin_path = hf_hub_download(args.model, "pytorch_model.bin")
    state = torch.load(bin_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded weights (missing={len(missing)}, unexpected={len(unexpected)})")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.out)
    print(f"Wrote safetensors checkpoint to {args.out}")
    print(f"Train from it with:  --base-model {args.out}")


if __name__ == "__main__":
    main()
