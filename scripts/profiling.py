#!/usr/bin/env python3
"""Per-layer timing profile for airllm-streaming.

Runs a single decode token and prints per-layer load / move / forward timings.
Useful for verifying that:
  - super-block leaders wait on the prefetched block (load ~0.1-0.3s), and
  - block-internal layers hit the promoted slot (load ~0.02s).

Example:
    python3 profile.py --model /path/to/Qwen3-32B --shards /data/airllm-shards
"""
import argparse
import os
import time

os.environ.setdefault("AIRLLM_PREFETCH_WITH_COMPRESSION", "1")  # before importing airllm
os.environ.setdefault("AIRLLM_TIMING", "1")

import torch
from transformers import AutoTokenizer

from airllm import AutoModel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="model directory (must contain config.json + tokenizer)")
    ap.add_argument("--shards", required=True, help="layer shards directory (layer_shards_saving_path)")
    ap.add_argument("--compression", default="4bit", choices=["4bit", "8bit"])
    ap.add_argument("--prompt", default="中国的首都是")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run airllm-streaming.")

    print(f"torch {torch.__version__} | CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(
        args.model,
        layer_shards_saving_path=args.shards,
        compression=args.compression,
    )

    ids = tokenizer(args.prompt, return_tensors="pt")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=1, do_sample=False)
    dt = time.time() - t0
    gen = out[0][ids["input_ids"].shape[1]:]
    print(f"GEN_IDS: {gen.tolist()}", flush=True)
    print(f"DONE {dt:.1f}s ({gen.shape[0]} token)", flush=True)


if __name__ == "__main__":
    main()