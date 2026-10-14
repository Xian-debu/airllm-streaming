#!/usr/bin/env python3
"""Benchmark airllm-streaming generation throughput.

This benchmark measures end-to-end decode throughput (tokens/s) for the
optimised AirLLM runtime (4-bit NF4 super-blocks + pinned-memory H2D).

Important: AIRLLM_PREFETCH_WITH_COMPRESSION is set here before importing airllm.
Without it AirLLM disables prefetching for compressed checkpoints and the
throughput drops to the unoptimised baseline.

Example:
    python3 benchmark.py --model /path/to/Qwen3-32B --shards /data/airllm-shards
    python3 benchmark.py --model /path/to/Qwen3-32B --shards /data/airllm-shards --max-new-tokens 64
"""
import argparse
import os
import time

os.environ.setdefault("AIRLLM_PREFETCH_WITH_COMPRESSION", "1")  # before importing airllm

import torch
from transformers import AutoTokenizer

from airllm import AutoModel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model directory (must contain config.json + tokenizer)")
    ap.add_argument("--shards", required=True, help="layer shards directory (layer_shards_saving_path)")
    ap.add_argument("--compression", default="4bit", choices=["4bit", "8bit"])
    ap.add_argument("--prompt", default="中国的首都是")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run airllm-streaming.")

    print(f"torch {torch.__version__} | CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    t0 = time.time()
    model = AutoModel.from_pretrained(
        args.model,
        layer_shards_saving_path=args.shards,
        compression=args.compression,
    )
    print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

    ids = tokenizer(args.prompt, return_tensors="pt")
    print(f"prompt: {args.prompt!r} -> {ids['input_ids'].shape}", flush=True)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=args.max_new_tokens, do_sample=False)
    dt = time.time() - t0
    gen = out[0][ids["input_ids"].shape[1]:]
    ntok = gen.shape[0]
    text = tokenizer.decode(gen)
    print(f"GEN_TEXT: {text!r}", flush=True)
    print(f"generated {ntok} tokens in {dt:.1f}s -> {ntok / dt:.2f} tok/s", flush=True)


if __name__ == "__main__":
    main()