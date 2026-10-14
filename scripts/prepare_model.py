#!/usr/bin/env python3
"""Prepare a HuggingFace model for airllm-streaming.

Pipeline:
  1. Download the model from HuggingFace (unless a local path is given).
  2. Split it into per-layer NF4 (4-bit) shards via AirLLM's split_and_save_layers.
  3. Pack the per-layer shards into super-blocks (4 decoder layers per block) so
     runtime I/O moves whole blocks instead of 67 small files.

The packed shards live in ``<shards>/splitted_model.4bit/`` and are consumed by
``AutoModel.from_pretrained(..., layer_shards_saving_path=<shards>, compression='4bit')``.

Example:
    python3 prepare_model.py --model Qwen/Qwen3-32B --shards /data/airllm-shards
    python3 prepare_model.py --model /local/Qwen3-32B --shards /data/airllm-shards
"""
import argparse
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from huggingface_hub import snapshot_download

from airllm import split_and_save_layers

# AirLLM runtime layer order (67 entries): embed_tokens, 64 decoder layers, norm, lm_head.
LAYER_NAMES = (
    ["model.embed_tokens"]
    + [f"model.layers.{i}" for i in range(64)]
    + ["model.norm", "lm_head"]
)


def download_model(model_id, cache_dir=None):
    print(f"Downloading {model_id} ...", flush=True)
    t0 = time.time()
    local = snapshot_download(
        repo_id=model_id,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.py"],
        cache_dir=cache_dir,
    )
    print(f"Downloaded in {time.time() - t0:.1f}s -> {local}", flush=True)
    return local


def split_shards(model_path, shards_dir, compression="4bit"):
    print(f"Splitting into {compression} per-layer shards ...", flush=True)
    t0 = time.time()
    split_and_save_layers(model_path, layer_shards_saving_path=shards_dir, compression=compression)
    print(f"Split done in {time.time() - t0:.1f}s", flush=True)


def pack_blocks(shards_dir, compression="4bit", layers_per_block=4):
    src = Path(shards_dir) / f"splitted_model.{compression}"
    total = (len(LAYER_NAMES) + layers_per_block - 1) // layers_per_block
    for b in range(total):
        start = b * layers_per_block
        end = min(start + layers_per_block, len(LAYER_NAMES))
        layers = LAYER_NAMES[start:end]
        out = src / f"model.block_{b:03d}.safetensors"
        if out.exists():
            print(f"block {b} exists, skip")
            continue
        t0 = time.time()
        merged = {}
        for ln in layers:
            sd = load_file(src / f"{ln}.safetensors", device="cpu")
            for k, v in sd.items():
                merged[k] = v.contiguous()
            del sd
        save_file(merged, out)
        size = out.stat().st_size / 1e9
        print(f"block {b}: {layers[0]}..{layers[-1]} {size:.2f}GB ({time.time() - t0:.1f}s)", flush=True)
        del merged
    # done markers (mirror AirLLM's <layer>.safetensors.done convention)
    for b in range(total):
        (src / f"model.block_{b:03d}.safetensors.done").touch()
    print("ALL DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF repo id (e.g. Qwen/Qwen3-32B) or local model directory")
    ap.add_argument("--shards", required=True, help="output directory for layer shards (runtime layer_shards_saving_path)")
    ap.add_argument("--compression", default="4bit", choices=["4bit", "8bit"])
    ap.add_argument("--cache-dir", default=None, help="HuggingFace download cache directory")
    ap.add_argument("--layers-per-block", type=int, default=4)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required: bitsandbytes quantization runs on GPU.")

    shards = Path(args.shards)
    shards.mkdir(parents=True, exist_ok=True)

    model_path = args.model
    if not Path(model_path).exists():
        model_path = download_model(args.model, args.cache_dir)
    else:
        print(f"Using local model: {model_path}")

    split_shards(model_path, shards, args.compression)
    pack_blocks(shards, args.compression, args.layers_per_block)


if __name__ == "__main__":
    main()