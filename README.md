# airllm-streaming

Drop-in performance patches for [AirLLM](https://github.com/Gavinlyonsimu/airllm) that make 30B+ models run on an 8 GB laptop GPU at useful interactive speed — **0.03 → 0.38 tokens/s** on Qwen3-32B (12.7× faster) with no quality loss.

```
8-bit baseline ............ 0.04 tok/s  (437.6 s / 16 tokens)
4-bit + deep prefetch ..... 0.28 tok/s  (57.3 s / 16 tokens)
+ pinned-memory H2D ...... 0.38 tok/s  (42.2 s / 16 tokens)   ← this repository
```

> **What this is.** airllm-streaming is **not a fork and not a new inference engine**. It is a small, self-contained deployment layer on top of AirLLM 4.0.0: three modified AirLLM source files plus model-preparation and benchmark scripts. You keep your normal AirLLM upgrade path — the patches apply to a pinned upstream version and are fully reversible.

## Why it is fast

AirLLM streams one 1 GB layer at a time from RAM to the GPU. With 4-bit NF4 weights the throughput ceiling is the H2D transfer of 18.7 GB per generated token — we measured ~7.5 GB/s for plain pageable memory, which capped decode at ~0.28 tok/s no matter how much you prefetch.

`airllm-streaming` removes that bottleneck with three changes:

1. ****Super-block packing**** — 67 per-layer files are packed into 17 super-blocks (4 decoder layers each), so one `safetensors` open + mmap covers a whole block instead of 4 file round-trips.
2. **Double-slot prefetching** — while block N is being consumed on the GPU, the next block is already loaded into a second slot; block leaders wait ~0.1 s instead of a cold 0.5–1.5 s read.
3. **Pinned-memory H2D (the key fix)** — prefetched blocks are copied into a CPU **pinned-memory pool** (256 B-aligned), and GPU dequantisation reads from pinned memory. Pinned H2D reaches ~28 GB/s vs ~8 GB/s pageable. More importantly, background `load_file(cuda)` no longer competes with the GPU's dequantisation stream — in the old design the two collided and background load degraded from 0.15 s to 1.5 s per block.

Result: each super-block is consumed in ~0.13 s, one decode token costs ~2.6 s end-to-end, and dequantisation + matmul (not I/O) is now the remaining bottleneck.

## Repository layout

```
patches/                        modified AirLLM 4.0.0 source files
  airllm_base.py                 prefetch future/promote, main-thread dequant, timing hooks
  utils.py                       CUDA-direct dequantisation path
  persist/safetensor_model_persister.py   super-block files + pinned-memory pool
  UPSTREAM_VERSION.txt           airllm version these patches apply to
scripts/
  prepare_model.py               download → 4-bit per-layer shards → super-block packing
  benchmark.py                   end-to-end generation benchmark (tok/s)
  profiling.py                   per-layer load/timing profile
install.sh                       apply patches to your airllm installation (backup + revert)
requirements.txt                 pinned dependencies
LICENSE                          Apache-2.0
```

## Requirements

- NVIDIA GPU with **≥ 8 GB VRAM** (developed on RTX 5070 Laptop 8.5 GB)
- Linux (developed on Ubuntu 24.04 under WSL2), CUDA-capable torch
- Python ≥ 3.10
- ~60 GB free disk for the model shards (35 GB packed 4-bit + working space)

## Install

```bash
pip install -r requirements.txt        # installs airllm==4.0.0 and friends
./install.sh                          # applies patches to the installed airllm package
```

`install.sh` backs up the original files with a timestamp next to them. To revert:

```bash
cp <airllm-pkg>/airllm_base.py.bak.<timestamp> <airllm-pkg>/airllm_base.py
# and the same for utils.py and persist/safetensor_model_persister.py
```

## Prepare a model

Model weights are **not** shipped in this repository. `prepare_model.py` downloads a HuggingFace model, converts it to AirLLM's 4-bit per-layer shards, and packs the shards into super-blocks:

```bash
python3 scripts/prepare_model.py \
  --model Qwen/Qwen3-32B \
  --shards /data/airllm-shards
```

- `--model` accepts a HF repo id (downloaded to the HF cache) or a local model directory.
- `--compression 8bit` is supported if you prefer 8-bit.
- `--layers-per-block 4` is the default packing; change only if you know what you are doing.
- The output goes to `/data/airllm-shards/splitted_model.4bit/` — this is exactly what AirLLM expects from `layer_shards_saving_path`.

You can also reuse an existing AirLLM shard directory produced by `split_and_save_layers` — the runtime patches work with the original per-layer layout, they are just less optimal because of more, smaller file reads.

## Run

```bash
# benchmark: 16-token generation
python3 scripts/benchmark.py --model /path/to/Qwen3-32B --shards /data/airllm-shards

# per-layer timing profile (1 token)
python3 scripts/profiling.py --model /path/to/Qwen3-32B --shards /data/airllm-shards
```

**Important:** set `AIRLLM_PREFETCH_WITH_COMPRESSION=1`. AirLLM disables its internal prefetcher for compressed checkpoints, which silently disables our prefetch path too. The benchmark script sets it for you; if you use the library yourself:

```python
import os
os.environ["AIRLLM_PREFETCH_WITH_COMPRESSION"] = "1"   # before importing airllm
os.environ["AIRLLM_TIMING"] = "1"                      # optional per-layer timing

from airllm import AutoModel
```

## Measured results

Environment: Ubuntu 24.04 (WSL2) · RTX 5070 Laptop 8.5 GB · torch 2.14.1+cu130 · bitsandbytes 0.50.2 · model **Qwen3-32B** · prompt `中国的首都是` · 16 generated tokens · 4-bit NF4.

| Configuration | Time (16 tok) | tok/s | Note |
|---|---|---|---|
| Official AirLLM 8-bit (baseline) | 437.6 s | 0.04 | original AirLLM, no optimisations |
| 4-bit super-blocks, no prefetch | 65.2 s | 0.25 | patches installed but prefetch disabled |
| 4-bit + double-slot prefetch | 57.3 s | 0.28 | previous step (pageable H2D) |
| **4-bit + super-blocks + pinned H2D (this repo)** | **42.2–46.2 s** | **0.35–0.38** | verified across runs |

Generation output stays identical to the baseline (`北京，对吗？ 是的，中国的首都是北京。北京不仅是政治…`).

## How it relates to upstream projects

- **[AirLLM](https://github.com/Gavinlyonsimu/airllm)** (Apache-2.0) — the base runtime. We patch three of its files; `patches/UPSTREAM_VERSION.txt` pins the exact version (4.0.0). All quantization and model code remains AirLLM's.
- **[bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)** (MIT) — NF4/8-bit dequantisation primitives used by `uncompress_layer_state_dict`.
- **[transformers](https://github.com/huggingface/transformers)** (Apache-2.0) and **HuggingFace Hub** — model loading and tokenization.
- **[Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)** (Apache-2.0) — the reference model used in benchmarks.

## Future work / improvement directions

The following directions were investigated or designed but are **not** in this release. They are listed to save the next contributor time.

- **Multi-token decode.** Every generated token currently re-streams 18.7 GB. Batching 2–4 tokens per pass amortizes the transfer and would push throughput past 1 tok/s; requires changing AirLLM's single-sequence decode loop.
- **Fused H2D transfer.** The pinned pool is copied to GPU in ~176 per-tensor `.cuda()` calls (measured ~12 GB/s effective). A single contiguous H2D copy for the whole block would reach the ~28 GB/s pinned ceiling.
- **Pipeline dequantisation.** Dequant (~0.04 s/layer) currently runs on the main thread after H2D; overlapping it with the previous layer's matmul on a second CUDA stream would shave another ~20%.
- **Resident GPU layers.** Keeping `embed`/`lm_head` permanently in VRAM failed in testing (VRAM fragmentation slowed every pass by ~1 s); revisiting with 8-bit residency may work.
- **CPU-side layer offload.** Evaluating a subset of layers on CPU would bypass H2D entirely for those layers (same idea as llm.cpp offload).
- **Framework integration.** Porting these ideas into vLLM/SGLang-style backends for AirLLM layer streaming.

## Known limitations

- Pinned-memory pool uses ~2.2 GB of host RAM permanently.
- Only the optimisations that survive end-to-end verification are included; several promising ideas (2-worker prefetch, resident embed, main-thread dequant with pageable loads) were tried and rejected because measurements showed they were slower — see the results table above.
- Tested with Qwen3 and Llama-style architectures; other architectures supported by AirLLM should work but are not benchmarked.

## License

Apache-2.0. The modified AirLLM files remain under AirLLM's Apache-2.0 license.