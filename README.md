<p align="center">
  <img src="https://raw.githubusercontent.com/sagnik3788/slm-turbo/main/assets/download.svg" width="320" alt="SLM Turbo logo">
</p>
<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>
<p align="center"><strong>Profile and auto-tune local AI for your specific GPU.</strong></p>

Automated inference optimizer for LLMs. Profiles your GPU, classifies the bottleneck with a roofline model, and prescribes targeted fixes — KV quantization, prefix caching, chunked prefill, backend selection. Outputs a version-controlled recipe, not magic flags.

## Architecture

![architecture](https://raw.githubusercontent.com/sagnik3788/slm-turbo/main/diagrams/arch.png)

## How it works

```
slm-turbo analyze   →  probes model topology + GPU. classifies bottleneck.
                        prints: "memory-bound. 2.1 GB KV cache at 4096 tokens."

slm-turbo optimize  →  generates recipe.yaml. 4 optimizers evaluated per (model, GPU).
                        prints: "+52% throughput, -34% latency expected."

slm-turbo warmup    →  downloads weights, JIT-compiles Triton kernels, verifies recipe.
                        one-time. every serve after is instant.

slm-turbo serve     →  reads recipe.yaml, injects config into vLLM, launches server.
                        metrics on :8000/metrics.

slm-turbo status    →  live: tokens/sec, p50/p99 latency, GPU%, KV cache usage.
```

## What's under the hood

**Roofline profiler** — analytical by default (math, no GPU load). Falls back to empirical forward-pass when model fits VRAM — real CUDA overhead, real fragmentation, real bandwidth numbers. No spreadsheet guesses.

![roofline](https://raw.githubusercontent.com/sagnik3788/slm-turbo/main/diagrams/roofline.png)

The roofline model classifies each inference phase as memory-bound or compute-bound by comparing arithmetic intensity (FLOPs per byte) against the GPU's ridge point (peak compute ÷ memory bandwidth).

**Custom CUDA kernels** — fused KV cache dequant + attention, hand-written CUDA (JIT-compiled via `torch.utils.cpp_extension`). Per-channel asymmetric quantization (keys 4-bit, values 4-bit) for the standalone path; per-token min/step quantization for the native packed vLLM KV cache. Dequant in registers — never writes fp16 back to DRAM. Targets sm_75+ (GTX 1650 and up).

**Recipe engine** — 4 optimizers evaluated per (model, GPU) pair. Declarative YAML output. Human-readable, editable, diffable in git.

**Pluggable backends** — one recipe, any serving runtime. vLLM today. SGLang and TensorRT-LLM planned. No forks — we inject at each backend's config boundary.

## Targets

| | |
|---|---|
| GPUs | NVIDIA Turing+ (sm_75). GTX 1650 stress-tested. |
| Models | Any HuggingFace transformer. VLMs, 1B–70B+. |
| Backends | vLLM · SGLang (planned) · TensorRT-LLM (planned) |

## Install

```bash
# From PyPI — base install (analyze / optimize / doctor)
pip install slm-turbo

# Add the vLLM serving backend (needed for `slm-turbo serve`)
pip install "slm-turbo[gpu]"

# Or with uv — run directly without installing:
uvx slm-turbo analyze --model meta-llama/Llama-2-7b-hf

# Or with uv — install the CLI globally:
uv tool install slm-turbo
uv tool install "slm-turbo[gpu]"    # with the vLLM backend
```

Requires Linux + NVIDIA GPU (Turing/sm_75 or newer). `analyze` needs PyTorch + Transformers; `serve` additionally needs vLLM (the `[gpu]` extra).

### From source (development)

```bash
git clone <repo-url> && cd slm-turbo
pip install -e ".[gpu]"     # or: uv sync --extra gpu
slm-turbo doctor
```

## Usage

```bash
slm-turbo doctor              # check env: CUDA, PyTorch, vLLM, GPU, permissions
slm-turbo analyze  --model meta-llama/Llama-2-7b-hf
slm-turbo optimize --model meta-llama/Llama-2-7b-hf
slm-turbo warmup   --model meta-llama/Llama-2-7b-hf
slm-turbo serve    --model meta-llama/Llama-2-7b-hf --config ~/.slm-turbo/recipes/llama-2-7b*.yaml
slm-turbo status
```

## Benchmarks

Measured on a **NVIDIA GTX 1650 (sm_75, 4 GB VRAM)** with PyTorch 2.11 + CUDA 13.1.
Reproduce with:

```bash
python kernels/benchmark/benchmark.py --op-only                 # op-level
python kernels/benchmark/benchmark.py --model-only --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
python kernels/benchmark/benchmark.py --model-only --model Qwen/Qwen2-0.5B-Instruct
python kernels/benchmark/benchmark.py --model-only --model Qwen/Qwen2-1.5B-Instruct
```

### Decode attention — 4-bit packed vs fp16 (op-level)

The 4-bit KV cache reads 4× less memory per decode step, which is the
memory-bandwidth-bound decode phase's bottleneck. Our hand-written CUDA kernel
is **2–4.7× faster than fp16 attention**:

| Sequence len | D=64 kernel | D=64 fp16 | speedup | D=128 kernel | D=128 fp16 | speedup |
|---|---|---|---|---|---|---|
| 512 | 60.5 µs | 286.5 µs | **4.74×** | 171.6 µs | 474.8 µs | 2.77× |
| 1024 | 136.1 µs | 525.4 µs | 3.86× | 396.3 µs | 908.9 µs | 2.29× |
| 2048 | 227.8 µs | 967.7 µs | 4.25× | 861.6 µs | 1794.0 µs | 2.08× |

![Decode speedup](https://raw.githubusercontent.com/sagnik3788/slm-turbo/main/docs/decode_speedup.png)

KV-cache memory per token drops **16–32×** (fp16 → 4-bit packed): e.g. 8 MB →
256 KB for D=64, KV=4 at 2048 tokens.

### Model-level decode throughput

Full-model decode (transformers eager, interleaved timing) — the kernel is
~0.84–0.89× eager here because at short context the non-attention layers
(matmuls) dominate; the 4-bit win compounds as context grows.

| Model | layers | head_dim | eager fp16 | slm-turbo 4-bit |
|---|---|---|---|---|
| TinyLlama-1.1B-Chat | 22 | 64 | 49.2 tok/s | **43.8 tok/s** (0.89×) |
| Qwen2-0.5B-Instruct | 24 | 64 | 45.9 tok/s | **38.7 tok/s** (0.84×) |
| Qwen2-1.5B-Instruct | 28 | 128 | 35.4 tok/s | **31.3 tok/s** (0.89×) |

![Model throughput](https://raw.githubusercontent.com/sagnik3788/slm-turbo/main/docs/model_throughput.png)

### Native packed KV cache in vLLM (`--custom` backend)

The vLLM adapter uses a **natively allocated 4-bit KV cache** (TurboQuant-style
hook: `get_kv_cache_shape` + quantize-on-write `do_kv_cache_update`), registered
at import time with **zero vLLM source modifications**.

| Metric | Stock vLLM | slm-turbo CUSTOM |
|---|---|---|
| KV cache capacity (same 0.72 GiB) | 34,160 tokens | **121,456 tokens (3.56×)** |
| TinyLlama decode | 48.3 tok/s | 40.3 tok/s |

**Honest trade-off:** at short context the packed backend trails stock vLLM on
throughput (~17%) because per-layer Python launch overhead dominates; the
memory win (3.56× context) and the 2–4.7× long-context decode speedup are
where the design pays off. Prefill is intentionally delegated to stock SDPA —
on sm_75 there are no tensor cores to win with.

