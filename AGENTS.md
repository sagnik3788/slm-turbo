# SLM Turbo — Agent Instructions

## Project

SLM Turbo is an automated LLM inference optimization platform. It profiles, diagnoses, and accelerates transformer model deployments across local GPUs and production environments. The core value proposition is **automated diagnosis + GPU-specific recipe generation** — not just exposing optimization flags, but telling the user *why* their model is slow on *their specific hardware* and applying the right fix.

**Why this matters:** Most LLM serving tools expose 50+ tuning knobs. Users guess. We measure, model, and prescribe.

## Target Scope (Locked)

| Dimension | Constraint |
|-----------|------------|
| **GPUs** | NVIDIA Turing and newer (sm_75+). GTX 1650 (4GB, no tensor cores) is the stress-test target. |
| **Models** | Any HuggingFace transformer with standard `config.json`. Non-attention (Mamba, RWKV) detected and passed through. VLMs supported — text backbone only. |
| **Size** | 1B to 70B+ parameters. Same engine, different recipes. |
| **Mode** | Offline optimization only for v1. `analyze` → `suggest` → `optimize` → `serve`. Hot-reconfigure deferred. |
| **Platform** | Linux only. CUDA 11.8 and 12.x. Python 3.9–3.12. |

## Product Philosophy

1. **Optimizations are data, not code paths.** Every optimizer emits a declarative `Recipe` (Pydantic model) that the engine applies and persists. A "run" is `{baseline_profile, recipe, applied_profile, delta}`.
2. **Model-agnostic via topology.** Optimizers take a `ModelTopology` object derived from HF `config.json`, never a model name.
3. **GPU-agnostic via capability detection.** A `DeviceProfile` probes the GPU (sm version, memory bandwidth, VRAM, features) and gates optimizations by measured capability, not hardcoded GPU names.
4. **vLLM as a library, not a fork.** We inject at vLLM's `AttentionBackend` and `CacheConfig` boundaries. The adapter is the only vLLM-coupled file.

## Architecture

### Layered Design

```
┌─ Interfaces ─────────────────────────────────────────┐
│  CLI (one-shot) · Dashboard (deferred) · API (deferred)
├─ Orchestrator ───────────────────────────────────────┤
│  Run lifecycle · recipe execution · snapshot store     │
├─ Engine Core ────────────────────────────────────────┤
│  Profiler · Roofline Analyzer · Recipe Engine        │
├─ Optimizers ─────────────────────────────────────────┤
│  KV Quant · Prefix Cache · Chunked Prefill ·         │
│  Attention Backend Selection                         │
├─ Adapters ───────────────────────────────────────────┤
│  vLLM (primary) · Mock (tests/CI)                    │
├─ Kernels ────────────────────────────────────────────┤
│  Triton: fused KV dequant + attention, layout transforms
├─ Topology ───────────────────────────────────────────┤
│  HF Config Probe · VLM-aware extensions              │
└──────────────────────────────────────────────────────┘
```

**Data Flow:**
1. User runs `slm-turbo analyze --model <id>`
2. `topology/probe.py` downloads `config.json` → builds `ModelTopology`
3. `engine/profiler.py` probes GPU → builds `DeviceProfile`
4. `engine/roofline.py` classifies bottleneck (memory-bound vs compute-bound)
5. `engine/recipe.py` generates `Recipe` with optimization steps
6. `optimizers/*.py` each contribute steps if `can_apply(topology, device)` returns True
7. `adapters/vllm_adapter.py` applies recipe to vLLM serving config
8. `slm-turbo serve` launches optimized vLLM instance

### Module Layout

```
slm_turbo/
├── cli.py                    # Typer: all CLI commands
├── engine/
│   ├── core.py               # OptimizationEngine (library entry point)
│   ├── profiler.py           # GPU timing + memory tracker
│   ├── roofline.py           # Bottleneck classifier
│   ├── recipe.py             # Recipe schema + validation + versioning
│   └── snapshot.py           # ProfileSnapshot, OptimizationRun
├── topology/
│   ├── probe.py              # HF config → ModelTopology
│   └── vlm.py                # VLM-aware topology extensions
├── optimizers/
│   ├── base.py               # ABC: can_apply(), apply(), estimate_delta()
│   ├── kv_quant.py           # Asymmetric KV quantization
│   ├── prefix_cache.py       # Radix-style prefix caching
│   ├── chunked_prefill.py    # Chunked prefill for long contexts
│   └── attention_backend.py  # Backend selection (FA2, custom, fallback)
├── adapters/
│   ├── base.py               # ABC: start(), stop(), get_metrics()
│   ├── vllm_adapter.py       # vLLM integration (AttentionBackend + CacheConfig hooks)
│   └── mock_adapter.py       # For tests / no-GPU CI
├── kernels/
│   └── triton/
│       ├── kv_dequant.py     # Fused dequant + attention kernel
│       └── layout.py         # Blocked / paged KV layouts
├── dashboard/                # deferred to v2
└── api/                      # deferred to v2
```

### Key Schemas

**ModelTopology** (from HF `config.json`):
```python
class ModelTopology(BaseModel):
    model_id: str
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int           # GQA/MQA support
    head_dim: int
    num_hidden_layers: int
    sliding_window: Optional[int]
    max_position_embeddings: int
    torch_dtype: str
    architectures: List[str]
    is_vlm: bool = False
    quantization_config: Optional[Dict]
    
    # Derived properties
    @property
    def is_gqa(self) -> bool:
        return self.num_kv_heads < self.num_attention_heads
    
    @property
    def kv_cache_size_per_token(self) -> int:
        """Bytes per token in KV cache (fp16)"""
        return 2 * self.num_kv_heads * self.head_dim * self.num_hidden_layers * 2  # 2 bytes per fp16
```

**DeviceProfile** (from GPU probe):
```python
class DeviceProfile(BaseModel):
    name: str
    sm_version: int             # e.g., 75 for sm_75
    memory_bandwidth_gbps: float
    vram_mb: int
    sm_count: int
    supports_tensor_cores: bool
    supports_async_copy: bool   # sm_80+
    
    @property
    def is_small_gpu(self) -> bool:
        """GTX 1650 class — no tensor cores, limited bandwidth"""
        return self.vram_mb < 6000 and not self.supports_tensor_cores
```

**Recipe** (versioned, validated):
```python
class OptimizationStep(BaseModel):
    name: str
    enabled: bool
    params: Dict[str, Any]
    target_phase: Literal["prefill", "decode", "both"]
    expected_delta: Dict[str, float]  # {"throughput": 1.5, "latency": 0.8}

class Recipe(BaseModel):
    version: str = "1.0"
    topology_hash: str            # hash of ModelTopology dict
    device_hash: str              # hash of DeviceProfile dict
    steps: List[OptimizationStep]
    
    def validate_compatibility(self, topology: ModelTopology, device: DeviceProfile) -> bool:
        """Check if recipe matches current topology + device"""
        return (
            self.topology_hash == hash_dict(topology.model_dump()) and
            self.device_hash == hash_dict(device.model_dump())
        )
```

### vLLM Integration Boundary

We do not fork vLLM. We inject at two hook points:

1. **AttentionBackend**: Subclass vLLM's backend interface. Return our fused kernel when:
   - `device.sm_version >= 75`
   - `topology.head_dim` is compatible with kernel tile sizes
   - Roofline says custom kernel beats FA2 on this device
   
2. **CacheConfig**: Wrap vLLM's cache manager to:
   - Use quantized KV layout (4-bit K, 2-bit V)
   - Adjust block size based on `topology.kv_cache_size_per_token`
   - Enable prefix caching if `optimizer.can_apply()` says yes

**Isolation guarantee:** If vLLM's interface changes, only `adapters/vllm_adapter.py` breaks. All other code is vLLM-agnostic.

## Core Subsystems

### KV Quantization Optimizer (`optimizers/kv_quant.py`)

**What it does:** Reduces KV cache memory footprint via asymmetric quantization.

**Algorithm:**
- Keys: 4-bit per-channel (per head) with learned/zero-point scales
- Values: 2-bit or 4-bit (configurable, 2-bit for aggressive compression)
- Dequantization happens in registers during attention computation — never writes fp16 back to DRAM
- Block size: 64 or 128 tokens (tuned per GPU bandwidth)

**When to apply:**
- `device.vram_mb < 8000` (always on small GPUs)
- `topology.kv_cache_size_per_token * max_seq_len > 0.5 * device.vram_mb`
- Decode phase only (prefill is compute-bound, KV cache not the bottleneck)

**Safety gate:** Perplexity validation on 100-token sample. If `perplexity_delta > 5%`, disable and warn.

### Prefix Cache Optimizer (`optimizers/prefix_cache.py`)

**What it does:** Reuses KV cache for repeated prompt prefixes (radix-style matching).

**Algorithm:**
- Hash prompt tokens → lookup in radix tree
- On match: skip prefill for matching prefix, only compute new suffix
- Automatic eviction when KV cache full (LRU within prefix cache)

**When to apply:**
- `topology.num_hidden_layers <= 80` (deep models have high prefix storage cost)
- Interactive/chat use cases (repeated system prompts)
- Not for single-turn batch inference

### Chunked Prefill Optimizer (`optimizers/chunked_prefill.py`)

**What it does:** Splits long prefill into chunks to reduce latency spikes.

**Algorithm:**
- Chunk size: 512 or 1024 tokens (configurable)
- Interleave prefill chunks with decode tokens to maintain responsiveness
- vLLM native feature — we just tune chunk size based on GPU compute capability

**When to apply:**
- `max_position_embeddings > 4096`
- Interactive serving (not batch)
- Always safe to enable, minimal downside

### Attention Backend Optimizer (`optimizers/attention_backend.py`)

**What it does:** Selects optimal attention implementation per (model, GPU) pair.

**Options:**
1. **FlashAttention-2**: Default for A100+, H100. Best for compute-bound prefill.
2. **Custom Triton kernel**: For small GPUs where FA2 overhead exceeds savings. Fused dequant + attention.
3. **PyTorch SDPA**: Fallback. Always available, never crashes.

**Selection logic:**
```python
if device.supports_tensor_cores and device.sm_version >= 80:
    return "flash_attention_2"
elif device.sm_version >= 75 and kernel_validation_passed:
    return "custom_triton"
else:
    return "sdpa"
```

## Key Concepts

- **Prefill vs decode**: The system profiles both phases separately and applies phase-specific optimizations. Prefill is compute-bound (big matmuls). Decode is memory-bandwidth-bound (small kernels + huge KV cache reads).
- **Asymmetric KV quantization**: Keys and values use different quantization schemes because keys are more sensitive to precision loss than values.
- **Roofline analysis**: Plots achievable throughput vs arithmetic intensity to classify whether a workload is memory-bound or compute-bound on a specific GPU.
- **PagedAttention**: vLLM's non-contiguous KV cache storage. Our optimizations respect the paging layout.
- **GQA (Grouped Query Attention)**: `num_kv_heads < num_attention_heads`. Reduces KV cache size. Our topology probe detects this automatically.
- **MQA (Multi-Query Attention)**: `num_kv_heads == 1`. Extreme KV cache reduction.

## Tech Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.9+ | Primary language |
| PyTorch | 2.0+ | Tensor operations, CUDA |
| Transformers | 4.40+ | HF config probing |
| vLLM | 0.4+ | Serving integration |
| Triton | (bundled with PyTorch) | GPU kernel development |
| Typer | latest | CLI framework |
| Rich | latest | Terminal output |
| Pydantic | v2 | Schemas and validation |
| SQLite | stdlib | Run storage (v1) |

## CLI Commands

### `slm-turbo analyze --model <id>`

**Purpose:** Profile prefill/decode, print topology + roofline diagnosis.

**Flow:**
1. Download `config.json` from HF Hub (or use local path)
2. Build `ModelTopology`
3. Probe GPU → `DeviceProfile`
4. Run analytical profiler (no model loading)
5. Run roofline classifier
6. Print diagnosis table

**Output:**
```
Model: meta-llama/Llama-2-7b-hf
Topology: 4096 hidden, 32 heads, 32 layers, GQA=False
GPU: NVIDIA GeForce GTX 1650 (sm_75, 4GB VRAM)
Bottleneck: Memory-bound (decode phase)
KV cache: 2.1 GB at 4096 tokens (53% of VRAM)
Recommendation: Apply KV quantization + chunked prefill
```

### `slm-turbo optimize --model <id> [--recipe <file>]`

**Purpose:** Generate or apply recipe, produce optimized config.

**Flow:**
1. If `--recipe` provided: validate and apply existing recipe
2. Else: generate new recipe via `OptimizationEngine`
3. Save recipe to `~/.slm-turbo/recipes/<model_id>_<gpu_hash>.yaml`
4. Print recipe summary + expected deltas

### `slm-turbo serve --model <id> --config <file>`

**Purpose:** Launch vLLM with optimized settings.

**Flow:**
1. Load recipe from config file
2. Build vLLM `CacheConfig` + `AttentionBackend` from recipe
3. Start vLLM server with injected adapters
4. Print serving URL + metrics endpoint

### `slm-turbo status`

**Purpose:** Show live serving metrics.

**Metrics:**
- Throughput (tokens/sec)
- Latency (p50, p99)
- GPU utilization (%)
- KV cache fill (%)
- Active requests

### `slm-turbo warmup --model <id>`

**Purpose:** Pre-download weights, JIT-compile Triton kernels, cache everything.

**Flow:**
1. Download model weights to HF cache
2. Run 1 forward pass to trigger Triton JIT compilation
3. Save compiled kernels to `~/.triton/cache/`
4. Verify all optimizers can_apply() with this (topology, device)

### `slm-turbo doctor`

**Purpose:** Diagnose environment.

**Checks:**
- Python version (3.9+)
- PyTorch + CUDA availability
- GPU detection (sm version, VRAM)
- vLLM version compatibility
- Triton version
- Write permissions for cache dirs (`~/.slm-turbo/`, `~/.triton/cache/`)
- Memory availability (RAM + VRAM)

## Build Plan (8-10 Weeks)

**Week 1: Environment + Topology**
- Install PyTorch, vLLM, verify on 1650
- `topology/probe.py` reads any HF config and builds `ModelTopology`
- `cli.py` skeleton (Typer + Rich)
- **Deliverable:** `slm-turbo analyze` prints topology (no GPU profiling yet)

**Week 2: Profiler + Roofline**
- `engine/profiler.py` (analytical + empirical branches)
- `engine/roofline.py` (bottleneck classifier)
- `cli analyze` command with full diagnosis
- **Deliverable:** `slm-turbo analyze` prints roofline diagnosis

**Week 3: STANDALONE KERNEL VALIDATION (KILL WEEK)**
- `kernels/triton/kv_dequant.py` (synthetic data only)
- Benchmark vs PyTorch eager
- **Decision gate:** If kernel slower than PyTorch on 1650 → pivot to optimizer-only mode
- **Deliverable:** Working kernel + benchmark script

**Week 4: Recipe Engine + Optimizers**
- `engine/recipe.py` (schema, validation, versioning)
- `optimizers/base.py` (ABC)
- `optimizers/kv_quant.py`
- `optimizers/attention_backend.py`
- **Deliverable:** `slm-turbo optimize` generates recipe

**Week 5: vLLM Adapter + Integration**
- `adapters/vllm_adapter.py`
- Integrate kernel (if Week 3 passed)
- `cli optimize + serve` commands
- **Deliverable:** End-to-end: analyze → optimize → serve

**Week 6: Remaining Optimizers + Testing**
- `optimizers/prefix_cache.py`
- `optimizers/chunked_prefill.py`
- Tier 1 tests (CPU, CI)
- Tier 2 tests (GPU smoke)
- **Deliverable:** All 4 optimizers + test suite

**Week 7: Observability + Polish**
- `cli status` command
- `cli warmup` command
- `cli doctor` command
- Metrics collection
- **Deliverable:** Full CLI suite

**Week 8: Benchmarks + Documentation**
- Before/after benchmarks on 1650
- README with architecture diagram
- Technical blog post draft
- **Deliverable:** Benchmarks + docs

**Week 9-10: Buffer / Kernel Debugging**
- If kernel has issues, this is fix time
- If kernel works, add polish / extra tests
- Final resume packaging
- **Deliverable:** Production-ready v1

## Testing Strategy

Three-tier testing to handle the "no GPU in CI" problem:

### Tier 1: CPU-only unit tests (GitHub Actions)

**Coverage target: 60% of codebase**

Tests:
- `topology.probe()` with cached `config.json` fixtures (20+ model configs)
- `roofline.py` with synthetic `ModelTopology` + `DeviceProfile` — pure math, no GPU
- Recipe schema validation, migration, serialization
- Optimizer `can_apply()` logic with mocked profiles
- Mock adapter lifecycle tests
- Recipe compatibility validation

**CI:** Run on every PR. `pytest tests/tier1/`

### Tier 2: GPU smoke tests (local 1650, manual)

Tests:
- Boot real vLLM adapter with a 1B model (e.g., `TinyLlama/TinyLlama-1.1B`)
- Run 10 tokens, verify no crash
- Verify output coherence (not gibberish)
- Check GPU memory stays within bounds

**Trigger:** Run before every commit via `make gpu-test`

### Tier 3: Kernel correctness tests (local 1650, manual)

Tests:
- Random Q/K/V tensors vs PyTorch reference
- `torch.allclose(your_output, reference, atol=1e-2, rtol=1e-2)`
- Multiple head_dim values (64, 80, 128)
- Multiple sequence lengths (128, 512, 2048, 4096)
- **This is the only test that catches kernel bugs**

**Trigger:** Run after any kernel change. `pytest tests/tier3/`

## Decision Framework for Agents

When implementing features, follow this priority order:

1. **Safety first:** Never compromise output quality. Perplexity gate is mandatory for quantization.
2. **Measure before optimize:** Analytical profiling always runs before empirical. Don't load a 70B model to profile a 1B.
3. **Graceful degradation:** Every optimization must have a fallback path. If Triton kernel fails, use PyTorch.
4. **GPU-agnostic:** Never hardcode GPU names. Use `DeviceProfile` capabilities.
5. **Model-agnostic:** Never hardcode model names. Use `ModelTopology` properties.
6. **Test at the right tier:** CPU tests for logic, GPU tests for integration, kernel tests for correctness.

## Code Style & Conventions

- **Type hints:** All functions must have type hints. Use `from __future__ import annotations` for forward references.
- **Pydantic:** All data schemas use Pydantic v2 BaseModel.
- **Error handling:** Use custom exception hierarchy under `slm_turbo.exceptions`. Never catch bare `Exception`.
- **Logging:** Use `structlog` or standard `logging` with consistent format. Log at DEBUG for internals, INFO for user-facing, WARNING for fallbacks, ERROR for failures.
- **Async:** CLI is synchronous. Adapters may use async internally but expose sync interfaces.
- **File naming:** `snake_case.py` for modules. Classes are `PascalCase`. Functions/variables are `snake_case`. Constants are `UPPER_SNAKE_CASE`.

## Error Handling Patterns

```python
# Custom exceptions
class SLMTurboError(Exception): pass
class TopologyError(SLMTurboError): pass
class GPUError(SLMTurboError): pass
class RecipeError(SLMTurboError): pass
class KernelError(SLMTurboError): pass

# Usage pattern
try:
    kernel_output = triton_kernel(q, k, v)
except triton.CompilationError as e:
    logger.warning(f"Triton compilation failed: {e}. Falling back to PyTorch.")
    kernel_output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
except CUDAError as e:
    logger.error(f"CUDA error in kernel: {e}")
    raise KernelError(f"Kernel execution failed: {e}") from e
```

## Packaging & Distribution

**v1 scope:** Linux only. CUDA 11.8 and 12.x supported. Python 3.9–3.12 (3.13 experimental).

**Install:** `pip install slm-turbo` with JIT Triton compilation on first run. No prebuilt wheels.

**First-run experience:** `slm-turbo warmup --model <id>` downloads weights + compiles kernels in the background. Subsequent runs are instant.

**`slm-turbo doctor` checks:**
- Python version
- PyTorch + CUDA availability
- GPU detection (sm version, VRAM)
- vLLM version
- Triton version
- Write permissions for cache dirs
- Memory availability

## Triton Risks & Mitigations

**Architecture support:** Triton core language works on sm_75+. We avoid advanced features (TMA, warp specialization, async copy) that require sm_80+ or sm_90. Tile sizing and loop ordering are the primary optimization levers.

**JIT compilation overhead:** 30–60 seconds on first kernel call. Cached to `~/.triton/cache/` thereafter. `warmup` command triggers this proactively.

**Compilation failures:** Different CUDA versions or driver mismatches can cause opaque Triton compile errors. Mitigation: catch `triton.CompilationError`, log the full traceback, and fall back to PyTorch eager path with a warning.

**Register pressure:** Triton compiler decides what lives in registers vs shared memory. If the compiler spills, performance collapses. Mitigation: benchmark multiple tile sizes, validate standalone before integration.

**Debugging:** CUDA illegal memory access from Triton kernels produces opaque errors. Mitigation: `CUDA_LAUNCH_BLOCKING=1` for deterministic error location, standalone kernel tests before vLLM integration.

## Resume / Interview Framing

**30-second pitch:**
> "I built an automated LLM inference optimizer. It profiles your GPU, diagnoses the bottleneck with a roofline model, and applies targeted optimizations — including a custom Triton kernel for asymmetric KV quantization. On a GTX 1650, it doubled throughput for a 1B model compared to stock vLLM."

**What it proves:**
- GPU kernel development (Triton)
- Performance engineering (measurement, diagnosis, optimization)
- Systems integration (vLLM plugin architecture)
- Product thinking (CLI design, schema versioning, user journey)

**Comparable projects:** vLLM (Berkeley), llama.cpp (Georgi Gerganov → Meta), FlashAttention (Stanford). This sits in the same systems+ML engineering space.

## Glossary

| Term | Definition |
|------|------------|
| **Prefill** | Processing the input prompt. Compute-bound (large matmuls). |
| **Decode** | Generating output tokens one at a time. Memory-bandwidth-bound (small kernels + KV cache reads). |
| **KV Cache** | Key-value tensors stored from previous tokens to avoid recomputation in attention. Grows with sequence length. |
| **GQA** | Grouped Query Attention. `num_kv_heads < num_attention_heads`. Reduces KV cache size. |
| **MQA** | Multi-Query Attention. `num_kv_heads == 1`. Extreme KV cache reduction. |
| **Roofline Model** | Performance model plotting achievable throughput vs arithmetic intensity. Classifies memory-bound vs compute-bound. |
| **Arithmetic Intensity** | FLOPs per byte of memory traffic. High = compute-bound, low = memory-bound. |
| **PagedAttention** | vLLM's non-contiguous KV cache storage. Enables efficient memory sharing and prefix caching. |
| **Triton** | Python-like language for writing GPU kernels. Compiled to CUDA/SASS via LLVM. |
| **SM** | Streaming Multiprocessor. NVIDIA GPU compute unit. `sm_75` = Turing architecture. |
| **Tensor Cores** | Specialized matrix multiply units on A100+ GPUs. Not available on GTX 1650. |
| **Quantization** | Reducing precision of tensors (e.g., fp16 → 4-bit) to save memory/bandwidth. |
| **Asymmetric Quantization** | Different quantization schemes for different tensors (e.g., keys 4-bit, values 2-bit). |
| **JIT Compilation** | Just-in-time compilation. Triton kernels compile on first use. Cached thereafter. |
| **Recipe** | Declarative optimization plan. Versioned, validated, GPU-specific. |
| **Topology** | Model architecture properties derived from `config.json`. Never requires loading weights. |

## Status

This repo is in early setup. Environment needs PyTorch + vLLM installation. Project skeleton not yet created.

**Current blockers:**
- PyTorch not installed (verified: `ModuleNotFoundError`)
- vLLM not installed
- No GPU detected in this environment

**Next immediate steps:**
1. Install PyTorch with CUDA support
2. Install vLLM
3. Create project skeleton (`slm_turbo/` package structure)
4. Implement `topology/probe.py`
5. Implement `cli.py` skeleton

## Open Questions for Future Brainstorming

1. **Recipe storage format**: YAML (human-editable, git-friendly) vs JSON (machine-friendly)? Engine uses Pydantic objects internally, so format is swappable.
2. **Profiler measurement strategy**: synthetic kernel screen (fast, approximate) vs real forward-pass validation (slow, accurate) vs both (quick rank + validate top-2)?
3. **Dashboard v2 scope**: React + design system (3x work, high UX ceiling) vs HTMX + minimal CSS (1 week, low ceiling)?
4. **Multi-GPU v2**: tensor parallelism, pipeline parallelism — how does the recipe adapt?
5. **Build/distribution**: `pip install slm-turbo` with JIT Triton compilation, or prebuilt CUDA-extension wheels per torch/CUDA version?
6. **Online serving v2**: Hot-reconfigure recipes without restarting vLLM. Architecturally possible via adapter reload, but needs careful state management.
7. **Model quantization v2**: INT8/INT4 weight quantization (GPTQ, AWQ) — orthogonal to KV quantization, can stack.
8. **Speculative decoding v2**: Draft model for faster decode — requires separate model loading, complex integration.
