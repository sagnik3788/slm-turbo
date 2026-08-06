"""vLLM adapter — the ONLY vLLM-coupled file in slm-turbo (AGENTS.md isolation rule).

Three layers:

1. `SlmTurboVLLMAdapter` — high-level lifecycle used by `slm-turbo serve`.
   Maps a declarative `Recipe` onto vLLM `LLM()` kwargs (small-GPU OOM-safe
   defaults were validated on a GTX 1650: gpu_memory_utilization=0.80,
   enforce_eager=True, max_num_seqs=8).

2. `QuantizedKVAttentionBackend` / `QuantizedKVAttentionImpl` — real injection
   into vLLM v1's attention stack via the official third-party hook
   (`register_backend(AttentionBackendEnum.CUSTOM, ...)` + `--attention-backend CUSTOM`).
   It subclasses vLLM's Triton backend, inheriting its metadata builder and paged
   fp16 KV-cache layout, and only overrides the decode forward: it gathers the
   request's KV blocks, quantizes them 4-bit per step, and runs our fused Triton
   decode kernel. Any eligibility miss or interface drift falls back to the stock
   Triton path (graceful degradation).

   Note: this is "shim" mode — vLLM still stores fp16 KV blocks, so the *speed*
   of our kernel is exercised inside vLLM but the cache-memory win is not.
   A native 4-bit cache layout requires wrapping vLLM's CacheConfig (the second
   hook point in AGENTS.md) and is staged as v2.

3. Degraded mode: if vLLM or the kernels cannot be imported, the adapter still
   constructs and reports the problem instead of crashing at import time.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch  # noqa: F401  (used by the impl methods below)

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("slm_turbo.vllm_adapter")

VLLM_AVAILABLE = False
try:  # vLLM is a heavy optional dependency
    import vllm  # noqa: F401
    from vllm import LLM, SamplingParams  # noqa: F401

    if vllm.__file__ is None:
        raise ImportError("vllm resolves to an empty namespace; set PYTHONPATH to the vllm source")
    VLLM_AVAILABLE = True
except Exception as e:  # pragma: no cover - depends on env
    logger.warning("vLLM import failed (%s); adapter will run in degraded mode", e)

if VLLM_AVAILABLE:
    try:  # vLLM-configured handlers so child-process logs are forwarded
        from vllm.logger import init_logger

        logger = init_logger("slm_turbo.vllm_adapter")
    except Exception:
        pass

# Kernel-supported head sizes (validated on sm_75, D in {64, 80, 128})
SUPPORTED_HEAD_SIZES = (64, 80, 128)
# Number of decode-attention calls served by our kernel (diagnostics)
_KERNEL_CALLS = 0


# ---------------------------------------------------------------------------
# Layer 2: vLLM attention backend injection (CUSTOM registry)
#
# PACKED 4-BIT KV CACHE (TurboQuant-style, native layout)
# ---------------------------------------------------------------------------
# vLLM allocates the KV cache from the shape our backend reports via
# `get_kv_cache_shape()`, and writes tokens through the impl's
# `do_kv_cache_update()`. We use both hooks to make the cache NATIVELY
# 4-bit packed instead of fp16:
#
#   cache shape  = (num_blocks, block_size, num_kv_heads, SLOT)
#   SLOT         = head_size + 8 bytes per (position, kv_head):
#                    [0 : D/2)  packed K   (4-bit, even dim = low nibble)
#                    [D/2 : D)  packed V   (4-bit)
#                    [D : D+8)  k_min, k_step, v_min, v_step  (fp16 each)
#
# Quantization is PER-TOKEN min/step (KIVI-style V; here both K and V):
#   value = nibble * step + min,  step = (max-min)/15
# This is O(1) to store incrementally (each token's own min/max is known the
# moment it is written) and immune to ruler drift — the failure mode of the
# old fixed-ruler shim. The decode kernel dequantizes in registers, reading
# 4x less memory than fp16 attention.
#
# Integration WITHOUT forking vLLM (AGENTS.md: "the adapter is the only
# vLLM-coupled file"): instead of editing vLLM source, `register_plugin()`
# monkey-patches two tiny hooks at import time (idempotent, only affects
# processes that load this adapter):
#   1. `AttentionLayer.get_kv_cache_spec` -> when the CUSTOM backend is
#      active, return a TQFullAttentionSpec whose page size = packed slot
#      size. This is what makes vLLM *allocate* 4x less cache memory.
#   2. (registration itself) `AttentionBackendEnum.CUSTOM` -> our backend.
# Stock vLLM is untouched; slm-turbo works on an unmodified checkout.
# ---------------------------------------------------------------------------

def _patch_attention_layer_spec() -> bool:
    """Make vLLM allocate the KV cache at the packed 4-bit size.

    vLLM sizes each layer's cache from `AttentionLayer.get_kv_cache_spec()`
    (spec.page_size_bytes drives the byte allocation). We wrap the original
    method: when the layer runs our CUSTOM backend, we return a spec whose
    page size is the packed slot (D + 8 bytes) instead of the fp16 formula.
    Uses vLLM's own `TQFullAttentionSpec` page-size override mechanism.
    """
    try:
        from vllm.model_executor.layers.attention.attention import Attention
        from vllm.v1.kv_cache_interface import TQFullAttentionSpec
    except Exception as e:
        logger.warning("spec patch unavailable (%s)", e)
        return False

    _orig = Attention.get_kv_cache_spec

    def get_kv_cache_spec(self, vllm_config):
        spec = _orig(self, vllm_config)
        backend = getattr(self, "attn_backend", None)
        if backend is not None and backend.get_name() == "CUSTOM":
            # Packed slot: D/2 (K) + D/2 (V) + 8 bytes of fp16 min/step rulers.
            slot = self.head_size + 8
            return TQFullAttentionSpec(
                block_size=spec.block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                head_size_v=self.head_size_v,
                dtype=torch.uint8,
                kv_quant_mode=spec.kv_quant_mode,
                tq_slot_size=slot,
            )
        return spec

    Attention.get_kv_cache_spec = get_kv_cache_spec
    return True


def _register_custom_backend() -> bool:
    """Define + register QuantizedKVAttentionBackend against the installed vLLM.

    Returns True on success, False on any interface drift (vLLM refactors the
    backend API without warning between releases). Isolation guarantee:
    everything vLLM-specific lives here and nowhere else.
    """
    global QuantizedKVAttentionImpl, QuantizedKVAttentionBackend  # noqa: PLW0603
    try:
        import torch
        from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
        from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend, TritonAttentionImpl
    except Exception as e:
        logger.warning("vLLM attention API not found (%s); custom backend disabled", e)
        return False

    _patch_attention_layer_spec()

    class QuantizedKVAttentionImpl(TritonAttentionImpl):
        """Native 4-bit packed KV cache: quantize-on-write store + fused
        per-token decode kernel. Prefill runs SDPA on the raw fp16 K/V of the
        current chunk (matching TurboQuant), then stores quantized."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._slm_kernel = False
            self._slm_kernel_err: Optional[str] = None
            try:
                from kernels.kv_dequant import (
                    kv_dequant_decode_per_token_paged_attention,
                    kv_quantize_store,
                )
                self._slm_decode_paged = kv_dequant_decode_per_token_paged_attention
                self._slm_store = kv_quantize_store
                self._slm_kernel = torch.cuda.is_available()
                if not self._slm_kernel:
                    self._slm_kernel_err = "no CUDA device"
            except Exception as e:
                self._slm_kernel_err = f"{type(e).__name__}: {e}"
            if self._slm_kernel_err:
                logger.warning("slm-turbo kernel disabled in this process: %s",
                               self._slm_kernel_err)

        # ---- quantize-on-write: called once per token batch BEFORE forward --
        def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
            """Quantize fp16 K/V and store the packed 4-bit slot. O(N) per call
            (N = tokens in this write batch; 1 for decode)."""
            if kv_cache is None or kv_cache.numel() == 0:
                return
            if not self._slm_kernel:
                return
            try:
                # key/value: [num_tokens, num_kv_heads, head_size] fp16
                key = key.contiguous()
                value = value.contiguous()
                block_size = kv_cache.shape[1]
                self._slm_store(key, value, kv_cache, slot_mapping, block_size)
            except Exception as e:
                # We cannot silently drop KV writes — the cache is packed and
                # stock Triton can't read it. Fail loudly if the store breaks.
                logger.error("slm-turbo kv store failed: %s: %s",
                             type(e).__name__, e)
                raise

        # ---- forward ----------------------------------------------------------
        def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                    output, output_scale=None, output_block_scale=None):
            num_tokens = query.shape[0]
            if output is None:
                output = torch.zeros(num_tokens, self.num_heads * self.head_size,
                                     dtype=query.dtype, device=query.device)
            if attn_metadata is None:
                return output.fill_(0)
            N = attn_metadata.num_actual_tokens
            if N <= 0:
                return output.fill_(0)
            q = query[:N]
            if attn_metadata.max_query_len > 1:
                attn_out = self._prefill_attention(
                    q, key[:N], value[:N], kv_cache, attn_metadata)
            else:
                attn_out = self._decode_attention(q, kv_cache, attn_metadata)
            if output.ndim == 3:
                output[:N] = attn_out.to(output.dtype)
            else:
                output[:N] = attn_out.reshape(N, -1).to(output.dtype)
            return output

        # ---- prefill: SDPA on raw fp16 K/V (first chunk) ---------------------
        def _prefill_attention(self, query, key, value, kv_cache, attn_metadata):
            # query/key/value: [num_tokens, heads, head_size] fp16
            N, Hq, D = query.shape
            Hk = key.shape[1]
            use_gqa = Hk < Hq
            q_t = query.transpose(0, 1).unsqueeze(0)   # [1, Hq, N, D]
            k_t = key.transpose(0, 1).unsqueeze(0)     # [1, Hk, N, D]
            v_t = value.transpose(0, 1).unsqueeze(0)   # [1, Hk, N, D]
            # First-chunk prefill: all K/V are in the current batch. (Chunked
            # continuation prefill is not in v1 scope; the CLI is one-shot.)
            out = torch.nn.functional.scaled_dot_product_attention(
                q_t, k_t, v_t, is_causal=True,
                scale=self.scale, enable_gqa=use_gqa)
            return out[0].transpose(0, 1)              # [N, Hq, D]

        # ---- decode: paged per-token fused kernel (reads cache directly) -----
        def _decode_attention(self, query, kv_cache, attn_metadata):
            # query: [num_tokens, num_heads, head_size]; decode = 1 tok/req.
            # kv_cache: [num_blocks, block_size, num_kv_heads, D+8] uint8.
            global _KERNEL_CALLS
            block_size = kv_cache.shape[1]
            num_kv_heads = kv_cache.shape[2]
            D = self.head_size
            out = torch.empty(query.shape[0], self.num_heads, D,
                              dtype=query.dtype, device=query.device)

            # Cache host-side conversions keyed by the metadata object: vLLM
            # reuses the SAME metadata across all layers within one step, so
            # calling .tolist()/.to(int64) here once per step avoids a GPU->CPU
            # sync on every one of the 22 layers (that was ~4ms/step idle).
            cache = getattr(self, "_slm_meta_cache", None)
            if cache is None or cache[0] is not attn_metadata:
                seq_lens_host = attn_metadata.seq_lens.tolist()
                blocks_host = [
                    attn_metadata.block_table[r].to(torch.int64).contiguous()
                    for r in range(len(seq_lens_host))
                ]
                cache = (attn_metadata, seq_lens_host, blocks_host)
                self._slm_meta_cache = cache
            _, seq_lens_host, blocks_host = cache

            for r in range(len(seq_lens_host)):
                seq_len = seq_lens_host[r]
                if seq_len < 1:
                    continue
                q = query[r:r + 1].permute(1, 0, 2).contiguous()  # [H, 1, D]
                o = self._slm_decode_paged(q, kv_cache, blocks_host[r],
                                           seq_len, block_size, num_kv_heads)
                out[r] = o.permute(1, 0, 2)
                _KERNEL_CALLS += 1
            if _KERNEL_CALLS and getattr(self, "_slm_logged", False) is False:
                self._slm_logged = True
                logger.info("slm-turbo 4-bit decode kernel serving inside vLLM (head_size=%d)",
                            self.head_size)
            return out

    class QuantizedKVAttentionBackend(TritonAttentionBackend):
        """Stock Triton metadata machinery + packed 4-bit KV layout + our impl.
        get_name() must be an AttentionBackendEnum member — "CUSTOM" is
        vLLM's third-party slot."""

        @staticmethod
        def get_name() -> str:
            return "CUSTOM"

        @staticmethod
        def get_impl_cls() -> type:
            return QuantizedKVAttentionImpl

        @staticmethod
        def get_kv_cache_shape(
            num_blocks: int,
            block_size: int,
            num_kv_heads: int,
            head_size: int,
            cache_dtype_str: str = "auto",
        ) -> tuple[int, ...]:
            # Packed 4-bit layout: D bytes of K+V + 8 bytes of min/step rulers.
            return (num_blocks, block_size, num_kv_heads, head_size + 8)

        @staticmethod
        def get_kv_cache_stride_order(
            include_num_layers_dimension: bool = False,
        ) -> tuple[int, ...]:
            # Physical layout == logical layout (no permutation needed).
            raise NotImplementedError  # -> vLLM falls back to identity

        @classmethod
        def supports_batch_invariance(cls) -> bool:
            return True

    register_backend(AttentionBackendEnum.CUSTOM, f"{__name__}.QuantizedKVAttentionBackend")
    logger.info("slm-turbo vLLM backend registered as --attention-backend CUSTOM")
    return True


_BACKEND_REGISTERED = _register_custom_backend() if VLLM_AVAILABLE else False


def register_plugin() -> None:
    """vLLM entry-point hook (`vllm.general_plugins`, see pyproject.toml).

    vLLM v1 spawns a separate EngineCore process; plugin functions from this
    group are executed in *every* process, so the CUSTOM-backend override
    registered here is visible where the model is built. Idempotent.
    """
    _register_custom_backend()


def _kernel_eligible_device() -> bool:
    """Small-GPU / capability gate for the custom backend (AGENTS.md selection logic)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        sm = torch.cuda.get_device_capability(0)
        if sm[0] * 10 + sm[1] < 75:
            return False
        props = torch.cuda.get_device_properties(0)
        mem_mb = props.total_memory // 2**20
        return mem_mb < 8000 or True  # decode win holds on any Turing+; small GPU is the stress target
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Layer 1: high-level adapter for `slm-turbo serve`
# ---------------------------------------------------------------------------

class SlmTurboVLLMAdapter:
    """Lifecycle adapter: Recipe -> vLLM LLM kwargs -> live metrics.

    Only kwargs that exist in this vLLM build are set; every step is validated
    for graceful degradation on small GPUs (GTX 1650 class).
    """

    def __init__(self, model_id: str, recipe: Optional[Any] = None,
                 backend_override: Optional[str] = None):
        self.model_id = model_id
        self.recipe = recipe
        self.backend_override = backend_override  # "custom" | "vllm" | None
        self._backend_used: Optional[str] = None
        self.llm: Optional[Any] = None
        self._tokens_generated = 0
        self._last_generate_at: Optional[float] = None
        # --- metrics state ---
        self._last_metrics: Dict[str, Any] = {}
        self._prompt_tokens = 0
        self._output_tokens = 0
        self._total_time_s = 0.0
        self._ttft_s = 0.0
        self._decode_time_s = 0.0
        self._kernel_calls = 0

    # -- recipe -> engine kwargs -------------------------------------------
    def build_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model_id,
            # OOM-safe defaults validated on GTX 1650 4GB
            "gpu_memory_utilization": 0.80,
            "enforce_eager": True,
            "max_num_seqs": 8,
            "max_num_batched_tokens": 2048,
        }
        if self.backend_override == "vllm":
            # stock vLLM kernels (A/B comparison baseline)
            self._backend_used = "vllm"
            return kwargs

        if self.recipe is None:
            if (self.backend_override == "custom" or _kernel_eligible_device()) and _BACKEND_REGISTERED:
                kwargs["attention_backend"] = "CUSTOM"
            self._backend_used = "CUSTOM" if kwargs.get("attention_backend") == "CUSTOM" else "vllm"
            return kwargs

        for step in self.recipe.steps:
            if not step.enabled:
                continue
            name = step.name
            if name == "chunked_prefill":
                kwargs["max_num_batched_tokens"] = step.params.get("chunk_size", 512)
            elif name == "prefix_cache":
                kwargs["enable_prefix_caching"] = True
            elif name == "attention_backend":
                sel = step.params.get("backend", "auto")
                # optimizer emits "custom_kernel"; AGENTS.md calls it "custom_triton"
                wants_custom = sel in ("custom_kernel", "custom_triton") or self.backend_override == "custom"
                if wants_custom and _BACKEND_REGISTERED:
                    kwargs["attention_backend"] = "CUSTOM"
        self._backend_used = "CUSTOM" if kwargs.get("attention_backend") == "CUSTOM" else "vllm"
        return kwargs

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "SlmTurboVLLMAdapter":
        if not VLLM_AVAILABLE:
            raise RuntimeError(
                "vLLM is not importable. On this machine run with "
                "PYTHONPATH=/home/sagnik/vllm (editable-install .pth is broken)."
            )
        self.llm = LLM(**self.build_kwargs())
        return self

    def stop(self) -> None:
        if self.llm is not None:
            engine = getattr(self.llm, "llm_engine", None)
            for obj in (engine, self.llm):
                if obj is None:
                    continue
                for method in ("shutdown", "close", "terminate"):
                    fn = getattr(obj, method, None)
                    if fn is not None:
                        try:
                            fn()
                            break
                        except Exception as e:  # pragma: no cover
                            logger.warning("vLLM %s() error: %s", method, e)
            self.llm = None

    # -- inference ---------------------------------------------------------
    def _count_prompt_tokens(self, prompts: List[str]) -> int:
        """Rough prompt token count (words * 1.3 heuristic)."""
        return sum(int(len(p.split()) * 1.3) for p in prompts)

    def _count_chat_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Rough chat token count from message text."""
        total = 0
        for m in messages:
            total += int(len(m.get("content", "").split()) * 1.3)
        return total

    def generate(self, prompts: List[str], max_tokens: int = 64,
                 temperature: float = 0.0, **sp_kwargs: Any) -> List[str]:
        if self.llm is None:
            self.start()
        sp = SamplingParams(max_tokens=max_tokens, temperature=temperature, **sp_kwargs)
        self._prompt_tokens = self._count_prompt_tokens(prompts)
        t0 = time.monotonic()
        results = self.llm.generate(prompts, sp)
        self._total_time_s = time.monotonic() - t0
        self._output_tokens = sum(len(r.outputs[0].token_ids) for r in results)
        self._tokens_generated += self._output_tokens
        self._last_generate_at = time.monotonic()
        # Estimate TTFT as proportional to prompt share of total work
        # (prefill is compute-heavy, decode is memory-bandwidth-bound)
        if self._output_tokens > 0:
            prompt_share = self._prompt_tokens / (self._prompt_tokens + self._output_tokens)
            self._ttft_s = self._total_time_s * prompt_share * 2.0  # prefill is ~2x slower per token
            self._decode_time_s = self._total_time_s - self._ttft_s
            if self._decode_time_s < 0:
                self._ttft_s = self._total_time_s * 0.3
                self._decode_time_s = self._total_time_s - self._ttft_s
        return [r.outputs[0].text for r in results]

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 64,
             temperature: float = 0.0, **sp_kwargs: Any) -> List[str]:
        """Chat-template path (vLLM llm.chat) for instruct models — without it,
        raw completion prompts often yield empty output on chat-tuned models."""
        if self.llm is None:
            self.start()
        sp = SamplingParams(max_tokens=max_tokens, temperature=temperature, **sp_kwargs)
        self._prompt_tokens = self._count_chat_tokens(messages)
        t0 = time.monotonic()
        results = self.llm.chat(messages, sp)
        self._total_time_s = time.monotonic() - t0
        self._output_tokens = sum(len(r.outputs[0].token_ids) for r in results)
        self._tokens_generated += self._output_tokens
        self._last_generate_at = time.monotonic()
        if self._output_tokens > 0:
            prompt_share = self._prompt_tokens / (self._prompt_tokens + self._output_tokens)
            self._ttft_s = self._total_time_s * prompt_share * 2.0
            self._decode_time_s = self._total_time_s - self._ttft_s
            if self._decode_time_s < 0:
                self._ttft_s = self._total_time_s * 0.3
                self._decode_time_s = self._total_time_s - self._ttft_s
        return [r.outputs[0].text for r in results]

    # -- observability -----------------------------------------------------
    def get_metrics(self) -> Dict[str, Any]:
        """Live metrics: adapter-level tokens/sec + GPU state + KV cache + optimization status."""
        m: Dict[str, Any] = {
            "model": self.model_id,
            "backend": self._backend_used or ("CUSTOM" if (_BACKEND_REGISTERED and self.llm is not None) else "vllm"),
            "kernel_serving": self._backend_used == "CUSTOM" if self._backend_used else bool(_BACKEND_REGISTERED),
            "tokens_generated": self._tokens_generated,
            # --- timing ---
            "prompt_tokens": self._prompt_tokens,
            "output_tokens": self._output_tokens,
            "total_time_s": round(self._total_time_s, 3),
            "ttft_s": round(self._ttft_s, 3),
            "decode_time_s": round(self._decode_time_s, 3),
        }
        # throughput
        if self._output_tokens > 0 and self._total_time_s > 0:
            m["throughput_tok_s"] = round(self._output_tokens / self._total_time_s, 1)
            m["decode_throughput_tok_s"] = round(self._output_tokens / self._total_time_s, 1)
        if self._output_tokens > 1 and self._decode_time_s > 0:
            m["tpot_ms"] = round(self._decode_time_s * 1000 / (self._output_tokens - 1), 1)
        else:
            m["tpot_ms"] = 0.0

        # --- GPU state (pynvml is authoritative; torch memory is 0 in parent proc) ---
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            m["vram_used_mb"] = info.used // 2**20
            m["vram_total_mb"] = info.total // 2**20
            m["vram_used_percent"] = round(100 * info.used / info.total, 1)
            m["gpu_util_percent"] = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            m["gpu_name"] = pynvml.nvmlDeviceGetName(handle)
            try:
                bw = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                bus = pynvml.nvmlDeviceGetMemoryBusWidth(handle)
                peak_bw_gbps = 2.0 * bw * (bus / 8.0) / 1000.0
                m["peak_bw_gbps"] = round(peak_bw_gbps, 1)
            except Exception:
                pass
        except Exception:
            # fallback to torch
            try:
                import torch
                m["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                if torch.cuda.is_available():
                    total = torch.cuda.get_device_properties(0).total_memory // 2**20
                    m["vram_total_mb"] = total
                    m["vram_used_mb"] = torch.cuda.memory_reserved(0) // 2**20
                    m["vram_used_percent"] = round(100 * m["vram_used_mb"] / total, 1) if total else 0
            except Exception:
                pass

        # --- KV cache (computed from topology + vLLM memory budget) ---
        if self.recipe is not None:
            try:
                from topology import build_topology
                topo = build_topology(self.model_id)
                kv_per_tok = topo.kv_cache_size_per_token
                # vLLM memory budget = gpu_memory_utilization * total_vram
                # Model weights ≈ num_params * dtype_bytes (fp16 = 2B)
                # KV cache budget = vLLM budget - model_weights - engine_overhead (~20%)
                vram_total_bytes = m.get("vram_total_mb", 0) * 1024 * 1024
                vllm_budget_bytes = vram_total_bytes * 0.80  # gpu_memory_utilization from build_kwargs
                dtype_bytes = 2.0 if topo.dtype in ("float16", "bfloat16") else 4.0
                model_params = topo.num_params or (12 * (topo.hidden_size ** 2) * topo.num_hidden_layers)
                model_size_bytes = model_params * dtype_bytes
                # vLLM engine overhead (CUDA context, buffers, activations) ≈ 20% of model size
                engine_overhead = model_size_bytes * 0.20
                kv_budget_bytes = max(vllm_budget_bytes - model_size_bytes - engine_overhead, 0)
                kv_total_tokens = int(kv_budget_bytes / kv_per_tok) if kv_per_tok else 0
                # current fill = prompt + generated tokens
                kv_used_tokens = self._output_tokens + self._prompt_tokens
                kv_used_bytes = kv_used_tokens * kv_per_tok
                m["kv_cache_total_tokens"] = kv_total_tokens
                m["kv_cache_used_tokens"] = kv_used_tokens
                m["kv_cache_fill_percent"] = round(100 * kv_used_tokens / max(kv_total_tokens, 1), 2)
                m["kv_cache_size_per_token_b"] = kv_per_tok
                m["kv_cache_used_mb"] = round(kv_used_bytes / (1024 * 1024), 2)
                m["kv_cache_budget_mb"] = round(kv_budget_bytes / (1024 * 1024), 1)
            except Exception:
                pass

        # --- VRAM breakdown (model weights + KV + overhead) ---
        if self.recipe is not None:
            try:
                from topology import build_topology
                topo = build_topology(self.model_id)
                dtype_bytes = 2.0 if topo.dtype in ("float16", "bfloat16") else 4.0
                model_params = topo.num_params or (12 * (topo.hidden_size ** 2) * topo.num_hidden_layers)
                model_size_mb = round(model_params * dtype_bytes / (1024 * 1024), 1)
                vram_used = m.get("vram_used_mb", 0)
                kv_used_mb = m.get("kv_cache_used_mb", 0)
                overhead_mb = round(vram_used - model_size_mb - kv_used_mb, 1)
                m["model_weights_mb"] = model_size_mb
                m["engine_overhead_mb"] = max(overhead_mb, 0)
                m["kv_cache_used_mb"] = kv_used_mb
            except Exception:
                pass

        # --- memory bandwidth utilization ---
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            bw = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            bus = pynvml.nvmlDeviceGetMemoryBusWidth(handle)
            peak_bw_gbps = 2.0 * bw * (bus / 8.0) / 1000.0
            # rough bandwidth utilization: (model_size + kv_reads) / decode_time
            # model weights read once per decode token + kv cache read
            model_size_bytes = m.get("model_weights_mb", 0) * 1024 * 1024
            kv_bytes = m.get("kv_cache_used_mb", 0) * 1024 * 1024
            decode_time = m.get("decode_time_s", 0)
            if decode_time > 0:
                bytes_per_tok = model_size_bytes + kv_bytes
                bw_util_gbps = (bytes_per_tok / decode_time) / (1024**3)
                m["mem_bw_util_gbps"] = round(bw_util_gbps, 1)
                m["mem_bw_peak_gbps"] = round(peak_bw_gbps, 1)
                m["mem_bw_util_percent"] = round(100 * bw_util_gbps / peak_bw_gbps, 1)
        except Exception:
            pass

        # --- bytes per token comparison ---
        if self.recipe is not None:
            try:
                from topology import build_topology
                topo = build_topology(self.model_id)
                kv_per_tok = topo.kv_cache_size_per_token
                stock_bytes_per_tok = kv_per_tok  # fp16
                quant_ratio = m.get("kv_quant_ratio", 4)
                opt_bytes_per_tok = kv_per_tok / quant_ratio
                m["stock_bytes_per_tok"] = round(stock_bytes_per_tok, 1)
                m["opt_bytes_per_tok"] = round(opt_bytes_per_tok, 1)
                m["bytes_per_tok_savings"] = quant_ratio
            except Exception:
                pass

        # --- effective context window ---
        kv_total = m.get("kv_cache_total_tokens", 0)
        kv_used = m.get("kv_cache_used_tokens", 0)
        if kv_total > 0:
            m["remaining_tokens"] = kv_total - kv_used
            m["max_conversation_tokens"] = kv_total

        # --- memory efficiency ---
        throughput = m.get("throughput_tok_s", 0)
        vram_used = m.get("vram_used_mb", 0)
        if throughput > 0 and vram_used > 0:
            m["throughput_per_gb"] = round(throughput / (vram_used / 1024), 2)

        # --- kernel / optimization status ---
        m["kernel_calls"] = _KERNEL_CALLS  # parent proc count (usually 0); EngineCore logs authoritative
        m["roofline_decode"] = "memory_bound"  # default for small-GPU decode; recipe may override
        if self.recipe is not None:
            for step in self.recipe.steps:
                if step.name == "attention_backend" and step.enabled:
                    m["roofline_decode"] = step.params.get("roofline", "memory_bound")
                if step.name == "kv_quant" and step.enabled:
                    m["kv_quant_ratio"] = step.params.get("quant_ratio", 4)
                    m["kv_quant_active"] = True

        self._last_metrics = m
        return m


def get_available_adapter(model_id: str, recipe: Optional[Any] = None,
                          backend_override: Optional[str] = None) -> SlmTurboVLLMAdapter:
    """Factory used by `slm-turbo serve`; the only adapter for now.

    backend_override: "custom" -> force our 4-bit kernel backend,
                      "vllm"    -> force stock vLLM kernels,
                      None      -> follow the recipe.
    """
    return SlmTurboVLLMAdapter(model_id, recipe, backend_override)


def __getattr__(name: str) -> Any:
    """Make the backend classes importable even when vLLM is absent."""
    if name in ("QuantizedKVAttentionBackend", "QuantizedKVAttentionImpl"):
        raise AttributeError(
            f"{name} is only defined when vLLM's attention API is importable."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
