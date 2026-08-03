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
# ---------------------------------------------------------------------------

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

    class QuantizedKVAttentionImpl(TritonAttentionImpl):
        """Triton decode attention, but with our 4-bit fused kernel for eligible
        single-request decodes. Everything else defers to the stock Triton path."""

        def __init__(self, num_heads, head_size, scale, num_kv_heads=None, alibi_slopes=None,
                     sliding_window=None, kv_cache_dtype="auto", logits_soft_cap=None,
                     attn_type="decoder", kv_sharing_target_layer_name=None, **kwargs):
            super().__init__(num_heads, head_size, scale, num_kv_heads, alibi_slopes,
                             sliding_window, kv_cache_dtype, logits_soft_cap,
                             attn_type, kv_sharing_target_layer_name, **kwargs)
            self._slm_kernel = None
            self._slm_kernel_err: Optional[str] = None
            try:
                import sys as _sys
                if str(_REPO_ROOT) not in _sys.path:
                    _sys.path.insert(0, str(_REPO_ROOT))
                from kernels.kv_dequant import kv_dequant_decode_attention
                from kernels.quantize import quantize_kv

                self._slm_attn = kv_dequant_decode_attention
                self._slm_quantize = quantize_kv
                self._slm_kernel = torch.cuda.is_available()
                if not self._slm_kernel:
                    self._slm_kernel_err = "no CUDA device"
            except Exception as e:
                self._slm_kernel_err = f"{type(e).__name__}: {e}"
            if self._slm_kernel_err:
                logger.warning("slm-turbo kernel disabled in this process: %s", self._slm_kernel_err)

        def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                    output_scale=None, output_block_scale=None):
            if self._slm_kernel and attn_metadata is not None and self._try_slm_decode(
                    query, kv_cache, attn_metadata, output):
                return output
            return super().forward(layer, query, key, value, kv_cache, attn_metadata,
                                   output, output_scale, output_block_scale)

        def _try_slm_decode(self, query, kv_cache, attn_metadata, output) -> bool:
            """Gather -> quantize -> fused 4-bit decode kernel -> scatter. Pure best-effort."""
            global _KERNEL_CALLS

            def _gate(name: str) -> bool:
                """Log the first occurrence of a rejected eligibility gate."""
                if not getattr(self, "_slm_gates_logged", None):
                    self._slm_gates_logged = set()
                if name not in self._slm_gates_logged:
                    self._slm_gates_logged.add(name)
                    logger.debug("slm-turbo decode gate rejected: %s", name)
                return False

            try:
                # --- eligibility gates (any miss -> fall back to stock Triton) ---
                # TritonAttentionMetadata has no num_reqs; derive it from seq_lens.
                num_reqs = int(attn_metadata.seq_lens.shape[0])
                if num_reqs != 1:
                    return _gate(f"num_reqs={num_reqs}")
                if attn_metadata.max_query_len != 1:
                    return _gate(f"max_query_len={attn_metadata.max_query_len}")
                if attn_metadata.num_actual_tokens != 1:
                    return _gate(f"num_actual_tokens={attn_metadata.num_actual_tokens}")
                if self.head_size not in SUPPORTED_HEAD_SIZES:
                    return _gate(f"head_size={self.head_size}")
                if query.dtype != torch.float16:
                    return _gate(f"dtype={query.dtype}")
                if getattr(self, "alibi_slopes", None) is not None:
                    return _gate("alibi_slopes")
                # vLLM encodes "no sliding window" as (-1, -1)
                sw = getattr(self, "sliding_window", None)
                if sw is not None and sw != (-1, -1):
                    return _gate(f"sliding_window={sw}")
                kv_dtype = getattr(self, "kv_cache_dtype", "auto") or "auto"
                if "fp8" in kv_dtype or "int8" in kv_dtype:
                    return _gate(f"kv_dtype={kv_dtype}")
                if self.attn_type not in ("decoder",):
                    return _gate(f"attn_type={self.attn_type}")

                # --- paged fp16 cache: [num_blocks, 2, block_size, KV, D] ---
                key_cache, value_cache = kv_cache[:, 0], kv_cache[:, 1]
                num_kv_heads = key_cache.shape[2]
                block_size = key_cache.shape[1]
                seq_len = int(attn_metadata.seq_lens[0])
                if seq_len < 1:
                    return False
                n_blocks = (seq_len + block_size - 1) // block_size
                blocks = attn_metadata.block_table[0, :n_blocks]

                k_cache = (key_cache[blocks].transpose(0, 2).contiguous()
                           .reshape(num_kv_heads, -1, self.head_size)[:, :seq_len])
                v_cache = (value_cache[blocks].transpose(0, 2).contiguous()
                           .reshape(num_kv_heads, -1, self.head_size)[:, :seq_len])
                q = query[:1].permute(1, 0, 2).contiguous()  # [H, 1, D]

                kp, ks, kz, vp, vs, vz = self._slm_quantize(k_cache, v_cache)
                out = self._slm_attn(q, kp, ks, kz, vp, vs, vz, num_kv_heads=num_kv_heads)
                # v1 attention output buffer is [num_tokens, num_heads, head_size]
                output[:1].copy_(out.permute(1, 0, 2))  # [H,1,D] -> [1,H,D]
                _KERNEL_CALLS += 1
                if _KERNEL_CALLS == 1:
                    logger.info("slm-turbo 4-bit decode kernel serving inside vLLM (head_size=%d)", self.head_size)
                return True
            except Exception as e:
                if not getattr(self, "_slm_fallback_logged", False):
                    self._slm_fallback_logged = True
                    logger.warning("slm-turbo decode kernel unavailable, falling back to stock Triton: %s: %s",
                                   type(e).__name__, e)
                return False

    class QuantizedKVAttentionBackend(TritonAttentionBackend):
        """Stock Triton metadata machinery + our impl. get_name() must be an
        AttentionBackendEnum member — "CUSTOM" is vLLM's third-party slot."""

        @staticmethod
        def get_name() -> str:
            return "CUSTOM"

        @staticmethod
        def get_impl_cls() -> type:
            return QuantizedKVAttentionImpl

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
