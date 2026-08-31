# =============================================================================
#  kernels/kv_dequant.py
#
#  Fused 4-bit KV-quantized attention kernels — CUDA.
#
#  The device code lives right here in the kernels package:
#
#      kernels/kv_dequant_kernels.cuh   # the __global__ kernels (pure CUDA)
#      kernels/kv_dequant_cuda.cu       # torch bindings + launch wrappers
#
#  This file is the Python API for all kernels:
#
#      kv_dequant_decode_attention(...)                -> [H, 1, D] fp32
#      kv_dequant_prefill_attention(...)               -> [H, M, D] fp32
#      kv_dequant_decode_per_token_attention(...)      -> [H, 1, D] fp32
#      kv_dequant_decode_per_token_paged_attention(...)-> [H, 1, D] fp32
#      kv_quantize_store(...)                          -> None (writes cache)
#
#  The first two are per-channel (used by the standalone benchmark); the
#  per-token + paged variants and the store kernel power the native packed
#  4-bit KV cache inside the vLLM adapter (adapters/vllm_adapter.py).
#
#  WHY JIT COMPILE (instead of shipping a prebuilt .so):
#  The extension is compiled with torch.utils.cpp_extension.load_inline() on
#  first import and cached under ~/.cache/torch_extensions. This mirrors JIT
#  kernel-compilation behavior: the first call is slow (nvcc build), subsequent
#  runs load the cached binary in milliseconds. It also means no packaging
#  changes and no prebuilt wheels for every torch/CUDA combination.
#
#  BUILD REQUIREMENTS: nvcc + a CUDA toolkit. If the build fails (no nvcc, no
#  torch CUDA), importing this module raises; callers such as
#  adapters/vllm_adapter.py catch import errors and degrade gracefully.
# =============================================================================
from __future__ import annotations

from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_EXT_NAME = "slm_turbo_kv_dequant_cuda"

_ext = None
_ext_error: Exception | None = None


def _load_cuda_extension():
    """Build (or load from cache) the CUDA extension. Idempotent."""
    global _ext, _ext_error
    if _ext is not None:
        return _ext
    if _ext_error is not None:
        raise _ext_error  # re-raise the original failure with its traceback

    from torch.utils.cpp_extension import load_inline

    # Pick the right gencode for the GPU actually present; without a GPU,
    # default to the project's stress-test target (GTX 1650 = sm_75, plus
    # sm_80 for newer cards). SLM_TURBO_CUDA_ARCH overrides everything.
    cuda_flags = ["-O3", "--expt-relaxed-constexpr"]
    arch = __import__("os").environ.get("SLM_TURBO_CUDA_ARCH")
    if arch:
        cuda_flags.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")
    elif torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        sm = f"{cap[0]}{cap[1]}"
        cuda_flags.append(f"-gencode=arch=compute_{sm},code=sm_{sm}")
    else:
        cuda_flags += [
            "-gencode=arch=compute_75,code=sm_75",
            "-gencode=arch=compute_80,code=sm_80",
        ]

    try:
        _ext = load_inline(
            name=_EXT_NAME,
            cpp_sources=["// CUDA bindings live in the .cu file (see kernels/)."],
            cuda_sources=[(_HERE / "kv_dequant_cuda.cu").read_text()],
            extra_include_paths=[str(_HERE)],
            extra_cuda_cflags=cuda_flags,
            verbose=False,
        )
    except Exception as e:  # pragma: no cover - depends on build environment
        _ext_error = e
        raise
    return _ext


# =============================================================================
#  DECODE attention
# =============================================================================
def kv_dequant_decode_attention(
    q, k_packed, k_scale, k_zero, v_packed, v_scale, v_zero,
    BLOCK_N=64, use_quant=True, num_kv_heads=None,
):
    """Run the fused 4-bit decode kernel. Returns [H, 1, D] fp32.

    Args:
        q:         [H, 1, D] fp16 query tokens (one per head, decode step).
        k_packed:  [KV, S, D//2] uint8 packed 4-bit keys.
        k_scale:   [KV, D] fp16 per-channel key scales.
        k_zero:    [KV, D] fp16 per-channel key zero points.
        v_packed:  [KV, S, D//2] uint8 packed 4-bit values.
        v_scale:   [KV, D] fp16 per-channel value scales.
        v_zero:    [KV, D] fp16 per-channel value zero points.
        BLOCK_N:   KV chunk size (32 or 64) — the kernel loops over S in these.
        use_quant: True -> dequantize (nibble - z) * s; False -> raw nibbles.
        num_kv_heads: KV count if q uses GQA; defaults to k_packed.shape[0].
    """
    H, _, D = q.shape
    KV, S, D2 = k_packed.shape
    assert D == D2 * 2, f"D {D} != 2*D2 {D2}"
    assert D in (64, 80, 128), f"head_dim must be 64/80/128, got {D}"
    assert BLOCK_N in (32, 64), f"BLOCK_N must be 32 or 64, got {BLOCK_N}"
    num_kv_heads = KV if num_kv_heads is None else num_kv_heads
    assert H % num_kv_heads == 0, f"H {H} not divisible by num_kv_heads {num_kv_heads}"
    assert q.shape == (H, 1, D), f"q {q.shape} != {(H, 1, D)}"

    ext = _load_cuda_extension()
    return ext.kv_dequant_decode_attention_cuda(
        q, k_packed, k_scale, k_zero, v_packed, v_scale, v_zero,
        BLOCK_N, use_quant, num_kv_heads,
    )


# =============================================================================
#  PREFILL attention
# =============================================================================
def kv_dequant_prefill_attention(
    q, k_packed, k_scale, k_zero, v_packed, v_scale, v_zero,
    BLOCK_M=None, BLOCK_N=None, causal=True, use_quant=True, num_kv_heads=None,
):
    """Run the fused 4-bit prefill kernel. Returns [H, M, D] fp32.

    Args:
        q:         [H, M, D] fp16 query tokens.
        k_packed:  [KV, S, D//2] uint8 packed 4-bit keys.
        k_scale:   [KV, D] fp16 per-channel key scales.
        k_zero:    [KV, D] fp16 per-channel key zero points.
        v_packed:  [KV, S, D//2] uint8 packed 4-bit values.
        v_scale:   [KV, D] fp16 per-channel value scales.
        v_zero:    [KV, D] fp16 per-channel value zero points.
        BLOCK_M:   query rows per block (16/32). Default: 32 for D=64, else 16
                   (larger tiles exceed the 48 KB shared-memory budget).
        BLOCK_N:   KV chunk size (32/64). Default: 64 for D=64, else 32.
        causal:    mask out future KV tokens (True for autoregressive prefill).
        use_quant: True -> dequantize; False -> raw nibbles.
        num_kv_heads: KV count if q uses GQA; defaults to k_packed.shape[0].
    """
    H, M, D = q.shape
    KV, S, D2 = k_packed.shape
    assert D == D2 * 2, f"D {D} != 2*D2 {D2}"
    assert D in (64, 80, 128), f"head_dim must be 64/80/128, got {D}"
    assert q.device == k_packed.device
    num_kv_heads = KV if num_kv_heads is None else num_kv_heads
    assert H % num_kv_heads == 0, f"H {H} not divisible by num_kv_heads {num_kv_heads}"

    # Shared-memory budget limits how big the tiles can be for D >= 80, and
    # measured on the GTX 1650 the sweet spot is a tall-ish (16, 64) tile for
    # D=64 and (16, 32) for larger head_dims.
    if BLOCK_M is None:
        BLOCK_M = 16
    if BLOCK_N is None:
        BLOCK_N = 64 if D == 64 else 32
    assert BLOCK_M in (16, 32), f"BLOCK_M must be 16 or 32, got {BLOCK_M}"
    assert BLOCK_N in (32, 64), f"BLOCK_N must be 32 or 64, got {BLOCK_N}"
    if D != 64:
        assert (BLOCK_M, BLOCK_N) == (16, 32), (
            f"head_dim {D}: only (BLOCK_M=16, BLOCK_N=32) fits the "
            f"48 KB shared budget; got ({BLOCK_M}, {BLOCK_N})"
        )

    ext = _load_cuda_extension()
    return ext.kv_dequant_prefill_attention_cuda(
        q, k_packed, k_scale, k_zero, v_packed, v_scale, v_zero,
        BLOCK_M, BLOCK_N, causal, use_quant, num_kv_heads,
    )


# =============================================================================
#  PER-TOKEN DECODE attention  (vLLM packed-cache integration)
# =============================================================================
#  Variant of the decode kernel that uses per-token (min, step) rulers instead
#  of per-channel scales. This is what the vLLM adapter stores in its packed
#  4-bit cache: each token's K/V is quantized with its OWN min/max at write
#  time (do_kv_cache_update), so there is no ruler drift and no need to
#  re-quantize on read.
# =============================================================================
def kv_dequant_decode_per_token_attention(
    q, k_packed, k_min, k_step, v_packed, v_min, v_step,
    BLOCK_N=64, num_kv_heads=None,
):
    """Run the per-token decode kernel. Returns [H, 1, D] fp32.

    Args:
        q:         [H, 1, D] fp16 query tokens.
        k_packed:  [KV, S, D//2] uint8 packed 4-bit keys.
        k_min:     [KV, S] fp16 per-token key min.
        k_step:    [KV, S] fp16 per-token key step ((max-min)/15).
        v_packed:  [KV, S, D//2] uint8 packed 4-bit values.
        v_min:     [KV, S] fp16 per-token value min.
        v_step:    [KV, S] fp16 per-token value step.
        BLOCK_N:   KV chunk size (32 or 64).
        num_kv_heads: KV count if q uses GQA; defaults to k_packed.shape[0].
    """
    H, _, D = q.shape
    KV, S, D2 = k_packed.shape
    assert D == D2 * 2, f"D {D} != 2*D2 {D2}"
    assert D in (64, 80, 128), f"head_dim must be 64/80/128, got {D}"
    assert BLOCK_N in (32, 64), f"BLOCK_N must be 32 or 64, got {BLOCK_N}"
    num_kv_heads = KV if num_kv_heads is None else num_kv_heads
    assert H % num_kv_heads == 0, f"H {H} not divisible by num_kv_heads {num_kv_heads}"

    ext = _load_cuda_extension()
    return ext.kv_dequant_decode_per_token_attention_cuda(
        q, k_packed, k_min, k_step, v_packed, v_min, v_step, BLOCK_N, num_kv_heads,
    )


def kv_quantize_store(key, value, kv_cache, slot_mapping, block_size):
    """Quantize-on-write: pack fp16 K/V rows into the 4-bit cache slots.

    Args:
        key:         [N, KV, D] fp16 keys (N tokens in the write batch).
        value:       [N, KV, D] fp16 values.
        kv_cache:    [num_blocks, block_size, KV, D+8] uint8 packed cache.
        slot_mapping: [N] int64 — flat slot per token (block*block_size + pos).
        block_size:  vLLM's KV block size.
    """
    assert key.dim() == 3, f"key must be [N, KV, D], got {key.shape}"
    assert kv_cache.is_contiguous(), "kv_cache must be contiguous"
    ext = _load_cuda_extension()
    ext.kv_quantize_store_cuda(key, value, kv_cache, slot_mapping, block_size)


def kv_dequant_decode_per_token_paged_attention(
    q, kv_cache, block_table, seq_len, block_size, num_kv_heads,
):
    """Paged per-token decode: reads the packed 4-bit cache directly via the
    request's block_table (no host-side gather). Returns [H, 1, D] fp32.

    Args:
        q:          [H, 1, D] fp16 query tokens.
        kv_cache:   [num_blocks, block_size, KV, D+8] uint8 packed cache.
        block_table: [max_blocks] int64 — this request's physical block ids.
        seq_len:    number of KV tokens for this request.
        block_size: vLLM's KV block size.
        num_kv_heads: number of KV heads (KV).
    """
    H, _, D = q.shape
    assert D in (64, 80, 128), f"head_dim must be 64/80/128, got {D}"
    assert H % num_kv_heads == 0, f"H {H} not divisible by num_kv_heads {num_kv_heads}"
    assert kv_cache.size(-1) == D + 8, f"cache slot must be D+8, got {kv_cache.size(-1)}"
    ext = _load_cuda_extension()
    return ext.kv_dequant_decode_per_token_paged_attention_cuda(
        q, kv_cache, block_table, seq_len, block_size, num_kv_heads,
    )
