# a byte can hold 8 bits means 4, 4 two separte channnel;s thats why even and odd jkust anming 
import torch
import triton
import triton.language as tl


@triton.jit
def kv_dequant_decode_kernel(
    q_ptr,            # query tokens 
    k_packed_ptr,     # [H,S,D//2]as 2 ,4 bits packed
    k_scale_ptr,      # fp16 per channel
    k_zero_ptr,       # fp16 per channel
    v_packed_ptr,     # 2 , 4 channels packed 
    v_scale_ptr,      # fp16
    v_zero_ptr,       # fp16
    out_ptr,          # [H, 1, D] fp32 output
    S,                # no of kv tokens
    D: tl.constexpr,           # head_dim
    BLOCK_D: tl.constexpr,     # padded head_dim 
    BLOCK_D2: tl.constexpr,    # padded packed dimension (BLOCK_D // 2)
    BLOCK_N: tl.constexpr,     # KV chunk size like 64/128
    USE_QUANT: tl.constexpr,   # bool: False -> plain attention
    GROUPS: tl.constexpr,      # query_heads // kv_heads (1 = MHA, >1 = GQA/MQA)
):
    head = tl.program_id(0)
    kv_head = head // GROUPS   # GQA: query head maps to its group's KV head

    d_idx = tl.arange(0, BLOCK_D)
    d2_idx = tl.arange(0, BLOCK_D2)

    # Load the query 
    q = tl.load(q_ptr + head * D + d_idx, mask=d_idx < D, other=0.0)
    q = q.to(tl.float32)[None, :]               # [1, BLOCK_D]

    # Load this head's rulers 
    if USE_QUANT:
        d2_mask = d2_idx < (D // 2)
        s_k_even = tl.load(k_scale_ptr + kv_head * D + d2_idx * 2,     mask=d2_mask, other=0.0).to(tl.float32)
        z_k_even = tl.load(k_zero_ptr + kv_head * D + d2_idx * 2,      mask=d2_mask, other=0.0).to(tl.float32)
        s_k_odd = tl.load(k_scale_ptr + kv_head * D + d2_idx * 2 + 1,  mask=d2_mask, other=0.0).to(tl.float32)
        z_k_odd = tl.load(k_zero_ptr + kv_head * D + d2_idx * 2 + 1,   mask=d2_mask, other=0.0).to(tl.float32)
        s_v_even = tl.load(v_scale_ptr + kv_head * D + d2_idx * 2,     mask=d2_mask, other=0.0).to(tl.float32)
        z_v_even = tl.load(v_zero_ptr + kv_head * D + d2_idx * 2,      mask=d2_mask, other=0.0).to(tl.float32)
        s_v_odd = tl.load(v_scale_ptr + kv_head * D + d2_idx * 2 + 1,  mask=d2_mask, other=0.0).to(tl.float32)
        z_v_odd = tl.load(v_zero_ptr + kv_head * D + d2_idx * 2 + 1,   mask=d2_mask, other=0.0).to(tl.float32)
    else:
        ones = tl.full([BLOCK_D2], 1.0, tl.float32)
        zeros = tl.full([BLOCK_D2], 0.0, tl.float32)
        s_k_even, s_k_odd, s_v_even, s_v_odd = ones, ones, ones, ones
        z_k_even, z_k_odd, z_v_even, z_v_odd = zeros, zeros, zeros, zeros

    # Online softmax state
    acc = tl.zeros([1, BLOCK_D], dtype=tl.float32)
    m = tl.full([1, 1], float("-inf"), tl.float32)
    l = tl.zeros([1, 1], dtype=tl.float32)

    kv_base = kv_head * S * (D // 2)             # this kv-head's cache offset
    scale = 1.0 / (D ** 0.5)

    # Loop over KV history in chunks of BLOCK_N 
    n_idx = tl.arange(0, BLOCK_N)
    for start in range(0, S, BLOCK_N):
        n_offs = start + n_idx
        n_mask = n_offs < S

        # 5. Load + unpack + dequant K chunk -> [BLOCK_N, BLOCK_D] 
        k_bytes = tl.load(
            k_packed_ptr + kv_base + n_offs[:, None] * (D // 2) + d2_idx[None, :],
            mask=n_mask[:, None] & (d2_idx[None, :] < (D // 2)),
            other=0,
        )                                       # [BLOCK_N, BLOCK_D2] uint8
        k_even = (k_bytes & 0x0F).to(tl.float32)
        k_odd = ((k_bytes >> 4) & 0x0F).to(tl.float32)
        k_even = (k_even - z_k_even[None, :]) * s_k_even[None, :]
        k_odd = (k_odd - z_k_odd[None, :]) * s_k_odd[None, :]
        k = tl.reshape(tl.join(k_even, k_odd), (BLOCK_N, BLOCK_D))
        # k[n, 2j] = k_even[n, j]  (low nibble)   k[n, 2j+1] = k_odd[n, j]  (high nibble)

        # Scores + online softmax update 
        scores = tl.dot(q, tl.trans(k)) * scale  # [1, BLOCK_N]
        # Mask out-of-range (padding) tokens: -inf -> exp -> 0, no softmax vote
        scores = tl.where(n_mask[None, :], scores, float("-inf"))
        chunk_max = tl.max(scores, axis=1)       # [1, 1]
        new_m = tl.maximum(m, chunk_max)
        p = tl.exp(scores - new_m)               # [1, BLOCK_N]
        rescale = tl.exp(m - new_m)              # shrink old accumulation to new scale
        l = l * rescale + tl.sum(p, axis=1)
        m = new_m

        # Load + unpack + dequant V chunk 
        v_bytes = tl.load(
            v_packed_ptr + kv_base + n_offs[:, None] * (D // 2) + d2_idx[None, :],
            mask=n_mask[:, None] & (d2_idx[None, :] < (D // 2)),
            other=0,
        )                                       # [BLOCK_N, BLOCK_D2] uint8
        v_even = (v_bytes & 0x0F).to(tl.float32)
        v_odd = ((v_bytes >> 4) & 0x0F).to(tl.float32)
        v_even = (v_even - z_v_even[None, :]) * s_v_even[None, :]
        v_odd = (v_odd - z_v_odd[None, :]) * s_v_odd[None, :]
        v = tl.reshape(tl.join(v_even, v_odd), (BLOCK_N, BLOCK_D))

        # Accumulate (rescale old acc to new max scale, like l above)
        acc = acc * rescale + tl.dot(p, v)       # [1, BLOCK_D]

    # Normalize and store 
    out = acc / l
    out1d = tl.reshape(out, (BLOCK_D,))
    tl.store(out_ptr + head * D + d_idx, out1d, mask=d_idx < D)


def kv_dequant_decode_attention(
    q, k_packed, k_scale, k_zero, v_packed, v_scale, v_zero,
    BLOCK_N=64, use_quant=True, num_kv_heads=None,
):
    """Launch the decode kernel. Returns [H, 1, D] fp32. GQA: k/v have num_kv_heads rows."""
    H, _, D = q.shape
    KV, S, D2 = k_packed.shape
    assert D == D2 * 2, f"D {D} != 2*D2 {D2}"
    num_kv_heads = KV if num_kv_heads is None else num_kv_heads
    assert H % num_kv_heads == 0, f"H {H} not divisible by num_kv_heads {num_kv_heads}"
    groups = H // num_kv_heads
    BLOCK_D = triton.next_power_of_2(D)
    assert q.shape == (H, 1, D), f"q {q.shape} != {(H, 1, D)}"
    out = torch.zeros((H, 1, D), dtype=torch.float32, device=q.device)
    grid = (H,)
    kv_dequant_decode_kernel[grid](
        q, k_packed, k_scale, k_zero, v_packed, v_scale, v_zero, out,
        S, D, BLOCK_D, BLOCK_D // 2, BLOCK_N, use_quant, groups,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# PREFILL kernel: many query tokens at once [H, M, D], causal mask.
# Same online-softmax math as the decode kernel, but each program handles a
# (head, BLOCK_M query block) pair and each query row attends to j <= i.
# ---------------------------------------------------------------------------

@triton.jit
def kv_dequant_prefill_kernel(
    q_ptr,            # [H, M, D] fp16 queries
    k_packed_ptr,     # [H, S, D//2] packed 4-bit keys
    k_scale_ptr,      # fp16 per channel
    k_zero_ptr,       # fp16 per channel
    v_packed_ptr,     # [H, S, D//2] packed 4-bit values
    v_scale_ptr,      # fp16 per channel
    v_zero_ptr,       # fp16 per channel
    out_ptr,          # [H, M, D] fp32 output
    M,                # number of query tokens
    S,                # number of KV tokens
    D: tl.constexpr,           # head_dim
    BLOCK_D: tl.constexpr,     # padded head_dim
    BLOCK_D2: tl.constexpr,    # padded packed dim (BLOCK_D // 2)
    BLOCK_M: tl.constexpr,     # query tokens per program
    BLOCK_N: tl.constexpr,     # KV chunk size
    CAUSAL: tl.constexpr,      # bool: mask out future tokens
    USE_QUANT: tl.constexpr,   # bool: False -> plain attention
    GROUPS: tl.constexpr,      # query_heads // kv_heads (1 = MHA, >1 = GQA/MQA)
):
    pid = tl.program_id(0)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    head = pid // num_m_blocks
    kv_head = head // GROUPS   # GQA: query head maps to its group's KV head
    m_block = pid % num_m_blocks
    m_start = m_block * BLOCK_M

    m_idx = m_start + tl.arange(0, BLOCK_M)  # [BLOCK_M] query rows
    d_idx = tl.arange(0, BLOCK_D)
    d2_idx = tl.arange(0, BLOCK_D2)
    m_mask = m_idx < M

    # Load this program's query block -> [BLOCK_M, BLOCK_D]
    q = tl.load(
        q_ptr + head * M * D + m_idx[:, None] * D + d_idx[None, :],
        mask=m_mask[:, None] & (d_idx[None, :] < D),
        other=0.0,
    )
    q = q.to(tl.float32)

    # Per-channel rulers (same as decode kernel)
    if USE_QUANT:
        d2_mask = d2_idx < (D // 2)
        s_k_even = tl.load(k_scale_ptr + kv_head * D + d2_idx * 2,     mask=d2_mask, other=0.0).to(tl.float32)
        z_k_even = tl.load(k_zero_ptr + kv_head * D + d2_idx * 2,      mask=d2_mask, other=0.0).to(tl.float32)
        s_k_odd = tl.load(k_scale_ptr + kv_head * D + d2_idx * 2 + 1,  mask=d2_mask, other=0.0).to(tl.float32)
        z_k_odd = tl.load(k_zero_ptr + kv_head * D + d2_idx * 2 + 1,   mask=d2_mask, other=0.0).to(tl.float32)
        s_v_even = tl.load(v_scale_ptr + kv_head * D + d2_idx * 2,     mask=d2_mask, other=0.0).to(tl.float32)
        z_v_even = tl.load(v_zero_ptr + kv_head * D + d2_idx * 2,      mask=d2_mask, other=0.0).to(tl.float32)
        s_v_odd = tl.load(v_scale_ptr + kv_head * D + d2_idx * 2 + 1,  mask=d2_mask, other=0.0).to(tl.float32)
        z_v_odd = tl.load(v_zero_ptr + kv_head * D + d2_idx * 2 + 1,   mask=d2_mask, other=0.0).to(tl.float32)
    else:
        ones = tl.full([BLOCK_D2], 1.0, tl.float32)
        zeros = tl.full([BLOCK_D2], 0.0, tl.float32)
        s_k_even, s_k_odd, s_v_even, s_v_odd = ones, ones, ones, ones
        z_k_even, z_k_odd, z_v_even, z_v_odd = zeros, zeros, zeros, zeros

    # Online softmax state, one row per query token
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    m = tl.full([BLOCK_M, 1], float("-inf"), tl.float32)
    l = tl.zeros([BLOCK_M, 1], dtype=tl.float32)

    kv_base = kv_head * S * (D // 2)   # this kv-head's cache offset
    scale = 1.0 / (D ** 0.5)

    n_idx = tl.arange(0, BLOCK_N)
    for start in range(0, S, BLOCK_N):
        n_offs = start + n_idx
        n_mask = n_offs < S

        # valid cell = real row & real column & (causal: query >= kv index)
        valid = m_mask[:, None] & n_mask[None, :]
        if CAUSAL:
            valid = valid & (m_idx[:, None] >= n_offs[None, :])

        # Load + unpack + dequant K chunk -> [BLOCK_N, BLOCK_D]
        k_bytes = tl.load(
            k_packed_ptr + kv_base + n_offs[:, None] * (D // 2) + d2_idx[None, :],
            mask=n_mask[:, None] & (d2_idx[None, :] < (D // 2)),
            other=0,
        )
        k_even = (k_bytes & 0x0F).to(tl.float32)
        k_odd = ((k_bytes >> 4) & 0x0F).to(tl.float32)
        k_even = (k_even - z_k_even[None, :]) * s_k_even[None, :]
        k_odd = (k_odd - z_k_odd[None, :]) * s_k_odd[None, :]
        k = tl.reshape(tl.join(k_even, k_odd), (BLOCK_N, BLOCK_D))

        # Scores -> apply mask (-inf on forbidden/phantom cells)
        scores = tl.dot(q, tl.trans(k)) * scale   # [BLOCK_M, BLOCK_N]
        scores = tl.where(valid, scores, float("-inf"))

        chunk_max = tl.max(scores, axis=1, keep_dims=True)  # [BLOCK_M, 1]
        new_m = tl.maximum(m, chunk_max)
        p = tl.exp(scores - new_m)                # [BLOCK_M, BLOCK_N]
        rescale = tl.exp(m - new_m)               # [BLOCK_M, 1]
        l = l * rescale + tl.sum(p, axis=1, keep_dims=True)
        m = new_m

        # Load + unpack + dequant V chunk
        v_bytes = tl.load(
            v_packed_ptr + kv_base + n_offs[:, None] * (D // 2) + d2_idx[None, :],
            mask=n_mask[:, None] & (d2_idx[None, :] < (D // 2)),
            other=0,
        )
        v_even = (v_bytes & 0x0F).to(tl.float32)
        v_odd = ((v_bytes >> 4) & 0x0F).to(tl.float32)
        v_even = (v_even - z_v_even[None, :]) * s_v_even[None, :]
        v_odd = (v_odd - z_v_odd[None, :]) * s_v_odd[None, :]
        v = tl.reshape(tl.join(v_even, v_odd), (BLOCK_N, BLOCK_D))

        acc = acc * rescale + tl.dot(p, v)        # [BLOCK_M, BLOCK_D]

    # Normalize and store [BLOCK_M, BLOCK_D]
    out = acc / l
    tl.store(
        out_ptr + head * M * D + m_idx[:, None] * D + d_idx[None, :],
        out,
        mask=m_mask[:, None] & (d_idx[None, :] < D),
    )


def kv_dequant_prefill_attention(
    q, k_packed, k_scale, k_zero, v_packed, v_scale, v_zero,
    BLOCK_M=None, BLOCK_N=64, causal=True, use_quant=True, num_kv_heads=None,
):
    """Launch the prefill kernel. Returns [H, M, D] fp32. GQA: k/v have num_kv_heads rows."""
    H, M, D = q.shape
    KV, S, D2 = k_packed.shape
    assert D == D2 * 2, f"D {D} != 2*D2 {D2}"
    assert q.device == k_packed.device
    num_kv_heads = KV if num_kv_heads is None else num_kv_heads
    assert H % num_kv_heads == 0, f"H {H} not divisible by num_kv_heads {num_kv_heads}"
    groups = H // num_kv_heads
    BLOCK_D = triton.next_power_of_2(D)
    # shared memory: [BLOCK_M, BLOCK_D] tiles are fp32; keep BLOCK_M small for big D.
    # GTX 1650 (14 SMs): smaller tiles -> more CTAs -> better SM occupancy (2x faster than BLOCK_M=64).
    if BLOCK_M is None:
        BLOCK_M = 32
    out = torch.zeros((H, M, D), dtype=torch.float32, device=q.device)
    num_m_blocks = triton.cdiv(M, BLOCK_M)
    grid = (H * num_m_blocks,)
    kv_dequant_prefill_kernel[grid](
        q, k_packed, k_scale, k_zero, v_packed, v_scale, v_zero, out,
        M, S, D, BLOCK_D, BLOCK_D // 2, BLOCK_M, BLOCK_N, causal, use_quant, groups,
        num_warps=4, num_stages=1,
    )
    return out