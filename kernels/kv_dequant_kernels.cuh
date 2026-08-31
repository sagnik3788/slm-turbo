// =============================================================================
//  kernels/kv_dequant_kernels.cuh
//
//  All slm-turbo CUDA device kernels in one header (pure CUDA, no torch):
//    - kv_dequant_decode_kernel          per-channel decode (standalone bench)
//    - kv_dequant_prefill_kernel         per-channel causal prefill (bench)
//    - kv_dequant_decode_per_token_kernel       per-token decode (vLLM cache)
//    - kv_dequant_decode_per_token_paged_kernel paged per-token decode (vLLM)
//    - kv_quantize_store_kernel          quantize-on-write store (vLLM)
//
//  ---------------------------------------------------------------------------
//  WHAT THE KERNELS DO
//  ---------------------------------------------------------------------------
//  The KV cache is stored PACKED as 4-bit values: two numbers share one byte.
//  A byte 0bHHHHLLLL holds two 4-bit values:
//      dim 2j (even)  -> low  nibble, extracted with  byte & 0x0F
//      dim 2j+1 (odd) -> high nibble, extracted with  (byte >> 4)
//  Two dequant formats are used:
//    - per-channel (decode/prefill, standalone): each (kv_head, dim) channel
//      has its own scale `s` and zero point `z` (fp16, host-computed):
//          value = (nibble - z) * s
//    - per-token (vLLM packed cache): each token has its own (min, step):
//          value = nibble * step + min        step = (max-min)/15
//      This is O(1) to store incrementally and immune to ruler drift.
//
//  The kernels FUSE everything into one pass:
//      unpack nibbles -> dequantize -> attention -> online softmax -> output
//  No fp16 (or fp32) KV is ever materialized in global memory; the 4-bit cache
//  is read directly, giving the 4x KV-memory / bandwidth win.
//
//  All kernels implement FLASH-ATTENTION style online softmax. Normal softmax
//  needs the max over ALL key tokens before you can compute exp(score - max).
//  Online softmax avoids two passes over the KV history by keeping running
//  statistics (m = running max, l = running sum of exp, acc = weighted output)
//  and RESCALING the old statistics whenever a new chunk has a bigger max:
//      new_m     = max(m, chunk_max)
//      rescale   = exp(m - new_m)          // shrink old stats to the new scale
//      p         = exp(score - new_m)
//      l         = l * rescale + sum(p)
//      acc       = acc * rescale + p @ V
//  At the end:  out = acc / l.
//
//  ---------------------------------------------------------------------------
//  CUDA MENTAL MODEL (one block = one "program")
//  ---------------------------------------------------------------------------
//  Each kernel maps work to blocks and threads explicitly:
//      - one BLOCK == one attention head (decode) or (head, tile) (prefill)
//      - 128 or 256 threads per block
//      - shared memory is the explicit scratch pad that stages the
//        dequantized K tile (and packed V bytes) so every thread can read any
//        (n, d) element cheaply and global memory is read coalesced.
//
//  BANK-CONFLICT PADDING: shared memory is split into 32 banks; if two threads
//  in a warp touch the same bank in the same cycle the access is serialized
//  ("bank conflict"). A tile row of BLOCK_D floats = 128 floats ≡ 0 (mod 32)
//  banks, so k_s[n][d] and k_s[n+1][d] would hit the SAME bank. Padding each
//  row to BLOCK_D + 1 floats shifts consecutive rows by one bank, so row n
//  lives in bank (n + d) % 32 -> conflict-free strided access.
//  ---------------------------------------------------------------------------
//  SUPPORTED SIZES (validated on sm_75, GTX 1650)
//      head_dim D          : 64, 80, 128     (must be even; D/2 packed dims)
//      KV chunk BLOCK_N    : 32, 64
//      prefill BLOCK_M     : 16, 32          (query rows per block)
//  Shared-memory budget: static shared memory is capped at 48 KB per block, so
//  not every (D, BLOCK_M, BLOCK_N) combination fits. The torch bindings in
//  kv_dequant_cuda.cu check the budget and reject over-budget combos with a
//  clear error before launching.
// =============================================================================

#ifndef SLM_TURBO_KV_DEQUANT_KERNELS_CUH
#define SLM_TURBO_KV_DEQUANT_KERNELS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <math.h>

namespace slm_turbo {

// One block = 4 warps = 128 threads.
constexpr int kDecodeThreads = 128;

// =============================================================================
//  DECODE KERNEL
//  =============================================================================
//  One block per query head (grid = (H,)).
//
//  Inputs (all row-major, contiguous):
//      q_ptr        [H, D]            fp16  query token (1 token per head)
//      k_packed_ptr [KV, S, D/2]      u8    4-bit packed keys
//      k_scale_ptr  [KV, D]           fp16  per-channel key scale
//      k_zero_ptr   [KV, D]           fp16  per-channel key zero point
//      v_packed_ptr [KV, S, D/2]      u8    4-bit packed values
//      v_scale_ptr  [KV, D]           fp16  per-channel value scale
//      v_zero_ptr   [KV, D]           fp16  per-channel value zero point
//      out_ptr      [H, D]            fp32  attention output (acc / l)
//  Runtime (non-template) args:
//      S             number of KV tokens (the whole KV history is looped over)
//      groups        H / num_kv_heads  (1 = MHA, >1 = GQA/MQA)
//      use_quant     True -> apply (nibble - z) * s ; False -> raw nibbles
//
//  Templates (compile-time constants):
//      D         actual head_dim (64/80/128)
//      BLOCK_D   next power of 2 >= D (128 for D=80,128; 64 for D=64)
//      BLOCK_N   KV chunk size (32/64) — the outer loop steps over S in these
//      THREADS   block width (128)
//
//  Thread mapping (decode is a single query row, so it is simple):
//      - step 3  (scores):     thread tid owns KV token n = tid
//      - step 5  (accumulate): thread tid owns output column d = tid
//  This works because D <= 128 == THREADS: every output column gets a thread.
// =============================================================================
template <int D, int BLOCK_D, int BLOCK_N, int THREADS = kDecodeThreads>
__global__ void kv_dequant_decode_kernel(
    const __half* __restrict__ q_ptr,
    const uint8_t* __restrict__ k_packed_ptr,
    const __half* __restrict__ k_scale_ptr,
    const __half* __restrict__ k_zero_ptr,
    const uint8_t* __restrict__ v_packed_ptr,
    const __half* __restrict__ v_scale_ptr,
    const __half* __restrict__ v_zero_ptr,
    float* __restrict__ out_ptr,
    int S, int groups, bool use_quant)
{
    constexpr int D2 = D / 2;      // packed (byte) width of one KV row

    // ---- Shared-memory tile for this block (the "scratch pad") --------------
    // q_s        [BLOCK_D]              query row, broadcast to all threads
    // k_s        [BLOCK_N][BLOCK_D + 1] dequantized K tile, bank-conflict padded
    // v_bytes    [BLOCK_N][D/2]         packed V bytes (unpacked on the fly)
    // s_k_s z_k_s s_v_s z_v_s [BLOCK_D] dequant rulers (scale/zero) per channel
    // p_s        [BLOCK_N]              softmax probabilities for this chunk
    // scratch    [THREADS]              cross-thread reduction workspace
    __shared__ float q_s[BLOCK_D];
    __shared__ float k_s[BLOCK_N][BLOCK_D + 1];
    __shared__ uint8_t v_bytes[BLOCK_N][D / 2];
    __shared__ float s_k_s[BLOCK_D], z_k_s[BLOCK_D], s_v_s[BLOCK_D], z_v_s[BLOCK_D];
    __shared__ float p_s[BLOCK_N];
    __shared__ float scratch[THREADS];

    const int tid  = threadIdx.x;  // 0..THREADS-1  (which thread am I?)
    const int head = blockIdx.x;   // which query head does this block own?
    const int kv_head = head / groups;   // GQA: a query head reads the KV row
                                         //      of its group's KV head

    // ---- 1. Stage the query row and the dequant "rulers" in shared memory ----
    if (tid < BLOCK_D) {
        float v = 0.0f;
        if (tid < D) v = __half2float(q_ptr[(size_t)head * D + tid]);
        q_s[tid] = v;
    }
    if (use_quant && tid < BLOCK_D) {
        if (tid < D) {
            s_k_s[tid] = __half2float(k_scale_ptr[(size_t)kv_head * D + tid]);
            z_k_s[tid] = __half2float(k_zero_ptr[(size_t)kv_head * D + tid]);
            s_v_s[tid] = __half2float(v_scale_ptr[(size_t)kv_head * D + tid]);
            z_v_s[tid] = __half2float(v_zero_ptr[(size_t)kv_head * D + tid]);
        } else {
            // Padded dims (BLOCK_D >= D): neutral rulers so they never matter.
            s_k_s[tid] = 1.0f; z_k_s[tid] = 0.0f;
            s_v_s[tid] = 1.0f; z_v_s[tid] = 0.0f;
        }
    }
    __syncthreads();   // q_s / rulers must be written before anyone reads them

    const size_t kv_base = (size_t)kv_head * (size_t)S * D2; // this KV head's slice
    const float  scale   = 1.0f / sqrtf((float)D);           // 1/sqrt(D)riton

    // ---- Online-softmax running state (all threads compute identical values,
    // so keeping them in registers is safe) -----------------------------------
    float m_cur = -INFINITY;   // running max of scores so far
    float l_cur = 0.0f;        // running sum of exp(score - m)
    float acc   = 0.0f;        // running output for THIS thread's column (d=tid)

    // ---- Main loop: walk the KV history in chunks of BLOCK_N -----------------
    for (int start = 0; start < S; start += BLOCK_N) {

        // ---- 2a. Load + unpack + dequant the K chunk into shared memory ----
        // Total bytes in a chunk: BLOCK_N rows * D2 bytes. Each thread loads a
        // strided slice; consecutive threads read consecutive bytes, which the
        // hardware coalesces into few memory transactions.
        for (int i = tid; i < BLOCK_N * D2; i += THREADS) {
            int n = i / D2;                 // row  (which KV token)
            int j = i % D2;                 // byte within the row
            int n_abs = start + n;          // absolute KV index
            uint8_t byte = 0;
            if (n_abs < S)                  // S may not be a multiple of BLOCK_N
                byte = k_packed_ptr[kv_base + (size_t)n_abs * D2 + j];
            float even = (float)(byte & 0x0Fu);   // dim 2j   (low nibble)
            float odd  = (float)(byte >> 4);      // dim 2j+1 (high nibble)
            if (use_quant) {
                even = (even - z_k_s[2 * j]) * s_k_s[2 * j];
                odd  = (odd  - z_k_s[2 * j + 1]) * s_k_s[2 * j + 1];
            }
            k_s[n][2 * j]     = even;       // dequantized tile in shared mem
            k_s[n][2 * j + 1] = odd;
        }

        // ---- 2b. Stage the V chunk (packed bytes; unpacked later in step 5) --
        for (int i = tid; i < BLOCK_N * D2; i += THREADS) {
            int n = i / D2, j = i % D2;
            int n_abs = start + n;
            v_bytes[n][j] = (n_abs < S)
                                ? v_packed_ptr[kv_base + (size_t)n_abs * D2 + j]
                                : (uint8_t)0;
        }
        __syncthreads();   // K/V tiles are ready for everyone

        // ---- 3. Scores: one thread per KV token -----------------------------
        // score[n] = (q . K[n]) * scale   — a length-D dot product.
        // Thread tid computes the score for token n = tid (tid < BLOCK_N).
        // (A vector x matrix product, so we write the dot out by hand.)
        float score = -INFINITY;           // masked tokens get -inf (-> p = 0)
        if (tid < BLOCK_N && (start + tid) < S) {
            float s = 0.0f;
            for (int d = 0; d < D; ++d) s += q_s[d] * k_s[tid][d];
            score = s * scale;
        }

        // ---- 3b. Block reduction: chunk_max = max over all scores ------------
        // Tree reduction over 128 threads: pairs halve each round. Every thread
        // ends up agreeing on the max in scratch[0].
        scratch[tid] = score;
        __syncthreads();
        for (int off = THREADS / 2; off > 0; off >>= 1) {
            if (tid < off) scratch[tid] = fmaxf(scratch[tid], scratch[tid + off]);
            __syncthreads();
        }
        const float chunk_max = scratch[0];
        __syncthreads();

        // ---- 4. Online-softmax update ----------------------------------------
        const float new_m   = fmaxf(m_cur, chunk_max);
        const float rescale = __expf(m_cur - new_m);  // 0 on the first chunk
        m_cur = new_m;
        l_cur *= rescale;

        // p[n] = exp(score[n] - new_m); store so step 5 can broadcast-read it.
        if (tid < BLOCK_N && (start + tid) < S) p_s[tid] = __expf(score - new_m);
        __syncthreads();

        // Block reduction #2: chunk_sum = sum of p over the chunk -> l_cur += it
        scratch[tid] = 0.0f;
        if (tid < BLOCK_N && (start + tid) < S) scratch[tid] = p_s[tid];
        __syncthreads();
        for (int off = THREADS / 2; off > 0; off >>= 1) {
            if (tid < off) scratch[tid] += scratch[tid + off];
            __syncthreads();
        }
        l_cur += scratch[0];
        __syncthreads();

        // ---- 5. Output accumulation -----------------------------------------
        // acc[d] = sum_n p[n] * V[n][d]. Thread tid owns ONE output column
        // d = tid, so it walks all n in the chunk (works because D <= THREADS).
        // V is unpacked + dequantized on the fly from the shared packed bytes.
        if (tid < D) {
            float a = acc * rescale;        // rescale the old accumulation
            for (int n = 0; n < BLOCK_N; ++n) {
                int n_abs = start + n;
                if (n_abs >= S) break;      // n_abs grows with n -> safe break
                uint8_t byte = v_bytes[n][tid >> 1];        // byte for dim tid
                float v = (tid & 1) ? (float)(byte >> 4)    // odd  dim
                                    : (float)(byte & 0x0Fu);// even dim
                if (use_quant) v = (v - z_v_s[tid]) * s_v_s[tid];
                a += p_s[n] * v;
            }
            acc = a;
        }
        __syncthreads();   // p_s / scratch must be fully consumed before reuse
    }

    // ---- 6. Epilogue: out = acc / l -----------------------------------------
    if (tid < D) out_ptr[(size_t)head * D + tid] = acc / l_cur;
}

// =============================================================================
//  PREFILL KERNEL
//  =============================================================================
//  One block per (head, BLOCK_M-query-block) pair. Grid = (H * num_m_blocks,).
//
//  Same math as decode, but BLOCK_M query tokens are processed at once and a
//  causal mask hides future key tokens. The online-softmax state becomes a
//  vector (one m, l per query row) and the output tile is [BLOCK_M, D].
//
//  Templates: D, BLOCK_D, BLOCK_M (query rows), BLOCK_N (KV chunk), THREADS.
//
//  Thread mapping (prefill is a 2D tile, so threads map to cells):
//      step 3  (scores):      thread c -> cell (r, n) with c = r*BLOCK_N + n
//      step 4  (softmax):     thread r owns query row r (r < BLOCK_M)
//      step 5  (accumulate):  thread c -> cell (r, d) with c = r*BLOCK_D + d
// =============================================================================
template <int D, int BLOCK_D, int BLOCK_M, int BLOCK_N, int THREADS = kDecodeThreads>
__global__ void kv_dequant_prefill_kernel(
    const __half* __restrict__ q_ptr,        // [H, M, D] fp16
    const uint8_t* __restrict__ k_packed_ptr,// [KV, S, D/2] u8
    const __half* __restrict__ k_scale_ptr,  // [KV, D] fp16
    const __half* __restrict__ k_zero_ptr,   // [KV, D] fp16
    const uint8_t* __restrict__ v_packed_ptr,// [KV, S, D/2] u8
    const __half* __restrict__ v_scale_ptr,  // [KV, D] fp16
    const __half* __restrict__ v_zero_ptr,   // [KV, D] fp16
    float* __restrict__ out_ptr,             // [H, M, D] fp32
    int M, int S, int groups, bool causal, bool use_quant)
{
    constexpr int D2 = D / 2;

    // ---- Shared-memory tiles ------------------------------------------------
    // q_s   [BLOCK_M][BLOCK_D + 1]   fp32 query tile (padded rows)
    // k_s   [BLOCK_N][BLOCK_D + 1]   fp32 dequantized K tile (padded rows)
    // acc_s [BLOCK_M][BLOCK_D + 1]   fp32 running output tile
    // p_s   [BLOCK_M][BLOCK_N]       fp32 scores first, then probabilities
    // m_s / l_s [BLOCK_M]            per-row online-softmax state
    // s_k_s z_k_s s_v_s z_v_s [BLOCK_D]  dequant rulers
    __shared__ float q_s[BLOCK_M][BLOCK_D + 1];
    __shared__ float k_s[BLOCK_N][BLOCK_D + 1];
    __shared__ float acc_s[BLOCK_M][BLOCK_D + 1];
    __shared__ float p_s[BLOCK_M][BLOCK_N];
    __shared__ float m_s[BLOCK_M], l_s[BLOCK_M];
    __shared__ float s_k_s[BLOCK_D], z_k_s[BLOCK_D], s_v_s[BLOCK_D], z_v_s[BLOCK_D];
    __shared__ float red_s[THREADS];   // per-thread scratch for row reductions

    const int tid  = threadIdx.x;
    const int pid  = blockIdx.x;

    // Decompose the 1D block index into (head, query-block):
    //   pid = head * num_m_blocks + m_block
    const int num_m_blocks = (M + BLOCK_M - 1) / BLOCK_M;   // ceil(M / BLOCK_M)
    const int head    = pid / num_m_blocks;
    const int kv_head = head / groups;         // GQA mapping
    const int m_block = pid % num_m_blocks;
    const int m_start = m_block * BLOCK_M;     // first query row of this block

    // ---- 1. Stage the query tile, rulers, and zero the accumulators ---------
    for (int i = tid; i < BLOCK_M * BLOCK_D; i += THREADS) {
        int r = i / BLOCK_D, d = i % BLOCK_D;  // (row, col) in the tile
        int m_idx = m_start + r;
        float v = 0.0f;
        if (m_idx < M && d < D)
            v = __half2float(q_ptr[(size_t)head * M * D + (size_t)m_idx * D + d]);
        q_s[r][d] = v;
    }
    if (use_quant) {
        for (int i = tid; i < BLOCK_D; i += THREADS) {
            if (i < D) {
                s_k_s[i] = __half2float(k_scale_ptr[(size_t)kv_head * D + i]);
                z_k_s[i] = __half2float(k_zero_ptr[(size_t)kv_head * D + i]);
                s_v_s[i] = __half2float(v_scale_ptr[(size_t)kv_head * D + i]);
                z_v_s[i] = __half2float(v_zero_ptr[(size_t)kv_head * D + i]);
            } else {
                s_k_s[i] = 1.0f; z_k_s[i] = 0.0f;
                s_v_s[i] = 1.0f; z_v_s[i] = 0.0f;
            }
        }
    }
    for (int i = tid; i < BLOCK_M * BLOCK_D; i += THREADS)
        acc_s[i / BLOCK_D][i % BLOCK_D] = 0.0f;
    if (tid < BLOCK_M) {
        m_s[tid] = -INFINITY;
        l_s[tid] = 0.0f;
    }
    __syncthreads();

    const size_t kv_base = (size_t)kv_head * (size_t)S * D2;
    const float  scale   = 1.0f / sqrtf((float)D);

    // ---- Main loop over KV chunks -------------------------------------------
    for (int start = 0; start < S; start += BLOCK_N) {

        // ---- 2. Load + unpack + dequant K chunk into shared ------------------
        for (int i = tid; i < BLOCK_N * D2; i += THREADS) {
            int n = i / D2, j = i % D2;
            int n_abs = start + n;
            uint8_t byte = 0;
            if (n_abs < S) byte = k_packed_ptr[kv_base + (size_t)n_abs * D2 + j];
            float even = (float)(byte & 0x0Fu), odd = (float)(byte >> 4);
            if (use_quant) {
                even = (even - z_k_s[2 * j]) * s_k_s[2 * j];
                odd  = (odd  - z_k_s[2 * j + 1]) * s_k_s[2 * j + 1];
            }
            k_s[n][2 * j] = even;
            k_s[n][2 * j + 1] = odd;
        }
        __syncthreads();

        // ---- 3. Scores: the [BLOCK_M, BLOCK_N] tile --------------------------
        // Each thread computes a strided subset of score cells:
        //   cell (r, n) = (q[r] . K[n]) * scale
        // Invalid cells (padding rows/tokens, or future tokens when causal)
        // are set to -inf so exp() turns them into 0 during softmax.
        for (int c = tid; c < BLOCK_M * BLOCK_N; c += THREADS) {
            int r = c / BLOCK_N, n = c % BLOCK_N;
            int m_idx = m_start + r, n_abs = start + n;
            float s = 0.0f;
            for (int d = 0; d < D; ++d) s += q_s[r][d] * k_s[n][d];
            s *= scale;
            bool valid = (m_idx < M) && (n_abs < S);
            if (causal) valid = valid && (m_idx >= n_abs);   // no attending to future
            p_s[r][n] = valid ? s : -INFINITY;
        }
        __syncthreads();

        // ---- 4. Online softmax, parallelized across TPR threads per row ------
        // The naive version gave one thread per query row, leaving most of the
        // block idle at the barrier while that thread walked the whole chunk
        // serially. Instead, split the row's work across TPR = THREADS/BLOCK_M
        // consecutive lanes (thread `tid` handles row `tid/TPR`, slice
        // `tid%TPR` of the n / d axes). The per-row reductions (chunk max,
        // chunk sum) happen in two barrier-separated passes over `red_s`.
        const int TPR = THREADS / BLOCK_M;   // threads per query row
        const int row = tid / TPR;           // which query row this thread helps
        const int part = tid % TPR;          // which slice of the row it owns
        float new_m = m_s[row];              // (re)assigned below; hoisted so
        float rescale = 1.0f;                // the later phase can read them
        if (row < BLOCK_M) {
            const int m_idx = m_start + row;
            if (m_idx < M) {
                // 4a. partial per-row chunk max over this thread's n-slice
                float part_max = -INFINITY;
                for (int n = part; n < BLOCK_N; n += TPR)
                    part_max = fmaxf(part_max, p_s[row][n]);   // p_s holds scores
                red_s[tid] = part_max;
            }
        }
        __syncthreads();   // every thread's partial max is now visible

        if (row < BLOCK_M) {
            const int m_idx = m_start + row;
            if (m_idx < M) {
                // 4b. merge the TPR partials (read-only over red_s -> no race),
                // then the online-softmax update. All TPR lanes compute the
                // same new_m / rescale / row_l so they all agree downstream.
                float row_max = -INFINITY;
                for (int k = 0; k < TPR; ++k)
                    row_max = fmaxf(row_max, red_s[row * TPR + k]);
                const float m_old  = m_s[row];
                new_m   = fmaxf(m_old, row_max);
                rescale = __expf(m_old - new_m);
                float part_l = 0.0f;
                for (int n = part; n < BLOCK_N; n += TPR) {
                    float p = __expf(p_s[row][n] - new_m);      // -inf -> 0
                    p_s[row][n] = p;                            // overwrite w/ p
                    part_l += p;
                }
                red_s[tid] = part_l;   // partial row-l (reuse scratch)
            }
        }
        __syncthreads();   // p is written; partial row-l sums are visible

        if (row < BLOCK_M) {
            const int m_idx = m_start + row;
            if (m_idx < M) {
                float row_l = 0.0f;
                for (int k = 0; k < TPR; ++k) row_l += red_s[row * TPR + k];
                // m/l are only written by lane part==0 (single writer, no race)
                if (part == 0) {
                    m_s[row] = new_m;
                    l_s[row] = l_s[row] * rescale + row_l;
                }
                // 4c. rescale this thread's d-slice of the acc row
                for (int d = part; d < BLOCK_D; d += TPR) acc_s[row][d] *= rescale;
            }
        }
        __syncthreads();   // p_s now holds probabilities; acc_s rescaled

        // ---- 5. Output accumulation ------------------------------------------
        // acc_s[r][d] += sum_n p_s[r][n] * V[n][d]
        // V is NOT staged in shared memory (shared budget); each thread reads
        // the packed V bytes straight from global memory. Threads in a warp
        // have consecutive d, so they touch consecutive bytes -> coalesced.
        for (int c = tid; c < BLOCK_M * BLOCK_D; c += THREADS) {
            int r = c / BLOCK_D, d = c % BLOCK_D;
            int m_idx = m_start + r;
            if (m_idx < M && d < D) {
                float a = acc_s[r][d];
                for (int n = 0; n < BLOCK_N; ++n) {
                    int n_abs = start + n;
                    if (n_abs >= S) break;       // n_abs grows -> safe break
                    bool use_cell = !causal || (m_idx >= n_abs);
                    if (use_cell) {
                        uint8_t byte = v_packed_ptr[kv_base + (size_t)n_abs * D2
                                                    + (d >> 1)];
                        float v = (d & 1) ? (float)(byte >> 4)
                                          : (float)(byte & 0x0Fu);
                        if (use_quant) v = (v - z_v_s[d]) * s_v_s[d];
                        a += p_s[r][n] * v;
                    }
                }
                acc_s[r][d] = a;
            }
        }
        __syncthreads();   // acc_s fully updated before next chunk
    }

    // ---- 6. Epilogue: out = acc / l (per row) --------------------------------
    for (int c = tid; c < BLOCK_M * BLOCK_D; c += THREADS) {
        int r = c / BLOCK_D, d = c % BLOCK_D;
        int m_idx = m_start + r;
        if (m_idx < M && d < D)
            out_ptr[(size_t)head * M * D + (size_t)m_idx * D + d]
                = acc_s[r][d] / l_s[r];
    }
}

// =============================================================================
//  PER-TOKEN DECODE KERNEL (vLLM packed-cache integration)
//  =============================================================================
//  Same online-softmax decode as above, but with PER-TOKEN min/step scaling
//  instead of per-channel rulers. This is the format used by the vLLM adapter's
//  packed 4-bit KV cache:
//
//      slot (per block, position, kv_head) = D + 8 bytes:
//          [0 : D/2)       packed K  (4-bit, even dim = low nibble)
//          [D/2 : D)       packed V  (4-bit)
//          [D : D+2)       k_min     (fp16, min over this token's K)
//          [D+2 : D+4)     k_step    (fp16, (max-min)/15)
//          [D+4 : D+6)     v_min     (fp16)
//          [D+6 : D+8)     v_step    (fp16)
//
//  Dequant is a plain affine map:  value = nibble * step + min.
//  Why per-token instead of per-channel? The cache is written INCREMENTALLY
//  (one new token per decode step via do_kv_cache_update). Per-channel rulers
//  need the whole sequence's min/max, which we don't have yet — but each
//  token's own min/max is available the moment it is written, so per-token
//  scaling is both O(1) to store and immune to ruler drift.
//
//  This kernel reads CONTIGUOUS [KV, S, D/2] packed buffers (the adapter
//  gathers the request's paged blocks into a contiguous staging buffer first).
template <int D, int BLOCK_D, int BLOCK_N, int THREADS = kDecodeThreads>
__global__ void kv_dequant_decode_per_token_kernel(
    const __half* __restrict__ q_ptr,        // [H, D] fp16 query token
    const uint8_t* __restrict__ k_packed_ptr,// [KV, S, D/2] packed 4-bit keys
    const __half* __restrict__ k_min_ptr,    // [KV, S] per-token min
    const __half* __restrict__ k_step_ptr,   // [KV, S] per-token step
    const uint8_t* __restrict__ v_packed_ptr,// [KV, S, D/2] packed 4-bit values
    const __half* __restrict__ v_min_ptr,    // [KV, S] per-token min
    const __half* __restrict__ v_step_ptr,   // [KV, S] per-token step
    float* __restrict__ out_ptr,             // [H, D] fp32 output
    int S, int groups)
{
    constexpr int D2 = D / 2;

    __shared__ float q_s[BLOCK_D];
    __shared__ float k_s[BLOCK_N][BLOCK_D + 1];
    __shared__ uint8_t v_bytes[BLOCK_N][D2];   // packed V staged in shared
    __shared__ float k_min_s[BLOCK_N], k_step_s[BLOCK_N];
    __shared__ float v_min_s[BLOCK_N], v_step_s[BLOCK_N];
    __shared__ float p_s[BLOCK_N];
    __shared__ float scratch[THREADS];

    const int tid  = threadIdx.x;
    const int head = blockIdx.x;
    const int kv_head = head / groups;

    if (tid < BLOCK_D) {
        float v = 0.0f;
        if (tid < D) v = __half2float(q_ptr[(size_t)head * D + tid]);
        q_s[tid] = v;
    }
    __syncthreads();

    const size_t kv_base = (size_t)kv_head * (size_t)S * D2;
    const float  scale   = 1.0f / sqrtf((float)D);

    float m_cur = -INFINITY;
    float l_cur = 0.0f;
    float acc   = 0.0f;

    for (int start = 0; start < S; start += BLOCK_N) {
        // Load the per-token rulers for this chunk (one thread per token).
        if (tid < BLOCK_N) {
            int n_abs = start + tid;
            if (n_abs < S) {
                k_min_s[tid]  = __half2float(k_min_ptr[(size_t)kv_head * S + n_abs]);
                k_step_s[tid] = __half2float(k_step_ptr[(size_t)kv_head * S + n_abs]);
                v_min_s[tid]  = __half2float(v_min_ptr[(size_t)kv_head * S + n_abs]);
                v_step_s[tid] = __half2float(v_step_ptr[(size_t)kv_head * S + n_abs]);
            } else {
                k_min_s[tid] = 0.0f; k_step_s[tid] = 1.0f;
                v_min_s[tid] = 0.0f; v_step_s[tid] = 1.0f;
            }
        }
        __syncthreads();   // rulers visible before the dequant loop below

        // Load packed K chunk and dequantize with the per-token rulers.
        for (int i = tid; i < BLOCK_N * D2; i += THREADS) {
            int n = i / D2, j = i % D2;
            int n_abs = start + n;
            uint8_t byte = 0;
            if (n_abs < S) byte = k_packed_ptr[kv_base + (size_t)n_abs * D2 + j];
            float k_step = k_step_s[n], k_min = k_min_s[n];
            k_s[n][2 * j]     = (float)(byte & 0x0Fu) * k_step + k_min;
            k_s[n][2 * j + 1] = (float)(byte >> 4) * k_step + k_min;
        }
        // Stage packed V bytes in shared (coalesced global read once per chunk).
        for (int i = tid; i < BLOCK_N * D2; i += THREADS) {
            int n = i / D2, j = i % D2;
            int n_abs = start + n;
            v_bytes[n][j] = (n_abs < S)
                                ? v_packed_ptr[kv_base + (size_t)n_abs * D2 + j]
                                : (uint8_t)0;
        }
        __syncthreads();

        float score = -INFINITY;
        if (tid < BLOCK_N && (start + tid) < S) {
            float s = 0.0f;
            for (int d = 0; d < D; ++d) s += q_s[d] * k_s[tid][d];
            score = s * scale;
        }

        scratch[tid] = score;
        __syncthreads();
        for (int off = THREADS / 2; off > 0; off >>= 1) {
            if (tid < off) scratch[tid] = fmaxf(scratch[tid], scratch[tid + off]);
            __syncthreads();
        }
        const float chunk_max = scratch[0];
        __syncthreads();

        const float new_m   = fmaxf(m_cur, chunk_max);
        const float rescale = __expf(m_cur - new_m);
        m_cur = new_m;
        l_cur *= rescale;

        if (tid < BLOCK_N && (start + tid) < S) p_s[tid] = __expf(score - new_m);
        __syncthreads();

        scratch[tid] = 0.0f;
        if (tid < BLOCK_N && (start + tid) < S) scratch[tid] = p_s[tid];
        __syncthreads();
        for (int off = THREADS / 2; off > 0; off >>= 1) {
            if (tid < off) scratch[tid] += scratch[tid + off];
            __syncthreads();
        }
        l_cur += scratch[0];
        __syncthreads();

        // Accumulate: V dequantized per token from SHARED packed bytes.
        if (tid < D) {
            float a = acc * rescale;
            for (int n = 0; n < BLOCK_N; ++n) {
                int n_abs = start + n;
                if (n_abs >= S) break;
                uint8_t byte = v_bytes[n][tid >> 1];
                float v = (tid & 1) ? (float)(byte >> 4) : (float)(byte & 0x0Fu);
                v = v * v_step_s[n] + v_min_s[n];
                a += p_s[n] * v;
            }
            acc = a;
        }
        __syncthreads();
    }

    if (tid < D) out_ptr[(size_t)head * D + tid] = acc / l_cur;
}

// =============================================================================
//  PAGED PER-TOKEN DECODE KERNEL (vLLM packed-cache integration, no gather)
//  =============================================================================
//  Same math as kv_dequant_decode_per_token_kernel, but reads the PAGED
//  vLLM cache directly through a block_table, so the adapter does not need to
//  gather + copy the request's blocks into contiguous staging buffers on every
//  step (that Python-side gather cost ~5ms/step in eager mode — more than the
//  kernel itself). Grid = (H,) as usual; one block per query head.
//
//  Layout of a slot (per block, position, kv_head) — D + 8 bytes:
//      [0 : D/2)       packed K
//      [D/2 : D)       packed V
//      [D : D+2)       k_min fp16, [D+2 : D+4) k_step fp16
//      [D+4 : D+6)     v_min fp16, [D+6 : D+8) v_step fp16
template <int D, int BLOCK_D, int BLOCK_N, int THREADS = kDecodeThreads>
__global__ void kv_dequant_decode_per_token_paged_kernel(
    const __half* __restrict__ q_ptr,        // [H, D] fp16 query token
    const uint8_t* __restrict__ cache_ptr,   // [num_blocks, block_size, KV, D+8]
    const int64_t* __restrict__ block_row,   // this request's row of block_table
    float* __restrict__ out_ptr,             // [H, D] fp32
    int S, int block_size, int groups, int num_kv_heads)
{
    constexpr int D2 = D / 2;
    constexpr int SLOT = D + 8;

    __shared__ float q_s[BLOCK_D];
    __shared__ float k_s[BLOCK_N][BLOCK_D + 1];
    __shared__ uint8_t v_bytes[BLOCK_N][D2];
    __shared__ float k_min_s[BLOCK_N], k_step_s[BLOCK_N];
    __shared__ float v_min_s[BLOCK_N], v_step_s[BLOCK_N];
    __shared__ float p_s[BLOCK_N];
    __shared__ float scratch[THREADS];

    const int tid  = threadIdx.x;
    const int head = blockIdx.x;
    const int kv_head = head / groups;

    if (tid < BLOCK_D) {
        float v = 0.0f;
        if (tid < D) v = __half2float(q_ptr[(size_t)head * D + tid]);
        q_s[tid] = v;
    }
    __syncthreads();

    // Map a global KV index n (0..S-1) to (block, slot-in-block).
    // block_table is [req_id, block_index] -> physical block id.
    const float scale = 1.0f / sqrtf((float)D);
    float m_cur = -INFINITY;
    float l_cur = 0.0f;
    float acc   = 0.0f;

    for (int start = 0; start < S; start += BLOCK_N) {
        // Load per-token rulers for the chunk.
        if (tid < BLOCK_N) {
            int n_abs = start + tid;
            if (n_abs < S) {
                int block_i = n_abs / block_size;
                int pos     = n_abs % block_size;
                int64_t phys_block = block_row[block_i];
                size_t base = ((size_t)phys_block * block_size + pos) * num_kv_heads * SLOT
                              + (size_t)kv_head * SLOT;
                const uint8_t* slot = cache_ptr + base;
                k_min_s[tid]  = __half2float(reinterpret_cast<const __half*>(slot + D)[0]);
                k_step_s[tid] = __half2float(reinterpret_cast<const __half*>(slot + D)[1]);
                v_min_s[tid]  = __half2float(reinterpret_cast<const __half*>(slot + D)[2]);
                v_step_s[tid] = __half2float(reinterpret_cast<const __half*>(slot + D)[3]);
            } else {
                k_min_s[tid] = 0.0f; k_step_s[tid] = 1.0f;
                v_min_s[tid] = 0.0f; v_step_s[tid] = 1.0f;
            }
        }
        __syncthreads();

        // Load packed K + V for the chunk straight from the paged cache.
        for (int i = tid; i < BLOCK_N * D2; i += THREADS) {
            int n = i / D2, j = i % D2;
            int n_abs = start + n;
            uint8_t kbyte = 0, vbyte = 0;
            if (n_abs < S) {
                int block_i = n_abs / block_size;
                int pos     = n_abs % block_size;
                int64_t phys_block = block_row[block_i];
                size_t base = ((size_t)phys_block * block_size + pos) * num_kv_heads * SLOT
                              + (size_t)kv_head * SLOT;
                kbyte = cache_ptr[base + j];
                vbyte = cache_ptr[base + D2 + j];
            }
            float k_step = k_step_s[n], k_min = k_min_s[n];
            k_s[n][2 * j]     = (float)(kbyte & 0x0Fu) * k_step + k_min;
            k_s[n][2 * j + 1] = (float)(kbyte >> 4) * k_step + k_min;
            v_bytes[n][j] = vbyte;
        }
        __syncthreads();

        float score = -INFINITY;
        if (tid < BLOCK_N && (start + tid) < S) {
            float s = 0.0f;
            for (int d = 0; d < D; ++d) s += q_s[d] * k_s[tid][d];
            score = s * scale;
        }

        scratch[tid] = score;
        __syncthreads();
        for (int off = THREADS / 2; off > 0; off >>= 1) {
            if (tid < off) scratch[tid] = fmaxf(scratch[tid], scratch[tid + off]);
            __syncthreads();
        }
        const float chunk_max = scratch[0];
        __syncthreads();

        const float new_m   = fmaxf(m_cur, chunk_max);
        const float rescale = __expf(m_cur - new_m);
        m_cur = new_m;
        l_cur *= rescale;

        if (tid < BLOCK_N && (start + tid) < S) p_s[tid] = __expf(score - new_m);
        __syncthreads();

        scratch[tid] = 0.0f;
        if (tid < BLOCK_N && (start + tid) < S) scratch[tid] = p_s[tid];
        __syncthreads();
        for (int off = THREADS / 2; off > 0; off >>= 1) {
            if (tid < off) scratch[tid] += scratch[tid + off];
            __syncthreads();
        }
        l_cur += scratch[0];
        __syncthreads();

        if (tid < D) {
            float a = acc * rescale;
            for (int n = 0; n < BLOCK_N; ++n) {
                int n_abs = start + n;
                if (n_abs >= S) break;
                uint8_t byte = v_bytes[n][tid >> 1];
                float v = (tid & 1) ? (float)(byte >> 4) : (float)(byte & 0x0Fu);
                v = v * v_step_s[n] + v_min_s[n];
                a += p_s[n] * v;
            }
            acc = a;
        }
        __syncthreads();
    }

    if (tid < D) out_ptr[(size_t)head * D + tid] = acc / l_cur;
}

// =============================================================================
//  QUANTIZE-ON-WRITE STORE KERNEL (vLLM packed-cache integration)
//  =============================================================================
//  Called from the adapter's `do_kv_cache_update` once per token (N tokens in
//  the current write batch, typically 1 for decode). For each (token, kv_head):
//      min = min over D of key row, max = max over D
//      step = (max - min) / 15        (guard: step >= tiny)
//      q    = clamp(round((x - min) / step), 0, 15)
//      pack pairs of dims into one byte (even = low nibble, odd = high)
//  and write the packed K + packed V + (min, step) rulers into the cache slot
//  addressed by slot_mapping (slot = block_idx * block_size + pos_in_block).
//
//  Grid: (N, KV)  — one block per (token, kv_head). Block: THREADS threads.
template <int D, int SLOT, int THREADS = kDecodeThreads>
__global__ void kv_quantize_store_kernel(
    const __half* __restrict__ key_ptr,        // [N, KV, D] fp16
    const __half* __restrict__ value_ptr,      // [N, KV, D] fp16
    uint8_t* __restrict__ cache_ptr,           // [num_blocks, block_size, KV, SLOT]
    const int64_t* __restrict__ slot_mapping_ptr, // [N]
    int block_size)
{
    constexpr int D2 = D / 2;
    __shared__ float s_min[THREADS];     // reduction scratch (min)
    __shared__ float s_max[THREADS];     // reduction scratch (max)
    __shared__ uint8_t s_packed[2 * D2]; // packed K then packed V

    const int tok = blockIdx.x;   // token index in the write batch
    const int kvh = blockIdx.y;   // kv head index
    const int tid = threadIdx.x;

    const __half* row_k = key_ptr + ((size_t)tok * gridDim.y + kvh) * D;
    const __half* row_v = value_ptr + ((size_t)tok * gridDim.y + kvh) * D;

    // ---- 1. min/max over the K row ------------------------------------------
    // Padding threads (tid >= D) get +inf/-inf so they never win the reduction.
    if (tid < D) {
        s_min[tid] = s_max[tid] = __half2float(row_k[tid]);
    } else {
        s_min[tid] = INFINITY;
        s_max[tid] = -INFINITY;
    }
    __syncthreads();
    for (int off = THREADS / 2; off > 0; off >>= 1) {
        if (tid < off) {
            s_min[tid] = fminf(s_min[tid], s_min[tid + off]);
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + off]);
        }
        __syncthreads();
    }
    const float k_min = s_min[0], k_max = s_max[0];
    float k_step = (k_max - k_min) / 15.0f;
    if (!(k_step > 0.0f)) k_step = 1.0f;   // constant row guard

    // ---- 2. quantize + pack K: each thread owns one dim pair -----------------
    if (tid < D2) {
        int d0 = 2 * tid, d1 = 2 * tid + 1;
        float x0 = __half2float(row_k[d0]);
        float x1 = __half2float(row_k[d1]);
        int q0 = (int)((x0 - k_min) / k_step + 0.5f);
        int q1 = (int)((x1 - k_min) / k_step + 0.5f);
        q0 = q0 < 0 ? 0 : (q0 > 15 ? 15 : q0);
        q1 = q1 < 0 ? 0 : (q1 > 15 ? 15 : q1);
        s_packed[tid] = (uint8_t)(q0 | (q1 << 4));
    }
    __syncthreads();

    // ---- 3. min/max + quantize + pack for the V row --------------------------
    if (tid < D) {
        s_min[tid] = s_max[tid] = __half2float(row_v[tid]);
    } else {
        s_min[tid] = INFINITY;
        s_max[tid] = -INFINITY;
    }
    __syncthreads();
    for (int off = THREADS / 2; off > 0; off >>= 1) {
        if (tid < off) {
            s_min[tid] = fminf(s_min[tid], s_min[tid + off]);
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + off]);
        }
        __syncthreads();
    }
    const float v_min = s_min[0], v_max = s_max[0];
    float v_step = (v_max - v_min) / 15.0f;
    if (!(v_step > 0.0f)) v_step = 1.0f;

    if (tid < D2) {
        int d0 = 2 * tid, d1 = 2 * tid + 1;
        float x0 = __half2float(row_v[d0]);
        float x1 = __half2float(row_v[d1]);
        int q0 = (int)((x0 - v_min) / v_step + 0.5f);
        int q1 = (int)((x1 - v_min) / v_step + 0.5f);
        q0 = q0 < 0 ? 0 : (q0 > 15 ? 15 : q0);
        q1 = q1 < 0 ? 0 : (q1 > 15 ? 15 : q1);
        s_packed[D2 + tid] = (uint8_t)(q0 | (q1 << 4));
    }
    __syncthreads();

    // ---- 4. scatter the slot ------------------------------------------------
    int64_t slot = slot_mapping_ptr[tok];
    if (slot < 0) return;   // padding token
    size_t block_idx = (size_t)(slot / block_size);
    size_t pos       = (size_t)(slot % block_size);
    size_t slot_base = ((block_idx * (size_t)block_size + pos) * gridDim.y + kvh) * SLOT;

    if (tid < D2) {
        cache_ptr[slot_base + tid]      = s_packed[tid];
        cache_ptr[slot_base + D2 + tid] = s_packed[D2 + tid];
    }
    if (tid < 4) {
        __half* ruler = reinterpret_cast<__half*>(cache_ptr + slot_base + D);
        __half vals[4] = {
            __float2half(k_min), __float2half(k_step),
            __float2half(v_min), __float2half(v_step),
        };
        ruler[tid] = vals[tid];
    }
}

}  // namespace slm_turbo

#endif  // SLM_TURBO_KV_DEQUANT_KERNELS_CUH
