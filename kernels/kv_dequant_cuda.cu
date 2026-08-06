// =============================================================================
//  kernels/kv_dequant_cuda.cu
//
//  Torch bindings + launch wrappers for the CUDA kernels in
//  kv_dequant_kernels.cuh. This file is JIT-compiled by
//  kernels/kv_dequant.py via torch.utils.cpp_extension.load_inline on first
//  import (cached under ~/.cache/torch_extensions afterwards).
//
//  Public API (called from kernels/kv_dequant.py):
//
//      kv_dequant_decode_attention_cuda(          per-channel decode (bench)
//          q, k_packed, k_scale, k_zero, v_packed, v_scale, v_zero,
//          block_n, use_quant, num_kv_heads)
//      kv_dequant_prefill_attention_cuda(         per-channel prefill (bench)
//          q, k_packed, k_scale, k_zero, v_packed, v_scale, v_zero,
//          block_m, block_n, causal, use_quant, num_kv_heads)
//      kv_dequant_decode_per_token_attention_cuda per-token decode (vLLM)
//      kv_dequant_decode_per_token_paged_attention_cuda  paged decode (vLLM)
//      kv_quantize_store_cuda                      quantize-on-write store
//
//  Each launcher: validates dtypes/shapes -> allocates the fp32 output tensor
//  -> picks the right template instantiation of the kernel -> launches.
// =============================================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>

#include "kv_dequant_kernels.cuh"

namespace slm_turbo {

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
namespace {

// Smallest power of two >= x (next_power_of_2).
int next_pow2(int x) {
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

// Shared-memory budget for one block. Static shared memory is capped at 48 KB
// (49152 B) on all supported architectures; over-budget launches fail.
constexpr size_t kMaxSharedBytes = 49152;

// Reject invalid shapes early, before we touch the GPU. This mirrors the
// asserts in the launcher functions.
void check_common(const torch::Tensor& q, const torch::Tensor& k_packed,
                  const torch::Tensor& k_scale, const torch::Tensor& k_zero,
                  const torch::Tensor& v_packed, const torch::Tensor& v_scale,
                  const torch::Tensor& v_zero,
                  const char* phase, int64_t num_kv_heads) {
    TORCH_CHECK(q.is_cuda(), phase, ": q must be a CUDA tensor");
    TORCH_CHECK(q.scalar_type() == at::kHalf, phase, ": q must be fp16");
    TORCH_CHECK(k_packed.is_cuda() && v_packed.is_cuda(), phase,
                ": packed KV must be CUDA tensors");
    TORCH_CHECK(k_packed.scalar_type() == at::kByte &&
                    v_packed.scalar_type() == at::kByte,
                phase, ": packed KV must be uint8");
    for (auto t : {k_scale, k_zero, v_scale, v_zero}) {
        TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kHalf, phase,
                    ": scale/zero tensors must be fp16 CUDA");
    }
    const int64_t D  = q.size(2);
    const int64_t D2 = k_packed.size(2);
    TORCH_CHECK(D == 2 * D2, phase, ": D (", D, ") must be 2 * D2 (", D2, ")");
    const int64_t H = q.size(0);
    const int64_t KV = k_packed.size(0);
    TORCH_CHECK(num_kv_heads > 0 && H % num_kv_heads == 0, phase,
                ": num_kv_heads ", num_kv_heads, " must divide query heads ", H);
}

}  // namespace

// =============================================================================
// DECODE launcher
// =============================================================================
torch::Tensor kv_dequant_decode_attention_cuda(
    torch::Tensor q, torch::Tensor k_packed, torch::Tensor k_scale,
    torch::Tensor k_zero, torch::Tensor v_packed, torch::Tensor v_scale,
    torch::Tensor v_zero, int64_t block_n, bool use_quant,
    int64_t num_kv_heads)
{
    check_common(q, k_packed, k_scale, k_zero, v_packed, v_scale, v_zero,
                 "kv_dequant_decode", num_kv_heads);

    TORCH_CHECK(q.dim() == 3 && q.size(1) == 1,
                "kv_dequant_decode: q must be [H, 1, D], got ",
                q.sizes());
    TORCH_CHECK(block_n == 32 || block_n == 64,
                "kv_dequant_decode: BLOCK_N must be 32 or 64, got ", block_n);

    const int H  = (int)q.size(0);
    const int D  = (int)q.size(2);
    const int S  = (int)k_packed.size(1);
    const int KV = (int)k_packed.size(0);
    const int groups = (int)(H / num_kv_heads);  // query heads per KV head

    TORCH_CHECK(D == 64 || D == 80 || D == 128,
                "kv_dequant_decode: head_dim must be 64/80/128, got ", D);

    const int BLOCK_D = next_pow2(D);
    const int THREADS = kDecodeThreads;

    // Allocate the fp32 output.
    auto out = torch::zeros({H, 1, D}, q.options().dtype(at::kFloat));

    // Keep the selected device in scope for the launch.
    const c10::cuda::CUDAGuard guard(q.device());

    q        = q.contiguous();
    k_packed = k_packed.contiguous();
    k_scale  = k_scale.contiguous();
    k_zero   = k_zero.contiguous();
    v_packed = v_packed.contiguous();
    v_scale  = v_scale.contiguous();
    v_zero   = v_zero.contiguous();

    const auto qp   = reinterpret_cast<const __half*>(q.data_ptr<at::Half>());
    const auto kpp  = k_packed.data_ptr<uint8_t>();
    const auto ksp  = reinterpret_cast<const __half*>(k_scale.data_ptr<at::Half>());
    const auto kzp  = reinterpret_cast<const __half*>(k_zero.data_ptr<at::Half>());
    const auto vpp  = v_packed.data_ptr<uint8_t>();
    const auto vsp  = reinterpret_cast<const __half*>(v_scale.data_ptr<at::Half>());
    const auto vzp  = reinterpret_cast<const __half*>(v_zero.data_ptr<at::Half>());
    const auto outp = out.data_ptr<float>();

    dim3 grid(H);
    dim3 block(THREADS);

    // Shared-memory budget check (same formula the kernel uses internally).
    const size_t shmem =
        (size_t)BLOCK_D * sizeof(float) +                             // q_s
        (size_t)block_n * (BLOCK_D + 1) * sizeof(float) +             // k_s
        (size_t)block_n * (D / 2) +                                   // v_bytes
        4 * (size_t)BLOCK_D * sizeof(float) +                   // rulers
        (size_t)block_n * sizeof(float) +                       // p_s
        (size_t)THREADS * sizeof(float);                        // scratch
    TORCH_CHECK(shmem <= kMaxSharedBytes,
                "kv_dequant_decode: shared memory ", shmem,
                " B exceeds the 48 KB budget for (D=", D,
                ", BLOCK_N=", block_n, ")");

    switch (D) {
        case 64:
            if (block_n == 32)
                kv_dequant_decode_kernel<64, 64, 32><<<grid, block>>>(
                    qp, kpp, ksp, kzp, vpp, vsp, vzp, outp, S, groups, use_quant);
            else
                kv_dequant_decode_kernel<64, 64, 64><<<grid, block>>>(
                    qp, kpp, ksp, kzp, vpp, vsp, vzp, outp, S, groups, use_quant);
            break;
        case 80:
            if (block_n == 32)
                kv_dequant_decode_kernel<80, 128, 32><<<grid, block>>>(
                    qp, kpp, ksp, kzp, vpp, vsp, vzp, outp, S, groups, use_quant);
            else
                kv_dequant_decode_kernel<80, 128, 64><<<grid, block>>>(
                    qp, kpp, ksp, kzp, vpp, vsp, vzp, outp, S, groups, use_quant);
            break;
        case 128:
            if (block_n == 32)
                kv_dequant_decode_kernel<128, 128, 32><<<grid, block>>>(
                    qp, kpp, ksp, kzp, vpp, vsp, vzp, outp, S, groups, use_quant);
            else
                kv_dequant_decode_kernel<128, 128, 64><<<grid, block>>>(
                    qp, kpp, ksp, kzp, vpp, vsp, vzp, outp, S, groups, use_quant);
            break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// =============================================================================
// PREFILL launcher
// =============================================================================
torch::Tensor kv_dequant_prefill_attention_cuda(
    torch::Tensor q, torch::Tensor k_packed, torch::Tensor k_scale,
    torch::Tensor k_zero, torch::Tensor v_packed, torch::Tensor v_scale,
    torch::Tensor v_zero, int64_t block_m, int64_t block_n, bool causal,
    bool use_quant, int64_t num_kv_heads)
{
    check_common(q, k_packed, k_scale, k_zero, v_packed, v_scale, v_zero,
                 "kv_dequant_prefill", num_kv_heads);

    TORCH_CHECK(q.dim() == 3, "kv_dequant_prefill: q must be [H, M, D], got ",
                q.sizes());
    TORCH_CHECK(block_m == 16 || block_m == 32,
                "kv_dequant_prefill: BLOCK_M must be 16 or 32, got ", block_m);
    TORCH_CHECK(block_n == 32 || block_n == 64,
                "kv_dequant_prefill: BLOCK_N must be 32 or 64, got ", block_n);

    const int H = (int)q.size(0);
    const int M = (int)q.size(1);
    const int D = (int)q.size(2);
    const int S = (int)k_packed.size(1);
    const int KV = (int)k_packed.size(0);
    const int groups = (int)(H / num_kv_heads);  // query heads per KV head

    TORCH_CHECK(D == 64 || D == 80 || D == 128,
                "kv_dequant_prefill: head_dim must be 64/80/128, got ", D);

    const int BLOCK_D = next_pow2(D);
    // Prefill uses 256 threads (8 warps) so the per-row softmax can be spread
    // over 8 lanes (TPR) instead of one thread per row. Decode stays at 128.
    const int THREADS = 256;

    // Only some (BLOCK_M, BLOCK_N) combos fit the 48 KB static-shared budget.
    const bool d64_ok  = (D == 64);
    const bool combo_ok = (block_m == 16 && block_n == 32) ||   // all D
                          (d64_ok && block_m == 16 && block_n == 64) ||
                          (d64_ok && block_m == 32 && block_n == 32) ||
                          (d64_ok && block_m == 32 && block_n == 64);
    TORCH_CHECK(combo_ok,
                "kv_dequant_prefill: (BLOCK_M=", block_m, ", BLOCK_N=", block_n,
                ") does not fit the 48 KB shared budget for head_dim=", D,
                ". Use BLOCK_M=16, BLOCK_N=32 for D>=80.");

    auto out = torch::zeros({H, M, D}, q.options().dtype(at::kFloat));
    const c10::cuda::CUDAGuard guard(q.device());

    q        = q.contiguous();
    k_packed = k_packed.contiguous();
    k_scale  = k_scale.contiguous();
    k_zero   = k_zero.contiguous();
    v_packed = v_packed.contiguous();
    v_scale  = v_scale.contiguous();
    v_zero   = v_zero.contiguous();

    const auto qp   = reinterpret_cast<const __half*>(q.data_ptr<at::Half>());
    const auto kpp  = k_packed.data_ptr<uint8_t>();
    const auto ksp  = reinterpret_cast<const __half*>(k_scale.data_ptr<at::Half>());
    const auto kzp  = reinterpret_cast<const __half*>(k_zero.data_ptr<at::Half>());
    const auto vpp  = v_packed.data_ptr<uint8_t>();
    const auto vsp  = reinterpret_cast<const __half*>(v_scale.data_ptr<at::Half>());
    const auto vzp  = reinterpret_cast<const __half*>(v_zero.data_ptr<at::Half>());
    const auto outp = out.data_ptr<float>();

    // Grid: one block per (head, query-block).
    const int num_m_blocks = (M + (int)block_m - 1) / (int)block_m;
    dim3 grid(H * num_m_blocks);
    dim3 block(THREADS);

    // Explicit dispatch table of the instantiated kernel combos.
    if (D == 64) {
        if (block_m == 16 && block_n == 32)
            kv_dequant_prefill_kernel<64, 64, 16, 32, 256><<<grid, block>>>(
                qp, kpp, ksp, kzp, vpp, vsp, vzp, outp, M, S, groups, causal, use_quant);
        else if (block_m == 16 && block_n == 64)
            kv_dequant_prefill_kernel<64, 64, 16, 64, 256><<<grid, block>>>(
                qp, kpp, ksp, kzp, vpp, vsp, vzp, outp, M, S, groups, causal, use_quant);
        else if (block_m == 32 && block_n == 32)
            kv_dequant_prefill_kernel<64, 64, 32, 32, 256><<<grid, block>>>(
                qp, kpp, ksp, kzp, vpp, vsp, vzp, outp, M, S, groups, causal, use_quant);
        else
            kv_dequant_prefill_kernel<64, 64, 32, 64, 256><<<grid, block>>>(
                qp, kpp, ksp, kzp, vpp, vsp, vzp, outp, M, S, groups, causal, use_quant);
    } else if (D == 80) {
        kv_dequant_prefill_kernel<80, 128, 16, 32, 256><<<grid, block>>>(
            qp, kpp, ksp, kzp, vpp, vsp, vzp, outp, M, S, groups, causal, use_quant);
    } else {  // D == 128
        kv_dequant_prefill_kernel<128, 128, 16, 32, 256><<<grid, block>>>(
            qp, kpp, ksp, kzp, vpp, vsp, vzp, outp, M, S, groups, causal, use_quant);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// =============================================================================
//  PER-TOKEN DECODE launcher (vLLM packed-cache integration)
//  =============================================================================
torch::Tensor kv_dequant_decode_per_token_attention_cuda(
    torch::Tensor q, torch::Tensor k_packed, torch::Tensor k_min,
    torch::Tensor k_step, torch::Tensor v_packed, torch::Tensor v_min,
    torch::Tensor v_step, int64_t block_n, int64_t num_kv_heads)
{
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kHalf,
                "per-token decode: q must be fp16 CUDA");
    TORCH_CHECK(q.dim() == 3 && q.size(1) == 1,
                "per-token decode: q must be [H, 1, D], got ", q.sizes());
    TORCH_CHECK(block_n == 32 || block_n == 64,
                "per-token decode: BLOCK_N must be 32 or 64, got ", block_n);

    const int H  = (int)q.size(0);
    const int D  = (int)q.size(2);
    const int S  = (int)k_packed.size(1);
    const int KV = (int)k_packed.size(0);
    const int groups = (int)(H / num_kv_heads);
    TORCH_CHECK(D == 64 || D == 80 || D == 128,
                "per-token decode: head_dim must be 64/80/128, got ", D);
    TORCH_CHECK(H % num_kv_heads == 0, "H not divisible by num_kv_heads");

    auto out = torch::zeros({H, 1, D}, q.options().dtype(at::kFloat));
    const c10::cuda::CUDAGuard guard(q.device());

    q        = q.contiguous();
    k_packed = k_packed.contiguous();
    k_min    = k_min.contiguous();
    k_step   = k_step.contiguous();
    v_packed = v_packed.contiguous();
    v_min    = v_min.contiguous();
    v_step   = v_step.contiguous();

    const auto qp   = reinterpret_cast<const __half*>(q.data_ptr<at::Half>());
    const auto kpp  = k_packed.data_ptr<uint8_t>();
    const auto kmp  = reinterpret_cast<const __half*>(k_min.data_ptr<at::Half>());
    const auto ksp  = reinterpret_cast<const __half*>(k_step.data_ptr<at::Half>());
    const auto vpp  = v_packed.data_ptr<uint8_t>();
    const auto vmp  = reinterpret_cast<const __half*>(v_min.data_ptr<at::Half>());
    const auto vsp  = reinterpret_cast<const __half*>(v_step.data_ptr<at::Half>());
    const auto outp = out.data_ptr<float>();

    const int BLOCK_D = next_pow2(D);
    dim3 grid(H);
    dim3 block(kDecodeThreads);

    switch (D) {
        case 64:
            if (block_n == 32)
                kv_dequant_decode_per_token_kernel<64, 64, 32><<<grid, block>>>(
                    qp, kpp, kmp, ksp, vpp, vmp, vsp, outp, S, groups);
            else
                kv_dequant_decode_per_token_kernel<64, 64, 64><<<grid, block>>>(
                    qp, kpp, kmp, ksp, vpp, vmp, vsp, outp, S, groups);
            break;
        case 80:
            if (block_n == 32)
                kv_dequant_decode_per_token_kernel<80, 128, 32><<<grid, block>>>(
                    qp, kpp, kmp, ksp, vpp, vmp, vsp, outp, S, groups);
            else
                kv_dequant_decode_per_token_kernel<80, 128, 64><<<grid, block>>>(
                    qp, kpp, kmp, ksp, vpp, vmp, vsp, outp, S, groups);
            break;
        case 128:
            if (block_n == 32)
                kv_dequant_decode_per_token_kernel<128, 128, 32><<<grid, block>>>(
                    qp, kpp, kmp, ksp, vpp, vmp, vsp, outp, S, groups);
            else
                kv_dequant_decode_per_token_kernel<128, 128, 64><<<grid, block>>>(
                    qp, kpp, kmp, ksp, vpp, vmp, vsp, outp, S, groups);
            break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// =============================================================================
//  PAGED PER-TOKEN DECODE launcher (vLLM packed-cache integration)
//  =============================================================================
torch::Tensor kv_dequant_decode_per_token_paged_attention_cuda(
    torch::Tensor q, torch::Tensor kv_cache, torch::Tensor block_table,
    int64_t seq_len, int64_t block_size, int64_t num_kv_heads)
{
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kHalf,
                "paged decode: q must be fp16 CUDA");
    TORCH_CHECK(q.dim() == 3 && q.size(1) == 1,
                "paged decode: q must be [H, 1, D], got ", q.sizes());
    TORCH_CHECK(kv_cache.scalar_type() == at::kByte,
                "paged decode: kv_cache must be uint8");

    const int H  = (int)q.size(0);
    const int D  = (int)q.size(2);
    const int KV = (int)num_kv_heads;
    const int groups = (int)(H / KV);
    TORCH_CHECK(D == 64 || D == 80 || D == 128,
                "paged decode: head_dim must be 64/80/128, got ", D);
    TORCH_CHECK(H % KV == 0, "H not divisible by num_kv_heads");
    TORCH_CHECK(kv_cache.size(-1) == D + 8,
                "paged decode: cache slot size must be D+8");

    auto out = torch::zeros({H, 1, D}, q.options().dtype(at::kFloat));
    const c10::cuda::CUDAGuard guard(q.device());

    q          = q.contiguous();
    kv_cache   = kv_cache.contiguous();
    block_table = block_table.contiguous();

    const auto qp   = reinterpret_cast<const __half*>(q.data_ptr<at::Half>());
    const auto cp   = kv_cache.data_ptr<uint8_t>();
    const auto btp  = block_table.data_ptr<int64_t>();
    const auto outp = out.data_ptr<float>();

    const int BLOCK_D = next_pow2(D);
    const int S  = (int)seq_len;
    const int BS = (int)block_size;
    dim3 grid(H);
    dim3 block(kDecodeThreads);

    switch (D) {
        case 64:
            kv_dequant_decode_per_token_paged_kernel<64, 64, 64><<<grid, block>>>(
                qp, cp, btp, outp, S, BS, groups, KV);
            break;
        case 80:
            kv_dequant_decode_per_token_paged_kernel<80, 128, 64><<<grid, block>>>(
                qp, cp, btp, outp, S, BS, groups, KV);
            break;
        case 128:
            kv_dequant_decode_per_token_paged_kernel<128, 128, 64><<<grid, block>>>(
                qp, cp, btp, outp, S, BS, groups, KV);
            break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// =============================================================================
//  QUANTIZE-ON-WRITE STORE launcher (vLLM packed-cache integration)
//  =============================================================================
void kv_quantize_store_cuda(
    torch::Tensor key, torch::Tensor value, torch::Tensor kv_cache,
    torch::Tensor slot_mapping, int64_t block_size)
{
    TORCH_CHECK(key.is_cuda() && key.scalar_type() == at::kHalf,
                "store: key/value must be fp16 CUDA");
    TORCH_CHECK(kv_cache.scalar_type() == at::kByte,
                "store: kv_cache must be uint8");
    const int N  = (int)key.size(0);
    const int KV = (int)key.size(1);
    const int D  = (int)key.size(2);
    TORCH_CHECK(D == 64 || D == 80 || D == 128,
                "store: head_dim must be 64/80/128, got ", D);
    // SLOT = D packed bytes + 8 bytes of (min, step) rulers
    const int SLOT = D + 8;
    TORCH_CHECK(kv_cache.size(-1) == SLOT,
                "store: cache slot size ", kv_cache.size(-1),
                " != expected ", SLOT);
    TORCH_CHECK(slot_mapping.scalar_type() == at::kLong &&
                    slot_mapping.numel() == N,
                "store: slot_mapping must be int64 of length N");

    const c10::cuda::CUDAGuard guard(key.device());
    key          = key.contiguous();
    value        = value.contiguous();
    kv_cache     = kv_cache.contiguous();
    slot_mapping = slot_mapping.contiguous();

    const auto kp  = reinterpret_cast<const __half*>(key.data_ptr<at::Half>());
    const auto vp  = reinterpret_cast<const __half*>(value.data_ptr<at::Half>());
    const auto cp  = kv_cache.data_ptr<uint8_t>();
    const auto smp = slot_mapping.data_ptr<int64_t>();

    dim3 grid(N, KV);
    dim3 block(kDecodeThreads);
    switch (D) {
        case 64:  kv_quantize_store_kernel<64, 72><<<grid, block>>>(kp, vp, cp, smp, (int)block_size); break;
        case 80:  kv_quantize_store_kernel<80, 88><<<grid, block>>>(kp, vp, cp, smp, (int)block_size); break;
        case 128: kv_quantize_store_kernel<128, 136><<<grid, block>>>(kp, vp, cp, smp, (int)block_size); break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace slm_turbo

// Explicit binding (load_inline's auto-generated bindings only work for
// global-scope functions, and these live in namespace slm_turbo). We define
// the module ourselves so the names are stable regardless of how the JIT
// build assembles the sources.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kv_dequant_decode_attention_cuda",
          &slm_turbo::kv_dequant_decode_attention_cuda,
          "Fused 4-bit KV dequant + attention, decode phase (CUDA)");
    m.def("kv_dequant_prefill_attention_cuda",
          &slm_turbo::kv_dequant_prefill_attention_cuda,
          "Fused 4-bit KV dequant + attention, prefill phase (CUDA)");
    m.def("kv_dequant_decode_per_token_attention_cuda",
          &slm_turbo::kv_dequant_decode_per_token_attention_cuda,
          "Per-token 4-bit KV decode attention (CUDA)");
    m.def("kv_dequant_decode_per_token_paged_attention_cuda",
          &slm_turbo::kv_dequant_decode_per_token_paged_attention_cuda,
          "Paged per-token 4-bit KV decode attention (CUDA)");
    m.def("kv_quantize_store_cuda",
          &slm_turbo::kv_quantize_store_cuda,
          "Quantize-on-write 4-bit KV store (CUDA)");
}
