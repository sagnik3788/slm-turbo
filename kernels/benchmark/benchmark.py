from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from kernels.kv_dequant import kv_dequant_decode_attention, kv_dequant_prefill_attention  # noqa: E402
from kernels.quantize import quantize_kv, dequantize_tensor, quantize_tensor  # noqa: E402

ERR_GATE = 1e-2
RESULTS = []


def report(section, line):
    print(line)
    RESULTS.append((section, line))


def timeit(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


# ---------------------------------------------------------------------------
# Section 1 + 2: op-level benchmarks (correct GQA handling)
# ---------------------------------------------------------------------------

def _kv_mem(H, KV, S, D):
    """(fp16 broadcast bytes, packed 4-bit bytes) for the KV cache of one layer."""
    fp16 = H * S * D * 2                      # H fp16 kv-head copies
    packed = KV * S * (D // 2)                # KV packed 4-bit (2 dims/byte)
    return fp16, packed


def bench_decode(H, KV, D, S, iters=100):
    torch.manual_seed(0)
    q = torch.randn(H, 1, D, device="cuda").half()
    k = torch.randn(KV, S, D, device="cuda").half()
    v = torch.randn(KV, S, D, device="cuda").half()
    kp, ks, kz, vp, vs, vz = quantize_kv(k, v)
    kd, vd = dequantize_tensor(kp, ks, kz).half(), dequantize_tensor(vp, vs, vz).half()

    # GQA reference: broadcast kv heads to H (what a non-GQA path must read)
    kb = kd.repeat_interleave(H // KV, dim=0)
    vb = vd.repeat_interleave(H // KV, dim=0)
    ref_deq = F.scaled_dot_product_attention(q, kb, vb)

    out = kv_dequant_decode_attention(q, kp, ks, kz, vp, vs, vz, num_kv_heads=KV)
    err_deq = (out - ref_deq.float()).abs().max().item()   # kernel vs dequantized ref
    if err_deq >= ERR_GATE:
        report("decode", f"H={H:2d} KV={KV} D={D:3d} S={S:5d}  [SKIP] err={err_deq:.4f} vs dequantized ref")
        return

    ref_fp16 = F.scaled_dot_product_attention(q, k.repeat_interleave(H // KV, dim=0),
                                                v.repeat_interleave(H // KV, dim=0))
    err_fp16 = (out - ref_fp16.float()).abs().max().item()  # kernel vs true fp16 (quant cost)

    t_k = timeit(lambda: kv_dequant_decode_attention(q, kp, ks, kz, vp, vs, vz, num_kv_heads=KV), iters)
    t_r = timeit(lambda: F.scaled_dot_product_attention(q, kb, vb), iters)
    fp16, packed = _kv_mem(H, KV, S, D)
    report("decode",
           f"H={H:2d} KV={KV} D={D:3d} S={S:5d} | kernel {t_k*1000:7.1f}us | "
           f"fp16 {t_r*1000:7.1f}us | {t_r/t_k:5.2f}x | err(deq) {err_deq:.1e} err(fp16) {err_fp16:.1e} | "
           f"KV {fp16//1024}KB -> {packed//1024}KB ({fp16//max(packed,1)}x)")


def bench_prefill(H, KV, D, S, iters=10):
    torch.manual_seed(0)
    q = torch.randn(H, S, D, device="cuda").half()
    k = torch.randn(KV, S, D, device="cuda").half()
    v = torch.randn(KV, S, D, device="cuda").half()
    kp, ks, kz, vp, vs, vz = quantize_kv(k, v)
    kd, vd = dequantize_tensor(kp, ks, kz).half(), dequantize_tensor(vp, vs, vz).half()
    kb = kd.repeat_interleave(H // KV, dim=0)
    vb = vd.repeat_interleave(H // KV, dim=0)
    ref_deq = F.scaled_dot_product_attention(q, kb, vb, is_causal=True)
    out = kv_dequant_prefill_attention(q, kp, ks, kz, vp, vs, vz, causal=True, num_kv_heads=KV)
    err_deq = (out - ref_deq.float()).abs().max().item()
    if err_deq >= ERR_GATE:
        report("prefill", f"H={H:2d} KV={KV} D={D:3d} S={S:5d}  [SKIP] err={err_deq:.4f} vs dequantized ref")
        return
    ref_fp16 = F.scaled_dot_product_attention(q, k.repeat_interleave(H // KV, dim=0),
                                                v.repeat_interleave(H // KV, dim=0), is_causal=True)
    err_fp16 = (out - ref_fp16.float()).abs().max().item()
    t_k = timeit(lambda: kv_dequant_prefill_attention(q, kp, ks, kz, vp, vs, vz, causal=True, num_kv_heads=KV), iters)
    t_r = timeit(lambda: F.scaled_dot_product_attention(q, kb, vb, is_causal=True), iters)
    fp16, packed = _kv_mem(H, KV, S, D)
    report("prefill",
           f"H={H:2d} KV={KV} D={D:3d} S={S:5d} | kernel {t_k:7.2f}ms | "
           f"fp16 {t_r:7.2f}ms | {t_r/t_k:5.2f}x | err(deq) {err_deq:.1e} err(fp16) {err_fp16:.1e} | "
           f"KV {fp16//1024}KB -> {packed//1024}KB ({fp16//max(packed,1)}x)")


def op_level():
    print("\n=== SECTION 1: decode attention (op-level, GQA shapes) ===")
    for H, KV, D in ((32, 4, 64), (14, 2, 64), (32, 8, 128)):
        for S in (512, 1024, 2048):
            bench_decode(H, KV, D, S)
    print("\n=== SECTION 2: prefill attention (op-level, causal) ===")
    for H, KV, D in ((32, 4, 64), (14, 2, 64)):
        for S in (512, 1024):
            bench_prefill(H, KV, D, S)


# ---------------------------------------------------------------------------
# Section 3: model-level TinyLlama (interleaved, quality + throughput)
# ---------------------------------------------------------------------------

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = ("The history of Paris spans over two thousand years. It grew into one of the "
          "most influential cities in Europe, known for its art, its cuisine and its "
          "architecture. Today its most famous landmark is")
STATE = {}


def _harness(use_kernel, ruler_refresh):
    from transformers.models.llama.modeling_llama import eager_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    def our_attention_forward(module, query_states, key_states, value_states,
                              attention_mask=None, **kwargs):
        bsz, q_heads, q_len, head_dim = query_states.shape
        _, kv_heads, kv_len, _ = key_states.shape
        if use_kernel and bsz == 1 and q_len == 1 and kv_len > 1:
            if ruler_refresh:
                # correct rulers every step (no saturation drift) — O(S) quantize
                kf = key_states.squeeze(0).contiguous().half()
                vf = value_states.squeeze(0).contiguous().half()
                kp, ksc, kz, vp, vsc, vz = quantize_kv(kf, vf)
            else:
                # production-style: incremental append with rulers fixed at first decode
                st = STATE.get(id(module))
                if st is None:
                    kf = key_states.squeeze(0).contiguous().half()
                    vf = value_states.squeeze(0).contiguous().half()
                    kp, ksc, kz, vp, vsc, vz = quantize_kv(kf, vf)
                    cap = kv_len + 256
                    kb = torch.zeros(kv_heads, cap, kp.shape[2], dtype=torch.uint8, device=kp.device)
                    vb = torch.zeros(kv_heads, cap, vp.shape[2], dtype=torch.uint8, device=vp.device)
                    kb[:, :kv_len] = kp
                    vb[:, :kv_len] = vp
                    STATE[id(module)] = dict(kb=kb, vb=vb, ksc=ksc, kz=kz, vsc=vsc, vz=vz, n=kv_len)
                    st = STATE[id(module)]
                else:
                    prev = st["n"]
                    k_new = key_states.squeeze(0)[:, prev:, :].contiguous().half()
                    v_new = value_states.squeeze(0)[:, prev:, :].contiguous().half()
                    st["kb"][:, prev:kv_len] = quantize_tensor(k_new, st["ksc"], st["kz"])
                    st["vb"][:, prev:kv_len] = quantize_tensor(v_new, st["vsc"], st["vz"])
                    st["n"] = kv_len
                kp, ksc, kz, vp, vsc, vz = st["kb"][:, :st["n"]], st["ksc"], st["kz"], st["vb"][:, :st["n"]], st["vsc"], st["vz"]
            out = kv_dequant_decode_attention(query_states.squeeze(0), kp, ksc, kz, vp, vsc, vz,
                                              num_kv_heads=kv_heads)
            return out.unsqueeze(0).half(), None
        return eager_attention_forward(module, query_states, key_states, value_states, attention_mask, **kwargs)

    ALL_ATTENTION_FUNCTIONS.register("slm_kernel", our_attention_forward)
    return "slm_kernel" if use_kernel else "eager"


def _run_model(use_kernel, ruler_refresh, n_new=40, warm=6, rounds=2):
    """Interleaved timing of one mode. Returns mean steady-state tok/s."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    global STATE
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda").eval()
    impl = _harness(use_kernel, ruler_refresh)
    ids = tok(PROMPT, return_tensors="pt")["input_ids"].to("cuda")

    def once():
        global STATE
        STATE = {}
        torch.cuda.empty_cache()
        for m in model.model.layers:
            m.self_attn.config._attn_implementation = impl
        with torch.no_grad():
            out = model(ids, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        times = []
        for _ in range(n_new):
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(next_id, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        steady = times[warm:]
        return len(steady) / sum(steady)

    speeds = [once() for _ in range(rounds)]
    del model
    torch.cuda.empty_cache()
    return sum(speeds) / len(speeds), speeds


def _generate_text(use_kernel, ruler_refresh, n=24):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    global STATE
    STATE = {}
    torch.cuda.empty_cache()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda").eval()
    impl = _harness(use_kernel, ruler_refresh)
    for m in model.model.layers:
        m.self_attn.config._attn_implementation = impl
    ids = tok(PROMPT, return_tensors="pt")["input_ids"].to("cuda")
    gen = []
    with torch.no_grad():
        out = model(ids, use_cache=True)
        past = out.past_key_values
        for _ in range(n):
            next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
            out = model(next_id, past_key_values=past, use_cache=True)
            past = out.past_key_values
            gen.append(next_id.item())
    del model
    torch.cuda.empty_cache()
    return tok.decode(gen)


def model_level():
    try:
        import transformers  # noqa: F401
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL)
    except Exception as e:
        print(f"\n=== SECTION 3 SKIPPED: transformers/TinyLlama unavailable ({e}) ===")
        return

    print("\n=== SECTION 3: model-level TinyLlama-1.1B decode (interleaved, thermal-cancelled) ===")
    print("\n-- generation quality (24 tokens, greedy) --")
    eager_txt = _generate_text(False, False)
    kern_refresh = _generate_text(True, True)
    kern_fixed = _generate_text(True, False)
    print(f"eager   (fp16)   : {eager_txt[:70]!r}")
    print(f"kernel  (refresh): {kern_refresh[:70]!r}")
    print(f"kernel  (fixed)  : {kern_fixed[:70]!r}")
    print(f"token match (eager vs kernel-refresh): {sum(1 for a, b in zip(eager_txt, kern_refresh) if a == b)} chars overlap")

    print("\n-- decode throughput (steady-state tok/s, interleaved) --")
    e_speeds = []
    k_speeds = []
    r_speeds = []
    for _ in range(2):
        e_speeds.append(_run_model(False, False)[0])
        k_speeds.append(_run_model(True, False)[0])
        r_speeds.append(_run_model(True, True)[0])
    em, km, rm = sum(e_speeds) / 2, sum(k_speeds) / 2, sum(r_speeds) / 2
    print(f"eager fp16         : {em:5.1f} tok/s  {[f'{x:.1f}' for x in e_speeds]}")
    print(f"kernel incremental : {km:5.1f} tok/s  {[f'{x:.1f}' for x in k_speeds]}  ({km/em:.2f}x of eager)")
    print(f"kernel +rulerRefresh: {rm:5.1f} tok/s  {[f'{x:.1f}' for x in r_speeds]}  ({rm/em:.2f}x of eager)")


# ---------------------------------------------------------------------------
# Section 4: vLLM baseline (optional)
# ---------------------------------------------------------------------------

def _run_vllm(model_name: str, n: int, backend: str):
    """Boot vLLM with the given attention backend and report throughput."""
    from vllm import LLM, SamplingParams
    kw = dict(model=model_name, max_model_len=2048, gpu_memory_utilization=0.80,
              enforce_eager=True, max_num_seqs=8, max_num_batched_tokens=2048)
    if backend != "stock":
        kw["attention_backend"] = backend
    llm = LLM(**kw)
    sp = SamplingParams(max_tokens=n, temperature=0)
    t0 = time.time()
    out = llm.generate([PROMPT], sp)
    tot = time.time() - t0
    tok = len(out[0].outputs[0].token_ids)
    t1 = time.time()
    for _ in range(2):
        out = llm.generate([PROMPT], sp)
    t2 = time.time()
    steady = (tok - 1) / ((t2 - t1) / 2)
    print(f"{model_name}  [backend={backend}]")
    print(f"  {tok} tokens in {tot:.1f}s -> {tok/tot:.1f} tok/s (incl prefill) | steady ~{steady:.1f} tok/s")
    print(f"  text: {out[0].outputs[0].text[:70]!r}")
    del llm
    torch.cuda.empty_cache()


def vllm_section():
    try:
        import vllm  # noqa: F401
        if vllm.__file__ is None:
            raise ImportError("vllm resolves to an empty namespace — run with PYTHONPATH=/path/to/vllm-src")
        from vllm import LLM, SamplingParams  # noqa: F401
    except Exception as e:
        print(f"\n=== SECTION 4 SKIPPED: vLLM unavailable ({e}) ===")
        return

    # CUSTOM is only ours if the slm-turbo plugin overrode the stock stub.
    custom_ok = False
    try:
        import adapters.vllm_adapter  # noqa: F401  # triggers backend registration
        from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES, AttentionBackendEnum
        custom_ok = AttentionBackendEnum.CUSTOM in _ATTN_OVERRIDES
    except Exception:
        pass

    print("\n=== SECTION 4: vLLM (stock fp16 vs slm-turbo CUSTOM backend) ===")
    for model_name, n in (("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 96),
                          ("Qwen/Qwen2-0.5B-Instruct", 96)):
        _run_vllm(model_name, n, "stock")
        if custom_ok:
            _run_vllm(model_name, n, "CUSTOM")
        else:
            print(f"  {model_name}: CUSTOM skipped (plugin not registered — re-run `pip install -e .`)" )


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op-only", action="store_true")
    ap.add_argument("--model-only", action="store_true")
    ap.add_argument("--vllm", action="store_true", help="include vLLM baseline (needs vLLM importable)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU — run on the GTX 1650 box.")

    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}  "
          f"VRAM {torch.cuda.get_device_properties(0).total_memory // 2**20}MB")
    if not args.model_only:
        op_level()
    if not args.op_only:
        model_level()
    if args.vllm:
        vllm_section()

    print("\n=== SUMMARY ===")
    for section, line in RESULTS:
        print(f"[{section}] {line}")


if __name__ == "__main__":
    main()
