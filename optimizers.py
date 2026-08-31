from schema import (
    Bottleneck,
    DeviceProfile,
    ModelTopology,
    OptimizationStep,
    ProfileSnapshot,
)


def optimizers(
    device: DeviceProfile,
    topology: ModelTopology,
    snapshot: ProfileSnapshot,
    bottleneck: Bottleneck,
) -> list[OptimizationStep]:
    """Gen opt for (model,gpu) pair"""
    steps: list[OptimizationStep] = []

    # Check if model weights alone fit in GPU VRAM
    if snapshot.model_size_bytes > device.vram_mb * 1024 * 1024:
        return [
            OptimizationStep(
                name="model_too_large",
                enabled=False,
                params={
                    "model_size_gb": round(snapshot.model_size_bytes / 1e9, 1),
                    "gpu_vram_mb": device.vram_mb,
                    "reason": f"Model {snapshot.model_size_bytes / 1e9:.1f}GB > GPU {device.vram_mb}MB VRAM",
                },
                target_phase="both",
            )
        ]

    if Kv_quant_can_apply(topology, device, snapshot, bottleneck):
        steps.append(kv_quant_apply(topology, device))

    steps.append(attention_backend_apply(device))

    if prefix_cache_can_apply(topology, device):
        steps.append(prefix_cache_apply())

    if chunked_prefill_can_apply(topology):
        steps.append(chunked_prefill_apply(topology))

    return steps


# KV Quantization Optimizer
def Kv_quant_can_apply(
    topology: ModelTopology,
    device: DeviceProfile,
    snapshot: ProfileSnapshot,
    bottleneck: Bottleneck,
) -> bool:
    # case1 : small gpus
    if device.is_small_gpu:
        return True
    # case2 : if decode is mem bound and kv cache is large
    if bottleneck.decode == "memory_bound":
        kv_at_4k = (
            topology.kv_cache_size_per_token * 4096
        )  # max 4096 tokens (rough estimate)

        # if kv cache size at 4k tokens is larger than 30% of device VRAM, apply kv quant
        if kv_at_4k > 0.3 * device.vram_mb * 1024 * 1024:
            return True

    # case3 : if model size + kv cache  exceeds device VRAM, apply kv quant
    kv_at_max = topology.kv_cache_size_per_token * topology.max_position_embeddings
    if snapshot.model_size_bytes + kv_at_max > device.vram_mb * 1024 * 1024:
        return True

    return False


def kv_quant_apply(
    topology: ModelTopology,
    device: DeviceProfile,
) -> OptimizationStep:

    # vLLM 8bit, later give options to users
    # if device.sm_version >= 89:
    #     return OptimizationStep(
    #         name="kv_quant",
    #         enabled=True,
    #         params={"kv_cache_dtype": "fp8"},
    #         target_phase="decode",
    #         expected_delta={"memory": 0.5},
    #     )

    # only for custom kernel
    # most consumer gpus wont cross 500 gb/s bandwidth
    block_size = 64 if device.memory_bandwidth_gbps < 500 else 128

    return OptimizationStep(
        name="kv_quant",
        params={
            # 4 bit quantized (change as per need)
            "k_bits": 4,
            "v_bits": 4,
            "block_size": block_size,
            "per_channel": True,
        },
        target_phase="decode",
        enabled=True,
        expected_delta={"memory": 0.5, "latency": 0.5},
    )


def attention_backend_apply(device: DeviceProfile) -> OptimizationStep:
    backend = "custom_kernel"

    return OptimizationStep(
        name="attention_backend",
        enabled=True,
        params={"backend": backend},
        target_phase="both",
        # useless
        expected_delta={"throughput": 1.0},
    )


def prefix_cache_can_apply(topology: ModelTopology, device: DeviceProfile) -> bool:
    # not apply for large models and small gpus (hardcoded for now)
    if topology.num_hidden_layers > 80:
        return False
    if device.vram_mb < 4000:
        return False
    return True


def prefix_cache_apply() -> OptimizationStep:
    return OptimizationStep(
        name="prefix_cache",
        enabled=True,
        params={"enable": True, "eviction_policy": "lru"},
        target_phase="decode",
        expected_delta={"ttft": 0.7},
    )


def chunked_prefill_can_apply(topology: ModelTopology) -> bool:
    # TODO: make it dynamic
    return topology.max_position_embeddings > 4096


def chunked_prefill_apply(topology: ModelTopology) -> OptimizationStep:
    # TODO: make it dynamic
    chunk_size = 512 if topology.num_hidden_layers > 40 else 1024

    return OptimizationStep(
        name="chunked_prefill",
        enabled=True,
        params={"chunk_size": chunk_size},
        target_phase="prefill",
        expected_delta={"latency": 0.9},
    )
