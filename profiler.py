import pynvml
import torch

from schema import DeviceProfile, ModelTopology, ProfileSnapshot


def device_info(device_id: int = 0) -> DeviceProfile:
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available")
    device = torch.cuda.get_device_properties(device_id)
    sm_version = device.major * 10 + device.minor

    return DeviceProfile(
        name=device.name,
        sm_version=device.major * 10 + device.minor,
        memory_bandwidth_gbps=get_bandwidth(device_id),
        vram_mb=device.total_memory // 1024 // 1024,
        sm_count=device.multi_processor_count,
        ## update these two
        supports_tensor_cores=sm_version >= 80,
        supports_async_copy=sm_version >= 80,
    )


## threoitical b/w , change it later
def get_bandwidth(device_id: int) -> float:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    mem_clock_mhz = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
    bus_width_bits = pynvml.nvmlDeviceGetMemoryBusWidth(handle)

    bandwidth = 2.0 * mem_clock_mhz * (bus_width_bits / 8.0) / 1000.0
    pynvml.nvmlShutdown()
    return bandwidth


def profile_analytical(topology: ModelTopology, seq_len: int = 1024) -> ProfileSnapshot:

    # rough estimate for decoder-only transformer
    model_params = topology.num_params
    if model_params is None:
        model_params = 12 * (topology.hidden_size**2) * topology.num_hidden_layers

    # total size
    model_size_bytes = model_params * get_dtype_size(topology)

    # kv cache for full seq/prompt (from the model topology)
    kv_total_bytes = topology.kv_cache_size_per_token * seq_len

    # memeroy traffic
    # for prefill we have full model , plus write kv to vram
    prefill_mem = model_size_bytes + kv_total_bytes
    # for decode model size , with prev kvs read, plus new kv  write
    decode_mem = model_size_bytes + kv_total_bytes + topology.kv_cache_size_per_token

    # FLOPs (2 coz each given prompt llm do 1 multiply of that prompt with params weights and 1 addition of mul with bias) also prefil work with full seq len
    prefill_flops = 2 * seq_len * model_params
    decode_flops = 2 * model_params

    return ProfileSnapshot(
        prefill_flops=prefill_flops,
        decode_flops=decode_flops,
        prefill_memory_bytes=prefill_mem,
        decode_memory_bytes=decode_mem,
        model_params=model_params,
        model_size_bytes=model_size_bytes,
        kv_cache_bytes_per_token=topology.kv_cache_size_per_token,
    )


def get_dtype_size(topology: ModelTopology) -> float:
    # standard dtype fallback
    DTYPE_TO_BYTES = {
        "float32": 4.0,
        "float16": 2.0,
        "bfloat16": 2.0,
    }
    dtype_bytes = DTYPE_TO_BYTES.get(topology.dtype, 2.0)

    # for quantized models
    if topology.quantization_config is not None:
        qc = topology.quantization_config
        QUANT_METHOD_TO_BYTES: dict[str, float] = {
            "fp8": 1.0,
            "awq": 0.5,
            "gptq": 0.5,
            "gguf": 0.5,
            "bitsandbytes": 0.5,
        }
        quant_method: str | None = qc.get("quant_method")
        if quant_method is None:
            return dtype_bytes
        return QUANT_METHOD_TO_BYTES.get(quant_method, dtype_bytes)

    return dtype_bytes
