from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ModelTopology(BaseModel):
    model_id: str
    hidden_size: int
    num_attention_heads: int = Field(..., gt=0)
    num_kv_heads: int = Field(..., gt=0)
    head_dim: int
    num_hidden_layers: int
    num_params: Optional[int] = None
    sliding_window: Optional[int] = None
    max_position_embeddings: int
    dtype: str
    architectures: List[str]
    is_vlm: bool = False
    quantization_config: Optional[Dict[str, Any]] = None

    ## for gqa models use less heads than q heads
    @property
    def is_gqa(self) -> bool:
        return self.num_kv_heads < self.num_attention_heads

    ## calculate kv cache size per token
    @property
    def kv_cache_size_per_token(self) -> int:
        dtype_map = {"float32": 4, "float16": 2, "bfloat16": 2}
        bytes_per_element = dtype_map.get(self.dtype, 2)

        return (
            2
            * self.num_kv_heads
            * self.head_dim
            * self.num_hidden_layers
            * bytes_per_element
        )


class DeviceProfile(BaseModel):
    name: str
    sm_version: int
    memory_bandwidth_gbps: float
    vram_mb: int
    sm_count: int
    supports_tensor_cores: bool
    supports_async_copy: bool

    ## 6gb rough guess for now for slms
    @property
    def is_small_gpu(self) -> bool:
        return self.vram_mb < 6000 and not self.supports_tensor_cores


class OptimizationStep(BaseModel):
    name: str
    enabled: bool = True
    params: Dict[str, Any] = {}
    target_phase: Literal["prefill", "decode", "both"] = "both"
    expected_delta: Dict[str, float] = {}


class Recipe(BaseModel):
    version: str = "1.0"
    topology_hash: str
    device_hash: str
    steps: List[OptimizationStep] = []

    ## check compatibilty so that when we change gpu/model it should be same as before ,
    ## we check this before serving the model
    def validate_compatibility(
        self,
        topology: ModelTopology,
        device: DeviceProfile,
    ) -> bool:
        return self.topology_hash == _hash_dict(
            topology.model_dump()
        ) and self.device_hash == _hash_dict(device.model_dump())


def _hash_dict(data: dict) -> str:
    import hashlib
    import json

    payload = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


class ProfileSnapshot(BaseModel):
    prefill_flops: float
    decode_flops: float  # flops per decode token
    prefill_memory_bytes: float
    decode_memory_bytes: float
    model_params: int
    model_size_bytes: float
    kv_cache_bytes_per_token: int
