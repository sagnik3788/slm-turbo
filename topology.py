from huggingface_hub import model_info
from transformers import AutoConfig

from schema import ModelTopology


def build_topology(model_id: str) -> ModelTopology:
    hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    num_attention_heads = hf_config.num_attention_heads
    ## old models kv heads so we use attn heads
    num_kv_heads = getattr(hf_config, "num_key_value_heads", num_attention_heads)
    head_dim = getattr(
        hf_config, "head_dim", hf_config.hidden_size // num_attention_heads
    )
    architectures = getattr(hf_config, "architectures", []) or []

    num_params = getattr(model_info(model_id).safetensors, "total", None)

    return ModelTopology(
        model_id=model_id,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=hf_config.hidden_size,
        head_dim=head_dim,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_params=num_params,
        sliding_window=getattr(hf_config, "sliding_window", None),
        dtype=str(getattr(hf_config, "dtype", "float16")).replace("torch.", ""),
        max_position_embeddings=getattr(hf_config, "max_position_embeddings", 2048),
        architectures=architectures,
        # is_vlm=?
        quantization_config=getattr(hf_config, "quantization_config", None),
    )
