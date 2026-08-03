import torch

#  compute_scale_zero takes each col/feature values and return the scale and  #   zero_point, so that we can map those in 0-15 values as 4 bit
def compute_scale_zero(x: torch.Tensor, bits: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    q_min = 0                
    q_max = (1 << bits) - 1   
    
    # For each column (channel): find its smallest and largest value
    x_min = x.min(dim=1).values   
    x_max = x.max(dim=1).values   
    
    # 0-15 marks
    scale = (x_max - x_min) / (q_max - q_min) 
    
    # find mark for 0.0
    zero_point = q_min - torch.round(x_min / scale)   

    # a col where max and min are same 
    is_constant = x_max == x_min
    scale = torch.where(is_constant, torch.ones_like(scale), scale)
    zero_point = torch.where(is_constant, torch.zeros_like(zero_point), zero_point)

    
    zero_point = zero_point.clamp(q_min, q_max)

    return scale.half(), zero_point.half()  
    
# we pack the values    
def quantize_tensor(x, scale, zero_point, bits=4):
    q_max = (1 << bits) - 1                     

    # Make the rulers line up with the grid's rows 
    scale = scale.unsqueeze(1)                 
    zero_point = zero_point.unsqueeze(1)        

    # how many steps from zero
    q = torch.round(x / scale) + zero_point     
    q = q.clamp(0, q_max)
    q = q.to(torch.uint8)
                           
    # pack pairs of tags into single bytes 
    q = q.view(x.shape[0], x.shape[1], x.shape[2] // 2, 2) 
    packed = q[..., 0] | (q[..., 1] << 4)       

    return packed
# deqauntize so that we can use in sm 
def dequantize_tensor(packed, scale, zero_point, bits=4):
    evens = packed & 0x0F
    odds = (packed >> 4) & 0x0F
    q = torch.stack([evens, odds], dim=-1).view(
        packed.shape[0], packed.shape[1], -1
    )
    scale = scale.unsqueeze(1)
    zero_point = zero_point.unsqueeze(1)
    return (q.float() - zero_point.float()) * scale.float()


def quantize_kv(k, v, bits_k=4, bits_v=4):
    k_scale, k_zero = compute_scale_zero(k, bits_k)
    k_packed = quantize_tensor(k, k_scale, k_zero, bits_k)

    v_scale, v_zero = compute_scale_zero(v, bits_v)
    v_packed = quantize_tensor(v, v_scale, v_zero, bits_v)

    return k_packed, k_scale, k_zero, v_packed, v_scale, v_zero

