from schema import Bottleneck, DeviceProfile, ProfileSnapshot


# TODO: Update with empheircal
def _estimate_peak_compute(device: DeviceProfile) -> float:
    # ops per SM per clock by architecture(will need to fully remove when we opt tensorrt technique)
    ops_per_sm = {
        75: 128,  # Turing
        80: 256,  # Ampere
        86: 256,  # Ampere
        89: 512,  # Ada Lovelace
        90: 512,  # Hopper
    }.get(device.sm_version, 128)

    clock_ghz = 1.5  # rough average
    return device.sm_count * ops_per_sm * clock_ghz


def classify_bottleneck(snapshot: ProfileSnapshot, device: DeviceProfile) -> Bottleneck:
    peak_compute = _estimate_peak_compute(device)
    # ridge point where bottlenech beggins
    ridge_point = peak_compute / device.memory_bandwidth_gbps

    # intensity tells us how good kernels are at utilizing compute resources// generally prefill:- high intensity , decode:- low intensity
    prefill_intensity = snapshot.prefill_flops / snapshot.prefill_memory_bytes
    decode_intensity = snapshot.decode_flops / snapshot.decode_memory_bytes

    # on the left side of ridge is mem bound and right side is compute bound
    return Bottleneck(
        prefill="compute_bound" if prefill_intensity > ridge_point else "memory_bound",
        decode="compute_bound" if decode_intensity > ridge_point else "memory_bound",
    )
