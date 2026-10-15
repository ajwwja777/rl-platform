"""Physical buffer identity for fully addressable single-host JAX arrays."""

def buffer_identity(value):
    # Global pointer APIs reject multi-device arrays. Include every local shard,
    # device and slice so only exact aliases may share a donated argument.
    if not value.is_fully_addressable:
        raise ValueError("EXPO supports one host with fully addressable arrays")
    return (value.shape, str(value.dtype), tuple(
        (shard.device.id, str(shard.index), shard.data.unsafe_buffer_pointer())
        for shard in value.addressable_shards))
