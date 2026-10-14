"""Bound activation memory while preserving full-batch losses and optimizer steps.

Map deterministic, example-independent network forwards over microbatches.
Random augmentations/noise and global loss reductions remain in the native
caller. Rematerialization recomputes each microbatch during reverse-mode AD.
No per-microbatch optimizer step, clipping or EMA is introduced.
"""
import jax


def mapped(function, arrays, size=4, *, remat=False):
    leaves = jax.tree.leaves(arrays)
    count = leaves[0].shape[0]
    if count <= size:
        return function(arrays)
    if count % size or any(x.shape[0] != count for x in leaves):
        raise ValueError('Inconsistent mapped batch dimensions')
    chunks = jax.tree.map(lambda x: x.reshape((count // size, size) + x.shape[1:]), arrays)
    body = jax.checkpoint(function) if remat else function
    result = jax.lax.map(body, chunks)
    return jax.tree.map(lambda x: x.reshape((count,) + x.shape[2:]), result)


def install(size=4):
    from expo_ft.agents.vla import pi05
    from expo_ft.agents.alg import realtime_expo_ft as alg
    if getattr(pi05, '_cobot_memory_mapping', False):
        return
    velocity = pi05._forward_get_velocity
    encode = alg.batch_encode
    sampler = pi05._jitted_sample_with_prefix.__wrapped__

    def bounded_velocity(model, observation, x_t, time):
        return mapped(lambda values: velocity(model, *values),
                      (observation, x_t, time), size, remat=True)

    def bounded_encode(encoder_fn, params, observations, stop_gradient=False):
        return mapped(lambda x: encode(encoder_fn, params, x, stop_gradient),
                      observations, size, remat=not stop_gradient)

    def bounded_sampler(inputs, state, rng, prefix, num_samples, noise=None):
        if num_samples != 1 or noise is None:
            return sampler(inputs, state, rng, prefix, num_samples, noise)
        return mapped(lambda values: sampler(values[0], state, rng, values[1], 1, values[2]),
                      (inputs, prefix, noise), size)

    pi05._forward_get_velocity = bounded_velocity
    pi05._jitted_sample_with_prefix = jax.jit(bounded_sampler, static_argnames=('num_samples',))
    alg.batch_encode = bounded_encode
    pi05._cobot_memory_mapping = True
