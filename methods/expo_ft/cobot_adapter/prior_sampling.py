"""Fixed demonstration quota per Critic minibatch; Actor sampling is untouched."""
import numpy as np


def draw_indices(buffer, total, seed_steps, minibatch=64):
    import jax
    if total <= 0 or minibatch <= 0 or minibatch % 2 or total % minibatch:
        raise ValueError('demo50 requires complete even-sized Critic minibatches')
    # Seed is the immutable prefix inserted by Learner.rebuild(). HIL/success
    # flags cannot identify source: successful online/HIL samples remain online.
    window = buffer._replan_steps
    seed_stop = seed_steps - window
    online_stop = len(buffer) - window
    if seed_stop <= 0 or online_stop <= seed_steps:
        raise ValueError('demo50 requires both demonstration and online windows')
    if getattr(buffer, '_insert_index', len(buffer)) < len(buffer):
        raise ValueError('demo50 requires chronological Replay without ring wrap')
    rng = getattr(buffer, 'rng', None)
    if rng is None:
        rng = jax.random.PRNGKey(buffer._seed or 42)
    next_rng, demo_key, online_key = jax.random.split(rng, 3)
    shape = (total // minibatch, minibatch // 2)
    demo = np.asarray(jax.random.randint(demo_key, shape, 0, seed_stop))
    online = np.asarray(jax.random.randint(online_key, shape, seed_steps, online_stop))
    indices = np.concatenate((demo, online), axis=1).reshape(-1)
    buffer.rng = next_rng
    buffer.prior_sampling_stats = dict(mode='demo50', demonstration_fraction=.5,
        minibatch=minibatch, demo_per_minibatch=minibatch//2,
        online_per_minibatch=minibatch//2, sampled=total, seed_steps=seed_steps,
        online_steps=len(buffer)-seed_steps)
    return indices


def sample_critic(buffer, total, seed_steps, minibatch=64):
    from stream_sample import functions
    _, gather = functions(buffer)
    indices = draw_indices(buffer, total, seed_steps, minibatch)
    return gather(buffer, total, indices=indices)
