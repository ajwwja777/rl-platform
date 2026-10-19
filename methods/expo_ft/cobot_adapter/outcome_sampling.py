"""Episode-outcome stratification: 50% demonstration, 30% success, 20% failure."""
import numpy as np


def pools(buffer, seed_steps, episodes):
    """episodes contains chronological (transition count, explicit outcome) pairs."""
    window = buffer._replan_steps
    result = [np.arange(max(0, seed_steps-window), dtype=np.int64), [], []]
    offset = seed_steps
    for count, outcome in episodes:
        if outcome not in ('success', 'failure'):
            raise ValueError('Online sampling requires explicit episode outcome')
        # Exclude backup windows crossing episode boundaries, including pool boundaries.
        result[1 if outcome == 'success' else 2].extend(range(offset, max(offset, offset+count-window)))
        offset += count
    if offset != len(buffer):
        raise ValueError('Outcome pool does not match Replay length')
    if getattr(buffer, '_insert_index', len(buffer)) < len(buffer):
        raise ValueError('Outcome sampling requires chronological Replay without ring wrap')
    return tuple(np.asarray(x, dtype=np.int64) for x in result)


def deferred_schedule(schedule, calls, available, interval=30):
    """Retain earned updates when either outcome has not been observed yet."""
    if not calls or all(len(p) for p in available):
        return schedule, calls
    from dataclasses import replace
    return replace(schedule, credit=schedule.credit+calls*interval), 0


def draw_indices(buffer, total, available, minibatch=64):
    import jax
    if minibatch != 64 or total <= 0 or total % 320:
        raise ValueError('532 sampling requires groups of five 64-row minibatches')
    if not all(len(p) for p in available):
        raise ValueError('532 sampling awaits demonstration, online success and online failure pools')
    rng = getattr(buffer, 'rng', None)
    if rng is None: rng = jax.random.PRNGKey(buffer._seed or 42)
    keys = jax.random.split(rng, 4)
    counts = (total//2, total*3//10, total//5)
    draws = [np.asarray(p)[np.asarray(jax.random.randint(k, (n,), 0, len(p)))]
             for p,k,n in zip(available,keys[1:],counts)]
    rows=[]; positions=[0,0,0]
    for i in range(total//64):
        success = ((i+1)*96)//5-(i*96)//5  # 19,19,19,19,20 per five minibatches
        quota=(32,success,32-success)
        rows.append(np.concatenate([x[pos:pos+n] for x,pos,n in zip(draws,positions,quota)]))
        positions=[pos+n for pos,n in zip(positions,quota)]
    buffer.rng=keys[0]
    buffer.prior_sampling_stats=dict(mode='demo50_success30_failure20',sampled=total,
        demonstration_samples=counts[0],online_success_samples=counts[1],online_failure_samples=counts[2],
        minibatch=64,demo_per_minibatch=32,success_per_minibatch=[19,20],failure_per_minibatch=[12,13],
        pool_sizes=[len(p) for p in available])
    return np.concatenate(rows)


def sample_critic(buffer, total, available):
    from stream_sample import functions
    _,gather=functions(buffer)
    return gather(buffer,total,indices=draw_indices(buffer,total,available))
