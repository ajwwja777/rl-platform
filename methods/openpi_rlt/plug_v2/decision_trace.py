"""Exact delayed-action decision evidence; pure arrays, no robot I/O."""
import numpy as np

def decision_evidence(request, reference, actor_plan, conditioned, *, actor_key=None):
    arrays = {}
    for key, value in (("reference_plan", reference), ("raw_action_plan", actor_plan),
                       ("conditioned_plan", conditioned)):
        value = np.asarray(value, np.float32)
        if value.shape != (50, 14) or not np.isfinite(value).all():
            raise ValueError("invalid decision evidence " + key)
        arrays[key] = value.copy()
    d = int(request.prefix_length)
    if d not in (0, 6):
        raise ValueError("unsupported RTC delay")
    for key, value in arrays.items():
        if not np.array_equal(value[:d], request.prefix[:d]):
            raise ValueError("decision evidence changed committed prefix: " + key)
    return dict(trace_schema=2, actor_key=actor_key,
                decision_tick=int(request.start_tick),
                action_start_tick=int(request.start_tick)+d,
                action_stop_tick=int(request.start_tick)+d+10,
                nominal_next_decision_tick=int(request.start_tick)+(4 if d==0 else 10),
                **arrays)

def delayed_target(rewards, duration, bootstrap_value, *, gamma, terminal):
    """Decision-interval SMDP target. Duration counts executed ticks, not actor tail.

    Diagnostic identity only; it does not replace the upstream learner.
    """
    r=np.asarray(rewards,np.float64)
    if r.ndim!=1 or len(r)!=duration or duration<=0 or not np.isfinite(r).all():
        raise ValueError("reward span must equal actual decision interval")
    if not 0<gamma<=1 or not np.isfinite(bootstrap_value):
        raise ValueError("invalid discount/value")
    return float(r @ gamma**np.arange(duration) +
                 (0. if terminal else gamma**duration*bootstrap_value))
