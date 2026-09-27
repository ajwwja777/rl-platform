
# Training-time RTC for fixed openpi-RLT Pi05; upstream remains unchanged.
# time=1 is noise, time=0 is clean. Prefixes are normalized/anchored actions.
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import nnx
from einops import einops
from openpi.models import gemma, pi0
from openpi.models import model as model_api
MAX_DELAY = 6
_INSTALLED = False
_ORIGINAL = {}

def prefix_mask(lengths, horizon):
    lengths = jnp.asarray(lengths, dtype=jnp.int32)
    return jnp.arange(horizon)[None, :] < lengths[:, None]

def training_inputs(rng, actions, max_delay):
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * .999 + .001
    delays = jax.random.randint(jax.random.fold_in(rng, 20260918),
                               actions.shape[:-2], 0, max_delay + 1)
    mask = prefix_mask(delays, actions.shape[-2])
    times = jnp.where(mask, 0., time[:, None])
    noisy = times[..., None] * noise + (1. - times[..., None]) * actions
    return preprocess_rng, noisy, noise - actions, times, mask

def masked_flow_loss(predicted, target, mask):
    loss = jnp.mean(jnp.square(predicted - target), axis=-1)
    postfix = ~mask
    scale = loss.shape[-1] / jnp.maximum(jnp.sum(postfix, axis=-1), 1)
    return jnp.where(postfix, loss * scale[:, None], 0.)

class RTCRMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x, cond):
        dtype = x.dtype
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normalized = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-6)))
        if cond is None:
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
            return (normalized * (1 + scale)).astype(dtype), None
        modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype)(cond)
        if modulation.ndim == 2:
            modulation = modulation[:, None, :]
        if modulation.ndim != 3:
            raise ValueError("RTC conditioning must be batch-level or per action")
        scale, shift, gate = jnp.split(modulation, 3, axis=-1)
        return (normalized * (1 + scale) + shift).astype(dtype), gate

class RTCGemmaModule(gemma.Module):
    def __call__(self, embedded, positions, mask, adarms_cond=None, *,
                 kv_cache=None, deterministic=True):
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)
        embedded, kv_cache = self.layers(embedded, kv_cache, positions, mask,
                                        adarms_cond, deterministic)
        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)
        outputs = [f(e, a)[0] if e is not None else e
                   for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)]
        return outputs, kv_cache

def embed_suffix(self, observation, noisy_actions, timestep):
    if timestep.ndim == 1:
        return _ORIGINAL["embed_suffix"](self, observation, noisy_actions, timestep)
    if not self.pi05 or timestep.shape != noisy_actions.shape[:2]:
        raise ValueError("per-action RTC timesteps require Pi05 and [batch,horizon]")
    action_tokens = self.action_in_proj(noisy_actions)
    batch, horizon = timestep.shape
    time_emb = pi0.posemb_sincos(timestep.reshape(-1), self.action_in_proj.out_features,
                                min_period=4e-3, max_period=4.)
    time_emb = time_emb.reshape(batch, horizon, -1)
    time_emb = nnx.swish(self.time_mlp_in(time_emb))
    time_emb = nnx.swish(self.time_mlp_out(time_emb))
    input_mask = jnp.ones(action_tokens.shape[:2], dtype=jnp.bool_)
    ar_mask = jnp.array([True] + [False] * (horizon - 1))
    return action_tokens, input_mask, ar_mask, time_emb

def flow_loss(self, rng, observation, actions, *, train=False, image_only=False):
    preprocess_rng, noisy, target, times, locked = training_inputs(rng, actions, MAX_DELAY)
    observation = model_api.preprocess_observation(preprocess_rng, observation, train=train)
    prefix, p_mask, p_ar = self.embed_prefix(observation)
    suffix, s_mask, s_ar, cond = self.embed_suffix(observation, noisy, times)
    mask = jnp.concatenate([p_mask, s_mask], axis=1)
    ar = jnp.concatenate([p_ar, s_ar], axis=0)
    attention = pi0.make_attn_mask(mask, ar)
    positions = jnp.cumsum(mask, axis=1) - 1
    (prefix_out, suffix_out), _ = self.PaliGemma.llm(
        [prefix, suffix], mask=attention, positions=positions,
        adarms_cond=[None, cond])
    prediction = self.action_out_proj(suffix_out[:, -self.action_horizon:])
    loss = masked_flow_loss(prediction, target, locked)
    if image_only:
        language_count = observation.tokenized_prompt.shape[1] if observation.tokenized_prompt is not None else 0
        count = prefix.shape[1] - language_count
        prefix_out, p_mask = prefix_out[:, :count], p_mask[:, :count]
    return loss, prefix_out, p_mask

def compute_loss(self, rng, observation, actions, *, train=False):
    if MAX_DELAY == 0:
        return _ORIGINAL["compute_loss"](self, rng, observation, actions, train=train)
    return flow_loss(self, rng, observation, actions, train=train)[0]

def compute_loss_with_prefix(self, rng, observation, actions, *, train=False, image_only=False):
    if MAX_DELAY == 0:
        return _ORIGINAL["compute_loss_with_prefix"](self, rng, observation, actions,
                                                    train=train, image_only=image_only)
    return flow_loss(self, rng, observation, actions, train=train, image_only=image_only)

def sample_actions_from_prefix_cache(self, rng, cache, *, num_steps=10, noise=None,
                                     action_prefix=None, prefix_lengths=None):
    if action_prefix is None:
        if prefix_lengths is not None:
            raise ValueError("prefix lengths without prefix actions")
        return _ORIGINAL["sample_actions_from_prefix_cache"](
            self, rng, cache, num_steps=num_steps, noise=noise)
    batch = cache.observation.state.shape[0]
    if action_prefix.shape != (batch, self.action_horizon, self.action_dim):
        raise ValueError("prefix must be padded to the model action shape")
    if prefix_lengths is None:
        raise ValueError("prefix actions require lengths")
    lengths = jnp.broadcast_to(jnp.asarray(prefix_lengths, jnp.int32), (batch,))
    if not isinstance(lengths, jax.core.Tracer):
        values = np.asarray(lengths)
        if (values < 0).any() or (values > MAX_DELAY).any():
            raise ValueError("prefix delay exceeds the trained range")
    locked = prefix_mask(lengths, self.action_horizon)
    if noise is None:
        noise = jax.random.normal(rng, action_prefix.shape)
    dt = -1. / num_steps
    def step(carry):
        x, time = carry
        x = jnp.where(locked[..., None], action_prefix, x)
        times = jnp.where(locked, 0., jnp.broadcast_to(time, locked.shape))
        suffix, mask, ar, cond = self.embed_suffix(cache.observation, x, times)
        attention = pi0.make_attn_mask(mask, ar)
        prefix_attention = einops.repeat(cache.prefix_mask, "b p -> b s p", s=mask.shape[1])
        full_attention = jnp.concatenate([prefix_attention, attention], axis=-1)
        positions = jnp.sum(cache.prefix_mask, axis=-1)[:, None] + jnp.cumsum(mask, axis=-1) - 1
        (_, output), _ = self.PaliGemma.llm(
            [None, suffix], mask=full_attention, positions=positions,
            kv_cache=cache.kv_cache, adarms_cond=[None, cond])
        velocity = self.action_out_proj(output[:, -self.action_horizon:])
        updated = x + dt * velocity
        return jnp.where(locked[..., None], action_prefix, updated), time + dt
    def cond(carry):
        return carry[1] >= -dt / 2
    result, _ = jax.lax.while_loop(cond, step, (noise, 1.))
    return jnp.where(locked[..., None], action_prefix, result)

def sample_actions(self, rng, observation, *, num_steps=10, noise=None,
                   action_prefix=None, prefix_lengths=None):
    if action_prefix is None:
        return _ORIGINAL["sample_actions"](self, rng, observation,
                                           num_steps=num_steps, noise=noise)
    cache = self.prepare_prefix_for_inference(observation)
    return self.sample_actions_from_prefix_cache(
        rng, cache, num_steps=num_steps, noise=noise,
        action_prefix=action_prefix, prefix_lengths=prefix_lengths)

def install(max_delay=6):
    global MAX_DELAY, _INSTALLED
    if not 0 <= max_delay < 50:
        raise ValueError("unsupported RTC training delay")
    MAX_DELAY = max_delay
    if _INSTALLED:
        return
    for name in ["embed_suffix", "compute_loss", "compute_loss_with_prefix",
                 "sample_actions", "sample_actions_from_prefix_cache"]:
        _ORIGINAL[name] = getattr(pi0.Pi0, name)
        setattr(pi0.Pi0, name, globals()[name])
    gemma.RMSNorm = RTCRMSNorm
    gemma.Module = RTCGemmaModule
    _INSTALLED = True
