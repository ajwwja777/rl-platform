"""Physical-time action diagnostics; no invented samples beyond Episode end."""
import numpy as np


def recover_uint8_image(normalized):
    """Recover the loader's exact uint8 pixels before service-side normalization."""
    normalized = np.asarray(normalized)
    if not np.isfinite(normalized).all() or np.any(np.abs(normalized) > 1 + 2e-7):
        raise ValueError('expected finite image pixels normalized to [-1, 1]')
    pixels = np.rint((normalized.astype(np.float64) + 1) * 127.5).astype(np.uint8)
    rebuilt = pixels.astype(np.float32) / 255.0 * 2.0 - 1.0
    if np.max(np.abs(rebuilt - normalized)) > 2e-7:
        raise ValueError('image does not lie on the uint8 normalization grid')
    return pixels


def interpolate_chunk(chunk, source_hz, target_hz, length):
    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.ndim < 2 or min(source_hz, target_hz, length) <= 0:
        raise ValueError('positive sampling rates and action sequence required')
    positions = np.arange(length, dtype=float) * source_hz / target_hz
    if positions[-1] > chunk.shape[-2] - 1:
        raise ValueError('model horizon cannot cover requested physical time')
    lower = positions.astype(int); upper = np.ceil(positions).astype(int)
    weight = (positions-lower)[..., None]
    return (chunk[..., lower, :] * (1-weight) + chunk[..., upper, :] * weight).astype(np.float32)


def time_targets(actions, source_hz, target_hz, length):
    actions = np.asarray(actions, dtype=np.float32)
    positions = np.arange(len(actions))[:, None] + np.arange(length)[None, :] * source_hz / target_hz
    valid = positions <= len(actions)-1
    positions = np.minimum(positions, len(actions)-1)
    lo = positions.astype(int); hi = np.ceil(positions).astype(int)
    w = (positions-lo)[..., None]
    values = actions[lo]*(1-w) + actions[hi]*w
    return values.astype(np.float32), valid


def action_summary(predictions, targets, valid):
    predictions = np.asarray(predictions, float)
    targets = np.asarray(targets, float)
    valid = np.asarray(valid, bool)
    if predictions.shape != targets.shape or predictions.shape[:-1] != valid.shape or predictions.shape[-1] != 7:
        raise ValueError('aligned N,H,7 action arrays and N,H valid mask required')
    if not valid.any() or not np.isfinite(predictions).all() or not np.isfinite(targets).all():
        raise ValueError('finite action predictions and real future samples required')
    residual = predictions-targets
    selected = residual[valid]
    return dict(valid_action_slots=int(valid.sum()), padded_slots_excluded=int((~valid).sum()),
        mae_per_dim=np.abs(selected).mean(0).tolist(), rmse_per_dim=np.sqrt((selected**2).mean(0)).tolist(),
        bias_per_dim=selected.mean(0).tolist(), p95_abs_per_dim=np.quantile(np.abs(selected), .95, axis=0).tolist(),
        p99_abs_per_dim=np.quantile(np.abs(selected), .99, axis=0).tolist(),
        horizon_mae_per_dim=[np.abs(residual[:, h][valid[:, h]]).mean(0).tolist() if valid[:, h].any() else None
                             for h in range(valid.shape[1])])
