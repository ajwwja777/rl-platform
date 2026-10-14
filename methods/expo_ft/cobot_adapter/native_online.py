"""Use xiqing's native RealTimeEXPOFTLearner with Cobot's 7D/two-view contract."""
import json
import sys
from pathlib import Path
import numpy as np

CAMERAS = ("left_wrist_0_rgb", "right_wrist_0_rgb")


def source_path(project):
    project = Path(project).resolve()
    for path in (project, project / "expo_ft/agents/vla/openpi/src",
                 project / "expo_ft/agents/vla/openpi/packages/openpi-client/src"):
        sys.path.insert(0, str(path))


def configuration(bundle, delay, aligned=False, edit_scale=None):
    from configs.model.realtime_expo_ft_pi_config import get_config
    from configs.openpi.cobot_pi05 import CONFIG_NAME, register
    register()
    config = get_config()
    config.pi05_config_name = CONFIG_NAME
    config.pi05_assets_dir = str(bundle)
    config.pi05_asset_id = "."
    config.pi05_weight_loader_path = str(bundle / "checkpoints/2000/params")
    if aligned:
        config.N=32
        config.n_edit_samples=32
        config.filter_N=32
        config.filter_n_edit=1
        config.edit_scale=0.1
        config.valids_keep_terminal_windows=True
    if edit_scale is not None:
        from sampling_config import edit_scale as validate
        if not aligned:raise ValueError("Custom edit scale requires native_sync")
        config.edit_scale=validate(edit_scale)
    config.delay = delay
    config.critic_camera_keys = CAMERAS
    return config


def load_native(bundle, delay=4, resume=False, aligned=False, edit_scale=None):
    import jax
    import openpi.training.sharding as sharding
    from configs.task.cobot_plug import get_config as task_config
    from expo_ft.agents.vla.pi05 import build_pi05
    from expo_ft.agents.alg.realtime_expo_ft import load_agent
    contract = json.loads((bundle / "deployment.json").read_text())
    if (contract["active_arm"], contract["control_hz"], contract["action_horizon"]) != ("left", 10, 16):
        raise ValueError("EXPO online requires the verified left-arm10Hz/H16 bundle")
    if jax.default_backend() != "gpu" or jax.device_count() != 1:
        raise RuntimeError("Reserve exactly one GPU for native EXPO online")
    config = configuration(bundle, delay, aligned, edit_scale)
    mesh = sharding.make_mesh(1)
    data = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    prompt = task_config().language_instruction
    actor, state, _, kwargs, metadata = build_pi05(config, 42, mesh, data, replicated,
                                                  resume, prompt, initialize_target=False)
    actor.action_dim, actor.state_dim = 7, 7
    target = actor.get_params(state)
    agent = load_agent(42, np.zeros((224,224,6), np.float32), np.zeros((16,7), np.float32),
                       np.zeros(7, np.float32), actor, state, target, kwargs, metadata,
                       mesh, data, replicated, resume, 8, prompt, False)
    return agent, config, prompt


def replay_buffer(config, prompt, capacity=12000):
    from expo_ft.data.replay_buffer import create_replay_buffer
    return create_replay_buffer(config, np.zeros((16,7), np.float32), capacity, prompt,
                                8, 42, delay=config.delay, critic_camera_keys=CAMERAS)


def insert_episode(buffer,data,terminal_observation=True):
    from online_protocol import transitions,KEYS
    for row in transitions(data):buffer.insert(row)
    if not terminal_observation:return
    # Native Replay obtains s[t+r] from the next stored row. This observed final
    # state exposes the last complete reward window without waiting for another
    # episode. Its own action window is a truncation and is never sampled.
    terminal={key:data['terminal_'+key] for key in KEYS}
    buffer.insert(dict(observations=terminal,actions=data['actions'][-1],rewards=0.,
                       masks=1.,dones=True,is_hil=False,is_success=False))


def valid_batch(buffer,size,success_only=False):
    import jax
    gathered=[]
    count=0
    for _ in range(32):
        raw=buffer.sample_jax(max(size,4),success_only=success_only)
        if raw is None:return None
        valid=np.flatnonzero(np.asarray(raw['valids'])>.5)[:size-count]
        if len(valid):
            gathered.append(jax.tree.map(lambda x:x[valid],raw))
            count+=len(valid)
        if count==size:
            batch=jax.tree.map(lambda *x:jax.numpy.concatenate(x,axis=0),*gathered)
            return buffer._convert_to_openpi_format(batch)
    return None
