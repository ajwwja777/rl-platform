"""Best-effort CPU diagnostic process. Never scores on a publication thread."""
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import queue
import time
from collections import deque


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.%s.tmp' % os.getpid())
    tmp.write_text(json.dumps(value, allow_nan=False))
    os.replace(tmp, path)


class CheckpointScorer:
    """One durable Critic checkpoint per episode; native normalization/network."""
    def __init__(self, config_path):
        from rlt_online_rl.config import load_system_config_yaml
        from rlt_online_rl.networks import TwinCritic
        from .online_runtime import CobotBimanualActionRepresentationAdapter
        import jax
        import jax.numpy as jnp
        import numpy as np
        config = load_system_config_yaml(config_path)
        self.path = Path(config.learner_service.checkpoint_dir)/'latest.pkl'
        with self.path.open('rb') as f:
            payload = pickle.load(f)  # Trusted local configured training artifact only.
        state = payload['state']
        self.version = int(state['global_step'])
        self.params = jax.tree.map(jnp.asarray, state['critic_params'])
        cfg = config.rl
        self.adapter = CobotBimanualActionRepresentationAdapter.from_config(cfg)
        self.critic = TwinCritic(cfg.z_dim, cfg.proprio_dim, cfg.chunk_len, cfg.action_dim,
                                 cfg.critic_hidden_dim, cfg.critic_num_layers)
        profile_path = os.environ.get('COBOT_RLT_SUPPORTED_PROFILE')
        if profile_path:
            profile = json.loads(Path(profile_path).read_text())
            if profile.get('schema') not in ('held-gripper-retention-v1', 'held-gripper-hil8-clip-v2'):
                raise ValueError('Unknown diagnostic Critic contract')
            if profile.get('held_gripper'):
                from methods.openpi_rlt.experiments.held_gripper import HeldGripperCritic
                self.critic = HeldGripperCritic(self.critic, float(self.adapter.stats.q01[6]), float(self.adapter.stats.q99[6]))
        self.score_fn = jax.jit(lambda z, p, a: self.critic.q_values(self.params, z, p, a)[0])
        self.np, self.jnp = np, jnp

    def score(self, event):
        np, jnp = self.np, self.jnp
        z, p, a = (np.asarray(event[k], np.float32) for k in ('z', 'proprio', 'actions'))
        if self.adapter is not None:
            a = self.adapter.normalize_chunk(a, p)
        value = float(self.score_fn(jnp.asarray(z[None]), jnp.asarray(p[None]), jnp.asarray(a[None]))[0])
        if not np.isfinite(value):
            raise ValueError('Nonfinite monitor Q')
        return value


def worker(events, root, config_path, parent_pid):
    # Spawn imports this dependency-light module before initializing JAX.
    os.environ.update(JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if hasattr(os, 'sched_setaffinity'):
        available = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, set(available[:2]))
    root = Path(root)/'telemetry'
    points = deque(maxlen=1500)
    current, scorer, last_dropped = None, None, 0
    while os.getppid() == parent_pid:
        try:
            event = events.get(timeout=1)
        except queue.Empty:
            continue
        if event is None:
            return
        kind = event['kind']
        if kind == 'begin':
            current = dict(schema=1, episode_uuid=event['episode_uuid'], actor_version=None,
                           critic_version=None, active=True, started_at=event['timestamp'],
                           semantics='adopted_chunk_q1_before_execution_filter', error=None)
            points.clear()
            scorer = None
            try:
                scorer = CheckpointScorer(config_path)
                current['critic_version'] = scorer.version
            except Exception as exc:
                current['error'] = type(exc).__name__ + ': ' + str(exc)
        if current is None or event['episode_uuid'] != current['episode_uuid']:
            continue
        if event.get('dropped', 0) > last_dropped:
            points.append(dict(seconds=max(0., event['timestamp']-current['started_at']), q=None, reason='monitor_queue_full'))
        last_dropped = event.get('dropped', 0)
        if kind == 'point':
            point = dict(seconds=max(0., event['timestamp']-current['started_at']),
                         timestamp=event['timestamp'], q=None, actor_version=event['actor_version'],
                         critic_version=current['critic_version'], source=event['source'])
            current['actor_version'] = event['actor_version']
            if scorer is not None and event['source'] == 1:
                try:
                    point['q'] = scorer.score(event)
                except Exception as exc:
                    current['error'] = type(exc).__name__ + ': ' + str(exc)
            points.append(point)
        elif kind == 'gap':
            points.append(dict(seconds=max(0., event['timestamp']-current['started_at']), q=None,
                               reason=event.get('reason', 'pause_or_hil')))
        elif kind == 'end':
            current['active'] = False
            summary = event['summary']
            root.mkdir(parents=True, exist_ok=True)
            with (root/'episodes.jsonl').open('a') as f:
                f.write(json.dumps(summary, allow_nan=False)+'\n')
        current.update(points=list(points), dropped_points=event.get('dropped', 0), updated_at=time.time())
        try:
            from integrations.cobot_runtime.rollout_archive import save_snapshot
            save_snapshot(root.parent, current)
            write_json(root/'rollout.json', current)
        except Exception:
            pass  # Monitoring storage cannot stop the actor/executor.


class AsyncQMonitor:
    def __init__(self, root, config_path):
        self.root = Path(root)
        context = mp.get_context('spawn')
        self.events = context.Queue(maxsize=32)
        self.process = context.Process(target=worker, args=(self.events, str(root), str(config_path), os.getpid()),
                                       name='rlt-q-monitor', daemon=True)
        self.process.start()
        self.identity = None
        self.started = None
        self.dropped = 0
        self.enabled = False
        self.phase = None

    def send(self, kind, **value):
        if not self.enabled:
            return
        try:
            self.events.put_nowait(dict(kind=kind, episode_uuid=self.identity,
                                        timestamp=time.time(), dropped=self.dropped, **value))
        except (queue.Full, OSError, ValueError):
            self.dropped += 1

    def begin(self, status, phase):
        self.enabled = status.get('session_use') == 'collection' and phase == 'online'
        self.identity = status.get('task5_episode_uuid')
        self.enabled = bool(self.enabled and self.identity)
        self.phase = phase
        self.started = time.time()
        self.send('begin')

    def adopted(self, plan, actions=None):
        if not self.enabled:
            return
        # CPU numpy arrays already produced by Actor RPC; no device synchronization.
        self.send('point', z=plan.start_features.z_rl.copy(), proprio=plan.start_features.proprio.copy(),
                  actions=(plan.action_chunk if actions is None else actions).copy(),
                  source=int(plan.source), actor_version=int(plan.actor_param_version))

    def end(self, status, result):
        count = sum(result.get(k, 0) for k in ('rl_steps', 'base_steps', 'human_steps', 'mixed_steps'))
        hil = result.get('human_steps', 0)+result.get('mixed_steps', 0)
        summary = dict(episode_uuid=self.identity, outcome=status.get('outcome'), count=count,
                       hil_steps=hil if result else None, collection_phase=self.phase,
                       session_use='collection', actor_version=result.get('actor_version_end'),
                       started_at=self.started, finished_at=time.time())
        # Terminal metadata is durable even if the optional scoring queue is full.
        # This seam runs after replay finalization, never on the publication loop.
        if self.enabled:
            try:
                write_json(self.root/'telemetry/episodes'/(self.identity+'.json'), summary)
            except OSError:
                pass
        self.send('end', summary=summary)
        self.enabled = False

    def close(self):
        self.enabled = False
        try:
            self.events.put_nowait(None)
        except (queue.Full, OSError, ValueError):
            pass
        self.process.join(timeout=.2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=.2)
        self.events.cancel_join_thread()
        self.events.close()


def attach(engine, driver):
    """Activate only for an explicitly training-enabled RLT runtime."""
    config = os.environ.get('COBOT_RLT_MONITOR_CONFIG')
    root = os.environ.get('RLT_OUTPUT_DIR')
    if not config or not root or not Path(config).is_file() or os.environ.get('RLT_DISABLE_LEARNER') == '1':
        return
    try:
        monitor = AsyncQMonitor(root, config)
    except Exception:
        return
    engine.q_monitor = monitor
    reset = driver._env.reset
    def observed_reset(*args, **kwargs):
        result = reset(*args, **kwargs)
        try:
            application = driver._env._io._session_application
            monitor.begin(application.status(), driver._current_collection_phase())
        except Exception:
            monitor.enabled = False
        return result
    driver._env.reset = observed_reset
