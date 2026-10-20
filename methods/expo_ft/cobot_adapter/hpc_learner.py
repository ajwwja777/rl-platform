"""Run one persistent Learner inside the explicitly configured Slurm job step."""
import argparse
import fcntl
import json
import os
from pathlib import Path
from remote_launcher import learner_command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if os.environ.get('SLURM_JOB_ID') != str(config['job_id']):
        raise ValueError('Learner must run inside its configured Slurm allocation')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(visible) != 4 or len(set(visible)) != 4 or not all(visible):
        raise ValueError('Slurm must expose exactly four assigned GPUs')
    root = Path(config['run_dir'])
    if not root.is_absolute() or not (root / 'committed.json').is_file():
        raise ValueError('Prepared, verified continuation required')
    owner = (root / 'owner.lock').open('a')
    fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.set_inheritable(owner.fileno(), True)
    allocator = config.get('gpu_allocator', 'platform')
    fraction = config.get('memory_fraction')
    if allocator not in ('platform', 'default'):
        raise ValueError('Invalid HPC GPU allocator')
    if fraction is not None and (type(fraction) not in (int, float) or not 0 < fraction <= 1):
        raise ValueError('Invalid HPC memory fraction')
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
                'http_proxy', 'https_proxy', 'all_proxy'):
        env.pop(key, None)
    env.update(XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_ALLOCATOR='platform',
               OMP_NUM_THREADS='8', JAX_COMPILATION_CACHE_DIR=config['cache_dir'])
    if allocator == 'default':
        env.pop('XLA_PYTHON_CLIENT_ALLOCATOR', None)
    if fraction is not None:
        env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(fraction)
    command = learner_command(config, Path(__file__).resolve().parent) + ['--num-devices', '4']
    from online_protocol import atomic_json
    atomic_json(root / 'hpc_runtime.json', dict(job_id=config['job_id'], pid=os.getpid(),
                visible_gpus=visible, gpu_allocator=allocator, memory_fraction=fraction,
                forward_microbatch=4, omp_threads=8))
    os.execve(command[0], command, env)


if __name__ == '__main__':
    main()
