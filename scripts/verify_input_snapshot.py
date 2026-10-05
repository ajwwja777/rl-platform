#!/usr/bin/env python3
"""Verify/restore bounded input artifacts, without loading a model or robot."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from methods.openpi_rlt.cobot_adapter.input_snapshots import load_and_verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--metadata-sha256')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a new report path; do not overwrite evidence')
    _, manifest = load_and_verify(args.input, metadata_sha256=args.metadata_sha256)
    receipt = manifest['receipt']
    result = {'status': 'verified', 'schema': manifest['schema'],
              'input': str(args.input.resolve()), 'array_file_sha256': manifest['array_file_sha256'],
              'episode_id': receipt['episode_id'], 'step_id': receipt['step_id'],
              'current_input_sha256': receipt['current_input']['sha256'],
              'next_input_sha256': receipt['next_input']['sha256'],
              'raw_array_bytes': manifest['raw_array_bytes'],
              'input_contract_checks_passed': bool(receipt['checks_passed']),
              'model_consistency_verified': None,
              'boundary': manifest['boundary']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
