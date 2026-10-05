#!/usr/bin/env python3
"""Assess supplied complete-Episode learning evidence; never starts a robot."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrations.cobot_runtime.learning_curve import assess_learning_curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = args.input.read_bytes()
    result = assess_learning_curve(json.loads(data))
    result['source'] = str(args.input)
    result['source_sha256'] = hashlib.sha256(data).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    # Evidence gaps and failures are recorded outcomes, not tool crashes.
    print(result['status'])


if __name__ == '__main__':
    main()
