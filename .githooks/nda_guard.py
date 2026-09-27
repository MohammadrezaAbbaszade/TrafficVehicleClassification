#!/usr/bin/env python3
"""Reject confidential data and notebook outputs in staged files or pushed history."""
import json
import subprocess
import sys
from pathlib import PurePosixPath


def git(*args):
    return subprocess.check_output(['git', *args], text=True)


def inspect(ref, paths):
    for path in paths:
        if (path.startswith(('dataset/', 'dataset_v2/', 'checkpoints/', '.local/', 'reports/baseline_v1/'))
            or path in {'reports/base_split.json', 'reports/data_v2_inventory.csv', 'reports/data_v2_duplicate_groups.json', 'reports/data_v2_review_decisions.csv'}
            or PurePosixPath(path).suffix.lower() in {'.pt', '.jpg', '.jpeg', '.webp', '.bmp'}):
            raise ValueError(f'Confidential artifact: {path}')
        if path.endswith('.ipynb'):
            notebook = json.loads(git('show', f'{ref}:{path}'))
            if any(c.get('outputs') for c in notebook.get('cells', [])):
                raise ValueError(f'Notebook outputs must stay local: {path}. Keep a private copy and clear outputs before staging.')

try:
    if '--push' in sys.argv:
        commits = set()
        for line in sys.stdin:
            local_ref, sha, remote_ref, remote_sha = line.split()
            if sha == '0' * 40:
                continue
            if not local_ref.startswith(('refs/heads/', 'refs/tags/')):
                raise ValueError('Private backup refs must never be pushed.')
            commits.update(git('rev-list', sha).splitlines())
        for sha in commits:
            inspect(sha, git('ls-tree', '-r', '--name-only', sha).splitlines())
    else:
        inspect('', git('ls-files').splitlines())
except (ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
    sys.exit(f'NDA check blocked this operation: {error}')
