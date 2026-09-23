"""Separate training command entry point; no service or GPU starts implicitly.

Shared feature/data code remains in src/windpower to prevent train/serve drift.
Existing scripts/ entry points remain compatible with old job launchers.
"""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODULES = {'fetch': 'windpower.archive', 'assemble': 'windpower.dataset', 'train': 'windpower.train'}
SCRIPTS = {'auto': 'auto_train.py', 'brev': 'brev_train.py', 'benchmark': 'benchmark_models.py', 'verify': 'verify_trained_model.py'}


def main():
    p = argparse.ArgumentParser(description=__doc__, epilog='Use COMMAND --help to see that command\'s arguments. Paths are relative to your current directory.')
    p.add_argument('command', choices=[*MODULES, *SCRIPTS])
    p.add_argument('arguments', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = [sys.executable]
    if a.command in MODULES:
        command += ['-m', MODULES[a.command]]
    else:
        command += [str(ROOT / 'scripts' / SCRIPTS[a.command])]
    return subprocess.call(command + a.arguments)


if __name__ == '__main__':
    raise SystemExit(main())
