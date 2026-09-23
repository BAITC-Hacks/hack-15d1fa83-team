"""Lightweight local controller: upload to an existing Brev GPU, train, retrieve.

Requires only Python's standard library and an authenticated Brev CLI on this PC.
It does not provision instances or spend credits until `submit` executes remotely.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def package(dataset, destination, root=ROOT):
    dataset = Path(dataset).resolve()
    metadata_path = dataset.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != metadata["dataset_sha256"]:
        raise ValueError("Dataset checksum differs from its assembly metadata")
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(root / "pyproject.toml", "pyproject.toml")
        for path in sorted((root / "src" / "windpower").glob("*.py")):
            archive.write(path, path.relative_to(root).as_posix())
        archive.write(dataset, "data/training.csv")
        archive.write(metadata_path, "data/training.metadata.json")
    return hashlib.sha256(destination.read_bytes()).hexdigest()

def cli(arguments, capture=False):
    if not shutil.which("brev"):
        raise RuntimeError("Install the NVIDIA Brev CLI, run brev login and brev refresh. On Windows use WSL.")
    return subprocess.run(["brev", *arguments], check=True, text=True, capture_output=capture)

def read_job(path):
    job = json.loads(Path(path).read_text())
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", job["instance"]):
        raise ValueError("Invalid instance name")
    if not re.fullmatch(r"/tmp/windpower-[a-f0-9]{12}", job["remote_dir"]):
        raise ValueError("Invalid remote job directory")
    return job

def submit(args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.instance):
        raise ValueError("Use the Brev instance name, not a URL or shell command")
    if args.epochs < 1: raise ValueError("Epochs must be positive")
    job_id = uuid.uuid4().hex[:12]
    local = Path(args.job_dir).resolve() / job_id; local.mkdir(parents=True, exist_ok=False)
    remote = f"/tmp/windpower-{job_id}"
    digest = package(args.dataset, local / "bundle.zip")
    # This script executes on Linux on the remote GPU, never on the local PC.
    script = "\n".join([
        "#!/usr/bin/env bash", "set -euo pipefail",
        "trap 'result=$?; printf \"%s\\n\" \"$result\" > exit_code' EXIT",
        "python3 -m zipfile -e bundle.zip .",
        "python3 -m venv --system-site-packages .venv",
        '.venv/bin/python -m pip install -e ".[train]"',
        f".venv/bin/python -m windpower.train --dataset data/training.csv --output-dir artifacts --device cuda --epochs {args.epochs}",
        ".venv/bin/python -m zipfile -c results.zip artifacts/model.json artifacts/metrics.json artifacts/evaluation_predictions.csv",
        "",
    ])
    (local / "run.sh").write_text(script, encoding="utf-8", newline="\n")
    receipt = dict(job_id=job_id, instance=args.instance, remote_dir=remote, bundle_sha256=digest, epochs=args.epochs, created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), submitted=False)
    receipt_path = local / "job.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    if args.prepare_only:
        print(f"Prepared only; no remote commands executed. Receipt: {receipt_path}")
        return
    cli(["exec", args.instance, f"mkdir -p {shlex.quote(remote)}"])
    for name in ["bundle.zip", "run.sh"]:
        cli(["copy", str(local / name), f"{args.instance}:{remote}/{name}"])
    cli(["exec", args.instance, f"cd {shlex.quote(remote)} && nohup bash run.sh > job.log 2>&1 < /dev/null &"])
    receipt["submitted"] = True
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"Submitted. Monitor with: python scripts/brev_train.py status --job {shlex.quote(str(receipt_path))}")

def status(args):
    job = read_job(args.job)
    cli(["exec", job["instance"], f"cd {shlex.quote(job['remote_dir'])} && tail -n 25 job.log && if test -f exit_code; then printf 'EXIT_CODE='; cat exit_code; else printf 'Worker has not recorded an exit code yet.\\n'; fi"])

def download(args):
    job = read_job(args.job)
    code = cli(["exec", job["instance"], f"cat {shlex.quote(job['remote_dir'] + '/exit_code')}"], capture=True)
    # CLI versions may emit informational lines. A sole terminal numeric line is required.
    if not code.stdout.strip().endswith("\n0") and code.stdout.strip() != "0":
        raise RuntimeError("Remote job did not report success; inspect status before downloading")
    destination = Path(args.output).resolve(); destination.parent.mkdir(parents=True, exist_ok=True)
    cli(["copy", f"{job['instance']}:{job['remote_dir']}/results.zip", str(destination)])
    print(f"Downloaded {destination}. Stop the GPU instance when it is no longer needed.")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--instance", required=True)
    s.add_argument("--dataset", required=True)
    s.add_argument("--epochs", type=int, default=80)
    s.add_argument("--job-dir", default="artifacts/brev-jobs")
    s.add_argument("--prepare-only", action="store_true")
    s.set_defaults(function=submit)
    s = sub.add_parser("status"); s.add_argument("--job", required=True); s.set_defaults(function=status)
    s = sub.add_parser("download"); s.add_argument("--job", required=True); s.add_argument("--output", default="artifacts/brev-results.zip"); s.set_defaults(function=download)
    args = p.parse_args(); args.function(args)

if __name__ == "__main__": main()

