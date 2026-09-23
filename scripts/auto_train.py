"""One-command Brev training, resumable monitoring, verified download and stop.

Uses the existing Brev login. Does not create instances or install local ML packages.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import tempfile
import zipfile

import brev_train as worker


def legacy_controllers(output, proc_root=Path('/proc')):
    """Identify old controllers before retiring their existence-only lock.

    Old versions hold no OS lock. Never infer their death from a timestamp.
    This migration is intentionally limited to Linux, where the launcher runs.
    """
    if not proc_root.is_dir():
        raise RuntimeError("An old launcher lock needs checking from WSL; run the Windows training launcher.")
    found = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            argv = entry.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')
            if not any(Path(arg).name == 'auto_train.py' for arg in argv):
                continue
            value = next((arg.split('=', 1)[1] for arg in argv if arg.startswith('--output-dir=')), None)
            if '--output-dir' in argv:
                value = argv[argv.index('--output-dir') + 1]
            if not value:
                raise RuntimeError("Cannot identify an older training window; close it before retrying.")
            target = Path(value)
            if not target.is_absolute():
                target = entry.joinpath('cwd').resolve(strict=True) / target
            if target.resolve() == output.resolve():
                found.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError):
            continue  # Process exited during inspection.
    return found


@contextmanager
def controller_lock(output):
    # Never unlink the guard: replacing its inode could permit two owners.
    # The OS releases the lock even when a terminal closes or Python crashes.
    guard = (output / 'automation.guard').open('a+b')
    try:
        if guard.seek(0, os.SEEK_END) == 0:
            guard.write(b'0'); guard.flush()
        guard.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(guard.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Training is already controlled by another window. Keep that window open; no duplicate job was started.") from error
        legacy = output / 'automation.lock'
        if legacy.exists():
            owners = legacy_controllers(output)
            if owners:
                raise RuntimeError("An older training window is still active (PID " + ', '.join(map(str, owners)) + "). Close that old training window, then reopen this launcher. No duplicate job was started.")
            legacy.unlink()
        yield
    finally:
        guard.close()


def run_command(command, report, timeout=300):
    """Bound local CLI waits without pipes that descendants can keep open."""
    with tempfile.TemporaryFile() as captured:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
            stdout=captured, stderr=subprocess.STDOUT, start_new_session=os.name != 'nt')
        deadline = time.monotonic() + timeout
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    process.wait(timeout=min(30, remaining))
                    break
                except subprocess.TimeoutExpired:
                    report('Still waiting for ' + ' '.join(command[:2]) + '...')
        except BaseException:
            if os.name != 'nt':
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait(timeout=10)
            captured.seek(0)
            detail = captured.read().decode(errors='replace').strip()
            if detail:
                report(detail[-6000:])
            raise
        captured.seek(0)
        combined = captured.read().decode(errors='replace')
        if process.returncode:
            raise RuntimeError(combined[-6000:] or 'Brev command failed')
        return combined


def remote_status(text):
    matches = re.findall(r"^WINDPOWER_EXIT=(\d+)$", text, re.MULTILINE)
    return int(matches[-1]) if matches else None


def instance_state(listing, instance):
    listing = re.sub(r"\x1b\[[0-9;]*m", "", listing)
    match = re.search(r"^\s*" + re.escape(instance) + r"\s+([A-Z]+)\b", listing, re.MULTILINE)
    if not match:
        raise ValueError("Selected instance is not in Brev's instance list; refusing to create a new one")
    return match.group(1)


def verify_results(archive_path, dataset_sha):
    """Read only the three expected members; never extract arbitrary ZIP paths."""
    with zipfile.ZipFile(archive_path) as archive:
        required = ["model.json", "metrics.json", "evaluation_predictions.csv"]
        # Python's zipfile CLI flattens individually supplied paths. Also accept
        # archives made by passing the artifacts directory as a whole.
        names = required if all(name in archive.namelist() for name in required) else ["artifacts/" + name for name in required]
        if any(archive.getinfo(name).file_size > 50_000_000 for name in names):
            raise ValueError("Unexpectedly large result file")
        files = {name.split("/")[-1]: archive.read(name) for name in names}
    model = json.loads(files["model.json"])
    metrics = json.loads(files["metrics.json"])
    for artifact in (model, metrics):
        if artifact["dataset_metadata"]["dataset_sha256"] != dataset_sha:
            raise ValueError("Downloaded results belong to a different dataset")
    if metrics["device"] != "cuda":
        raise ValueError("Result does not record CUDA training")
    if not model.get("layers") or not files["evaluation_predictions.csv"]:
        raise ValueError("Incomplete model results")
    return files, model, metrics


def find_job(roots, instance, epochs, dataset_sha):
    candidates = []
    for root in roots:
        for receipt in root.glob("*/job.json"):
            job = worker.read_job(receipt)
            if job["instance"] != instance or job["epochs"] != epochs:
                continue
            bundle = receipt.parent / "bundle.zip"
            if not bundle.exists():
                continue
            with zipfile.ZipFile(bundle) as archive:
                metadata = json.loads(archive.read("data/training.metadata.json"))
            if metadata["dataset_sha256"] == dataset_sha:
                candidates.append((job["created_at_utc"], receipt))
    if not candidates:
        return None
    receipt = max(candidates)[1]
    if not worker.read_job(receipt).get("submitted"):
        raise RuntimeError(f"An earlier submission was interrupted: {receipt}. Inspect it before retrying; no duplicate job was started.")
    return receipt


def run(args):
    os.environ["PATH"] = str(Path.home() / ".local/bin") + os.pathsep + os.environ.get("PATH", "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.instance):
        raise ValueError("Invalid instance name")
    if args.epochs < 1 or args.poll_seconds < 1 or args.timeout_minutes < 1:
        raise ValueError("Epochs and time limits must be positive")
    dataset = Path(args.dataset).resolve()
    dataset_sha = hashlib.sha256(dataset.read_bytes()).hexdigest()
    metadata = json.loads(dataset.with_suffix(".metadata.json").read_text())
    if dataset_sha != metadata["dataset_sha256"]:
        raise ValueError("Dataset checksum mismatch")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with controller_lock(output):
        return run_locked(args, dataset, dataset_sha, output)


def run_locked(args, dataset, dataset_sha, output):
    log = (output / "progress.log").open("a", encoding="utf-8")

    def report(message):
        line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + message
        print(line, flush=True)
        log.write(line + "\n"); log.flush()

    def brev(arguments):
        return run_command(["brev", *arguments], report)

    try:
        report("Checking existing jobs. Keep this window open; no commands are needed.")
        receipt = find_job([output / "jobs", worker.ROOT / "artifacts" / "brev-jobs"],
                           args.instance, args.epochs, dataset_sha)
        completed = output / "complete.json"
        if completed.exists():
            saved = json.loads(completed.read_text())
            if saved["dataset_sha256"] != dataset_sha or saved["epochs"] != args.epochs or saved["instance"] != args.instance:
                raise ValueError("This output folder contains a different training run; choose another folder")
            verify_results(output / "results.zip", dataset_sha)
            if args.stop_instance and not saved.get("instance_stopped"):
                report("Results already verified. Stopping the selected GPU instance.")
                brev(["stop", args.instance])
                saved["instance_stopped"] = True
                completed.write_text(json.dumps(saved, indent=2))
            report("Already complete. Results are saved here: " + str(output))
            return
        if getattr(args, "start_instance", False):
            state = instance_state(brev(["list"]), args.instance)
            if state == "STOPPED":
                report("Starting the existing GPU instance; compute billing will resume.")
                try:
                    report(brev(["start", args.instance, "--detached"]).strip())
                except subprocess.TimeoutExpired:
                    # A timed-out client may already have started the instance.
                    # Read current state; do not blindly repeat the mutation.
                    state = instance_state(brev(["list"]), args.instance)
                    if state not in ("RUNNING", "STARTING"):
                        raise RuntimeError("GPU startup timed out; current state is " + state)
                    report("GPU startup was accepted. Waiting for remote access.")
                brev(["refresh"])
            elif state not in ("RUNNING", "STARTING"):
                raise RuntimeError("Instance is in state " + state + "; inspect it in Brev")
        if receipt is None:
            report("Checking remote GPU access.")
            ready_deadline = time.monotonic() + 600
            while True:
                try:
                    brev(["exec", args.instance, "nvidia-smi"])
                    break
                except (RuntimeError, subprocess.TimeoutExpired):
                    if time.monotonic() >= ready_deadline:
                        raise
                    report("GPU SSH is not ready yet; checking again shortly.")
                    time.sleep(15)
            report("Submitting full training job.")
            worker.submit(argparse.Namespace(instance=args.instance, dataset=str(dataset),
                epochs=args.epochs, job_dir=str(output / "jobs"), prepare_only=False))
            receipt = find_job([output / "jobs"], args.instance, args.epochs, dataset_sha)
        else:
            report("Resuming monitoring of existing job: " + str(receipt))
        job = worker.read_job(receipt)
        remote = shlex.quote(job["remote_dir"])
        deadline = time.monotonic() + args.timeout_minutes * 60
        previous = None
        failures = 0
        while time.monotonic() < deadline:
            try:
                status = brev(["exec", args.instance,
                    f"cd {remote} && tail -n 8 job.log && if test -f exit_code; then printf '\\nWINDPOWER_EXIT='; cat exit_code; fi"])
                failures = 0
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                failures += 1
                report(f"Connection check failed ({failures}/3): {error}")
                if failures >= 3:
                    raise
                time.sleep(args.poll_seconds)
                continue
            if status != previous:
                report(status.strip()); previous = status
            code = remote_status(status)
            if code is not None:
                if code != 0:
                    raise RuntimeError(f"Training failed with exit code {code}; see progress.log")
                break
            time.sleep(args.poll_seconds)
        else:
            raise TimeoutError("Monitoring timed out; relaunch to resume the existing job")
        report("Training finished. Downloading results.")
        temporary = output / "results.partial.zip"
        brev(["copy", f"{args.instance}:{job['remote_dir']}/results.zip", str(temporary)])
        files, model, metrics = verify_results(temporary, dataset_sha)
        temporary.replace(output / "results.zip")
        for name, content in files.items():
            (output / name).write_bytes(content)
        summary = {"model_version": model["model_version"], "device": metrics["device"],
            "dataset_sha256": dataset_sha, "epochs": args.epochs, "instance": args.instance,
            "best_epoch": metrics["best_epoch"], "test_metrics": metrics["metrics"]["test"],
            "instance_stopped": False, "job_receipt": str(receipt)}
        completed.write_text(json.dumps(summary, indent=2))
        report("Results verified and saved: " + str(output))
        if args.stop_instance:
            report("Stopping the GPU instance now that results are safely downloaded.")
            brev(["stop", args.instance])
            summary["instance_stopped"] = True
            completed.write_text(json.dumps(summary, indent=2))
        report("COMPLETE. Model: " + model["model_version"])
    except BaseException as error:
        report("NEEDS ATTENTION: " + str(error))
        report("The GPU was not automatically stopped on this error. Check Brev before leaving it idle. Relaunching resumes a submitted job.")
        raise
    finally:
        log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--timeout-minutes", type=int, default=90)
    parser.add_argument("--stop-instance", action="store_true")
    parser.add_argument("--start-instance", action="store_true", help="Start the selected existing stopped instance (resumes billing)")
    try:
        run(parser.parse_args())
    except Exception as error:
        print("Automation stopped: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
