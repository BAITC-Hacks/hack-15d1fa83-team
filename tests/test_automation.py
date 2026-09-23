import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("auto_train", SCRIPTS / "auto_train.py")
auto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auto)


def result_zip(path, digest="correct", device="cuda", prefix="artifacts/"):
    # Minimal simulated worker outputs test orchestration, not model accuracy.
    metadata = {"dataset_sha256": digest}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(prefix + "model.json", json.dumps(dict(dataset_metadata=metadata, layers=[{}], model_version="fixture")))
        archive.writestr(prefix + "metrics.json", json.dumps(dict(dataset_metadata=metadata, device=device, best_epoch=2, metrics={"test": {}})))
        archive.writestr(prefix + "evaluation_predictions.csv", "target_power,predicted_power\n0.1,0.2\n")
        archive.writestr("../../escape.txt", "must never be extracted")


@pytest.mark.parametrize("prefix", ["", "artifacts/"])
def test_result_verification_rejects_wrong_dataset_and_cpu(tmp_path, prefix):
    path = tmp_path / "results.zip"
    result_zip(path, prefix=prefix)
    with pytest.raises(ValueError, match="different dataset"):
        auto.verify_results(path, "wrong")
    files, _, _ = auto.verify_results(path, "correct")
    assert set(files) == {"model.json", "metrics.json", "evaluation_predictions.csv"}
    result_zip(path, device="cpu")
    with pytest.raises(ValueError, match="CUDA"):
        auto.verify_results(path, "correct")


def test_exit_marker_is_not_confused_by_log_text():
    assert auto.remote_status("exit_code 0\nWINDPOWER_EXIT=1\n") == 1
    assert auto.remote_status("still training\n") is None


def test_instance_start_requires_an_existing_exact_name():
    assert auto.instance_state(" NAME STATUS\n wind-training STOPPED READY\n", "wind-training") == "STOPPED"
    with pytest.raises(ValueError, match="refusing to create"):
        auto.instance_state("wind-training-other STOPPED", "wind-training")


def test_resumed_job_downloads_before_stop_and_rerun_does_not_train(tmp_path, monkeypatch):
    data = tmp_path / "training.csv"
    data.write_text("example")
    digest = auto.hashlib.sha256(data.read_bytes()).hexdigest()
    data.with_suffix(".metadata.json").write_text(json.dumps({"dataset_sha256": digest}))
    receipt = tmp_path / "job.json"
    receipt.write_text(json.dumps({"instance": "wind-training", "remote_dir": "/tmp/windpower-0123456789ab"}))
    monkeypatch.setattr(auto, "find_job", lambda *args: receipt)
    monkeypatch.setattr(auto.worker, "submit", lambda *args: pytest.fail("Must resume, not submit"))
    calls = []
    output = tmp_path / "out"
    def fake_run(command, report, **kwargs):
        calls.append(command[1])
        if command[1] == "copy":
            result_zip(command[-1], digest)
        if command[1] == "stop":
            assert (output / "model.json").exists()
            assert json.loads((output / "complete.json").read_text())["dataset_sha256"] == digest
        return "WINDPOWER_EXIT=0\n"
    monkeypatch.setattr(auto, "run_command", fake_run)
    args = argparse.Namespace(instance="wind-training", dataset=str(data), output_dir=str(output), epochs=80, poll_seconds=1, timeout_minutes=1, stop_instance=True)
    auto.run(args)
    assert calls == ["exec", "copy", "stop"]
    assert not (tmp_path / "escape.txt").exists()
    auto.run(args)
    assert calls == ["exec", "copy", "stop"]
    assert not (output / "automation.lock").exists()


def test_failed_job_does_not_download_or_stop(tmp_path, monkeypatch):
    data = tmp_path / "training.csv"
    data.write_text("example")
    digest = auto.hashlib.sha256(data.read_bytes()).hexdigest()
    data.with_suffix(".metadata.json").write_text(json.dumps({"dataset_sha256": digest}))
    receipt = tmp_path / "job.json"
    receipt.write_text(json.dumps({"instance": "wind-training", "remote_dir": "/tmp/windpower-0123456789ab"}))
    monkeypatch.setattr(auto, "find_job", lambda *args: receipt)
    calls = []
    def fake_run(command, report, **kwargs):
        calls.append(command[1])
        return "Training error\nWINDPOWER_EXIT=1\n"
    monkeypatch.setattr(auto, "run_command", fake_run)
    args = argparse.Namespace(instance="wind-training", dataset=str(data), output_dir=str(tmp_path / "out"), epochs=80, poll_seconds=1, timeout_minutes=1, stop_instance=True)
    with pytest.raises(RuntimeError, match="Training failed"):
        auto.run(args)
    assert calls == ["exec"]


def test_incomplete_submission_is_not_automatically_duplicated(tmp_path):
    directory = tmp_path / "0123456789ab"
    directory.mkdir()
    (directory / "job.json").write_text(json.dumps(dict(instance="wind-training", remote_dir="/tmp/windpower-0123456789ab", epochs=80, submitted=False, created_at_utc="2026-09-23")))
    with zipfile.ZipFile(directory / "bundle.zip", "w") as archive:
        archive.writestr("data/training.metadata.json", json.dumps({"dataset_sha256": "correct"}))
    with pytest.raises(RuntimeError, match="interrupted"):
        auto.find_job([tmp_path], "wind-training", 80, "correct")


def test_legacy_lock_removed_only_after_no_controller_found(tmp_path, monkeypatch):
    legacy = tmp_path / 'automation.lock'
    legacy.touch()
    monkeypatch.setattr(auto, 'legacy_controllers', lambda output: [123])
    with pytest.raises(RuntimeError, match='older training window is still active'):
        with auto.controller_lock(tmp_path):
            pytest.fail('Must not enter while old controller is alive')
    assert legacy.exists()
    monkeypatch.setattr(auto, 'legacy_controllers', lambda output: [])
    with auto.controller_lock(tmp_path):
        assert not legacy.exists()
    # The persistent guard pathname is harmless on subsequent launches.
    with auto.controller_lock(tmp_path):
        pass


def test_lock_excludes_other_process_and_releases_after_crash(tmp_path):
    program = (
        'import sys,time; from pathlib import Path; '
        'sys.path.insert(0,sys.argv[1]); import auto_train; '
        'lock=auto_train.controller_lock(Path(sys.argv[2])); '
        'lock.__enter__(); print("locked",flush=True); time.sleep(30)'
    )
    child = subprocess.Popen([sys._base_executable, '-c', program, str(SCRIPTS), str(tmp_path)],
                             stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'locked'
        with pytest.raises(RuntimeError, match='already controlled'):
            with auto.controller_lock(tmp_path):
                pytest.fail('Duplicate controller acquired guard')
    finally:
        child.kill()
        child.wait(timeout=5)
        child.stdout.close()
    with auto.controller_lock(tmp_path):
        pass


def test_command_timeout_is_bounded_and_preserves_output():
    messages = []
    with pytest.raises(subprocess.TimeoutExpired):
        auto.run_command([sys._base_executable, '-c',
            'import time; print("starting GPU",flush=True); time.sleep(30)'], messages.append, timeout=0.5)
    assert any('starting GPU' in line for line in messages)
    assert auto.run_command([sys._base_executable, '-c', 'print("ready")'], messages.append).strip() == 'ready'


def test_detached_start_timeout_checks_state_and_resumes_receipt(tmp_path, monkeypatch):
    data = tmp_path / 'training.csv'
    data.write_text('example')
    digest = auto.hashlib.sha256(data.read_bytes()).hexdigest()
    data.with_suffix('.metadata.json').write_text(json.dumps({'dataset_sha256': digest}))
    receipt = tmp_path / 'job.json'
    receipt.write_text(json.dumps({'instance': 'wind-training', 'remote_dir': '/tmp/windpower-0123456789ab'}))
    monkeypatch.setattr(auto, 'find_job', lambda *args: receipt)
    monkeypatch.setattr(auto.worker, 'submit', lambda *args: pytest.fail('Must resume receipt'))
    calls = []
    def fake_run(command, report, **kwargs):
        calls.append(command[1:])
        if command[1] == 'list':
            return 'wind-training ' + ('STOPPED' if len(calls) == 1 else 'STARTING')
        if command[1] == 'start':
            assert command == ['brev', 'start', 'wind-training', '--detached']
            raise subprocess.TimeoutExpired(command, 300)
        if command[1] == 'copy':
            result_zip(command[-1], digest)
        return 'WINDPOWER_EXIT=0\n'
    monkeypatch.setattr(auto, 'run_command', fake_run)
    auto.run(argparse.Namespace(instance='wind-training', dataset=str(data),
        output_dir=str(tmp_path / 'out'), epochs=80, poll_seconds=1,
        timeout_minutes=1, stop_instance=True, start_instance=True))
    assert [call[0] for call in calls] == ['list', 'start', 'list', 'refresh', 'exec', 'copy', 'stop']


def test_legacy_scan_matches_output_folder(tmp_path, monkeypatch):
    proc = tmp_path / 'proc'
    proc.mkdir()
    output = tmp_path / 'results'
    # Use synthetic /proc entries to exercise argument parsing on Windows too.
    monkeypatch.setattr(auto.os, 'getuid', lambda: (proc.stat().st_uid), raising=False)
    for pid, target in [(1, str(output)), (2, str(tmp_path / 'different'))]:
        entry = proc / str(pid + 900000)
        entry.mkdir()
        entry.joinpath('cmdline').write_bytes(
            ('python3\0scripts/auto_train.py\0--output-dir\0' + target + '\0').encode())
    assert auto.legacy_controllers(output, proc) == [900001]
