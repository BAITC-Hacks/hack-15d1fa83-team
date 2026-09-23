import importlib.util
import json
from pathlib import Path
import zipfile
import pytest

spec = importlib.util.spec_from_file_location("brev_train", Path(__file__).parents[1]/"scripts"/"brev_train.py")
brev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(brev)

def test_upload_bundle_contains_only_code_and_selected_dataset(trained_fixture, tmp_path):
    root, _ = trained_fixture
    destination = tmp_path/"bundle.zip"
    digest = brev.package(root/"training.csv", destination)
    assert len(digest) == 64
    with zipfile.ZipFile(destination) as archive:
        names = archive.namelist()
        assert "data/training.csv" in names
        assert "data/training.metadata.json" in names
        assert "src/windpower/train.py" in names
        assert not any(".env" in n or "test" in n or "billing" in n for n in names)

def test_upload_rejects_changed_dataset(trained_fixture, tmp_path):
    root, _ = trained_fixture
    target = tmp_path/"training.csv"
    target.write_text((root/"training.csv").read_text()+"\n")
    target.with_suffix(".metadata.json").write_text((root/"training.metadata.json").read_text())
    with pytest.raises(ValueError, match="checksum"):
        brev.package(target, tmp_path/"bundle.zip")

def test_receipt_rejects_remote_shell_injection(tmp_path):
    target = tmp_path/"job.json"
    target.write_text(json.dumps(dict(instance="gpu; echo secret", remote_dir="/tmp/windpower-0123456789ab")))
    with pytest.raises(ValueError, match="instance"):
        brev.read_job(target)
