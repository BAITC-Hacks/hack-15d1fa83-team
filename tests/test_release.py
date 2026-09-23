"""Integration regressions against the committed real joint model."""
import hashlib
import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from windpower.api import create_app

ROOT = Path(__file__).resolve().parents[1]


def test_committed_release_identity_and_both_reference_forecasts():
    directory = ROOT / 'models/production'
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert hashlib.sha256((directory / 'model.json').read_bytes()).hexdigest() == manifest['artifact_sha256']
    headers = {'Authorization': 'Bearer release-test-token'}
    with TestClient(create_app(directory / 'model.json', allow_provisional=True, service_token='release-test-token')) as client:
        metadata = client.get('/v1/metadata', headers=headers).json()
        assert metadata['model_version'] == manifest['model_version']
        assert metadata['supported_turbines'] == manifest['turbines']
        assert metadata['alignment_confirmed'] is False
        for turbine in manifest['turbines']:
            body = json.loads((ROOT / f'inference/examples/{turbine}-48h-request.json').read_text())
            expected = json.loads((ROOT / f'inference/examples/{turbine}-48h-response.json').read_text())
            response = client.post('/v1/predict', json=body, headers=headers)
            assert response.status_code == 200
            actual = response.json()
            assert actual['model_version'] == manifest['model_version']
            assert [r['target_time'] for r in actual['records']] == [r['target_time'] for r in expected['records']]
            np.testing.assert_allclose([r['predicted_normalized_power'] for r in actual['records']],
                [r['predicted_normalized_power'] for r in expected['records']], atol=2e-6, rtol=0)


def test_default_model_path_runs_from_repository_root(monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.delenv('MODEL_PATH', raising=False)
    with TestClient(create_app(allow_provisional=True, service_token='')) as client:
        response = client.get('/health')
        assert response.status_code == 200
        assert response.json()['model_version'] == 'mlp-76540e5972a6'
