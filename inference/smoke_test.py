"""Check a running service against both committed, real-weather reference cases.

Uses only the Python standard library. ML_SERVICE_TOKEN is read from the environment.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def check(url):
    token = os.getenv('ML_SERVICE_TOKEN', '')
    def request(path, payload=None, authenticated=True):
        headers = {'Content-Type': 'application/json'}
        if token and authenticated:
            headers['Authorization'] = 'Bearer ' + token
        req = urllib.request.Request(url.rstrip('/') + path,
            data=None if payload is None else json.dumps(payload).encode(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)

    def require(condition, description):
        if not condition:
            raise RuntimeError(description)

    code, health = request('/health', authenticated=False)
    require(code == 200 and health['status'] == 'ready', 'Health check failed')
    code, metadata = request('/v1/metadata')
    require(code == 200, 'Metadata check failed')
    require(set(metadata['supported_turbines']) == {'turbine_1', 'turbine_2'}, 'Expected both turbines')
    report = {'model_version': metadata['model_version'], 'checks': []}
    for turbine in metadata['supported_turbines']:
        body = json.loads((ROOT / f'inference/examples/{turbine}-48h-request.json').read_text())
        expected = json.loads((ROOT / f'inference/examples/{turbine}-48h-response.json').read_text())
        code, actual = request('/v1/predict', body)
        require(code == 200, turbine + ': prediction failed')
        require(actual['model_version'] == expected['model_version'], 'Loaded model differs from reference release')
        for key in ['schema_version', 'turbine_id', 'weather_model', 'alignment_confirmed']:
            require(actual[key] == expected[key], 'Response metadata mismatch: ' + key)
        require(len(actual['records']) == 48, 'Incorrect output length')
        for a, b in zip(actual['records'], expected['records']):
            value = a['predicted_normalized_power']
            require(a['target_time'] == b['target_time'], 'Timestamp mismatch')
            require(math.isfinite(value) and 0 <= value <= 1, 'Invalid power value')
            require(abs(value - b['predicted_normalized_power']) <= 2e-6, 'Prediction differs from release reference')
        require(request('/v1/predict', body) == (code, actual), 'Repeated request changed')
        bad = copy.deepcopy(body); bad['records'].pop()
        require(request('/v1/predict', bad)[0] == 422, 'Incomplete forecast was accepted')
        bad = copy.deepcopy(body); bad['weather_model'] = 'gfs_global'
        require(request('/v1/predict', bad)[0] == 422, 'Wrong weather source was accepted')
        if token:
            require(request('/v1/predict', body, authenticated=False)[0] == 401, 'Unauthenticated request was accepted')
        report['checks'].append(turbine + ': reference parity, repeatability, invalid horizon/source rejection' + (', authentication' if token else ''))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8001')
    p.add_argument('--output')
    a = p.parse_args()
    result = check(a.url)
    if a.output:
        Path(a.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
