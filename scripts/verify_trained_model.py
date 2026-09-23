"""Verify downloaded GPU predictions and both turbine API paths on the CPU."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from windpower.api import create_app
from windpower.features import WEATHER_COLUMNS
from windpower.model import Predictor


def verify(dataset, results):
    dataset, results = Path(dataset), Path(results)
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    predictor = Predictor(results / 'model.json', allow_provisional=True)
    saved = json.loads((results / 'metrics.json').read_text())
    for artifact in (predictor.bundle, saved):
        if artifact['dataset_metadata']['dataset_sha256'] != digest:
            raise ValueError('Dataset does not match downloaded artifacts')
    frame = pd.read_csv(dataset)
    held = pd.read_csv(results / 'evaluation_predictions.csv')
    keys = ['turbine_id', 'valid_time_utc', 'provider_offset_days']
    joined = held.merge(frame, on=keys, validate='one_to_one', suffixes=('_export', ''), how='left')
    np.testing.assert_allclose(joined.target_power, joined.target_power_export, atol=1e-7)
    times = pd.to_datetime(frame.valid_time_utc, utc=True)
    split = saved['split_boundaries_utc']
    mask = (times >= pd.Timestamp(split['test_start'], tz='UTC')) & (times < pd.Timestamp(split['test_end'], tz='UTC'))
    if len(held) != int(mask.sum()) or not len(held):
        raise ValueError('Export does not cover the complete test split')
    actual_times = pd.to_datetime(joined.valid_time_utc, utc=True)
    if not ((actual_times >= pd.Timestamp(split['test_start'], tz='UTC')) & (actual_times < pd.Timestamp(split['test_end'], tz='UTC'))).all():
        raise ValueError('Export contains rows outside test split')
    predicted = predictor.predict(joined, predictor.bundle['weather_model'])
    np.testing.assert_allclose(predicted, joined.predicted_power, atol=2e-6, rtol=1e-5)
    if not np.isfinite(predicted).all() or ((predicted < 0) | (predicted > 1)).any():
        raise ValueError('Invalid output range')
    error = predicted - joined.target_power.to_numpy()
    metrics = dict(mae=float(np.abs(error).mean()), rmse=float(np.sqrt(np.mean(error**2))), bias=float(error.mean()))
    for key, value in metrics.items():
        np.testing.assert_allclose(value, saved['metrics']['test']['neural'][key], atol=2e-6)
    by_turbine = {}
    for turbine in predictor.turbines:
        selected = joined.turbine_id.eq(turbine)
        e = error[selected]
        by_turbine[turbine] = dict(mae=float(np.abs(e).mean()), rmse=float(np.sqrt(np.mean(e**2))), bias=float(e.mean()), rows=int(selected.sum()), unique_target_hours=int(joined.loc[selected, 'valid_time_utc'].nunique()))
        for key in ('mae', 'rmse', 'bias'):
            np.testing.assert_allclose(by_turbine[turbine][key], saved['metrics_by_turbine']['test'][turbine]['models']['neural'][key], atol=2e-6)
    checks = []
    token = 'local-verification-only'
    headers = {'Authorization': 'Bearer ' + token}
    with TestClient(create_app(results / 'model.json', allow_provisional=True, service_token=token)) as client:
        capabilities = client.get('/v1/metadata', headers=headers)
        capabilities.raise_for_status()
        assert capabilities.json()['supported_turbines'] == predictor.turbines
        for turbine in predictor.turbines:
            # Real archived rows, with one archive offset so there is one value per hour.
            sample = joined.loc[joined.turbine_id.eq(turbine) & joined.provider_offset_days.eq(1)].sort_values('valid_time_utc').head(48)
            records = []
            for row in sample.to_dict('records'):
                records.append({'target_time': pd.Timestamp(row['valid_time_utc']).isoformat(), **{key: row[key] for key in WEATHER_COLUMNS}})
            body = dict(schema_version='windpower.input.v1', turbine_id=turbine, weather_model=predictor.bundle['weather_model'], records=records)
            response = client.post('/v1/predict', json=body, headers=headers)
            response.raise_for_status()
            result = response.json()
            assert len(result['records']) == 48
            assert result['model_version'] == predictor.bundle['model_version']
            np.testing.assert_allclose([r['predicted_normalized_power'] for r in result['records']], predictor.predict(sample, body['weather_model']), atol=2e-6)
            assert client.post('/v1/predict', json=body, headers=headers).json() == result
            assert client.post('/v1/predict', json=body).status_code == 401
            assert client.post('/v1/predict', json={**body, 'weather_model': 'gfs_global'}, headers=headers).status_code == 422
            for label, content in [('request', body), ('response', result)]:
                (results / f'{turbine}-48h-{label}.json').write_text(json.dumps(content, indent=2) + '\n')
            checks.append(f'{turbine}: real archived 48-hour input, prediction parity, repeatability, authentication and provider mismatch checks passed')
    report = dict(model_version=predictor.bundle['model_version'], dataset_sha256=digest,
        training_device=saved['device'], best_epoch=saved['best_epoch'], epochs_run=len(saved['history']),
        evaluation_rows=len(held), unique_turbine_hours=len(held.drop_duplicates(['turbine_id', 'valid_time_utc'])),
        max_export_prediction_difference=float(np.abs(predicted - joined.predicted_power).max()),
        recomputed_metrics=metrics, metrics_by_turbine=by_turbine,
        wind_curve_mae=saved['metrics']['test']['wind_curve']['mae'],
        mae_reduction_vs_wind_curve=1-metrics['mae']/saved['metrics']['test']['wind_curve']['mae'],
        api_checks=checks, api_verification='In-process FastAPI requests using real archived weather; live teammate integration pending',
        alignment_confirmed=predictor.bundle['dataset_metadata']['alignment_confirmed'],
        archive_semantics=predictor.bundle['dataset_metadata']['archive_semantics'])
    (results / 'independent_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--results', required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.dataset, args.results), indent=2))
