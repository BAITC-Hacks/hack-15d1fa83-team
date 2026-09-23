"""Real committed model + Django orchestration; only weather/HTTP transport are fixtures."""
import json
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

from django.test import TestCase, override_settings
from fastapi.testclient import TestClient
from common.contracts import DomainError, feature_record, timestamp, WEATHER_SCHEMA
from forecasting.ml_client import get_metadata
from weather_archive.models import Turbine, WeatherSnapshot
from windpower.api import create_app

ROOT = Path(__file__).resolve().parents[1]


@override_settings(ML_BACKEND='http', ML_SERVICE_URL='http://model.test/v1/predict',
                   ML_METADATA_URL='', ML_SERVICE_TOKEN='integration-only', DEBUG=True)
class IntegratedModelTests(TestCase):
    def setUp(self):
        self.ml = self.enterContext(TestClient(create_app(
            ROOT / 'models/production/model.json', allow_provisional=True,
            service_token='integration-only')))
        self.enterContext(patch('forecasting.ml_client.requests.get', side_effect=lambda url, **kw:
            self.ml.get(urlsplit(url).path, headers=kw.get('headers'))))
        self.posts = self.enterContext(patch('forecasting.ml_client.requests.post', side_effect=lambda url, **kw:
            self.ml.post(urlsplit(url).path, headers=kw.get('headers'), json=kw['json'])))

    def test_real_metadata_and_both_turbines_roundtrip(self):
        metadata = get_metadata()
        self.assertEqual(metadata['weather_model'], 'jma_gsm')
        self.assertEqual(metadata['supported_turbines'], ['turbine_1', 'turbine_2'])
        self.assertIn('wind_speed_10m_ms', metadata['field_definitions'])
        self.assertFalse(metadata['alignment_confirmed'])
        for turbine_id, lat, lon in [('turbine_1',43.645150,78.535604), ('turbine_2',43.643198,78.538828)]:
            with self.subTest(turbine=turbine_id):
                turbine = Turbine.objects.create(id=turbine_id, name=turbine_id, latitude=lat, longitude=lon)
                payload = json.loads((ROOT / f'inference/examples/{turbine_id}-48h-request.json').read_text())
                expected = self.ml.post('/v1/predict', json=payload,
                    headers={'Authorization':'Bearer integration-only'}).json()
                start = timestamp(payload['records'][0]['target_time'])
                # Historical feature fixture, not a claimed live retrieval or exact archived issue.
                snapshot = WeatherSnapshot.objects.create(id=turbine_id+'-fixture', turbine=turbine,
                    as_of=start, start_time=start, provider='open_meteo', weather_model='jma_gsm',
                    records=payload['records'], provenance={'schema_version':WEATHER_SCHEMA,
                        'wind_height_m':10, 'temperature_height_m':2, 'timing':'test_fixture'})
                with patch('forecasting.services.get_weather', return_value=snapshot):
                    response = self.client.post('/api/v1/forecasts/', data=json.dumps({
                        'turbine_id':turbine_id, 'mode':'live', 'provider':'open_meteo'}), content_type='application/json')
                    self.assertEqual(response.status_code, 201, response.content)
                    run = response.json()
                    self.assertEqual(run['status'], 'completed', run)
                    self.assertFalse(run['is_demo'])
                    self.assertEqual([r['predicted_normalized_power'] for r in run['records']],
                                     [r['predicted_normalized_power'] for r in expected['records']])
                    self.assertEqual([timestamp(r['target_time']) for r in run['records']],
                                     [timestamp(r['target_time']) for r in expected['records']])
                    self.assertEqual(run['model_version'], 'mlp-76540e5972a6')
                    self.assertFalse(run['alignment_confirmed'])
                    self.assertTrue(run['analysis']['warnings'])
                    self.assertEqual(self.client.get(f"/api/v1/forecasts/{run['id']}/ml-input/").json(), payload)
                    self.assertEqual(len(self.client.get(f"/api/v1/forecasts/{run['id']}/export.csv").content.splitlines()),49)
                    calls = self.posts.call_count
                    refreshed = self.client.post(f"/api/v1/forecasts/{run['id']}/refresh/", data='{}', content_type='application/json')
                    self.assertEqual(refreshed.status_code,201)
                    self.assertEqual(self.posts.call_count, calls)

    def test_service_token_mismatch_fails_closed(self):
        with override_settings(ML_SERVICE_TOKEN='wrong'), self.assertRaises(DomainError) as error:
            get_metadata()
        self.assertEqual(error.exception.code,'ml_unauthorized')

    def test_feature_bounds_match_model(self):
        row = {'target_time':'2026-01-01T00:00:00Z','wind_speed_10m_ms':5,
               'wind_direction_10m_deg':180,'temperature_2m_c':0}
        for field, bad in [('wind_speed_10m_ms',101),('temperature_2m_c',-101),('temperature_2m_c',71)]:
            with self.subTest(field=field,value=bad), self.assertRaises(DomainError):
                feature_record({**row,field:bad})
