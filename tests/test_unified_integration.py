"""Django orchestrator against the actual committed model over loopback HTTP.
Weather is the committed reference fixture, not fabricated measurements or an external API.
"""
import json
import socket
import threading
import time
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np
import uvicorn
from django.test import TestCase, override_settings
from windpower.api import create_app
from common.contracts import timestamp, WEATHER_SCHEMA, validate_ml_input
from forecasting.ml_client import get_metadata
from forecasting.services import ml_payload
from weather_archive.models import Turbine, WeatherSnapshot

ROOT = Path(__file__).resolve().parents[1]


@override_settings(ML_BACKEND='http', ML_SERVICE_TOKEN='integration-test-token', ML_METADATA_URL='', DEBUG=True)
class UnifiedIntegrationTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.sock = socket.socket()
        cls.sock.bind(('127.0.0.1', 0))
        cls.port = cls.sock.getsockname()[1]
        app = create_app(ROOT/'models/production/model.json', allow_provisional=True,
                         service_token='integration-test-token', provider_url='')
        cls.server = uvicorn.Server(uvicorn.Config(app, log_level='error'))
        cls.thread = threading.Thread(target=cls.server.run, kwargs={'sockets': [cls.sock]}, daemon=True)
        cls.thread.start()
        deadline = time.monotonic() + 15
        while not cls.server.started and cls.thread.is_alive() and time.monotonic() < deadline:
            time.sleep(.05)
        if not cls.server.started:
            cls.server.should_exit = True
            cls.thread.join(timeout=5)
            cls.sock.close()
            super().tearDownClass()
            raise RuntimeError('Real model service did not start')
        cls.settings_override = override_settings(ML_SERVICE_URL=f'http://127.0.0.1:{cls.port}/v1/predict')
        cls.settings_override.enable()

    @classmethod
    def tearDownClass(cls):
        cls.settings_override.disable()
        cls.server.should_exit = True
        cls.thread.join(timeout=5)
        cls.sock.close()
        super().tearDownClass()

    def setUp(self):
        from datetime import timedelta
        self.inputs = {}
        self.snapshots = {}
        for turbine in ('turbine_1', 'turbine_2'):
            body = json.loads((ROOT/f'inference/examples/{turbine}-48h-request.json').read_text())
            self.inputs[turbine] = body
            start = timestamp(body['records'][0]['target_time'])
            t = Turbine.objects.create(id=turbine, name=turbine, latitude=43, longitude=78)
            self.snapshots[turbine] = WeatherSnapshot.objects.create(id='reference-'+turbine, turbine=t,
                as_of=start-timedelta(hours=6), start_time=start, provider='open_meteo', weather_model='jma_gsm',
                records=body['records'], provenance={'schema_version': WEATHER_SCHEMA, 'wind_height_m': 10,
                    'temperature_height_m': 2, 'timing': 'estimated_fixed_lead', 'mode': 'replay',
                    'notice': 'Committed historical weather reference; exact issuance is unverified.'})

    def request(self, turbine='turbine_2'):
        s = self.snapshots[turbine]
        return {'mode': 'replay', 'turbine_id': turbine, 'provider': 'open_meteo',
                'weather_model': 'jma_gsm', 'as_of': s.as_of.isoformat(), 'start_time': s.start_time.isoformat()}

    def post(self, turbine='turbine_2'):
        with patch('forecasting.services.get_weather', return_value=self.snapshots[turbine]):
            return self.client.post('/api/v1/forecasts/', json.dumps(self.request(turbine)), content_type='application/json')

    def test_actual_metadata_contract_and_two_turbines(self):
        meta = get_metadata()
        self.assertEqual(meta['model_version'], 'mlp-76540e5972a6')
        self.assertEqual(meta['supported_turbines'], ['turbine_1','turbine_2'])
        self.assertEqual(meta['weather_model'], 'jma_gsm')
        self.assertEqual(meta['field_definitions']['wind_speed_10m_ms']['height_m'], 10)
        self.assertIs(meta['alignment_confirmed'], False)

    def test_both_turbines_from_django_match_release_over_real_http(self):
        for turbine in self.inputs:
            with self.subTest(turbine=turbine):
                response = self.post(turbine)
                self.assertEqual(response.status_code, 201, response.content)
                result = response.json()
                self.assertFalse(result['is_demo'])
                self.assertIs(result['alignment_confirmed'], False)
                self.assertEqual(result['model_version'], 'mlp-76540e5972a6')
                expected = json.loads((ROOT/f'inference/examples/{turbine}-48h-response.json').read_text())
                self.assertEqual([timestamp(r['target_time']) for r in result['records']], [timestamp(r['target_time']) for r in expected['records']])
                np.testing.assert_allclose([r['predicted_normalized_power'] for r in result['records']],
                    [r['predicted_normalized_power'] for r in expected['records']], atol=2e-6, rtol=0)
                self.assertEqual(ml_payload(self.snapshots[turbine]), self.inputs[turbine])
                self.assertEqual(self.client.get(f"/api/v1/forecasts/{result['id']}/").json()['records'], result['records'])
                csv = self.client.get(f"/api/v1/forecasts/{result['id']}/export.csv").content.decode()
                self.assertEqual(len(csv.splitlines()), 49)
                self.assertIn('alignment_confirmed', csv)

    def test_real_model_cache_and_token_failure_without_fallback(self):
        self.assertEqual(self.post().status_code, 201)
        with patch('forecasting.services.predict', side_effect=AssertionError('Cache miss')):
            self.assertEqual(self.post().status_code, 201)
        with override_settings(ML_SERVICE_TOKEN='wrong'):
            failure = self.post()
        self.assertEqual(failure.status_code, 401)
        self.assertEqual(failure.json()['records'], [])
        self.assertEqual(failure.json()['status'], 'failed')

    def test_django_rejects_out_of_model_ranges_before_inference(self):
        for field, value in [('wind_speed_10m_ms',101), ('temperature_2m_c',71), ('temperature_2m_c',-101)]:
            body = json.loads(json.dumps(self.inputs['turbine_2']))
            body['records'][0][field] = value
            from common.contracts import DomainError
            with self.assertRaises(DomainError): validate_ml_input(body)

    def test_incompatible_metadata_is_not_silently_accepted(self):
        from common.contracts import DomainError
        actual = get_metadata()['raw']
        for changed in [{'required_records': 24}, {'time_step_hours': 3},
                        {'input_schema_version': 'unknown'}, {'alignment_confirmed': None}]:
            with patch('forecasting.ml_client.requests.get', return_value=Mock(status_code=200, json=lambda: {**actual, **changed})):
                with self.assertRaises(DomainError): get_metadata()
        bad = json.loads(json.dumps(actual))
        bad['fields']['wind_speed_10m_ms']['height_m'] = 100
        with patch('forecasting.ml_client.requests.get', return_value=Mock(status_code=200, json=lambda: bad)):
            with self.assertRaises(DomainError): get_metadata()
