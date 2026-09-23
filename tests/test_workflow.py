import copy
import json
from datetime import timedelta
from unittest.mock import Mock, patch
import requests
from django.test import TestCase, Client, override_settings
from common.contracts import DomainError, timestamp, iso, validate_weather, validate_ml_input, validate_predictions, INPUT_SCHEMA, OUTPUT_SCHEMA, WEATHER_SCHEMA
from weather_archive.models import Turbine, WeatherSnapshot, ArchivedRun
from weather_archive.services import demo_weather, get_weather, import_run, previous_runs, provider_forecast, VARIABLES
from forecasting.models import Forecast
from forecasting.services import execute_forecast, ml_payload, refresh_request

AS_OF = timestamp('2026-01-31T18:00:00Z')
START = timestamp('2026-02-01T00:00:00Z')
REQUEST = {'mode': 'replay', 'turbine_id': 'turbine_1', 'as_of': iso(AS_OF), 'start_time': iso(START), 'provider': 'demo'}


@override_settings(ML_BACKEND='demo', DEBUG=True)
class WorkflowTests(TestCase):
    def setUp(self):
        self.turbine = Turbine.objects.create(id='turbine_1', name='Turbine 1', latitude=43.64515, longitude=78.535604)

    def post(self, data=None, path='/api/v1/forecasts/'):
        return self.client.post(path, data=json.dumps(REQUEST if data is None else data), content_type='application/json')

    def test_complete_roundtrip_and_exports(self):
        response = self.post()
        self.assertEqual(response.status_code, 201)
        run = response.json()
        self.assertEqual(len(run['records']), 48)
        self.assertTrue(run['is_demo'])
        self.assertEqual(run['status'], 'completed')
        self.assertEqual(len(run['trace']), 6)
        self.assertEqual(self.client.get(f"/api/v1/forecasts/{run['id']}/").json()['records'], run['records'])
        csv = self.client.get(f"/api/v1/forecasts/{run['id']}/export.csv")
        self.assertEqual(len(csv.content.decode().splitlines()), 49)
        payload = self.client.get(f"/api/v1/forecasts/{run['id']}/ml-input/").json()
        self.assertEqual(set(payload), {'schema_version', 'turbine_id', 'weather_model', 'records'})
        self.assertEqual(set(payload['records'][0]), {'target_time', 'wind_speed_10m_ms', 'wind_direction_10m_deg', 'temperature_2m_c'})
        self.assertEqual(self.client.get('/api/v1/forecasts/').json()['forecasts'][0]['id'], run['id'])

    def test_reuses_immutable_snapshot(self):
        a, b = self.post().json(), self.post().json()
        self.assertNotEqual(a['id'], b['id'])
        self.assertEqual(a['snapshot_id'], b['snapshot_id'])
        self.assertEqual(WeatherSnapshot.objects.count(), 1)

    def test_refresh_without_change_skips_ml(self):
        initial = self.post().json()
        with patch('forecasting.services.predict') as predict:
            result = self.post({}, f"/api/v1/forecasts/{initial['id']}/refresh/")
        self.assertEqual(result.status_code, 201)
        predict.assert_not_called()

    def test_refresh_changed_weather_recalculates(self):
        initial = self.post().json()
        rows = demo_weather(START, AS_OF, 'turbine_1')
        rows[0]['wind_speed_10m_ms'] += 1
        with patch('weather_archive.services.demo_weather', return_value=rows):
            result = self.post({}, f"/api/v1/forecasts/{initial['id']}/refresh/").json()
        self.assertNotEqual(initial['snapshot_id'], result['snapshot_id'])
        self.assertNotEqual(initial['records'][0], result['records'][0])

    def test_bad_json_and_unknown_turbine(self):
        self.assertEqual(self.client.post('/api/v1/forecasts/', data='{', content_type='application/json').status_code, 400)
        self.assertEqual(self.post({'hello': 'world'}).status_code, 422)
        self.assertEqual(self.post({**REQUEST, 'turbine_id': 'missing'}).status_code, 404)
        self.assertEqual(self.post([]).status_code, 400)

    def test_time_must_be_aware_hourly_and_future(self):
        for changed in [{'as_of': '2026-01-31T18:00:00'}, {'start_time': iso(AS_OF)}, {'start_time': '2026-02-01T00:30:00Z'}]:
            with self.subTest(changed=changed):
                self.assertEqual(self.post({**REQUEST, **changed}).status_code, 422)

    def test_weather_rejects_gaps_null_nan_and_future_data(self):
        base = demo_weather(START, AS_OF, 'turbine_1')
        mutations = [('wind_speed_10m_ms', None), ('temperature_2m_c', float('nan')), ('pressure', 101325), ('available_at', '2026-02-01T00:00:00Z'), ('target_time', base[1]['target_time'])]
        for key, value in mutations:
            rows = copy.deepcopy(base); rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(DomainError):
                validate_weather(rows, START, AS_OF)
        with self.assertRaises(DomainError):
            validate_weather(base[:47], START, AS_OF)

    def test_invalid_ml_output_rejected(self):
        baseline = {'schema_version': OUTPUT_SCHEMA, 'turbine_id': 'turbine_1', 'weather_model': 'jma_gsm', 'alignment_confirmed': False, 'model_version': 'demo-contract-v2', 'records': [{'target_time': iso(START+timedelta(hours=i)), 'predicted_normalized_power': .5} for i in range(48)]}
        invalid = [None, {'records': []}, {**baseline, 'records': baseline['records'][:47]}]
        for key, value in [('predicted_normalized_power', 1.5), ('predicted_normalized_power', float('nan')), ('target_time', iso(START + timedelta(hours=1)))]:
            changed = copy.deepcopy(baseline); changed['records'][0][key] = value; invalid.append(changed)
        for output in invalid:
            with self.subTest(output=type(output)), patch('forecasting.services.predict', return_value=output):
                self.assertEqual(self.post().status_code, 502)

    def test_csrf_and_bearer(self):
        client = Client(enforce_csrf_checks=True)
        self.assertEqual(client.post('/api/v1/forecasts/', data=json.dumps(REQUEST), content_type='application/json').status_code, 403)
        client.get('/')
        csrf = client.cookies['csrftoken'].value
        self.assertEqual(client.post('/api/v1/forecasts/', data=json.dumps(REQUEST), content_type='application/json', HTTP_X_CSRFTOKEN=csrf).status_code, 201)
        with override_settings(API_TOKEN='secret'):
            self.assertEqual(client.post('/api/v1/forecasts/', data=json.dumps(REQUEST), content_type='application/json', HTTP_AUTHORIZATION='Bearer secret').status_code, 201)
            self.assertEqual(client.get('/api/v1/turbines/', HTTP_AUTHORIZATION='Bearer wrong').status_code, 401)

    @override_settings(DEBUG=False, API_TOKEN='secret', SECURE_SSL_REDIRECT=False)
    def test_production_requires_authentication(self):
        self.assertEqual(self.client.get('/api/v1/turbines/').status_code, 401)
        self.assertEqual(self.client.get('/api/v1/turbines/', HTTP_AUTHORIZATION='Bearer secret').status_code, 200)
        self.assertEqual(self.client.get('/').status_code, 302)

    def test_import_and_point_in_time_selection(self):
        rows = demo_weather(START, AS_OF, 'turbine_1')
        data = {'turbine_id': 'turbine_1', 'issued_at': rows[0]['issued_at'], 'available_at': rows[0]['available_at'],
                'wind_height_m': 10, 'temperature_height_m': 2, 'weather_model': 'jma_gsm', 'data_kind': 'archived_forecast', 'source_url': 'https://example.org/archive', 'records': rows}
        run, created = import_run(data)
        self.assertTrue(created)
        self.assertFalse(import_run(data)[1])
        weather = get_weather('turbine_1', START, AS_OF, 'archive', 'jma_gsm')
        self.assertEqual(weather.provenance['run_id'], run.pk)
        newer = copy.deepcopy(data)
        newer['issued_at'] = iso(AS_OF)
        newer['available_at'] = iso(AS_OF + timedelta(hours=1))
        newer['records'][0]['wind_speed_10m_ms'] = 20
        import_run(newer)
        self.assertEqual(get_weather('turbine_1', START, AS_OF, 'archive', 'jma_gsm').pk, weather.pk)
        with self.assertRaises(DomainError):
            import_run({**data, 'data_kind': 'reanalysis'})
        broken = copy.deepcopy(data); broken['records'][3]['target_time'] = broken['records'][2]['target_time']
        before = ArchivedRun.objects.count()
        with self.assertRaises(DomainError): import_run(broken)
        self.assertEqual(ArchivedRun.objects.count(), before)

    def test_open_meteo_uses_forecast_archive_and_conservative_offsets(self):
        hourly = {'time': [(START+timedelta(hours=i)).strftime('%Y-%m-%dT%H:%M') for i in range(48)]}
        units = {}
        values = {'wind_speed_10m_ms': (8, 'm/s'), 'wind_direction_10m_deg': (220, '°'), 'temperature_2m_c': (-4, '°C'), 'pressure': (950, 'hPa')}
        for day in (1, 2, 3):
            for field, variable in VARIABLES.items():
                key = f'{variable}_previous_day{day}'
                hourly[key] = [values[field][0]] * 48; units[key] = values[field][1]
        response = Mock(json=lambda: {'hourly': hourly, 'hourly_units': units})
        with patch('weather_archive.services.requests.get', return_value=response) as get:
            rows, provenance = previous_runs(self.turbine, START, AS_OF, 'jma_gsm')
        self.assertIn('previous-runs-api', get.call_args.args[0])
        self.assertEqual(provenance['timing'], 'estimated_fixed_lead')
        for row in validate_weather(rows, START, AS_OF):
            self.assertLessEqual(timestamp(row['available_at']), AS_OF)
            self.assertGreaterEqual(row['lead_time_hours'], 24)
        self.assertTrue(all('_previous_day' in variable for variable in get.call_args.kwargs['params']['hourly'].split(',')))

    def test_provider_timeout_does_not_create_snapshot(self):
        with patch('weather_archive.services.requests.get', side_effect=requests.Timeout), self.assertRaises(DomainError):
            get_weather('turbine_1', START, AS_OF, 'open_meteo')
        self.assertEqual(WeatherSnapshot.objects.count(), 0)

    def test_alignment_false_is_preserved_and_exported(self):
        run = self.post().json()
        self.assertIs(run['alignment_confirmed'], False)
        self.assertEqual(run['schema_version'], OUTPUT_SCHEMA)
        saved = Forecast.objects.get(pk=run['id'])
        self.assertIs(saved.alignment_confirmed, False)
        self.assertIn('alignment_confirmed', self.client.get(f"/api/v1/forecasts/{run['id']}/export.csv").content.decode())

    def test_live_uses_server_time_and_refresh_recomputes_it(self):
        with patch('common.contracts.django_timezone.now', return_value=AS_OF):
            run = self.post({'turbine_id': 'turbine_1', 'provider': 'demo'}).json()
        self.assertEqual(run['as_of'], iso(AS_OF))
        self.assertEqual(run['start_time'], iso(AS_OF + timedelta(hours=1)))
        self.assertNotIn('as_of', refresh_request(Forecast.objects.get(pk=run['id'])))
        self.assertEqual(self.post({**REQUEST, 'mode': 'live'}).status_code, 422)
        self.assertEqual(self.post({**REQUEST, 'issued_at': iso(AS_OF)}).status_code, 422)

    def test_ml_input_strict_contract(self):
        payload = ml_payload(get_weather('turbine_1', START, AS_OF, 'demo'))
        self.assertEqual(validate_ml_input(payload), payload)
        for value in [True, None, '5.2', float('inf'), float('nan')]:
            bad = copy.deepcopy(payload); bad['records'][0]['wind_speed_10m_ms'] = value
            with self.subTest(value=value), self.assertRaises(DomainError): validate_ml_input(bad)
        for extra in ['pressure', 'forecast_age_hours', 'issued_at', 'wind_speed_100m']:
            bad = copy.deepcopy(payload); bad['records'][0][extra] = 1
            with self.subTest(extra=extra), self.assertRaises(DomainError): validate_ml_input(bad)
        for changed in [{'issued_at': iso(AS_OF)}, {'weather_model': 'gfs_global'}, {'schema_version': '1.0'}, {'records': payload['records'][:24]}]:
            with self.assertRaises(DomainError): validate_ml_input({**payload, **changed})
        for time in [payload['records'][1]['target_time'], '2026-02-01T00:30:00Z', '2026-02-01T00:00:00']:
            bad = copy.deepcopy(payload); bad['records'][0]['target_time'] = time
            with self.assertRaises(DomainError): validate_ml_input(bad)

    def test_output_requires_boolean_alignment_and_matching_identity(self):
        payload = ml_payload(get_weather('turbine_1', START, AS_OF, 'demo'))
        from forecasting.services import predict
        response = predict(payload)
        for changed in [{'alignment_confirmed': None}, {'alignment_confirmed': 0}, {'turbine_id': 'turbine_2'}, {'weather_model': 'icon_global'}, {'schema_version': '1.0'}, {'unknown': 1}]:
            with self.assertRaises(DomainError): validate_predictions({**response, **changed}, payload)
        with self.assertRaises(DomainError): validate_predictions({k:v for k,v in response.items() if k != 'alignment_confirmed'}, payload)

    def test_legacy_weather_cannot_be_relabelled(self):
        snap = WeatherSnapshot.objects.create(id='legacy-100m', turbine=self.turbine,
            as_of=AS_OF, start_time=START, provider='demo', weather_model='jma_gsm',
            provenance={'wind_height_m': 100}, records=[])
        with self.assertRaises(DomainError): ml_payload(snap)
        fresh = get_weather('turbine_1', START, AS_OF, 'demo')
        self.assertNotEqual(fresh.pk, snap.pk)
        validate_ml_input(ml_payload(fresh))

    def test_legacy_archive_is_not_selected(self):
        ArchivedRun.objects.create(id='legacy-run', turbine=self.turbine,
            issued_at=AS_OF-timedelta(hours=12), available_at=AS_OF,
            weather_model='jma_gsm', source_url='https://example.org/archive',
            records=demo_weather(START, AS_OF, 'turbine_1'))
        with self.assertRaises(DomainError): get_weather('turbine_1', START, AS_OF, 'archive')

    def test_schemas_examples_and_metadata_endpoint(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        payload = json.loads((root/'docs/examples/ml-input.json').read_text(encoding='utf-8'))
        response = json.loads((root/'docs/examples/ml-output.json').read_text(encoding='utf-8'))
        validate_ml_input(payload)
        validate_predictions(response, payload)
        self.assertEqual(self.client.get('/api/v1/ml/metadata/').json()['weather_model'], 'jma_gsm')

    def test_live_provider_uses_exact_jma_features(self):
        hourly = {'time': [(START+timedelta(hours=i)).strftime('%Y-%m-%dT%H:%M') for i in range(48)]}
        units = {}
        for field, variable in VARIABLES.items():
            hourly[variable] = [5] * 48
            units[variable] = {'wind_speed_10m_ms': 'm/s', 'wind_direction_10m_deg': '°', 'temperature_2m_c': '°C'}[field]
        with patch('weather_archive.services.requests.get', return_value=Mock(json=lambda: {'hourly': hourly, 'hourly_units': units})) as get:
            rows, provenance = provider_forecast(self.turbine, START, AS_OF, 'jma_gsm', live=True)
        params = get.call_args.kwargs['params']
        self.assertEqual(params['models'], 'jma_gsm')
        self.assertEqual(set(params['hourly'].split(',')), set(VARIABLES.values()))
        self.assertEqual((params['wind_speed_unit'], params['temperature_unit'], params['timezone']), ('ms','celsius','UTC'))
        self.assertIsNone(rows[0]['issued_at'])
        self.assertEqual(provenance['wind_height_m'], 10)
        validate_weather(rows, START, AS_OF, historical=False)


@override_settings(ML_BACKEND='http', ML_SERVICE_URL='http://ml.test/v1/predict', ML_METADATA_URL='', ML_SERVICE_TOKEN='', DEBUG=True)
class ExternalMLTests(TestCase):
    def setUp(self):
        for n in (1, 2): Turbine.objects.create(id=f'turbine_{n}', name=f'Turbine {n}', latitude=43, longitude=78)
        self.snapshot = get_weather('turbine_2', START, AS_OF, 'demo')
        self.request = {**REQUEST, 'turbine_id': 'turbine_2', 'provider': 'archive'}
        self.metadata = {'model_version': 'mlp-test', 'supported_turbines': ['turbine_2'],
                         'weather_model': 'jma_gsm', 'field_definitions': list(VARIABLES)}
        self.output = {'schema_version': OUTPUT_SCHEMA, 'turbine_id': 'turbine_2', 'weather_model': 'jma_gsm',
                       'model_version': 'mlp-test', 'alignment_confirmed': False,
                       'records': [{'target_time': r['target_time'], 'predicted_normalized_power': .31} for r in self.snapshot.records]}

    def execute(self):
        with patch('forecasting.services.get_weather', return_value=self.snapshot):
            return execute_forecast(self.request)

    def test_http_metadata_predict_auth_and_exact_payload(self):
        with override_settings(ML_SERVICE_TOKEN='shared'), patch('forecasting.ml_client.requests.get', return_value=Mock(status_code=200, json=lambda: self.metadata)) as get, patch('forecasting.ml_client.requests.post', return_value=Mock(status_code=200, json=lambda: self.output)) as post:
            run = self.execute()
        self.assertEqual(run.status, 'completed')
        self.assertIs(run.alignment_confirmed, False)
        self.assertEqual(get.call_args.args[0], 'http://ml.test/v1/metadata')
        for call in (get, post): self.assertEqual(call.call_args.kwargs['headers']['Authorization'], 'Bearer shared')
        validate_ml_input(post.call_args.kwargs['json'])

    def test_cache_includes_loaded_model_version(self):
        with patch('forecasting.ml_client.requests.get', return_value=Mock(status_code=200, json=lambda: self.metadata)) as get, patch('forecasting.ml_client.requests.post', return_value=Mock(status_code=200, json=lambda: self.output)) as post:
            self.assertEqual(self.execute().status, 'completed')
            self.assertEqual(self.execute().status, 'completed')
            self.assertEqual(post.call_count, 1)
            self.metadata['model_version'] = self.output['model_version'] = 'mlp-new'
            self.assertEqual(self.execute().model_version, 'mlp-new')
            self.assertEqual(post.call_count, 2)
            self.assertEqual(get.call_count, 3)
            self.assertNotIn('Authorization', post.call_args.kwargs['headers'])

    def test_supported_turbines_follow_artifact(self):
        self.request['turbine_id'] = 'turbine_1'
        with patch('forecasting.ml_client.requests.get', return_value=Mock(status_code=200, json=lambda: self.metadata)), patch('forecasting.ml_client.requests.post') as post:
            run = self.execute()
            self.assertEqual(run.error['code'], 'unsupported_turbine')
            post.assert_not_called()
        from forecasting.ml_client import check_capability
        self.metadata['supported_turbines'].append('turbine_1')
        check_capability(self.metadata, 'turbine_1', 'jma_gsm')

    def test_failures_visible_without_fallback(self):
        for status in (401, 422, 503):
            with self.subTest(status=status), patch('forecasting.ml_client.requests.get', return_value=Mock(status_code=200, json=lambda: self.metadata)), patch('forecasting.ml_client.requests.post', return_value=Mock(status_code=status)):
                run = self.execute()
                self.assertEqual(run.status, 'failed')
                self.assertEqual(run.error['status'], status if status in (401,422) else 502)
                self.assertEqual(run.records, [])
                self.assertIsNone(run.alignment_confirmed)

    def test_timeout_and_invalid_metadata(self):
        with patch('forecasting.ml_client.requests.get', side_effect=requests.Timeout):
            self.assertEqual(self.execute().error['status'], 504)
        with patch('forecasting.ml_client.requests.get', return_value=Mock(status_code=200, json=lambda: {})):
            self.assertEqual(self.execute().error['code'], 'invalid_ml_metadata')

    def test_rollout_race_is_not_cached(self):
        self.output['model_version'] = 'mlp-new'
        with patch('forecasting.ml_client.requests.get', return_value=Mock(status_code=200, json=lambda: self.metadata)), patch('forecasting.ml_client.requests.post', return_value=Mock(status_code=200, json=lambda: self.output)):
            self.assertEqual(self.execute().error['code'], 'ml_model_changed')
        self.assertFalse(Forecast.objects.filter(status='completed').exists())

    def test_weather_source_mismatch_and_no_synthetic_to_real_ml(self):
        with patch('forecasting.ml_client.requests.get', return_value=Mock(status_code=200, json=lambda: self.metadata)):
            self.metadata['weather_model'] = 'icon_global'
            self.assertEqual(self.execute().error['code'], 'weather_source_mismatch')
            self.metadata['weather_model'] = 'jma_gsm'
            self.request['provider'] = 'demo'
            self.assertEqual(self.execute().error['code'], 'synthetic_weather_rejected')
