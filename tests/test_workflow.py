import copy
import json
from datetime import timedelta
from unittest.mock import Mock, patch
import requests
from django.test import TestCase, Client, override_settings
from common.contracts import DomainError, timestamp, iso, validate_weather
from weather_archive.models import Turbine, WeatherSnapshot, ArchivedRun
from weather_archive.services import demo_weather, get_weather, import_run, previous_runs, VARIABLES
from forecasting.models import Forecast
from forecasting.services import execute_forecast, ml_payload

AS_OF = timestamp('2026-01-31T18:00:00Z')
START = timestamp('2026-02-01T00:00:00Z')
REQUEST = {'turbine_id': 'turbine_1', 'as_of': iso(AS_OF), 'start_time': iso(START), 'provider': 'demo'}


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
        self.assertEqual(len(run['trace']), 5)
        self.assertEqual(self.client.get(f"/api/v1/forecasts/{run['id']}/").json()['records'], run['records'])
        csv = self.client.get(f"/api/v1/forecasts/{run['id']}/export.csv")
        self.assertEqual(len(csv.content.decode().splitlines()), 49)
        payload = self.client.get(f"/api/v1/forecasts/{run['id']}/ml-input/").json()
        self.assertEqual(set(payload), {'turbine_id', 'records'})
        self.assertEqual(set(payload['records'][0]), {'target_time', 'wind_speed', 'wind_direction', 'temperature', 'pressure', 'weather_model', 'forecast_age_hours'})
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
        rows[0]['wind_speed'] += 1
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
        mutations = [('wind_speed', None), ('temperature', float('nan')), ('pressure', 101325), ('available_at', '2026-02-01T00:00:00Z'), ('target_time', base[1]['target_time'])]
        for key, value in mutations:
            rows = copy.deepcopy(base); rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(DomainError):
                validate_weather(rows, START, AS_OF)
        with self.assertRaises(DomainError):
            validate_weather(base[:47], START, AS_OF)

    @override_settings(ML_BACKEND='http', ML_SERVICE_TOKEN='test-ml-token')
    def test_http_ml_contract(self):
        output = {'model_version': 'wind-v1', 'records': [{'target_time': iso(START + timedelta(hours=i)), 'predicted_normalized_power': .42} for i in range(48)]}
        with patch('forecasting.services.requests.post', return_value=Mock(json=lambda: output)) as post:
            response = self.post()
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()['model_version'], 'wind-v1')
        self.assertEqual(post.call_args.kwargs['headers']['Authorization'], 'Bearer test-ml-token')
        self.assertEqual(len(post.call_args.kwargs['json']['records']), 48)

    @override_settings(ML_BACKEND='http')
    def test_ml_failure_is_saved_and_never_falls_back(self):
        with patch('forecasting.services.requests.post', side_effect=requests.Timeout):
            response = self.post()
        self.assertEqual(response.status_code, 504)
        self.assertEqual(response.json()['status'], 'failed')
        self.assertEqual(Forecast.objects.get().status, 'failed')
        self.assertFalse(Forecast.objects.get().records)

    def test_invalid_ml_output_rejected(self):
        baseline = {'model_version': 'bad', 'records': [{'target_time': iso(START+timedelta(hours=i)), 'predicted_normalized_power': .5} for i in range(48)]}
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
                'weather_model': 'gfs_global', 'data_kind': 'archived_forecast', 'source_url': 'https://example.org/archive', 'records': rows}
        run, created = import_run(data)
        self.assertTrue(created)
        self.assertFalse(import_run(data)[1])
        weather = get_weather('turbine_1', START, AS_OF, 'archive', 'gfs_global')
        self.assertEqual(weather.provenance['run_id'], run.pk)
        newer = copy.deepcopy(data)
        newer['issued_at'] = iso(AS_OF)
        newer['available_at'] = iso(AS_OF + timedelta(hours=1))
        newer['records'][0]['wind_speed'] = 20
        import_run(newer)
        self.assertEqual(get_weather('turbine_1', START, AS_OF, 'archive', 'gfs_global').pk, weather.pk)
        with self.assertRaises(DomainError):
            import_run({**data, 'data_kind': 'reanalysis'})
        broken = copy.deepcopy(data); broken['records'][3]['target_time'] = broken['records'][2]['target_time']
        before = ArchivedRun.objects.count()
        with self.assertRaises(DomainError): import_run(broken)
        self.assertEqual(ArchivedRun.objects.count(), before)

    def test_open_meteo_uses_forecast_archive_and_conservative_offsets(self):
        hourly = {'time': [(START+timedelta(hours=i)).strftime('%Y-%m-%dT%H:%M') for i in range(48)]}
        units = {}
        values = {'wind_speed': (8, 'm/s'), 'wind_direction': (220, '°'), 'temperature': (-4, '°C'), 'pressure': (950, 'hPa')}
        for day in (1, 2, 3):
            for field, variable in VARIABLES.items():
                key = f'{variable}_previous_day{day}'
                hourly[key] = [values[field][0]] * 48; units[key] = values[field][1]
        response = Mock(json=lambda: {'hourly': hourly, 'hourly_units': units})
        with patch('weather_archive.services.requests.get', return_value=response) as get:
            rows, provenance = previous_runs(self.turbine, START, AS_OF, 'gfs_global')
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
