"""Development contract server, not a trained model. Default artifact supports turbine 2 only."""
import json
import os
import sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.contracts import DomainError, validate_ml_input, OUTPUT_SCHEMA
from forecasting.ml_client import FIELD_DEFINITIONS

TOKEN = os.getenv('ML_SERVICE_TOKEN', '')
TURBINES = os.getenv('MOCK_SUPPORTED_TURBINES', 'turbine_2').split(',')
VERSION = os.getenv('MOCK_MODEL_VERSION', 'mock-contract-v2')


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, data):
        payload = json.dumps(data, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def authorized(self):
        if TOKEN and self.headers.get('Authorization') != 'Bearer ' + TOKEN:
            self.reply(401, {'error': 'service token mismatch'})
            return False
        return True

    def do_GET(self):
        if not self.authorized(): return
        if self.path != '/v1/metadata':
            self.reply(404, {'error': 'not found'}); return
        self.reply(200, {'model_version': VERSION, 'supported_turbines': TURBINES,
                         'weather_model': 'jma_gsm', 'field_definitions': FIELD_DEFINITIONS})

    def do_POST(self):
        if not self.authorized(): return
        if self.path != '/v1/predict':
            self.reply(404, {'error': 'not found'}); return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size < 1_000_000: raise ValueError('body size')
            data = validate_ml_input(json.loads(self.rfile.read(size)))
            if data['turbine_id'] not in TURBINES: raise ValueError('unsupported turbine')
            self.reply(200, {'schema_version': OUTPUT_SCHEMA, 'turbine_id': data['turbine_id'],
                'weather_model': data['weather_model'], 'model_version': VERSION, 'alignment_confirmed': False,
                'records': [{'target_time': row['target_time'], 'predicted_normalized_power': .42} for row in data['records']]})
        except (ValueError, KeyError, TypeError, DomainError) as exc:
            self.reply(422, {'error': str(exc)})


if __name__ == '__main__':
    port = int(os.getenv('MOCK_PORT', '8001'))
    print(f'Development mock: http://127.0.0.1:{port}/v1/predict', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()
