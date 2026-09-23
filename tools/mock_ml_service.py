"""Development-only HTTP contract stub. Not a trained forecasting model."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != '/v1/predict':
            self.send_error(404)
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size < 1_000_000:
                raise ValueError('Body size')
            data = json.loads(self.rfile.read(size))
            if len(data['records']) != 48:
                raise ValueError('48 records required')
            result = {'model_version': 'mock-contract-v1', 'records': [
                {'target_time': row['target_time'], 'predicted_normalized_power': 0.42} for row in data['records']]}
        except (ValueError, KeyError, TypeError):
            self.send_error(422, 'Invalid contract input')
            return
        payload = json.dumps(result).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == '__main__':
    print('Development mock: http://127.0.0.1:8001/v1/predict')
    HTTPServer(('127.0.0.1', 8001), Handler).serve_forever()
