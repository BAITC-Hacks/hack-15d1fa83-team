"""Authenticated local web health check; does not print the token."""
import json
import os
import urllib.request

request = urllib.request.Request('http://127.0.0.1:8000/api/v1/health/',
    headers={'Authorization': 'Bearer ' + os.environ['API_TOKEN']})
with urllib.request.urlopen(request, timeout=3) as response:
    data = json.load(response)
    if data.get('status') != 'ok' or data.get('ml_backend') != 'http':
        raise SystemExit('Web health check failed')
