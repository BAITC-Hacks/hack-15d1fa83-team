"""Start two isolated real HTTP services, verify contracts, optionally fetch live JMA.

python deployment/verify_stack.py --live --report reports/local-stack.json
Requires the web + inference dependencies, no Torch or GPU. No existing servers
are stopped or databases reused. --live uses the public Open-Meteo service.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def request(url, token='', payload=None):
    headers = {'Authorization': 'Bearer '+token} if token else {}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        headers['Content-Type'] = 'application/json'
    try:
        response = urlopen(Request(url, data=data, headers=headers), timeout=45)
    except HTTPError as exc:
        response = exc
    with response:
        body = response.read()
        return response.code, (json.loads(body) if 'application/json' in response.headers.get('Content-Type','') else body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = {'model_version':'mlp-76540e5972a6','transport':'real_loopback_HTTP',
              'debug':False,'live_requested':args.live,'checks':[], 'live':[]}
    processes = []
    with tempfile.TemporaryDirectory(prefix='windpower-stack-') as temp:
        temp = Path(temp)
        mlport, webport = free_port(), free_port()
        while mlport == webport:
            webport = free_port()
        ml, web = f'http://127.0.0.1:{mlport}', f'http://127.0.0.1:{webport}'
        token = secrets.token_urlsafe(32)
        env = {**os.environ,'ML_SERVICE_TOKEN':token,'API_TOKEN':token,
               'DJANGO_SECRET_KEY':secrets.token_urlsafe(48),'DJANGO_DEBUG':'0',
               'DJANGO_ALLOWED_HOSTS':'127.0.0.1,localhost','DJANGO_SECURE_COOKIES':'0',
               'DJANGO_SECURE_SSL_REDIRECT':'0','DJANGO_DB_PATH':str(temp/'db.sqlite3'),
               'ML_BACKEND':'http','ML_SERVICE_URL':ml+'/v1/predict','ML_METADATA_URL':'',
               'MODEL_PATH':str(ROOT/'models/production/model.json'),'ALLOW_PROVISIONAL_MODEL':'1',
               'WEB_HOST':'127.0.0.1','PORT':str(webport),'PYTHONUNBUFFERED':'1',
               'WEATHER_PROVIDER_URL':'',
               'PYTHONPATH':os.pathsep.join([str(ROOT/'src'),str(ROOT),*sys.path])}
        logs = []
        try:
            for label, command in [('ml',['-m','uvicorn','windpower.api:app','--host','127.0.0.1','--port',str(mlport)]),
                                   ('web',[str(ROOT/'deployment/start_web.py')])]:
                log = open(temp/f'{label}.log','w+',encoding='utf-8')
                logs.append(log)
                # Direct interpreter avoids orphaning a Windows venv launcher child.
                processes.append(subprocess.Popen([getattr(sys,'_base_executable',sys.executable),*command],
                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT))
            for endpoint in [ml+'/health',web+'/api/v1/health/']:
                deadline = time.monotonic()+45
                while True:
                    try:
                        if request(endpoint,token)[0] == 200:
                            break
                    except (URLError,TimeoutError):
                        pass
                    if time.monotonic()>deadline or any(p.poll() is not None for p in processes):
                        raise RuntimeError('Server failed to start')
                    time.sleep(.2)
            assert request(web+'/api/v1/turbines/')[0] == 401
            assert request(ml+'/v1/metadata')[0] == 401
            assert request(web+'/static/app.js')[0] == 200
            assert request(web+'/accounts/login/')[0] == 200
            status, meta = request(web+'/api/v1/ml/metadata/',token)
            assert status == 200 and meta['model_version'] == report['model_version'], meta
            report['checks'].extend(['both_servers_healthy','api_authentication','production_static_assets',
                                     'login_page','real_metadata_mapping'])
            for turbine in ['turbine_1','turbine_2']:
                payload = json.loads((ROOT/f'inference/examples/{turbine}-48h-request.json').read_text())
                code, predicted = request(ml+'/v1/predict',token,payload)
                assert code == 200 and len(predicted['records']) == 48, predicted
                report['checks'].append(turbine+'_real_model_48_hours')
                if args.live:
                    code, run = request(web+'/api/v1/forecasts/',token,{
                        'turbine_id':turbine,'mode':'live','provider':'open_meteo'})
                    if code != 201 or run.get('status') != 'completed':
                        report['live'].append({'turbine_id':turbine,'passed':False,'http_status':code,
                                               'error':run.get('error',run)})
                        continue
                    assert run['is_demo'] is False and len(run['records']) == 48
                    assert run['model_version'] == report['model_version']
                    assert run['alignment_confirmed'] is False
                    assert run['provenance']['timing'] == 'live_retrieval'
                    assert all(row['issued_at'] is None for row in run['weather'])
                    base = web+'/api/v1/forecasts/'+run['id']
                    assert request(base+'/',token)[1]['records'] == run['records']
                    assert len(request(base+'/export.csv',token)[1].splitlines()) == 49
                    forwarded = request(base+'/ml-input/',token)[1]
                    direct = request(ml+'/v1/predict',token,forwarded)[1]
                    assert [r['predicted_normalized_power'] for r in direct['records']] == [r['predicted_normalized_power'] for r in run['records']]
                    report['live'].append({'turbine_id':turbine,'passed':True,'hours':48,
                        'start_time':run['start_time'],'provider':run['provenance']['source_url'],
                        'coordinates':{k:run['provenance']['request_params'][k] for k in ['latitude','longitude']},
                        'model_version':run['model_version'],'alignment_confirmed':False,
                        'saved_detail_and_csv':True,'direct_model_matches':True})
            report['passed'] = all(row['passed'] for row in report['live'])
        except Exception as exc:
            report['passed'] = False
            report['error'] = type(exc).__name__+': '+str(exc)
            for log in logs:
                log.flush(); log.seek(0)
                print(log.read()[-5000:],file=sys.stderr)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=10)
            for log in logs:
                log.close()
    encoded = json.dumps(report,ensure_ascii=False,indent=2)
    print(encoded)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(encoded+'\n',encoding='utf-8')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
