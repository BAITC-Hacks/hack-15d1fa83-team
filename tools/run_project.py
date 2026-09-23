"""Run the local Django + CPU ML stack. No training, cloud jobs or hidden demo fallback."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def available_port(port):
    if not 1 <= port <= 65535:
        raise ValueError('Ports must be between 1 and 65535')
    with socket.socket() as sock:
        try:
            sock.bind(('127.0.0.1', port))
        except OSError:
            raise RuntimeError(f'Port {port} is occupied; choose another port. Existing processes are not stopped.') from None


def wait_ready(process, url, timeout=40):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'Service exited with code {process.returncode}; see its log above')
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(.2)
    raise RuntimeError(f'Service did not become ready within {timeout}s')


def stop_children(children):
    for child in reversed(children):
        if child.poll() is None:
            child.terminate()
    for child in reversed(children):
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-provisional', action='store_true',
                        default=os.getenv('ALLOW_PROVISIONAL_MODEL') == '1',
                        help='Explicitly accept the committed model\'s unconfirmed timestamp alignment')
    parser.add_argument('--demo', action='store_true', help='Explicit offline synthetic UI mode, no real ML')
    parser.add_argument('--web-port', type=int, default=8000)
    parser.add_argument('--ml-port', type=int, default=8001)
    args = parser.parse_args()
    children = []
    try:
        if not args.demo:
            if args.web_port == args.ml_port:
                raise ValueError('Django and ML must have different ports')
            model = Path(os.getenv('MODEL_PATH') or ROOT / 'models/production/model.json')
            if not model.is_absolute():
                model = ROOT / model
            bundle = json.loads(model.read_text(encoding='utf-8'))
            if not bundle['dataset_metadata'].get('alignment_confirmed') and not args.allow_provisional:
                raise RuntimeError('Model alignment is unconfirmed. Review README limitations, then use --allow-provisional explicitly.')
            available_port(args.ml_port)
        available_port(args.web_port)
        env = dict(os.environ)
        env['ML_BACKEND'] = 'demo' if args.demo else 'http'
        env['ML_SERVICE_URL'] = f'http://127.0.0.1:{args.ml_port}/v1/predict'
        env['ML_METADATA_URL'] = ''
        # Integrated flow must not activate the legacy ML weather downloader.
        env['WEATHER_PROVIDER_URL'] = ''
        if not args.demo:
            command = [sys.executable, str(ROOT/'inference/serve.py'), '--port', str(args.ml_port), '--model', str(model)]
            if args.allow_provisional:
                command.append('--allow-provisional')
            children.append(subprocess.Popen(command, cwd=ROOT, env=env))
            wait_ready(children[-1], f'http://127.0.0.1:{args.ml_port}/health')
        children.append(subprocess.Popen([sys.executable, str(ROOT/'manage.py'), 'runserver',
            f'127.0.0.1:{args.web_port}', '--noreload'], cwd=ROOT, env=env))
        wait_ready(children[-1], f'http://127.0.0.1:{args.web_port}/')
        print(f'ARYS ready: http://127.0.0.1:{args.web_port} | Ctrl+C stops both processes', flush=True)
        while all(p.poll() is None for p in children):
            time.sleep(.25)
        raise RuntimeError('A service stopped; the other service will also be stopped.')
    except KeyboardInterrupt:
        return 0
    except (RuntimeError, ValueError, OSError, KeyError) as exc:
        print(f'Startup failed: {exc}', file=sys.stderr)
        return 1
    finally:
        stop_children(children)


if __name__ == '__main__':
    raise SystemExit(main())
