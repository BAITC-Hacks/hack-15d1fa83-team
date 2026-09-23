"""Serve the committed model. No training packages, weather API or GPU needed."""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8001)
    p.add_argument('--model', default=os.getenv('MODEL_PATH') or str(ROOT / 'models/production/model.json'))
    p.add_argument('--allow-provisional', action='store_true', default=os.getenv('ALLOW_PROVISIONAL_MODEL') == '1')
    a = p.parse_args()
    import uvicorn
    from windpower.api import create_app
    uvicorn.run(create_app(model_path=a.model, allow_provisional=a.allow_provisional), host=a.host, port=a.port)


if __name__ == '__main__':
    main()
