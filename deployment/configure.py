"""Create a private pre-final Compose environment; never overwrite existing secrets."""
import os
from pathlib import Path
import secrets

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    path = ROOT / '.env'
    content = '\n'.join([
        'COMPOSE_PROJECT_NAME=windpower-pre-final', 'WEB_PORT=18080',
        'DJANGO_SECRET_KEY=' + secrets.token_urlsafe(48),
        'API_TOKEN=' + secrets.token_urlsafe(32),
        'ML_SERVICE_TOKEN=' + secrets.token_urlsafe(32), '',
    ])
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        print('.env already exists; existing values were preserved.')
    else:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as file:
            file.write(content)
        print('Created private .env for pre-final. Tokens were not printed. Do not commit this file.')
