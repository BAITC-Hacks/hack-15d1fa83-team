"""Initialize the isolated web database/static assets and start a WSGI server."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

def main():
    import django
    django.setup()
    from django.core.management import call_command
    from waitress import serve
    call_command('migrate', interactive=False)
    # The historical command name seeds turbine identities/coordinates only.
    call_command('seed_demo')
    call_command('collectstatic', interactive=False, verbosity=0)
    from config.wsgi import application
    serve(application, host=os.getenv('WEB_HOST', '0.0.0.0'), port=int(os.getenv('PORT', '8000')), threads=4)

if __name__ == '__main__':
    main()
