import json
from pathlib import Path
from django.core.management.base import BaseCommand, CommandError
from common.contracts import DomainError
from weather_archive.services import import_run


class Command(BaseCommand):
    help = 'Import a JSON archived forecast run (never observations or reanalysis).'

    def add_arguments(self, parser):
        parser.add_argument('path')

    def handle(self, *args, **options):
        try:
            data = json.loads(Path(options['path']).read_text(encoding='utf-8'))
            if not isinstance(data, dict):
                raise ValueError('Expected JSON object')
            run, created = import_run(data)
        except (OSError, ValueError, DomainError) as exc:
            raise CommandError(str(exc))
        self.stdout.write(f'{run.pk}: ' + ('created' if created else 'already exists'))
