import csv
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from django.core.management.base import BaseCommand, CommandError
from forecasting.services import execute_forecast
from common.contracts import iso


class Command(BaseCommand):
    help = 'Sequential daily 48h forecast replay; default February 2026. No training or metrics without actual targets.'

    def add_arguments(self, parser):
        parser.add_argument('--start', default='2026-02-01')
        parser.add_argument('--end', default='2026-02-28')
        parser.add_argument('--provider', choices=['demo', 'open_meteo', 'archive'], default='archive')
        parser.add_argument('--weather-model', default='gfs_global')
        parser.add_argument('--turbine', default='turbine_1')
        parser.add_argument('--output', default='output/backtest.csv')

    def handle(self, *args, **opts):
        try:
            day, end = date.fromisoformat(opts['start']), date.fromisoformat(opts['end'])
        except ValueError:
            raise CommandError('Dates must use YYYY-MM-DD.')
        if end < day or (end-day).days > 365:
            raise CommandError('Choose an ordered period of up to 366 days.')
        target = Path(opts['output'])
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['forecast_id', 'turbine_id', 'as_of', 'target_time', 'predicted_normalized_power', 'model_version', 'is_demo', 'snapshot_id'])
            while day <= end:
                start = datetime.combine(day, datetime.min.time(), timezone.utc)
                as_of = start - timedelta(hours=6)
                run = execute_forecast({'turbine_id': opts['turbine'], 'as_of': iso(as_of), 'start_time': iso(start),
                    'provider': opts['provider'], 'weather_model': opts['weather_model']})
                if run.status != 'completed':
                    raise CommandError(f'{day}: {run.error}. Partial results retained at {target}.')
                for record in run.records:
                    writer.writerow([str(run.pk), run.turbine_id, iso(as_of), record['target_time'], record['predicted_normalized_power'], run.model_version, run.is_demo, run.snapshot_id])
                self.stdout.write(f'{day}: {run.pk}')
                day += timedelta(days=1)
        self.stdout.write(self.style.SUCCESS(str(target)))
