from django.core.management.base import BaseCommand
from forecasting.models import Forecast
from forecasting.services import execute_forecast, refresh_request


class Command(BaseCommand):
    help = 'One agent sweep: refetch weather for latest unique windows and recalculate changed inputs. Schedule externally.'

    def add_arguments(self, parser):
        parser.add_argument('--limit', type=int, default=20)

    def handle(self, *args, **opts):
        seen = set()
        count = 0
        candidates = list(Forecast.objects.filter(status='completed')[:max(1, min(opts['limit'], 100)) * 10])
        for previous in candidates:
            key = (previous.turbine_id, previous.as_of, previous.start_time, previous.request_params.get('provider'), previous.request_params.get('weather_model'))
            if key in seen:
                continue
            seen.add(key)
            run = execute_forecast(refresh_request(previous), refresh=True)
            self.stdout.write(f'{run.pk}: {run.status}; changed={run.snapshot_id != previous.snapshot_id}')
            count += 1
            if count >= opts['limit']:
                break
