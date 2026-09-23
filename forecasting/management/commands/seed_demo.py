from django.core.management.base import BaseCommand
from weather_archive.models import Turbine


class Command(BaseCommand):
    help = 'Create the two turbines from the case; never overwrite existing coordinates.'

    def handle(self, *args, **options):
        entries = [('turbine_1', 'Турбина 01', 43.645150, 78.535604, 'https://maps.app.goo.gl/iN6svMt69D5qRpFU9'),
                   ('turbine_2', 'Турбина 02', 43.643198, 78.538828, 'https://maps.app.goo.gl/8UQMwsYavY6nLvFY8')]
        for pk, name, lat, lon, url in entries:
            _, created = Turbine.objects.get_or_create(pk=pk, defaults={'name': name, 'latitude': lat, 'longitude': lon, 'location_reference': url})
            self.stdout.write(f'{pk}: ' + ('created' if created else 'already exists'))
