from django.db import models


class Turbine(models.Model):
    id = models.CharField(primary_key=True, max_length=64)
    name = models.CharField(max_length=100)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    location_reference = models.URLField(blank=True)

    def __str__(self):
        return self.name


class WeatherSnapshot(models.Model):
    id = models.CharField(primary_key=True, max_length=64)
    turbine = models.ForeignKey(Turbine, on_delete=models.PROTECT)
    as_of = models.DateTimeField()
    start_time = models.DateTimeField()
    provider = models.CharField(max_length=60)
    weather_model = models.CharField(max_length=160)
    provenance = models.JSONField()
    records = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [models.Index(fields=['turbine', 'as_of', 'start_time'])]


class ArchivedRun(models.Model):
    id = models.CharField(primary_key=True, max_length=64)
    turbine = models.ForeignKey(Turbine, on_delete=models.PROTECT)
    issued_at = models.DateTimeField()
    available_at = models.DateTimeField()
    weather_model = models.CharField(max_length=160)
    source_url = models.URLField(max_length=1000)
    records = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-issued_at']
