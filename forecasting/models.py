import uuid
from django.db import models


class Forecast(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    turbine = models.ForeignKey('weather_archive.Turbine', on_delete=models.PROTECT)
    snapshot = models.ForeignKey('weather_archive.WeatherSnapshot', null=True, on_delete=models.PROTECT)
    as_of = models.DateTimeField()
    start_time = models.DateTimeField()
    status = models.CharField(max_length=20, default='running')
    alignment_confirmed = models.BooleanField(null=True, default=None)
    response_schema_version = models.CharField(max_length=40, blank=True)
    model_version = models.CharField(max_length=160, blank=True)
    is_demo = models.BooleanField(default=False)
    records = models.JSONField(default=list)
    trace = models.JSONField(default=list)
    analysis = models.JSONField(default=dict)
    error = models.JSONField(default=dict)
    request_params = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
