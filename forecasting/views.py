import csv
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from common.contracts import DomainError, request_window
from common.http import api, body
from weather_archive.models import Turbine, WeatherSnapshot
from weather_archive.services import get_weather, import_run, serialize as weather_json
from .models import Forecast
from .services import execute_forecast, serialize, ml_payload


@ensure_csrf_cookie
def index(request):
    def page(req):
        return render(req, 'dashboard.html', {'ml_backend': settings.ML_BACKEND})
    return page(request) if settings.DEBUG else login_required(page)(request)


@api()
def health(request):
    return JsonResponse({'status': 'ok', 'schema_version': '1.0', 'ml_backend': settings.ML_BACKEND})


@api()
def turbines(request):
    return JsonResponse({'turbines': list(Turbine.objects.order_by('id').values('id', 'name', 'latitude', 'longitude', 'location_reference'))})


@api(('GET', 'POST'))
def forecasts(request):
    if request.method == 'GET':
        return JsonResponse({'forecasts': [serialize(r, full=False) for r in Forecast.objects.all()[:100]]})
    run = execute_forecast(body(request))
    return JsonResponse(serialize(run), status=201 if run.status == 'completed' else run.error['status'])


def find_run(pk):
    try:
        return Forecast.objects.select_related('snapshot').get(pk=pk)
    except Forecast.DoesNotExist:
        raise DomainError('not_found', 'Прогноз не найден.', 404)


@api()
def forecast_detail(request, pk):
    return JsonResponse(serialize(find_run(pk)))


@api(('POST',))
def refresh_forecast(request, pk):
    previous = find_run(pk)
    run = execute_forecast(previous.request_params, refresh=True, previous=previous)
    return JsonResponse(serialize(run), status=201 if run.status == 'completed' else run.error['status'])


@api()
def export_forecast(request, pk):
    run = find_run(pk)
    if run.status != 'completed':
        raise DomainError('not_ready', 'Доступен экспорт только завершённых прогнозов.', 409)
    response = HttpResponse(content_type='text/csv; charset=utf-8')
    response['Content-Disposition'] = f'attachment; filename="forecast-{run.pk}.csv"'
    writer = csv.writer(response)
    writer.writerow(['turbine_id', 'as_of', 'target_time', 'predicted_normalized_power', 'model_version', 'is_demo', 'snapshot_id'])
    for row in run.records:
        # Prevent spreadsheet formula execution through an external model_version.
        safe = lambda value: "'" + value if str(value).startswith(('=', '+', '-', '@', '\t', '\r')) else value
        writer.writerow([safe(run.turbine_id), run.as_of.isoformat(), row['target_time'], row['predicted_normalized_power'], safe(run.model_version), run.is_demo, run.snapshot_id])
    return response


@api(('POST',))
def weather_forecast(request):
    data = body(request)
    turbine, as_of, start = request_window(data)
    provider, model = data.get('provider', 'open_meteo'), data.get('weather_model', 'gfs_global')
    if not isinstance(provider, str) or not isinstance(model, str):
        raise DomainError('invalid_value', 'provider и weather_model должны быть строками.')
    return JsonResponse(weather_json(get_weather(turbine, start, as_of, provider, model)))


@api()
def weather_detail(request, pk):
    try:
        return JsonResponse(weather_json(WeatherSnapshot.objects.get(pk=pk)))
    except WeatherSnapshot.DoesNotExist:
        raise DomainError('not_found', 'Снимок не найден.', 404)


@api(('POST',))
def weather_import(request):
    run, created = import_run(body(request))
    return JsonResponse({'run_id': run.pk, 'created': created}, status=201 if created else 200)


@api()
def integration_payload(request, pk):
    run = find_run(pk)
    if not run.snapshot_id:
        raise DomainError('not_ready', 'Погодные данные ещё не получены.', 409)
    return JsonResponse(ml_payload(run.snapshot))
