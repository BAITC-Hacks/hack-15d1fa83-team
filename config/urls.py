from django.contrib import admin
from django.urls import path, include
from forecasting import views

urlpatterns = [
    path('', views.index), path('admin/', admin.site.urls), path('accounts/', include('django.contrib.auth.urls')),
    path('api/v1/health/', views.health), path('api/v1/turbines/', views.turbines),
    path('api/v1/ml/metadata/', views.ml_metadata),
    path('api/v1/forecasts/', views.forecasts), path('api/v1/forecasts/<uuid:pk>/', views.forecast_detail),
    path('api/v1/forecasts/<uuid:pk>/refresh/', views.refresh_forecast),
    path('api/v1/forecasts/<uuid:pk>/export.csv', views.export_forecast),
    path('api/v1/forecasts/<uuid:pk>/ml-input/', views.integration_payload),
    path('api/v1/weather/forecast/', views.weather_forecast),
    path('api/v1/weather/snapshots/<str:pk>/', views.weather_detail),
    path('api/v1/weather/import/', views.weather_import),
]
