from django.contrib import admin
from .models import Turbine, ArchivedRun, WeatherSnapshot

@admin.register(Turbine)
class TurbineAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'latitude', 'longitude')

@admin.register(ArchivedRun, WeatherSnapshot)
class ArchiveAdmin(admin.ModelAdmin):
    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False
