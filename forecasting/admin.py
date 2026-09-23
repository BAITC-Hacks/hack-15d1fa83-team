from django.contrib import admin
from .models import Forecast

@admin.register(Forecast)
class ForecastAdmin(admin.ModelAdmin):
    list_display = ('id', 'turbine', 'as_of', 'status', 'model_version', 'is_demo')
    list_filter = ('status', 'is_demo', 'turbine')

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False
