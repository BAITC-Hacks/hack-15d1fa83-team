import json
import secrets
from functools import wraps
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt, csrf_protect
from .contracts import DomainError


def body(request):
    try:
        data = json.loads(request.body)
    except (ValueError, UnicodeDecodeError):
        raise DomainError('invalid_json', 'Некорректный JSON.', 400)
    if not isinstance(data, dict):
        raise DomainError('invalid_body', 'Ожидается JSON-объект.', 400)
    return data


def api(methods=('GET',)):
    def decorate(view):
        @wraps(view)
        def dispatch(request, *args, **kwargs):
            if request.method not in methods:
                response = JsonResponse({'error': {'code': 'method_not_allowed', 'message': 'Метод не поддерживается.'}}, status=405)
                response['Allow'] = ', '.join(methods)
                return response
            auth = request.headers.get('Authorization', '')
            authenticated = bool(settings.API_TOKEN) and secrets.compare_digest(auth, 'Bearer ' + settings.API_TOKEN)
            if auth and not authenticated:
                return JsonResponse({'error': {'code': 'unauthorized', 'message': 'Неверный API-токен.'}}, status=401)
            if not settings.DEBUG and not authenticated and not request.user.is_authenticated:
                return JsonResponse({'error': {'code': 'unauthorized', 'message': 'Требуется авторизация.'}}, status=401)
            def execute(req, *a, **kw):
                try:
                    return view(req, *a, **kw)
                except DomainError as exc:
                    return JsonResponse({'error': {'code': exc.code, 'message': exc.message}}, status=exc.status)
            if authenticated:
                return execute(request, *args, **kwargs)
            return csrf_protect(execute)(request, *args, **kwargs)
        return csrf_exempt(dispatch)
    return decorate
