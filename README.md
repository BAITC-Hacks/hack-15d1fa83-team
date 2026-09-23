# ARYS / Wind Intelligence — pre-final

**Complete application / Вся система:** [English setup](docs/PRE_FINAL.en.md) · [Запуск на русском](docs/PRE_FINAL.ru.md)

Django dashboard + JMA GSM weather + the trained model are integrated. Real HTTP forecasts, database persistence and CSV exports passed for both turbine locations. / Django-портал, погода JMA GSM и обученная модель объединены. Реальный HTTP-цикл, сохранение в БД и CSV проверены для обеих точек.

With Python 3.12+ and Docker Engine running, from this checkout / Из корня этой ветки с Python 3.12+ и запущенным Docker Engine:

```sh
python deployment/configure.py
docker compose up --build -d
docker compose exec web python manage.py createsuperuser
```

Open / Откройте **http://localhost:18080** and log in / войдите. `configure.py` generates private secrets and refuses to overwrite an existing `.env`. / Скрипт создаёт секреты и не перезаписывает существующий `.env`.

**Verification / Проверки:** 93 tests + 31 subtests; [real live HTTP results / результаты](reports/pre-final-live.json). Compose configuration validated; image build/run unverified because Docker Engine was unavailable. / Compose валиден; сборка и запуск контейнеров не проверены из-за недоступного Docker Engine.

The selected model for **both turbines**, **mlp-76540e5972a6**, is committed in [models/production/](models/production/). CPU inference needs no NVIDIA credentials.

Выбранная модель **mlp-76540e5972a6** для **обеих турбин** включена в [models/production/](models/production/). Инференс работает на CPU без ключей NVIDIA.

| Component / Компонент | English | Русский |
|---|---|---|
| Complete setup, API, integration, results and limitations | [README.en.md](README.en.md) | [README.ru.md](README.ru.md) |
| Separate model-creation toolkit and commands | [Training toolkit](training_toolkit/README.en.md) | [Инструментарий обучения](training_toolkit/README.ru.md) |

After installing runtime dependencies / После установки runtime-зависимостей:

```sh
python inference/serve.py --allow-provisional
```

The flag explicitly accepts unconfirmed measurement interval alignment. This is an integration release; full competition replay compliance remains unverified.

Флаг явно разрешает пока не подтверждённую привязку измерений к началу/концу интервала. Это релиз для интеграции; полное соответствие исторического replay условиям конкурса ещё не проверено.
