# Wind-power ML / Прогноз мощности ВЭС

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
