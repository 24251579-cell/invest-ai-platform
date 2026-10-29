# invest-ai-platform

MVP инвестиционной аналитической AI-платформы (СРО2/СРО3): свечи MOEX ISS, новости RSS,
тональность FinBERT / RuBERT, прогноз цены закрытия на t+1 моделью LSTM с индексом
тональности (F=4) и без него (F=3), сравнение с наивным прогнозом.

Документы: `docs/mvp-plan.md`, `docs/ai-part-plan.md`, `docs/sro2-srk.docx` (+ `.txt`).

## Что уже есть

| Компонент | Статус |
|---|---|
| ETL: MOEX candles + RSS news → CSV | готово (`scripts/load_*.py`) |
| Sentiment FinBERT/RuBERT + дневной индекс | готово |
| LSTM F=3 / F=4, Model Registry, hypothesis JSON | готово |
| Web UI (FastAPI) | готово (`web/`, `scripts/run_web.py`) |
| Docker Compose: app + PostgreSQL 16 + Adminer | готово |
| Схема БД СРО2 + sync CSV → Postgres | готово (`db/init/`, `scripts/sync_db.py`) |
| Прямая запись ETL/AI в Postgres (без CSV) | ещё нет — пайплайн пишет CSV, в БД через sync |
| Streamlit Э1–Э4 / APScheduler | ещё нет (в UI сейчас FastAPI-панель) |

## Быстрый старт (Docker — рекомендуемый)

Нужны: Docker + Docker Compose, скачанные HF-модели на хосте
(путь в `config/settings.yaml` → `sentiment.local_models_dir`, по умолчанию `/home/vkotov/hf_models`).

```bash
cp .env.example .env
# при необходимости поправьте HF_HOME и local_models_dir

docker compose up --build -d
```

| Сервис | URL / параметры |
|---|---|
| Web UI | http://localhost:8000 |
| Adminer | http://localhost:8080 → System: **PostgreSQL**, Server: **`db`**, User/Pass/DB: **`invest`** |
| PostgreSQL | `localhost:5432`, user/pass/db: `invest` |

При старте контейнер `invest-ai` ждёт Postgres и синхронизирует `./data` → БД (`SYNC_DB_ON_START=1`).

```bash
docker compose logs -f invest-ai   # логи
docker compose down                # стоп (volume pgdata сохраняется)
docker compose down -v             # стоп + сброс БД
```

Тома: `./data`, `./models`, `./reports`, `./config` (ro), named volumes `pgdata` и `hf_cache`,
HF-чекпоинты с хоста (`/home/vkotov/hf_models`).

## Установка без Docker (venv, CPU)

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
cp .env.example .env               # HF_HOME, DATABASE_URL
```

Скачать RuBERT + FinBERT (~1.1 ГБ) в `local_models_dir` (см. `scripts/download_models.ps1`
или `huggingface_hub.snapshot_download`). Для Linux путь задайте в `config/settings.yaml`.

Для доступа к Postgres с хоста поднимите хотя бы БД: `docker compose up -d db adminer`.

## Пайплайн (CLI)

```bash
source .venv/bin/activate

python scripts/load_candles.py --tickers SBER      # data/raw/candles/SBER.csv
python scripts/load_news.py                        # data/raw/news.csv
python scripts/score_sentiment.py --limit 50       # sentiment_scores + daily_sentiment
python scripts/train_lstm.py --ticker SBER         # models/lstm_sber_f{3,4}/
python scripts/predict.py --ticker SBER            # data/processed/forecasts.csv
python scripts/sync_db.py                          # CSV → PostgreSQL
pytest -q
```

Те же шаги доступны кнопками в Web UI (включая «Sync → Postgres» и полный пайплайн).

## Веб-интерфейс без Docker

```bash
source .venv/bin/activate
python scripts/run_web.py          # http://127.0.0.1:8000
```

В UI: тикер, шаги пайплайна, график test (факт / LSTM / naive), метрики, прогноз t+1,
новости, лог задачи, статус Postgres.

## База данных

Схема: `db/init/01_schema.sql` — таблицы `candles`, `news`, `sentiment_scores`, `forecasts`,
`daily_sentiment`, view `v_daily_sentiment`.

```
DATABASE_URL=postgresql+psycopg://invest:invest@localhost:5432/invest
```

В контейнере URL переопределяется на `@db:5432`. Пока модули ИИ читают/пишут **CSV**
в `data/` (те же поля, что у таблиц); в Postgres данные попадают через `scripts/sync_db.py`.

## Структура репозитория

```
config/          tickers, RSS, settings
src/invest_ai/   ETL, sentiment, forecast, db
scripts/         CLI + docker_entrypoint + run_web
web/             FastAPI UI
db/init/         SQL schema (Postgres init)
models/          Model Registry
data/            CSV-артефакты (gitignore)
reports/         hypothesis_*.json
tests/           smoke / unit
```

## Принятые правила

- Язык новости: доля кириллицы ≥ 0,5 → ru (RuBERT), латиницы ≥ 0,5 → en (FinBERT).
- Оценка новости `s = p_pos − p_neg` (2.1); новость относится к торговому дню по отсечке
  19:00 МСК (2.2), после отсечки и в выходные — к следующему торговому дню.
- Дневной индекс `S = mean(s)` по тикеру (2.3); общерыночные новости — отдельный ряд `MARKET`.
- **Дни без новостей: S = 0** (нейтральный фон), число новостей хранится в `n_news`.
- Наивный прогноз: цена завтра = цена сегодня (2.19). `RMSE_rel < 1` — модель лучше наивного.
- Разбиение 70/15/15 по времени, MinMaxScaler только по train, seed = 42.

## Model Registry

`models/lstm_{ticker}_f{F}/`: `model.pt`, `config.json` (версия, признаки, гиперпараметры,
периоды выборок), `scaler.json`, `metrics.json` (MAE, RMSE, DA, RMSE_rel, эталоны),
`history.csv`, `test_predictions.csv`. Вывод по гипотезе — `reports/hypothesis_{ticker}.json`.
