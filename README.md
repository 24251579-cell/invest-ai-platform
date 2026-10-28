# invest-ai-platform

MVP инвестиционной аналитической AI-платформы (СРО2/СРО3): свечи MOEX ISS, новости RSS,
тональность FinBERT / RuBERT, прогноз цены закрытия на t+1 моделью LSTM с индексом
тональности (F=4) и без него (F=3), сравнение с наивным прогнозом.

Документы: `docs/` (`mvp-plan.md`, `ai-part-plan.md`, `sro2-srk.docx`).

## Установка (Windows, CPU)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
copy .env.example .env
python scripts\download_models.py   # ~1.1 ГБ в C:\hf_cache, с докачкой
```

## Запуск

```powershell
python scripts\load_candles.py --tickers SBER      # data/raw/candles/SBER.csv
python scripts\load_news.py                        # data/raw/news.csv
python scripts\score_sentiment.py --limit 50       # sentiment_scores.csv + daily_sentiment.csv
python scripts\train_lstm.py --ticker SBER         # models/lstm_sber_f3, models/lstm_sber_f4
python scripts\predict.py --ticker SBER            # data/processed/forecasts.csv
pytest
```

Пока база данных PostgreSQL не готова, все этапы обмениваются CSV-файлами в `data/`
с теми же полями, что и таблицы `candles`, `news`, `sentiment_scores`, `forecasts`.

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
