-- invest-ai schema (СРО2 tables: candles, news, sentiment_scores, forecasts)
SET TIME ZONE 'Europe/Moscow';

CREATE TABLE candles (
    ticker      VARCHAR(12)     NOT NULL,
    trade_date  DATE            NOT NULL,
    open        NUMERIC(18, 6)  NOT NULL CHECK (open > 0),
    high        NUMERIC(18, 6)  NOT NULL CHECK (high > 0),
    low         NUMERIC(18, 6)  NOT NULL CHECK (low > 0),
    close       NUMERIC(18, 6)  NOT NULL CHECK (close > 0),
    volume      BIGINT          NOT NULL CHECK (volume >= 0),
    value       NUMERIC(20, 2)  NOT NULL CHECK (value >= 0),
    loaded_at   TIMESTAMPTZ     NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, trade_date),
    CHECK (high >= open AND high >= close),
    CHECK (low <= open AND low <= close)
);
CREATE INDEX candles_trade_date_idx ON candles (trade_date);

CREATE TABLE news (
    id           BIGSERIAL PRIMARY KEY,
    source       VARCHAR(64)  NOT NULL,
    url          TEXT         NOT NULL UNIQUE,
    title        TEXT         NOT NULL,
    body         TEXT,
    published_at TIMESTAMPTZ  NOT NULL,
    lang         CHAR(2),
    ticker       VARCHAR(12),
    content_hash CHAR(64)     NOT NULL UNIQUE,
    loaded_at    TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX news_published_at_idx ON news (published_at);
CREATE INDEX news_ticker_idx ON news (ticker);

CREATE TABLE sentiment_scores (
    id         BIGSERIAL PRIMARY KEY,
    news_id    BIGINT       NOT NULL REFERENCES news (id) ON DELETE CASCADE,
    model_name VARCHAR(128) NOT NULL,
    label      VARCHAR(8)   NOT NULL CHECK (label IN ('positive', 'negative', 'neutral')),
    p_pos      REAL         NOT NULL CHECK (p_pos >= 0 AND p_pos <= 1),
    p_neg      REAL         NOT NULL CHECK (p_neg >= 0 AND p_neg <= 1),
    p_neu      REAL         NOT NULL CHECK (p_neu >= 0 AND p_neu <= 1),
    score      REAL         NOT NULL CHECK (score >= -1 AND score <= 1),
    scored_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    UNIQUE (news_id, model_name),
    CHECK (abs((p_pos + p_neg + p_neu) - 1.0) <= 1e-3)
);

CREATE TABLE forecasts (
    id             BIGSERIAL PRIMARY KEY,
    ticker         VARCHAR(12)    NOT NULL,
    forecast_date  DATE           NOT NULL,
    target_date    DATE           NOT NULL,
    pred_close     NUMERIC(18, 6) NOT NULL,
    pred_direction SMALLINT       NOT NULL CHECK (pred_direction IN (0, 1)),
    model_version  VARCHAR(64)    NOT NULL,
    use_sentiment  BOOLEAN        NOT NULL,
    created_at     TIMESTAMPTZ    NOT NULL DEFAULT now(),
    UNIQUE (ticker, target_date, model_version, use_sentiment),
    FOREIGN KEY (ticker, forecast_date) REFERENCES candles (ticker, trade_date)
);
CREATE INDEX forecasts_ticker_target_idx ON forecasts (ticker, target_date);

-- Approximate daily index (cutoff/trading calendar applied in Python; this view is a fallback)
CREATE OR REPLACE VIEW v_daily_sentiment AS
SELECT
    COALESCE(n.ticker, 'MARKET') AS ticker,
    ((n.published_at AT TIME ZONE 'Europe/Moscow')::date) AS trade_date,
    AVG(s.score)::DOUBLE PRECISION AS sentiment,
    COUNT(*)::INTEGER AS n_news
FROM sentiment_scores s
JOIN news n ON n.id = s.news_id
GROUP BY 1, 2;

-- Materialized daily index written by the Python aggregator (exact formula 2.2–2.3)
CREATE TABLE daily_sentiment (
    ticker     VARCHAR(12) NOT NULL,
    trade_date DATE        NOT NULL,
    sentiment  DOUBLE PRECISION NOT NULL,
    n_news     INTEGER     NOT NULL CHECK (n_news >= 0),
    PRIMARY KEY (ticker, trade_date)
);
