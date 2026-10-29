(() => {
  const TICKERS = Object.fromEntries(window.INVEST_AI.tickers);
  const PAGES = { dashboard: "Дашборд", ticker: "Тикер", news: "Новости", quality: "Качество модели", pipeline: "Пайплайн" };
  const PERIODS = [["1w", "1 нед"], ["1m", "1 мес"], ["3m", "3 мес"], ["6m", "6 мес"], ["1y", "1 год"]];
  const MODELS = [["4", "LSTM F=4 (с тональностью)"], ["3", "LSTM F=3 (без тональности)"]];
  const TONES = { positive: "Положительная", neutral: "Нейтральная", negative: "Отрицательная", none: "не оценена" };
  const FEATURES = { close: "цена", volume: "объём", ret: "доходность", sentiment: "тональность" };
  const STEPS = [
    ["candles", "Свечи", "MOEX ISS → data/raw/candles"],
    ["news", "Новости", "RSS-ленты → data/raw/news.csv"],
    ["sentiment", "Тональность", "RuBERT / FinBERT, дневной индекс"],
    ["train", "Обучение LSTM", "модели F=3 и F=4 для тикера"],
    ["predict", "Прогноз t+1", "data/processed/forecasts.csv"],
    ["sync_db", "Sync → PostgreSQL", "ручной полный sync CSV → БД"],
  ];
  const NEWS_PAGE = 40;

  const state = {
    page: "dashboard",
    ticker: "SBER",
    newsTicker: "ALL",
    period: "3m",
    features: 4,
    forecast: true,
    tones: { positive: true, neutral: true, negative: true },
  };
  let status = null;
  let renderSeq = 0;
  let newsQuery = null;
  let newsShown = 0;

  const $ = (sel, root = document) => root.querySelector(sel);
  const content = $("#content");

  // --- formatting ----------------------------------------------------------

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const num = (v, d = 2) => Number(v).toLocaleString("ru-RU", { minimumFractionDigits: d, maximumFractionDigits: d });
  const round = (v, d = 2) => Math.round(v * 10 ** d) / 10 ** d;
  const signed = (v, d = 2) => {
    const r = round(v, d);
    return (r > 0 ? "+" : r < 0 ? "−" : "") + num(Math.abs(r), d);
  };
  const rub = (v) => `${num(v, Math.abs(v) >= 1000 ? 0 : 2)} ₽`;
  const pct = (v) => `${signed(v)} %`;
  const cls = (v) => (v == null ? "muted" : round(v) > 0 ? "positive" : round(v) < 0 ? "negative" : "");
  const periodLabel = () => PERIODS.find(([k]) => k === state.period)[1];

  const MSK = { timeZone: "Europe/Moscow" };
  const fmtDateTime = (iso) => new Intl.DateTimeFormat("ru-RU", { ...MSK, day: "2-digit", month: "2-digit", year: "numeric", hour: "2-digit", minute: "2-digit" }).format(new Date(iso));
  const fmtShortDateTime = (iso) => new Intl.DateTimeFormat("ru-RU", { ...MSK, day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" }).format(new Date(iso));
  const fmtDay = (d) => d.split("-").reverse().join(".");
  const fmtAxisDay = (d) => new Intl.DateTimeFormat("ru-RU", { day: "2-digit", month: "short", timeZone: "UTC" }).format(new Date(`${d}T00:00:00Z`)).replace(".", "");

  async function api(path, opts) {
    const res = await fetch(path, opts);
    if (!res.ok) {
      let msg = res.statusText;
      try { msg = (await res.json()).detail || msg; } catch { /* plain text */ }
      throw new Error(msg);
    }
    return res.json();
  }

  // --- small building blocks -----------------------------------------------

  const metric = (label, value, accent = "") => `<div class="metric"><span>${label}</span><strong class="${accent}">${value}</strong></div>`;
  const sectionHead = (title, sub, right = "") => `<div class="section-head"><div><h2>${title}</h2><p>${sub}</p></div>${right}</div>`;
  const notice = (title, text) => `<div class="notice neutral-notice"><b>${title}</b><span>${text}</span></div>`;
  const emptyState = (icon, title, text, action = "") => `<div class="empty"><div>${icon}</div><h2>${title}</h2><p>${text}</p>${action}</div>`;
  const goPipeline = `<button class="primary" data-go="pipeline">Открыть пайплайн</button>`;
  const toneMark = (t) => (t === "positive" ? "＋" : t === "negative" ? "−" : "●");

  function loadingHtml() {
    return `<div class="loading"><div class="spinner"></div><span>Загружаем данные…</span>
      <div class="skeleton metrics-skel">${"<i></i>".repeat(4)}</div>
      <div class="skeleton chart-skel"></div>
      <div class="skeleton rows-skel">${"<i></i>".repeat(5)}</div></div>`;
  }

  // --- SVG charts ----------------------------------------------------------

  const scale = (d0, d1, r0, r1) => (v) => (d1 === d0 ? (r0 + r1) / 2 : r0 + ((v - d0) * (r1 - r0)) / (d1 - d0));
  const tickDigits = (range) => (range < 5 ? 2 : range < 50 ? 1 : 0);

  function xLabels(dates, x, y, count = 5) {
    if (!dates.length) return "";
    const idx = [...new Set(Array.from({ length: Math.min(count, dates.length) }, (_, k) =>
      Math.round((k * (dates.length - 1)) / Math.max(count - 1, 1))))];
    return idx.map((i, k) => {
      const anchor = k === 0 ? "start" : k === idx.length - 1 ? "end" : "middle";
      return `<text x="${x(i)}" y="${y}" text-anchor="${anchor}">${fmtAxisDay(dates[i])}</text>`;
    }).join("");
  }

  function candleChart(c, forecast) {
    const L = 38, R = 852, top = 25, bottom = 248, vTop = 266, vBottom = 300;
    const n = c.dates.length;
    const slots = n + (forecast ? 2 : 0);
    const step = (R - L) / slots;
    const x = (i) => L + step * (i + 0.5);
    let lo = Math.min(...c.low), hi = Math.max(...c.high);
    if (forecast) { lo = Math.min(lo, forecast.pred_close); hi = Math.max(hi, forecast.pred_close); }
    const pad = (hi - lo) * 0.06 || 1;
    lo -= pad; hi += pad;
    const y = scale(lo, hi, bottom, top);
    const vmax = Math.max(...c.volume) || 1;
    const bw = Math.max(1, Math.min(10, step * 0.62));
    const digits = tickDigits(hi - lo);

    const grid = [0, 1, 2, 3, 4].map((k) => {
      const v = lo + ((hi - lo) * k) / 4;
      return `<line x1="${L}" y1="${y(v)}" x2="${R}" y2="${y(v)}" class="gridline"/><text x="${R + 6}" y="${y(v) + 3}">${num(v, digits)}</text>`;
    }).join("");
    const candles = c.dates.map((_, i) => {
      const up = c.close[i] >= c.open[i];
      const y0 = y(Math.max(c.open[i], c.close[i])), y1 = y(Math.min(c.open[i], c.close[i]));
      return `<g class="${up ? "candle-up" : "candle-down"}"><line x1="${x(i)}" y1="${y(c.high[i])}" x2="${x(i)}" y2="${y(c.low[i])}"/><rect x="${x(i) - bw / 2}" y="${y0}" width="${bw}" height="${Math.max(1, y1 - y0)}"/></g>`;
    }).join("");
    const volume = c.dates.map((_, i) => {
      const h = Math.max(1, (c.volume[i] / vmax) * (vBottom - vTop));
      return `<rect x="${x(i) - bw / 2}" y="${vBottom - h}" width="${bw}" height="${h}" class="${c.close[i] >= c.open[i] ? "volume-up" : "volume-down"}"/>`;
    }).join("");
    let fc = "";
    if (forecast) {
      const x0 = x(n - 1), y0 = y(c.close[n - 1]), x1 = x(n), y1 = y(forecast.pred_close);
      fc = `<rect x="${x0 + step / 2}" y="${top}" width="${R - x0 - step / 2}" height="${bottom - top}" class="horizon"/>
        <path d="M ${x0} ${y0} C ${x0 + step * 0.6} ${y0}, ${x1 - step * 0.6} ${y1}, ${x1} ${y1}" class="forecast-line"/>
        <circle cx="${x1}" cy="${y1}" r="4" class="forecast-point"/>
        <text x="${x1}" y="${y1 - 10}" text-anchor="end" class="forecast-label">${rub(forecast.pred_close)}</text>`;
    }
    return `<div class="chart-wrap"><svg viewBox="0 0 900 320" class="price-chart">
      ${grid}${fc}${candles}<line x1="${L}" y1="258" x2="${R}" y2="258" class="axis"/>${volume}
      ${xLabels(c.dates, x, 316)}
    </svg><div class="legend"><span><i class="legend-actual"></i>Цена</span>${forecast ? `<span><i class="legend-forecast"></i>Прогноз LSTM F=${forecast.features} на ${fmtDay(forecast.target_date)}</span>` : ""}<span class="chart-period">1D</span></div></div>`;
  }

  function lineChart({ dates, series, zero = false, points = false, W = 500, H = 205 }) {
    const L = 40, R = W - 20, top = 18, bottom = H - 38;
    const values = series.flatMap((s) => s.values);
    let lo = Math.min(...values, ...(zero ? [0] : [])), hi = Math.max(...values, ...(zero ? [0] : []));
    const pad = (hi - lo) * 0.08 || 0.1;
    lo -= pad; hi += pad;
    const n = dates.length;
    const x = (i) => (n <= 1 ? (L + R) / 2 : L + ((R - L) * i) / (n - 1));
    const y = scale(lo, hi, bottom, top);
    const digits = tickDigits(hi - lo);
    const grid = [0, 1, 2, 3].map((k) => {
      const v = lo + ((hi - lo) * k) / 3;
      return `<line x1="${L}" y1="${y(v)}" x2="${R}" y2="${y(v)}" class="gridline"/><text x="${L - 6}" y="${y(v) + 3}" text-anchor="end">${num(v, digits)}</text>`;
    }).join("");
    const lines = series.map((s) => {
      const pts = s.values.map((v, i) => `${x(i)},${y(v)}`).join(" ");
      const dots = points && n <= 45 ? s.values.map((v, i) => `<circle cx="${x(i)}" cy="${y(v)}" r="3" class="tone-point"/>`).join("") : "";
      return `<polyline points="${pts}" class="${s.cls}"/>${dots}`;
    }).join("");
    return `<svg viewBox="0 0 ${W} ${H}" class="small-chart">${grid}${zero ? `<line x1="${L}" y1="${y(0)}" x2="${R}" y2="${y(0)}" class="zero-line"/>` : ""}${lines}${xLabels(dates, x, H - 14, 3)}</svg>`;
  }

  function barChart(counts) {
    const keys = ["positive", "neutral", "negative"];
    const names = ["Положит.", "Нейтр.", "Отрицат."];
    const max = Math.max(...keys.map((k) => counts[k]), 1);
    const grid = [35, 75, 115, 155].map((y) => `<line x1="40" y1="${y}" x2="440" y2="${y}" class="gridline"/>`).join("");
    const bars = keys.map((k, i) => {
      const h = (counts[k] / max) * 110;
      return `<rect x="${80 + i * 130}" y="${160 - h}" width="62" height="${h}" class="bar-${k}"/>
        <text x="${111 + i * 130}" y="${151 - h}" text-anchor="middle" class="bar-value">${counts[k].toLocaleString("ru-RU")}</text>
        <text x="${111 + i * 130}" y="181" text-anchor="middle">${names[i]}</text>`;
    }).join("");
    return `<svg viewBox="0 0 460 190" class="small-chart">${grid}${bars}</svg>`;
  }

  // --- screen 1: dashboard -------------------------------------------------

  async function pageDashboard() {
    const d = await api(`/api/overview?period=${state.period}`);
    const hasPrices = d.tickers.some((r) => r.last_price != null);
    if (!hasPrices && !d.latest_news.length) {
      const last = status?.updated_at ? fmtDateTime(status.updated_at) : "—";
      return `<div class="empty state-empty"><div>∅</div><h2>Данные не загружены. Последнее успешное обновление: ${last}</h2>
        <p>Загрузите котировки и новости в разделе «Пайплайн».</p>${goPipeline}</div>`;
    }
    const s = d.summary;
    const upShare = s.n_forecast ? `${Math.round((100 * s.n_up) / s.n_forecast)} % <small>(${s.n_up} из ${s.n_forecast})</small>` : "—";
    const rows = d.tickers.map((r) => {
      const fc = r.forecast;
      const arrow = fc ? (fc.pred_direction === 1 ? "▲" : "▼") : "—";
      return `<tr data-go="ticker" data-ticker="${r.ticker}">
        <td><b>${r.ticker}</b><small>${esc(r.name)}</small></td>
        <td class="${r.last_price == null ? "muted" : ""}">${r.last_price == null ? "нет свечей" : rub(r.last_price)}</td>
        <td class="${cls(r.change_pct)}">${r.change_pct == null ? "—" : pct(r.change_pct)}</td>
        <td class="${cls(r.tone)}">${r.tone == null ? "нет данных" : signed(r.tone)}</td>
        <td class="${fc ? (fc.pred_direction === 1 ? "positive forecast" : "negative forecast") : "muted"}" title="${fc ? `F=${fc.features}: ${rub(fc.pred_close)} на ${fmtDay(fc.target_date)}` : "модель не обучена"}">${arrow}</td>
      </tr>`;
    }).join("");
    const news = d.latest_news.map((n) => `<button class="news-mini" data-go="news" data-news-ticker="${n.ticker}">
        <span class="tone-bar ${n.tone}"></span><span><b>${esc(n.title)}</b><small>${fmtShortDateTime(n.published_at)} <em>${n.ticker}</em></small></span></button>`).join("");
    return `<div class="metrics three">
        ${metric("Отслеживаемых тикеров", s.tracked)}
        ${metric("Средняя тональность за период", s.avg_tone == null ? "—" : signed(s.avg_tone), cls(s.avg_tone))}
        ${metric("Тикеров с прогнозом роста", upShare)}
      </div>
      <div class="dashboard-grid">
        <section class="panel table-panel">${sectionHead("Обзор инструментов", "Котировки и сигналы за выбранный период", `<span class="period-pill">${periodLabel()}</span>`)}
          <table><thead><tr><th>Тикер</th><th>Последняя цена</th><th>Изменение</th><th>Тональность</th><th>Прогноз</th></tr></thead><tbody>${rows}</tbody></table>
        </section>
        <section class="panel latest">${sectionHead("Последние новости", "По отслеживаемым компаниям")}
          ${news || `<div class="empty"><p>Новостей по компаниям пока нет</p></div>`}
          <button class="link" data-go="news" data-news-ticker="ALL">Все новости <span>→</span></button>
        </section>
      </div>`;
  }

  // --- screen 2: ticker ----------------------------------------------------

  async function pageTicker() {
    const t = state.ticker;
    const d = await api(`/api/ticker/${t}?period=${state.period}&features=${state.features}`);
    const quote = d.last_price == null ? "" : `<div class="quote"><strong>${rub(d.last_price)}</strong><span class="${cls(d.change_pct)}">${pct(d.change_pct)} за ${periodLabel()}</span></div>`;
    const chart = d.candles
      ? candleChart(d.candles, state.forecast ? d.forecast : null)
      : emptyState("⌁", `Котировки ${t} не загружены`, `Загрузите свечи: «Пайплайн» → шаг «Свечи» с тикером ${t}.`, goPipeline);

    let cards;
    if (!d.trained) {
      cards = notice("Модель для тикера не обучена", `История котировок и новости доступны без прогноза. Модель LSTM F=${state.features} можно обучить в разделе «Пайплайн».`);
    } else if (!d.forecast) {
      cards = notice("Прогноз ещё не построен", "Модель обучена, но прогноза нет — запустите шаг «Прогноз t+1» в разделе «Пайплайн».");
    } else {
      const f = d.forecast;
      cards = `<div class="metrics four">
        ${metric("Последняя цена", d.last_price == null ? "—" : rub(d.last_price))}
        ${metric(`Прогноз на ${fmtDay(f.target_date)}`, rub(f.pred_close))}
        ${metric("Прогнозируемое изменение", f.change_pct == null ? "—" : pct(f.change_pct), cls(f.change_pct))}
        ${metric("Средняя тональность за период", d.tone == null ? "нет данных" : signed(d.tone), cls(d.tone))}
      </div>`;
    }
    const news = d.news.map((n) => `<a href="${esc(n.url)}" target="_blank" rel="noreferrer">
        <time>${fmtShortDateTime(n.published_at)}</time><span>${esc(n.title)}</span>
        <b class="tone-icon ${n.tone}" title="${TONES[n.tone]}${n.score != null ? ` ${signed(n.score)}` : ""}">${toneMark(n.tone)}</b></a>`).join("");
    return `<section class="panel chart-panel">${sectionHead(`${t} · ${esc(d.name)}`, "Свечной график и объём торгов, ₽", quote)}${chart}</section>
      ${cards}
      <section class="panel ticker-news">${sectionHead("Последние новости", `Новости и оценка тональности для ${t}`, `<button class="secondary" data-go="quality">Качество модели →</button>`)}
        <div class="news-table">${news || `<div class="empty"><p>Новостей по ${t} пока нет</p></div>`}</div>
        <button class="link" data-go="news" data-news-ticker="${t}">Все новости по ${t} <span>→</span></button>
      </section>`;
  }

  // --- screen 3: news ------------------------------------------------------

  function newsCard(n) {
    const chip = n.ticker ? `<button class="chip" data-go="ticker" data-ticker="${n.ticker}">${n.ticker}</button>` : `<span class="chip market">РЫНОК</span>`;
    const link = n.url ? `<a href="${esc(n.url)}" target="_blank" rel="noreferrer">Открыть оригинал ↗</a>` : "";
    const score = n.score != null ? ` ${signed(n.score)}` : "";
    return `<article class="news-card"><div class="tone-stripe ${n.tone}"></div><div class="news-main">
      <div class="news-meta"><span>${esc(n.source)}</span><time>${fmtDateTime(n.published_at)}</time></div>
      <h3>${esc(n.title)}</h3><div class="news-bottom">${chip}${link}</div></div>
      <span class="tone-badge ${n.tone}">${TONES[n.tone]}${score}</span></article>`;
  }

  async function pageNews() {
    const tones = Object.keys(state.tones).filter((k) => state.tones[k]).join(",");
    newsQuery = new URLSearchParams({ ticker: state.newsTicker, period: state.period, tones });
    const d = await api(`/api/news?${newsQuery}&offset=0&limit=${NEWS_PAGE}`);
    newsShown = d.items.length;
    const scope = state.newsTicker === "ALL" ? "Все отслеживаемые тикеры и рынок" : `Фильтр: ${state.newsTicker}`;
    const daily = d.daily.dates.length
      ? lineChart({ dates: d.daily.dates, series: [{ values: d.daily.tone, cls: "tone-line" }], zero: true, points: true, W: 460, H: 190 })
      : emptyState("≡", "Нет оценённых новостей", "Тональность ещё не посчитана для выбранного периода.");
    const feed = d.total
      ? `<div class="news-feed" id="news-feed">${d.items.map(newsCard).join("")}</div>
         <button class="secondary more" data-action="more-news" ${newsShown >= d.total ? "hidden" : ""}>Показать ещё</button>`
      : `<div class="empty"><div>≡</div><h2>Нет новостей по заданным фильтрам</h2><p>Измените параметры тикера, периода или тональности.</p><button class="primary" data-action="reset-news">Сбросить фильтры</button></div>`;
    return `<div class="chart-grid">
        <section class="panel">${sectionHead("Распределение новостей", "По классам тональности", `<span class="count">${d.total.toLocaleString("ru-RU")} новостей</span>`)}${barChart(d.counts)}</section>
        <section class="panel">${sectionHead("Средняя тональность по дням", "Оценка s = p_pos − p_neg, от −1 до +1")}${daily}</section>
      </div>
      <div class="feed-head"><div><h2>Лента новостей</h2><p>${scope} · ${periodLabel()}</p></div><span>Найдено: <b id="news-found">${d.total.toLocaleString("ru-RU")}</b></span></div>
      ${feed}`;
  }

  async function moreNews(btn) {
    btn.disabled = true;
    const d = await api(`/api/news?${newsQuery}&offset=${newsShown}&limit=${NEWS_PAGE}`);
    $("#news-feed").insertAdjacentHTML("beforeend", d.items.map(newsCard).join(""));
    newsShown += d.items.length;
    btn.disabled = false;
    btn.hidden = newsShown >= d.total;
  }

  // --- screen 4: model quality ----------------------------------------------

  function sentimentPanel(sm) {
    const head = sectionHead("Качество модели тональности", "Матрица ошибок · фактический / предсказанный класс");
    if (!sm) {
      return `<section class="panel">${head}${notice("Нет размеченной выборки",
        "Метрики появятся после разметки ≥400 новостей (СРО3): положите результат в reports/sentiment_metrics.json — матрица ошибок, accuracy, macro-F1.")}</section>`;
    }
    const labels = sm.labels || ["positive", "neutral", "negative"];
    const max = Math.max(...sm.confusion.flat(), 1);
    const level = (v) => `m${Math.min(4, 1 + Math.floor((4 * v) / (max + 1)))}`;
    const cells = sm.confusion.map((row, i) => `<b>${labels[i]}</b>${row.map((v) => `<i class="${level(v)}">${v}</i>`).join("")}`).join("");
    const byLang = sm.n_by_lang ? ` (${Object.entries(sm.n_by_lang).map(([k, v]) => `${v} ${k}`).join(", ")})` : "";
    return `<section class="panel">${head}<div class="matrix-wrap">
        <div class="matrix"><span></span>${labels.map((l) => `<b>${l}</b>`).join("")}${cells}</div>
        <div class="matrix-metrics"><div><span>Accuracy</span><strong>${num(sm.accuracy)}</strong></div><div><span>Macro-F1</span><strong>${num(sm.macro_f1)}</strong></div>
        <p>Размеченная выборка:<br/><b>${sm.n_labeled} новостей</b>${byLang}</p></div></div></section>`;
  }

  async function pageQuality() {
    const t = state.ticker, f = state.features;
    const d = await api(`/api/quality/${t}?features=${f}&period=${state.period}`);
    if (!d.trained) {
      return `<div class="empty state-empty"><div>⌗</div><h2>Модель LSTM F=${f} для ${t} не обучена. Метрики недоступны</h2>
        <p>Обучите модель и постройте прогноз в разделе «Пайплайн».</p>${goPipeline}</div>
        <div class="quality-grid" style="margin-top:18px">${sentimentPanel(d.sentiment)}</div>`;
    }
    const m = d.metrics, s = d.series, cfg = d.config;
    let body;
    if (!m) {
      body = `<div class="empty state-empty"><div>∅</div><h2>Нет прогнозов с наступившим горизонтом. Метрики недоступны</h2><p>Выберите более длинный период оценки.</p></div>`;
    } else {
      const better = m.rmse_rel != null && m.rmse_rel < 1;
      body = `<div class="metrics four">
          ${metric("MAE", rub(m.mae))}
          ${metric("RMSE", rub(m.rmse))}
          ${metric("Доля верных направлений", `${num(m.da, 0)} %`)}
          ${metric("RMSE к наивному прогнозу", m.rmse_rel == null ? "—" : num(m.rmse_rel), better ? "positive" : "negative")}
        </div>
        <div class="chart-grid quality-charts">
          <section class="panel">${sectionHead("Факт против прогноза", `Цена закрытия, ₽ · прогнозов на тесте: ${m.n_test}`, `<span class="tag ${better ? "positive-soft" : "negative-soft"}">${better ? "лучше наивного" : "хуже наивного"}</span>`)}
            ${lineChart({ dates: s.dates, series: [{ values: s.actual, cls: "fact-line" }, { values: s.pred, cls: "prediction-line" }] })}
            <div class="legend"><span><i class="legend-actual"></i>Факт</span><span><i class="legend-forecast"></i>Прогноз LSTM F=${f}</span></div></section>
          <section class="panel">${sectionHead("Ошибка прогноза по времени", "Факт минус прогноз, ₽")}${lineChart({ dates: s.dates, series: [{ values: s.error, cls: "error-line" }], zero: true })}</section>
        </div>`;
    }
    const h = d.hypothesis;
    const about = [
      ["Дата обучения", cfg.trained_at ? fmtDateTime(cfg.trained_at) : "—"],
      ["Длина окна", cfg.window ? `${cfg.window} торговых дней` : "—"],
      ["Слоёв LSTM", cfg.num_layers ?? "—"],
      ["Скрытое состояние", cfg.hidden_size ?? "—"],
      ["Горизонт", "1 торговый день"],
      ["Признаки", (cfg.features || []).map((x) => FEATURES[x] || x).join(", ") || "—"],
      ["Тестовый период", cfg.test_period ? cfg.test_period.map(fmtDay).join(" — ") : "—"],
      ["Версия", cfg.model_version || "—"],
    ];
    if (h) {
      about.push(["Гипотеза (F=4 vs F=3)", `<span class="${h.improved_by_sentiment ? "positive" : "negative"}">${h.improved_by_sentiment ? "тональность улучшила RMSE и DA" : "не подтвердилась"}</span>`]);
    }
    return `${body}<div class="quality-grid">${sentimentPanel(d.sentiment)}
        <section class="panel about">${sectionHead("О модели", `LSTM F=${f} для ${t}`)}
          ${about.map(([a, b]) => `<div><span>${a}</span><b>${b}</b></div>`).join("")}
          <button class="secondary full" data-go="ticker">Открыть тикер →</button>
        </section></div>`;
  }

  // --- screen 5: pipeline ----------------------------------------------------

  function jobBadge(job) {
    if (job.running) return `<span class="state-badge busy">выполняется: ${esc(job.step)}</span>`;
    if (job.ok === true) return `<span class="state-badge ok">готово: ${esc(job.step)}</span>`;
    if (job.ok === false) return `<span class="state-badge fail">ошибка: ${esc(job.step)}</span>`;
    return `<span class="state-badge">нет активной задачи</span>`;
  }

  function statusRows(st) {
    const yes = (v) => (v ? `<b class="positive">есть</b>` : `<b class="muted">нет</b>`);
    const db = st.db || {};
    const dbText = db.ok
      ? `<b class="positive">OK</b>`
      : `<b class="negative" title="${esc(db.error || "")}">нет связи</b>`;
    const rows = [
      ["Свечи", yes(st.has_candles)],
      ["Новости", yes(st.has_news)],
      ["Тональность", yes(st.has_sentiment)],
      ["Модели", st.models.length ? `<b>${st.models.join(", ")}</b>` : `<b class="muted">нет</b>`],
      ["Прогнозы", yes(st.has_forecasts)],
      ["Последние котировки", `<b>${st.last_market_date ? fmtDay(st.last_market_date) : "—"}</b>`],
      [`PostgreSQL${db.url_host ? ` · ${esc(db.url_host)}` : ""}`, dbText],
    ];
    if (db.ok) rows.push(...Object.entries(db.counts || {}).map(([k, v]) => [`&nbsp;&nbsp;${k}`, `<b>${v.toLocaleString("ru-RU")}</b>`]));
    return rows.map(([a, b]) => `<div><span>${a}</span>${b}</div>`).join("");
  }

  async function pagePipeline() {
    const [st, log] = await Promise.all([refreshStatus(), api("/api/job/log")]);
    const busy = st.job.running ? "disabled" : "";
    const steps = STEPS.map(([id, title, desc], i) => `<button class="step-card" data-action="run-step" data-step="${id}" ${busy}>
        <em>Шаг ${i + 1}</em><b>${title}</b>${desc}</button>`).join("");
    return `<div class="pipeline-grid"><div>
        <section class="panel">${sectionHead("Шаги пайплайна", `Тикер для свечей, обучения и прогноза: ${state.ticker}. После шагов 1–3 и 5 данные автоматически синхронизируются в PostgreSQL.`,
          `<button class="primary" data-action="run-pipeline" ${busy}>Запустить полный пайплайн</button>`)}
          <div class="steps-grid">${steps}</div></section>
        <section class="panel">${sectionHead("Лог задачи", "stdout / stderr скрипта", `<span id="job-badge">${jobBadge(st.job)}</span>`)}
          <pre class="log" id="log">${esc(log.log || "Нажмите шаг или «Запустить полный пайплайн».")}</pre></section>
      </div>
      <section class="panel status-list">${sectionHead("Артефакты и база", "Что уже посчитано")}<div class="rows" id="status-rows">${statusRows(st)}</div></section>
    </div>`;
  }

  function updatePipelineView(st, log) {
    if (state.page !== "pipeline") return;
    const badge = $("#job-badge"), rows = $("#status-rows"), logEl = $("#log");
    if (badge) badge.innerHTML = jobBadge(st.job);
    if (rows) rows.innerHTML = statusRows(st);
    if (logEl && log?.log) {
      const atBottom = logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 20;
      logEl.textContent = log.log;
      if (atBottom) logEl.scrollTop = logEl.scrollHeight;
    }
    document.querySelectorAll("[data-action='run-step'], [data-action='run-pipeline']").forEach((b) => {
      b.disabled = st.job.running;
      b.classList.toggle("running", st.job.running && (b.dataset.step === st.job.step));
    });
  }

  async function startJob(path) {
    try {
      await api(path, { method: "POST" });
      poll();
    } catch (e) {
      const logEl = $("#log");
      if (logEl) logEl.textContent = String(e.message || e);
    }
  }

  let polling = false;
  async function poll() {
    if (polling) return;
    polling = true;
    try {
      for (;;) {
        const [st, log] = await Promise.all([refreshStatus(), api("/api/job/log")]);
        updatePipelineView(st, log);
        if (!st.job.running) break;
        await new Promise((r) => setTimeout(r, 1500));
      }
    } finally {
      polling = false;
    }
  }

  // --- shell: status, banners, sidebar, routing --------------------------------

  async function refreshStatus(recheck = false) {
    status = await api(`/api/status${recheck ? "?recheck=true" : ""}`);
    renderShell();
    return status;
  }

  function renderShell() {
    if (!status) return;
    $("#updated").innerHTML = `Данные обновлены: <b>${status.updated_at ? `${fmtDateTime(status.updated_at)} МСК` : "—"}</b>`;
    const src = $("#source");
    src.textContent = status.db?.ok ? "PostgreSQL подключена" : "Локальные CSV";
    src.className = status.db?.ok ? "ok" : "";
    $("#job-dot").hidden = !status.job.running;

    const banners = [];
    if (status.db && !status.db.ok) {
      banners.push(`<div class="banner error"><span><b>Не удалось подключиться к базе данных</b>
        <small>Показаны данные из локальных CSV. Проверьте, что PostgreSQL запущен (docker compose up -d db), и повторите попытку.</small></span>
        <button data-action="retry-db">Повторить</button></div>`);
    }
    if (status.stale) {
      const last = status.last_market_date ? `последние котировки за ${fmtDay(status.last_market_date)}` : "котировки не загружены";
      banners.push(`<div class="banner warning"><span><b>Данные устарели: ${last}</b><small>Обновите свечи и новости в разделе «Пайплайн».</small></span></div>`);
    }
    $("#banners").innerHTML = banners.join("");
  }

  function selectField(label, key, value, options) {
    return `<label class="field">${label}<select data-key="${key}">${options.map(([v, text]) =>
      `<option value="${v}" ${String(v) === String(value) ? "selected" : ""}>${text}</option>`).join("")}</select></label>`;
  }

  function renderControls() {
    const p = state.page;
    const tickerOptions = Object.keys(TICKERS).map((t) => [t, `${t} · ${TICKERS[t]}`]);
    const parts = [];
    if (p === "news") parts.push(selectField("Тикер", "newsTicker", state.newsTicker, [["ALL", "Все"], ...tickerOptions]));
    else if (p !== "dashboard") parts.push(selectField("Тикер", "ticker", state.ticker, tickerOptions));
    if (p !== "pipeline") parts.push(selectField(p === "quality" ? "Период оценки" : "Период", "period", state.period, PERIODS));
    if (p === "ticker" || p === "quality") parts.push(selectField("Модель", "features", state.features, MODELS));
    if (p === "ticker") {
      parts.push(`<label class="switch-row"><span>Показать прогноз</span><input type="checkbox" data-key="forecast" ${state.forecast ? "checked" : ""}/><i></i></label>`);
    }
    if (p === "news") {
      parts.push(`<fieldset><legend>Класс тональности</legend>${["positive", "neutral", "negative"].map((t) =>
        `<label class="check"><input type="checkbox" data-tone="${t}" ${state.tones[t] ? "checked" : ""}/><span class="dot ${t}"></span>${TONES[t]}</label>`).join("")}</fieldset>`);
    }
    $("#controls").innerHTML = parts.join("");
  }

  const RENDERERS = { dashboard: pageDashboard, ticker: pageTicker, news: pageNews, quality: pageQuality, pipeline: pagePipeline };

  async function render() {
    const seq = ++renderSeq;
    document.querySelectorAll("#nav button").forEach((b) => b.classList.toggle("active", b.dataset.page === state.page));
    $("#title").textContent = PAGES[state.page] + (state.page === "ticker" ? ` · ${state.ticker}` : "");
    document.title = `${PAGES[state.page]} · InvestLab`;
    renderControls();
    const spinner = setTimeout(() => { if (seq === renderSeq) content.innerHTML = loadingHtml(); }, 150);
    let html;
    try {
      html = await RENDERERS[state.page]();
    } catch (e) {
      html = `<div class="empty state-empty"><div>!</div><h2>Не удалось загрузить данные</h2><p>${esc(e.message || e)}</p>
        <button class="primary" data-action="reload">Повторить</button></div>`;
    }
    clearTimeout(spinner);
    if (seq === renderSeq) content.innerHTML = html;
  }

  function go(page) {
    state.page = page;
    if (location.hash !== `#${page}`) history.pushState(null, "", `#${page}`);
    window.scrollTo(0, 0);
    render();
  }

  // --- events ----------------------------------------------------------------

  document.addEventListener("click", (e) => {
    const nav = e.target.closest("#nav button");
    if (nav) return go(nav.dataset.page);

    const link = e.target.closest("[data-go]");
    if (link) {
      e.preventDefault();
      if (link.dataset.ticker) state.ticker = link.dataset.ticker;
      if (link.dataset.newsTicker) state.newsTicker = link.dataset.newsTicker;
      return go(link.dataset.go);
    }

    const action = e.target.closest("[data-action]");
    if (!action) return;
    const q = new URLSearchParams({ ticker: state.ticker, limit: "50" });
    switch (action.dataset.action) {
      case "run-step": return startJob(`/api/run/${action.dataset.step}?${q}`);
      case "run-pipeline": return startJob(`/api/run/pipeline?${q}`);
      case "more-news": return moreNews(action);
      case "retry-db": action.disabled = true; return refreshStatus(true);
      case "reload": return render();
      case "reset-news":
        state.newsTicker = "ALL";
        state.tones = { positive: true, neutral: true, negative: true };
        return render();
      default: return undefined;
    }
  });

  $("#controls").addEventListener("change", (e) => {
    const el = e.target;
    if (el.dataset.tone) state.tones[el.dataset.tone] = el.checked;
    else if (el.dataset.key === "forecast") state.forecast = el.checked;
    else if (el.dataset.key === "features") state.features = Number(el.value);
    else if (el.dataset.key) state[el.dataset.key] = el.value;
    if (state.page === "ticker") $("#title").textContent = `${PAGES.ticker} · ${state.ticker}`;
    render();
  });

  window.addEventListener("popstate", () => {
    const page = location.hash.slice(1);
    state.page = PAGES[page] ? page : "dashboard";
    render();
  });

  const initial = location.hash.slice(1);
  if (PAGES[initial]) state.page = initial;
  refreshStatus().then((st) => { if (st.job.running) poll(); }).catch(() => {});
  render();
})();
