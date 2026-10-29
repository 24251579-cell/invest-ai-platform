(() => {
  const $ = (sel) => document.querySelector(sel);
  const tickerEl = $("#ticker");
  const logEl = $("#log");
  const jobState = $("#job-state");
  const artifactsEl = $("#artifacts");
  const metricsEl = $("#metrics");
  const forecastsEl = $("#forecasts");
  const newsEl = $("#news");
  let featureSet = 3;
  let chart;

  async function api(path, opts) {
    const res = await fetch(path, opts);
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || res.statusText);
    }
    return res.json();
  }

  function setBusy(busy) {
    document.querySelectorAll("button[data-step], #btn-pipeline").forEach((b) => {
      b.disabled = busy;
    });
  }

  async function refreshStatus() {
    const st = await api("/api/status");
    const job = st.job;
    if (job.running) {
      jobState.textContent = `выполняется: ${job.step}`;
      jobState.className = "busy";
      setBusy(true);
    } else if (job.ok === true) {
      jobState.textContent = `готово (${job.step})`;
      jobState.className = "ok";
      setBusy(false);
    } else if (job.ok === false) {
      jobState.textContent = `ошибка (${job.step})`;
      jobState.className = "fail";
      setBusy(false);
    } else {
      jobState.textContent = "готово";
      jobState.className = "";
      setBusy(false);
    }
    const db = st.db || {};
    const dbText = db.ok
      ? `Postgres OK (${Object.entries(db.counts || {}).map(([k, v]) => `${k}:${v}`).join(", ")})`
      : `Postgres: ${db.error || "нет связи"}`;
    artifactsEl.textContent = [
      st.has_candles ? "свечи" : null,
      st.has_news ? "новости" : null,
      st.has_sentiment ? "sentiment" : null,
      st.models.length ? `модели: ${st.models.join(", ")}` : "модели: нет",
      st.has_forecasts ? "прогнозы" : null,
      dbText,
    ].filter(Boolean).join(" · ");
    return st;
  }

  async function refreshLog() {
    const data = await api("/api/job/log");
    if (data.log) logEl.textContent = data.log;
  }

  function renderMetrics(data) {
    if (!data.models || !Object.keys(data.models).length) {
      metricsEl.innerHTML = "модели ещё не обучены";
      return;
    }
    const rows = [];
    for (const [key, m] of Object.entries(data.models)) {
      rows.push(`<div class="row"><span class="label">${key.toUpperCase()} RMSE</span><span>${Number(m.rmse).toFixed(3)} (rel ${Number(m.rmse_rel).toFixed(3)})</span></div>`);
      rows.push(`<div class="row"><span class="label">${key.toUpperCase()} DA</span><span>${Number(m.da).toFixed(1)}%</span></div>`);
    }
    if (data.hypothesis) {
      const h = data.hypothesis;
      const ok = h.improved_by_sentiment;
      rows.push(`<div class="verdict">${ok ? "Тональность улучшила RMSE и DA" : "Гипотеза не подтвердилась на тесте"} · news days: ${h.news_days_with_sentiment ?? "—"}</div>`);
    }
    metricsEl.innerHTML = rows.join("");
  }

  function renderForecasts(rows) {
    if (!rows.length) {
      forecastsEl.textContent = "нет прогнозов — запустите Predict";
      return;
    }
    forecastsEl.innerHTML = rows.slice(-4).reverse().map((r) => {
      const up = Number(r.pred_direction) === 1;
      return `<div class="forecast-card">
        <div>${r.ticker} · F=${r.use_sentiment === true || r.use_sentiment === "True" || r.use_sentiment === 1 ? 4 : 3} · ${r.target_date}</div>
        <strong class="${up ? "up" : "down"}">${Number(r.pred_close).toFixed(2)} ₽ ${up ? "↑" : "↓"}</strong>
        <div class="news-meta">${r.model_version || ""}</div>
      </div>`;
    }).join("");
  }

  function renderNews(rows) {
    if (!rows.length) {
      newsEl.textContent = "нет новостей";
      return;
    }
    newsEl.innerHTML = rows.map((n) => {
      const when = (n.published_at || "").replace("T", " ").slice(0, 16);
      return `<article class="news-item">
        <div class="news-meta">${when}<br>${n.source || ""} · ${n.ticker || "MARKET"} · ${n.lang || ""}</div>
        <p class="news-title">${n.title || ""}</p>
      </article>`;
    }).join("");
  }

  async function loadChart() {
    const ticker = tickerEl.value;
    let pred = null;
    try {
      pred = await api(`/api/predictions/${ticker}?features=${featureSet}`);
    } catch {
      try {
        const candles = await api(`/api/candles/${ticker}?last=120`);
        drawChart(candles.dates, candles.close, null, null);
        return;
      } catch {
        if (chart) chart.destroy();
        return;
      }
    }
    drawChart(pred.dates, pred.y_true, pred.y_pred, pred.naive);
  }

  function drawChart(labels, yTrue, yPred, naive) {
    const ctx = $("#chart-price");
    const datasets = [
      { label: "факт", data: yTrue, borderColor: "#9bb0a4", tension: 0.15, pointRadius: 0, borderWidth: 1.5 },
    ];
    if (yPred) {
      datasets.push({ label: "LSTM", data: yPred, borderColor: "#3ecf8e", tension: 0.15, pointRadius: 0, borderWidth: 2 });
    }
    if (naive && naive.length) {
      datasets.push({ label: "naive", data: naive, borderColor: "#d6b26a", borderDash: [4, 4], tension: 0, pointRadius: 0, borderWidth: 1.2 });
    }
    if (chart) chart.destroy();
    chart = new Chart(ctx, {
      type: "line",
      data: { labels, datasets },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: "#c5d8cc" } } },
        scales: {
          x: { ticks: { color: "#7f9488", maxTicksLimit: 8 }, grid: { color: "#1e2c26" } },
          y: { ticks: { color: "#7f9488" }, grid: { color: "#1e2c26" } },
        },
      },
    });
  }

  async function refreshAll() {
    const ticker = tickerEl.value;
    await refreshStatus();
    await refreshLog();
    try { renderMetrics(await api(`/api/metrics/${ticker}`)); } catch { metricsEl.textContent = "нет метрик"; }
    try { renderForecasts(await api(`/api/forecasts?ticker=${ticker}`)); } catch { forecastsEl.textContent = "нет прогнозов"; }
    try { renderNews(await api("/api/news?limit=25")); } catch { newsEl.textContent = "нет новостей"; }
    await loadChart();
  }

  async function start(path) {
    try {
      await api(path, { method: "POST" });
      setBusy(true);
      poll();
    } catch (e) {
      logEl.textContent = String(e.message || e);
    }
  }

  let polling = false;
  async function poll() {
    if (polling) return;
    polling = true;
    try {
      while (true) {
        const st = await refreshStatus();
        await refreshLog();
        if (!st.job.running) {
          await refreshAll();
          break;
        }
        await new Promise((r) => setTimeout(r, 1500));
      }
    } finally {
      polling = false;
    }
  }

  document.querySelectorAll("button[data-step]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const step = btn.dataset.step;
      const ticker = tickerEl.value;
      const q = new URLSearchParams({ ticker, limit: "50" });
      start(`/api/run/${step}?${q}`);
    });
  });

  $("#btn-pipeline").addEventListener("click", () => {
    const q = new URLSearchParams({ ticker: tickerEl.value, limit: "50" });
    start(`/api/run/pipeline?${q}`);
  });

  $("#feat-toggle").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-f]");
    if (!btn) return;
    featureSet = Number(btn.dataset.f);
    $("#feat-toggle").querySelectorAll("button").forEach((b) => b.classList.toggle("active", b === btn));
    loadChart();
  });

  tickerEl.addEventListener("change", refreshAll);
  refreshAll();
})();
