"use strict";

(function () {
  const root = document.querySelector(".viz-root");
  const $ = (id) => document.getElementById(id);
  // Charts update every second, so they never animate (also respects reduced-motion settings).
  const state = { info: null, fleet: null, selected: null, detail: null, socket: null, charts: {} };

  // ---------- helpers ----------
  const css = (name) => getComputedStyle(root).getPropertyValue(name).trim();
  const palette = () => ({
    surface: css("--surface-1"),
    text: css("--text-primary"),
    text2: css("--text-secondary"),
    muted: css("--text-muted"),
    grid: css("--grid"),
    axis: css("--axis"),
    series: css("--series-1"),
    wash: css("--series-1-wash"),
    truth: css("--truth"),
    neg: css("--div-neg"),
    mid: css("--div-mid"),
    pos: css("--div-pos"),
    zone: css("--zone"),
    serious: css("--status-serious"),
  });
  const fmt = {
    num: (v, d = 1) => (Number.isFinite(v) ? v.toFixed(d) : "–"),
    int: (v) => (Number.isFinite(v) ? Math.round(v).toLocaleString("en-GB") : "–"),
    pct: (v, d = 0) => (Number.isFinite(v) ? `${(v * 100).toFixed(d)}%` : "–"),
    signed: (v, d = 1) => (Number.isFinite(v) ? `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(d)}` : "–"),
    // An interval whose upper end reached the RUL cap is open-ended: "at least lower cycles".
    range: (r, d = 0) => (r.open_ended ? `≥ ${fmt.num(r.lower, d)}` : `${fmt.num(r.lower, d)}–${fmt.num(r.upper, d)}`),
  };
  const escapeHtml = (s) =>
    String(s).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);
  function el(tag, opts = {}, ...children) {
    const node = document.createElement(tag);
    if (opts.className) node.className = opts.className;
    if (opts.text != null) node.textContent = opts.text;
    for (const [k, v] of Object.entries(opts.attrs || {})) node.setAttribute(k, v);
    for (const child of children) if (child != null) node.append(child);
    return node;
  }
  const setText = (id, text) => {
    const node = $(id);
    if (node) node.textContent = text;
  };
  const STATUS = {
    healthy: ["✓", "Healthy"],
    watch: ["!", "Watch"],
    alert: ["▲", "Alert"],
    failed: ["✕", "Failed"],
  };
  function chip(status) {
    const [icon, label] = STATUS[status] || ["?", status];
    return el(
      "span",
      { className: `chip chip-${status}` },
      el("span", { className: "chip-icon", text: icon, attrs: { "aria-hidden": "true" } }),
      document.createTextNode(label),
    );
  }
  function chart(id) {
    if (!state.charts[id]) {
      state.charts[id] = echarts.init($(id), null, { renderer: "canvas" });
      if (id === "chart-rul" || id === "chart-prob") {
        state.charts[id].group = "life";
        echarts.connect("life");
      }
    }
    return state.charts[id];
  }
  const tooltipBase = (c) => ({
    backgroundColor: c.surface,
    borderColor: c.grid,
    borderWidth: 1,
    padding: [8, 10],
    textStyle: { color: c.text, fontSize: 12, fontFamily: "system-ui, -apple-system, 'Segoe UI', sans-serif" },
    extraCssText: "box-shadow: 0 4px 16px rgba(0,0,0,0.12); border-radius: 8px;",
  });
  const key = (color) =>
    `<span style="display:inline-block;width:12px;height:2px;border-radius:1px;background:${color};margin-right:6px;vertical-align:middle"></span>`;
  const row = (color, value, label) =>
    `<div style="display:flex;align-items:center;gap:2px;margin-top:2px">${color ? key(color) : '<span style="display:inline-block;width:18px"></span>'}<strong style="font-variant-numeric:tabular-nums">${escapeHtml(value)}</strong><span style="opacity:0.75;margin-left:6px">${escapeHtml(label)}</span></div>`;
  const axisCommon = (c) => ({
    axisLine: { lineStyle: { color: c.axis } },
    axisTick: { show: false },
    axisLabel: { color: c.muted, fontSize: 11, hideOverlap: true },
    splitLine: { lineStyle: { color: c.grid, width: 1 } },
  });

  // ---------- KPI row and model card ----------
  function renderInfo(info) {
    state.info = info;
    const k = info.kpis;
    setText("subset-name", info.subset);
    setText("kpi-rmse", fmt.num(k.rmse, 1));
    setText(
      "kpi-rmse-caption",
      `${info.data.test_engines} test engines${Number.isFinite(k.lightgbm_rmse) ? ` · LightGBM ${fmt.num(k.lightgbm_rmse, 1)}` : ""}`,
    );
    setText("kpi-score", fmt.int(k.nasa_score));
    setText("kpi-coverage", fmt.pct(k.coverage));
    setText("kpi-coverage-caption", `Test engines · nominal ${fmt.pct(info.interval)} · ${fmt.pct(k.open_ended_share)} open-ended`);
    setText("kpi-ap", fmt.num(k.alert_ap, 2));
    setText(
      "kpi-ap-caption",
      `Failure within ${info.alert_cycles} cycles · ${k.alert_positives ?? "–"} of ${info.data.test_engines} test engines positive`,
    );
    setText("kpi-lead", fmt.num(k.lead_median, 0));
    setText("kpi-lead-caption", `${k.lead_engines ?? "–"} engines, out of fold · ${k.missed ?? "–"} missed · ${k.early ?? "–"} early`);
    document.querySelectorAll(".max-rul").forEach((n) => (n.textContent = info.max_rul));
    document.querySelectorAll(".horizon").forEach((n) => (n.textContent = info.alert_cycles));
    const commit = info.code.source_sha256 ? ` · source ${info.code.source_sha256}` : "";
    setText(
      "model-card",
      `Model: ${info.ensemble_size} cross-fitted LSTM + attention networks (windows ${info.windows.join(", ")} cycles), ` +
        `${info.sensors.length} sensors, trained ${info.created_at.slice(0, 10)}${commit}. ` +
        `RMSE and score use ground truth capped at ${info.max_rul} cycles; uncapped RMSE ${fmt.num(k.rmse_uncapped, 1)}. ` +
        `An interval that reaches ${info.max_rul} is open-ended: at least the lower bound.`,
    );
  }

  // ---------- fleet table ----------
  function rangeBar(e, maxRul) {
    const pos = (v) => Math.max(0, Math.min(100, (v / maxRul) * 100));
    const wrap = el("div", { className: "range", attrs: { "aria-hidden": "true" } });
    const track = el("span", { className: "range-track" });
    const band = el("span", { className: "range-band" });
    band.style.left = `${pos(e.lower)}%`;
    band.style.width = `${Math.max(1, pos(e.open_ended ? maxRul : e.upper) - pos(e.lower))}%`;
    if (e.open_ended) band.classList.add("open");
    const truth = el("span", { className: "range-true" });
    truth.style.left = `${pos(Math.min(e.rul_true, maxRul))}%`;
    const dot = el("span", { className: "range-dot" });
    dot.style.left = `${pos(e.point)}%`;
    wrap.append(track, band, truth, dot);
    return wrap;
  }

  function renderFleet(snapshot) {
    state.fleet = snapshot;
    const maxRul = state.info ? state.info.max_rul : 125;
    const rows = snapshot.engines.map((e) => {
      const tr = el("tr", {
        attrs: {
          tabindex: "0",
          "data-unit": String(e.unit),
          "aria-selected": String(e.unit === state.selected),
          "aria-label": `Engine ${e.unit}, ${STATUS[e.status]?.[1] || e.status}, cycle ${e.cycle}, predicted RUL ${fmt.num(e.point, 0)} cycles`,
        },
      });
      tr.append(
        el(
          "td",
          {},
          el("span", { className: "engine-id", text: `#${e.unit}` }),
          e.lap > 1 ? el("span", { className: "lap", text: `run ${e.lap}` }) : null,
        ),
      );
      tr.append(el("td", {}, chip(e.status)));
      tr.append(el("td", { className: "num", text: String(e.cycle) }));
      const rul = el("td", { className: "rul-cell" });
      rul.append(
        el(
          "div",
          { className: "rul-text" },
          el("strong", { text: fmt.num(e.point, 0) }),
          el("span", { className: "muted", text: ` (${fmt.range(e)})` }),
        ),
        rangeBar(e, maxRul),
      );
      tr.append(rul);
      tr.append(el("td", { className: "num", text: fmt.pct(e.probability) }));
      return tr;
    });
    $("fleet-body").replaceChildren(...rows);
  }

  function onFleetActivate(event) {
    const tr = event.target.closest("tr[data-unit]");
    if (!tr) return;
    if (event.type === "keydown" && event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    selectEngine(Number(tr.dataset.unit));
  }

  // ---------- detail panel ----------
  function lifeSeries(d) {
    const maxRul = state.info.max_rul;
    const total = d.total_cycles;
    const categories = Array.from({ length: total }, (_, i) => String(i + 1));
    const blank = () => new Array(total).fill(null);
    const s = { categories, lower: blank(), span: blank(), point: blank(), truth: blank(), prob: blank(), rows: blank() };
    for (const r of d.history) {
      const i = r.cycle - 1;
      if (i < 0 || i >= total) continue;
      s.lower[i] = r.lower;
      s.span[i] = Math.max(0, (r.open_ended ? maxRul : r.upper) - r.lower);
      s.point[i] = r.point;
      s.truth[i] = Math.min(r.rul_true, maxRul);
      s.prob[i] = r.probability;
      s.rows[i] = r;
    }
    return s;
  }

  function lifeTooltip(s) {
    return (params) => {
      const list = Array.isArray(params) ? params : [params];
      const r = list.length ? s.rows[list[0].dataIndex] : null;
      if (!r) return "";
      const c = palette();
      return (
        `<div style="font-weight:600;margin-bottom:2px">Cycle ${r.cycle}</div>` +
        row(c.series, fmt.num(r.point, 0), "Predicted RUL") +
        row(null, fmt.range(r), `Nominal ${fmt.pct(state.info.interval)} interval`) +
        row(c.truth, String(Math.min(r.rul_true, state.info.max_rul)), `True RUL${r.rul_true > state.info.max_rul ? ` (uncapped ${r.rul_true})` : ""}`) +
        row(null, fmt.pct(r.probability), `P(failure within ${state.info.alert_cycles} cycles)`)
      );
    };
  }

  function renderLife(d) {
    const c = palette();
    const s = lifeSeries(d);
    const horizon = state.info.alert_cycles;
    const firstAlert = d.history.find((r) => r.probability >= state.info.alert_threshold);
    const xAxis = {
      type: "category",
      data: s.categories,
      boundaryGap: false,
      ...axisCommon(c),
      splitLine: { show: false },
      axisLabel: { ...axisCommon(c).axisLabel, interval: "auto" },
    };
    const alertLine = firstAlert
      ? {
          silent: true,
          symbol: "none",
          lineStyle: { color: c.serious, width: 1, type: "solid" },
          label: {
            formatter: `Alert raised · ${firstAlert.rul_true} cycles before failure`,
            color: c.text2,
            fontSize: 11,
            position: "end",
            align: "right",
            distance: 6,
          },
          data: [{ xAxis: String(firstAlert.cycle) }],
        }
      : undefined;
    chart("chart-rul").setOption(
      {
        animation: false,
        grid: { left: 40, right: 12, top: 26, bottom: 24 },
        tooltip: { ...tooltipBase(c), trigger: "axis", axisPointer: { type: "line", lineStyle: { color: c.axis } }, formatter: lifeTooltip(s) },
        xAxis,
        yAxis: { type: "value", min: 0, max: state.info.max_rul, interval: 25, ...axisCommon(c), axisLine: { show: false } },
        series: [
          { name: "lower", type: "line", data: s.lower, stack: "band", symbol: "none", silent: true, lineStyle: { opacity: 0 }, connectNulls: false },
          { name: "90% interval", type: "line", data: s.span, stack: "band", symbol: "none", silent: true, lineStyle: { opacity: 0 }, areaStyle: { color: c.wash, opacity: 1 } },
          { name: "True RUL", type: "line", data: s.truth, symbol: "none", lineStyle: { color: c.truth, width: 2 }, emphasis: { disabled: true } },
          {
            name: "Predicted RUL",
            type: "line",
            data: s.point,
            symbol: "none",
            lineStyle: { color: c.series, width: 2 },
            emphasis: { disabled: true },
            markArea: {
              silent: true,
              itemStyle: { color: c.zone },
              label: { show: true, position: "insideBottomLeft", color: c.muted, fontSize: 11, formatter: `Alert horizon · ${horizon} cycles` },
              data: [[{ yAxis: 0 }, { yAxis: horizon }]],
            },
            markLine: alertLine,
          },
        ],
      },
      { notMerge: true },
    );
    chart("chart-prob").setOption(
      {
        animation: false,
        grid: { left: 40, right: 12, top: 8, bottom: 24 },
        tooltip: { ...tooltipBase(c), trigger: "axis", axisPointer: { type: "line", lineStyle: { color: c.axis } }, formatter: lifeTooltip(s) },
        xAxis,
        yAxis: {
          type: "value",
          min: 0,
          max: 1,
          interval: 0.5,
          ...axisCommon(c),
          axisLine: { show: false },
          axisLabel: { ...axisCommon(c).axisLabel, formatter: (v) => `${Math.round(v * 100)}%` },
        },
        series: [
          {
            name: "P(failure within horizon)",
            type: "line",
            data: s.prob,
            symbol: "none",
            lineStyle: { color: c.series, width: 2 },
            areaStyle: { color: c.wash, opacity: 1 },
            emphasis: { disabled: true },
            markLine: {
              silent: true,
              symbol: "none",
              lineStyle: { color: c.muted, width: 1, type: "solid" },
              label: { formatter: "Alert threshold", color: c.muted, fontSize: 11, position: "insideEndTop" },
              data: [{ yAxis: state.info.alert_threshold }],
            },
          },
        ],
      },
      { notMerge: true },
    );
    const last = d.history[d.history.length - 1];
    $("chart-rul").setAttribute(
      "aria-label",
      `Engine ${d.unit} at cycle ${last.cycle}: predicted RUL ${fmt.num(last.point, 0)} cycles ` +
        `(nominal 90% interval ${fmt.range(last)}), true RUL ${last.rul_true}.`,
    );
  }

  function renderHeat(d) {
    const c = palette();
    const a = d.attribution;
    const data = [];
    let extent = 0.25;
    a.values.forEach((rowValues, t) =>
      rowValues.forEach((v, j) => {
        data.push([t, j, v]);
        extent = Math.max(extent, Math.abs(v));
      }),
    );
    const sensors = state.info.sensors;
    chart("chart-heat").setOption(
      {
        animation: false,
        grid: { left: 64, right: 12, top: 6, bottom: 70 },
        tooltip: {
          ...tooltipBase(c),
          trigger: "item",
          formatter: (p) => {
            const [t, j, v] = p.data;
            const meta = sensors[j];
            return (
              `<div style="font-weight:600">${escapeHtml(meta.symbol)} · cycle ${a.cycles[t]}</div>` +
              `<div style="opacity:0.75">${escapeHtml(meta.description)}</div>` +
              row(null, `${fmt.signed(v, 2)} cycles`, v < 0 ? "pulls RUL down" : "pushes RUL up")
            );
          },
        },
        xAxis: {
          type: "category",
          data: a.cycles.map(String),
          ...axisCommon(c),
          splitLine: { show: false },
          name: "Cycle",
          nameLocation: "middle",
          nameGap: 24,
          nameTextStyle: { color: c.muted, fontSize: 11 },
        },
        yAxis: {
          type: "category",
          data: a.sensors,
          inverse: true,
          ...axisCommon(c),
          axisLine: { show: false },
          splitLine: { show: false },
        },
        visualMap: {
          type: "continuous",
          min: -extent,
          max: extent,
          calculable: false,
          orient: "horizontal",
          left: "center",
          bottom: 0,
          itemWidth: 10,
          itemHeight: 180,
          text: ["Pushes RUL up", "Pulls RUL down"],
          textGap: 8,
          textStyle: { color: c.text2, fontSize: 11 },
          inRange: { color: [c.neg, c.mid, c.pos] },
          formatter: (v) => fmt.signed(v, 1),
        },
        series: [
          {
            type: "heatmap",
            data,
            itemStyle: { borderColor: c.surface, borderWidth: 1 },
            emphasis: { itemStyle: { borderColor: c.text, borderWidth: 1 } },
          },
        ],
      },
      { notMerge: true },
    );
  }

  function renderDrivers(d) {
    const c = palette();
    const items = d.contributions;
    // Leave room beyond the longest bar for its value label so it never meets the axis labels.
    const extent = Math.max(1, ...items.map((i) => Math.abs(i.contribution_cycles))) * 1.6;
    chart("chart-drivers").setOption(
      {
        animation: false,
        grid: { left: 64, right: 16, top: 6, bottom: 24 },
        tooltip: {
          ...tooltipBase(c),
          trigger: "item",
          formatter: (p) => {
            const item = items[p.dataIndex];
            return (
              `<div style="font-weight:600">${escapeHtml(item.symbol)}</div>` +
              `<div style="opacity:0.75">${escapeHtml(item.description)}</div>` +
              row(null, `${fmt.signed(item.contribution_cycles, 1)} cycles`, "net effect on predicted RUL")
            );
          },
        },
        xAxis: { type: "value", min: -extent, max: extent, ...axisCommon(c), axisLine: { show: false }, axisLabel: { ...axisCommon(c).axisLabel, formatter: (v) => fmt.signed(v, 0) } },
        yAxis: {
          type: "category",
          inverse: true,
          data: items.map((i) => i.symbol),
          ...axisCommon(c),
          axisLine: { show: false, onZero: false },
          splitLine: { show: false },
        },
        series: [
          {
            type: "bar",
            barMaxWidth: 14,
            data: items.map((i) => {
              const v = i.contribution_cycles;
              return {
                value: v,
                itemStyle: { color: v < 0 ? c.neg : c.pos, borderRadius: v < 0 ? [4, 0, 0, 4] : [0, 4, 4, 0] },
                label: { show: true, position: v < 0 ? "left" : "right", color: c.text2, fontSize: 11, formatter: fmt.signed(v, 1) },
              };
            }),
            markLine: { silent: true, symbol: "none", label: { show: false }, lineStyle: { color: c.axis, width: 1, type: "solid" }, data: [{ xAxis: 0 }] },
          },
        ],
      },
      { notMerge: true },
    );
    const total = items.reduce((acc, i) => acc + i.contribution_cycles, 0);
    setText(
      "drivers-sub",
      `Red pulls the prediction towards failure; blue pushes it away. Together they move it ${fmt.signed(total, 0)} cycles from an average engine.`,
    );
  }

  function renderTraces(d) {
    const c = palette();
    const names = Object.keys(d.traces.sensors);
    for (let k = 0; k < 3; k += 1) {
      const titleNode = $(`trace-title-${k}`);
      const symbol = names[k];
      if (!symbol) {
        titleNode.textContent = "";
        chart(`trace-${k}`).clear();
        continue;
      }
      const trace = d.traces.sensors[symbol];
      titleNode.replaceChildren(
        el("strong", { text: symbol }),
        document.createTextNode(` · ${trace.description}${trace.unit ? ` (${trace.unit})` : ""}`),
      );
      chart(`trace-${k}`).setOption(
        {
          animation: false,
          grid: { left: 52, right: 10, top: 8, bottom: 22 },
          tooltip: {
            ...tooltipBase(c),
            trigger: "axis",
            axisPointer: { type: "line", lineStyle: { color: c.axis } },
            formatter: (params) => {
              const p = Array.isArray(params) ? params[0] : params;
              return `<div style="font-weight:600">Cycle ${escapeHtml(p.name)}</div>` + row(c.series, fmt.num(p.value, 2), symbol);
            },
          },
          xAxis: { type: "category", data: d.traces.cycles.map(String), boundaryGap: false, ...axisCommon(c), splitLine: { show: false } },
          yAxis: { type: "value", scale: true, ...axisCommon(c), axisLine: { show: false }, splitNumber: 3 },
          series: [{ type: "line", data: trace.values, symbol: "none", lineStyle: { color: c.series, width: 2 }, emphasis: { disabled: true } }],
        },
        { notMerge: true },
      );
    }
  }

  function renderTables(d) {
    const rows = d.history.slice(-30).reverse();
    const head = el("tr", {}, ...["Cycle", "Predicted RUL", "Nominal 90% interval", "True RUL", "P(fail)"].map((h) => el("th", { text: h, attrs: { scope: "col" } })));
    const body = rows.map((r) =>
      el(
        "tr",
        {},
        el("td", { text: String(r.cycle) }),
        el("td", { text: fmt.num(r.point, 1) }),
        el("td", { text: fmt.range(r, 1) }),
        el("td", { text: String(r.rul_true) }),
        el("td", { text: fmt.pct(r.probability, 1) }),
      ),
    );
    $("table-rul").replaceChildren(el("thead", {}, head), el("tbody", {}, ...body));
    const dHead = el("tr", {}, ...["Sensor", "Description", "Net effect (cycles)"].map((h) => el("th", { text: h, attrs: { scope: "col" } })));
    const dBody = d.contributions.map((i) =>
      el("tr", {}, el("td", { text: i.symbol }), el("td", { text: i.description }), el("td", { text: fmt.signed(i.contribution_cycles, 2) })),
    );
    $("table-drivers").replaceChildren(el("thead", {}, dHead), el("tbody", {}, ...dBody));
  }

  function renderDetail(d) {
    if (!state.info || !d || !d.history.length) return;
    state.detail = d;
    const last = d.history[d.history.length - 1];
    const fleetEntry = state.fleet?.engines.find((e) => e.unit === d.unit);
    setText("detail-title", `Engine #${d.unit}`);
    $("detail-status").replaceChildren(chip(fleetEntry ? fleetEntry.status : last.status));
    setText(
      "detail-meta",
      `Cycle ${last.cycle} of ${d.total_cycles} · predicted RUL ${fmt.num(last.point, 0)} (${fmt.range(last)}) · true RUL ${last.rul_true}${d.lap > 1 ? ` · run ${d.lap}` : ""}`,
    );
    renderLife(d);
    renderHeat(d);
    renderDrivers(d);
    renderTraces(d);
    renderTables(d);
    document.querySelector(".detail").classList.remove("stale");
  }

  // ---------- selection, data flow ----------
  async function fetchJson(url) {
    const response = await fetch(url, { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
    return response.json();
  }

  function selectEngine(unit) {
    state.selected = unit;
    document.querySelectorAll("#fleet-body tr").forEach((tr) => tr.setAttribute("aria-selected", String(Number(tr.dataset.unit) === unit)));
    // Dim the old engine's charts only while a different engine loads (keeps the frame, no flash).
    if (state.detail && state.detail.unit !== unit) document.querySelector(".detail").classList.add("stale");
    if (state.socket && state.socket.connected) {
      state.socket.emit("watch_engine", { unit }, (ack) => {
        if (ack && ack.ok) renderDetail(ack.detail);
      });
    } else {
      fetchJson(`/api/engines/${unit}`).then(renderDetail).catch(() => {});
    }
  }

  function setConnection(stateName, text) {
    $("connection").dataset.state = stateName;
    setText("connection-text", text);
  }

  function connect() {
    if (typeof io !== "function") {
      setConnection("lost", "Live updates unavailable");
      return;
    }
    const socket = io();
    state.socket = socket;
    socket.on("connect", () => {
      setConnection("live", "Live");
      if (state.selected != null) selectEngine(state.selected);
    });
    socket.on("disconnect", () => setConnection("lost", "Reconnecting…"));
    socket.on("fleet_update", (snapshot) => {
      renderFleet(snapshot);
      setConnection("live", `Live · tick ${snapshot.tick}`);
    });
    socket.on("engine_update", (detail) => {
      if (detail.unit === state.selected) renderDetail(detail);
    });
  }

  function effectiveTheme() {
    const forced = document.documentElement.dataset.theme;
    if (forced) return forced;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function rerender() {
    if (state.fleet) renderFleet(state.fleet);
    if (state.detail) renderDetail(state.detail);
    setText("theme-label", effectiveTheme() === "dark" ? "Light" : "Dark");
  }

  function toggleTheme() {
    const next = effectiveTheme() === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("pm-theme", next);
    } catch (e) {
      /* storage disabled */
    }
    rerender();
  }

  async function init() {
    $("fleet-body").addEventListener("click", onFleetActivate);
    $("fleet-body").addEventListener("keydown", onFleetActivate);
    $("theme-toggle").addEventListener("click", toggleTheme);
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", rerender);
    window.addEventListener("resize", () => Object.values(state.charts).forEach((ch) => ch.resize()));
    setText("theme-label", effectiveTheme() === "dark" ? "Light" : "Dark");
    try {
      const [info, fleet] = await Promise.all([fetchJson("/api/model-info"), fetchJson("/api/fleet")]);
      renderInfo(info);
      renderFleet(fleet);
      const priority = ["alert", "watch", "failed", "healthy"];
      const first = [...fleet.engines].sort((a, b) => priority.indexOf(a.status) - priority.indexOf(b.status))[0];
      selectEngine(first.unit);
    } catch (error) {
      setText("model-card", "The model is not loaded. Train it first: python -m predmaint.train");
      setConnection("lost", "No model");
      return;
    }
    connect();
  }

  init();
})();
