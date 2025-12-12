// static/chart.js
// Lightweight Charts with toggleable PacePro histogram (neutral colors)

const INITIAL_LIMIT = 300;
let chart, candleSeries, paceProSeries;
let allCandles = [];
let loadedMin = null;
let loadedMax = null;
let loading = false;
let paceProEnabled = false;

// ---------- create chart ----------
function createChart() {
  chart = LightweightCharts.createChart(document.getElementById('chart'), {
    layout: { background: { color: '#ffffff' }, textColor: '#111' },
    grid: { vertLines: { color: '#f0f0f0' }, horzLines: { color: '#f7f7f7' } },
    rightPriceScale: { borderVisible: true },
    timeScale: { timeVisible: true, secondsVisible: false, rightOffset: 12, barSpacing: 20, minBarSpacing: 15 },
    handleScroll: { mouseWheel: true, pressedMouseMove: true },
    handleScale: { mouseWheel: true, pinch: true }
  });

  candleSeries = chart.addCandlestickSeries({
    upColor: '#26a69a', downColor: '#ef5350',
    wickUpColor: '#26a69a', wickDownColor: '#ef5350',
    borderVisible: false
  });

  paceProSeries = chart.addHistogramSeries({
    color: '#2196F3', // default color; will override per bar
    priceFormat: { type: 'volume' },
    priceScaleId: '', // overlay
    scaleMargins: { top: 0.85, bottom: 0 }
  });

  chart.timeScale().subscribeVisibleLogicalRangeChange(range => {
    if (!range || !allCandles.length) return;
    const from = Math.floor(range.from);
    const to = Math.ceil(range.to);
    const THRESH = 6;

    if (from <= THRESH && !loading && loadedMin > window.DATA_MIN_TS) loadBefore();
    if (to >= (allCandles.length - THRESH) && !loading && loadedMax < window.DATA_MAX_TS) loadAfter();
  });
}

// ---------- initialize UI ----------
async function initUI() {
  const info = await fetch('/data-info').then(r => r.json());
  window.DATA_MIN_TS = info.min_ts;
  window.DATA_MAX_TS = info.max_ts;

  const startEl = document.getElementById('start');
  const endEl = document.getElementById('end');

  startEl.min = info.min_str;
  startEl.max = info.max_str;
  endEl.min = info.min_str;
  endEl.max = info.max_str;

  const maxDate = new Date(info.max_ts * 1000);
  const defaultStart = new Date((info.max_ts - 24 * 3600) * 1000);

  startEl.value = toLocalInput(defaultStart);
  endEl.value = toLocalInput(maxDate);

  document.getElementById('btnLatest').onclick = () => loadLatest();
  document.getElementById('btnLoad').onclick = () => loadRange(startEl.value, endEl.value);

  document.getElementById('paceproToggle').addEventListener('change', e => {
    paceProEnabled = e.target.checked;
    updatePaceProSeries();
  });
}

// ---------- helper ----------
function toLocalInput(d) {
  const pad = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

// ---------- normalize & smooth ----------
function normalizeAndSmooth(values, period = 5) {
  if (!values.length) return [];
  const absMax = Math.max(...values.map(v => Math.abs(v))) || 1;
  const normalized = values.map(v => (v / absMax) * 100);
  const smoothed = [];
  for (let i = 0; i < normalized.length; i++) {
    let sum = 0, count = 0;
    for (let j = i - period + 1; j <= i; j++) {
      if (j >= 0) { sum += normalized[j]; count++; }
    }
    smoothed.push(sum / count);
  }
  return smoothed;
}

// ---------- PacePro calculation ----------
function calculatePacePro(candles) {
  const pace = [];
  for (let i = 1; i < candles.length; i++) {
    const prev = candles[i-1];
    const curr = candles[i];
    pace.push(((curr.close - curr.open) / curr.open) * 100);
  }
  pace.unshift(0);

  const smoothPace = normalizeAndSmooth(pace, 5);

  return smoothPace.map((val, idx) => {
    let color = val >= 0 ? '#1E88E5' : '#FF9800'; // blue for positive, orange for negative
    return { time: candles[idx].time, value: val, color };
  });
}

// ---------- update PacePro series ----------
function updatePaceProSeries() {
  if (!paceProSeries) return;
  if (paceProEnabled && allCandles.length) {
    paceProSeries.setData(calculatePacePro(allCandles));
  } else {
    paceProSeries.setData([]);
  }
}

// ---------- load functions ----------
async function loadLatest(limit = INITIAL_LIMIT) {
  if (loading) return;
  loading = true;
  try {
    const res = await fetch(`/data-latest?limit=${limit}`);
    const j = await res.json();
    allCandles = j.candles || [];
    loadedMin = j.min_time;
    loadedMax = j.max_time;
    candleSeries.setData(allCandles);
    updatePaceProSeries();
    chart.timeScale().fitContent();
  } finally { loading = false; }
}

async function loadBefore(limit = INITIAL_LIMIT) {
  if (loading || loadedMin === null || loadedMin <= window.DATA_MIN_TS) return;
  const oldRange = chart.timeScale().getVisibleLogicalRange();
  loading = true;
  try {
    const res = await fetch(`/data-before?time=${loadedMin}&limit=${limit}`);
    const j = await res.json();
    if (j.candles.length) {
      const newCount = j.candles.length;
      allCandles = [...j.candles, ...allCandles];
      loadedMin = j.min_time;
      loadedMax = Math.max(loadedMax, j.max_time);
      candleSeries.setData(allCandles);
      updatePaceProSeries();
      if (oldRange) chart.timeScale().setVisibleLogicalRange({ from: oldRange.from + newCount, to: oldRange.to + newCount });
    }
  } finally { loading = false; }
}

async function loadAfter(limit = INITIAL_LIMIT) {
  if (loading || loadedMax === null || loadedMax >= window.DATA_MAX_TS) return;
  const oldRange = chart.timeScale().getVisibleLogicalRange();
  loading = true;
  try {
    const res = await fetch(`/data-after?time=${loadedMax}&limit=${limit}`);
    const j = await res.json();
    if (j.candles.length) {
      allCandles = [...allCandles, ...j.candles];
      loadedMin = Math.min(loadedMin, j.min_time);
      loadedMax = j.max_time;
      candleSeries.setData(allCandles);
      updatePaceProSeries();
      if (oldRange) chart.timeScale().setVisibleLogicalRange({ from: oldRange.from, to: oldRange.to });
    }
  } finally { loading = false; }
}

async function loadRange(startIso, endIso) {
  if (!startIso) return alert("Please select a start datetime");
  const qs = new URLSearchParams({ start: startIso, end: endIso });
  const res = await fetch(`/data-range?${qs.toString()}`);
  const j = await res.json();
  allCandles = j.candles || [];
  if (!allCandles.length) {
    candleSeries.setData([]);
    updatePaceProSeries();
    alert("No data found for this range");
    return;
  }
  loadedMin = j.min_time;
  loadedMax = j.max_time;
  candleSeries.setData(allCandles);
  updatePaceProSeries();
  chart.timeScale().fitContent();
}

// ---------- boot ----------
(async function init() {
  createChart();
  await initUI();
  await loadLatest();
})();
