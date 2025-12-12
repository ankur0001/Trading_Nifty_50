// static/chart.js
// Client-side chart logic: initial latest, lazy load before/after, custom range.
// Uses setData() and preserves visible logical range on prepend so chart does not jump.

const INITIAL_LIMIT = 300;
let chart, candleSeries;
let allCandles = [];   // client-side in-memory dataset (sorted ascending by time)
let loadedMin = null;  // unix seconds (oldest in allCandles)
let loadedMax = null;  // unix seconds (newest in allCandles)
let loading = false;

function createChart() {
  chart = LightweightCharts.createChart(document.getElementById('chart'), {
    layout: { background: { color: '#ffffff' }, textColor: '#111' },
    grid: { vertLines: { color: '#f0f0f0' }, horzLines: { color: '#f7f7f7' } },
    rightPriceScale: { borderVisible: true },
    timeScale: {
      timeVisible: true,
      secondsVisible: false,
      rightOffset: 12,
      barSpacing: 20,        // make candles visible (not thin lines)
      minBarSpacing: 15
    },
    handleScroll: { mouseWheel: true, pressedMouseMove: true },
    handleScale: { mouseWheel: true, pinch: true }
  });

  candleSeries = chart.addCandlestickSeries({
    upColor: '#26a69a', downColor: '#ef5350',
    wickUpColor: '#26a69a', wickDownColor: '#ef5350',
    borderVisible: false
  });

  // detect near-edges via logical indices
  chart.timeScale().subscribeVisibleLogicalRangeChange(range => {
    if (!range || !allCandles.length) return;
    const from = Math.floor(range.from);
    const to = Math.ceil(range.to);
    const THRESH = 6;

    if (from <= THRESH) {
      if (!loading && loadedMin > window.DATA_MIN_TS) loadBefore();
    }

    if (to >= (allCandles.length - THRESH)) {
      if (!loading && loadedMax < window.DATA_MAX_TS) loadAfter();
    }
  });
}

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
  document.getElementById('btnLoad').onclick = () => {
    const s = document.getElementById('start').value;
    const e = document.getElementById('end').value;
    loadRange(s, e);
  };
}

function toLocalInput(d) {
  const pad = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

// ---------- initial load
async function loadLatest(limit = INITIAL_LIMIT) {
  if (loading) return;
  loading = true;
  try {
    const res = await fetch(`/data-latest?limit=${limit}`);
    const j = await res.json();
    allCandles = j.candles || [];
    if (!allCandles.length) {
      candleSeries.setData([]);
      loading = false;
      return;
    }
    loadedMin = j.min_time;
    loadedMax = j.max_time;
    candleSeries.setData(allCandles);
    chart.timeScale().fitContent();
  } finally {
    loading = false;
  }
}

// ---------- prepend older candles while preserving viewport
async function loadBefore(limit = INITIAL_LIMIT) {
  if (loading) return;
  if (loadedMin === null) return;
  if (loadedMin <= window.DATA_MIN_TS) return;
  // get current visible logical range
  const oldRange = chart.timeScale().getVisibleLogicalRange();
  const oldFrom = oldRange ? oldRange.from : null;
  const oldTo = oldRange ? oldRange.to : null;

  loading = true;
  try {
    const res = await fetch(`/data-before?time=${loadedMin}&limit=${limit}`);
    const j = await res.json();
    if (j.candles && j.candles.length) {
      const newCount = j.candles.length;
      // prepend
      allCandles = [...j.candles, ...allCandles];
      // update loaded bounds
      loadedMin = j.min_time;
      loadedMax = Math.max(loadedMax, j.max_time);
      // setData with full dataset
      candleSeries.setData(allCandles);
      // preserve viewport by shifting visible logical range right by newCount
      if (oldFrom !== null && oldTo !== null) {
        chart.timeScale().setVisibleLogicalRange({
          from: oldFrom + newCount,
          to: oldTo + newCount
        });
      } else {
        chart.timeScale().fitContent();
      }
    }
  } finally {
    loading = false;
  }
}

// ---------- append newer candles while preserving viewport
async function loadAfter(limit = INITIAL_LIMIT) {
  if (loading) return;
  if (loadedMax === null) return;
  if (loadedMax >= window.DATA_MAX_TS) return;

  const oldRange = chart.timeScale().getVisibleLogicalRange();
  const oldFrom = oldRange ? oldRange.from : null;
  const oldTo = oldRange ? oldRange.to : null;

  loading = true;
  try {
    const res = await fetch(`/data-after?time=${loadedMax}&limit=${limit}`);
    const j = await res.json();
    if (j.candles && j.candles.length) {
      const newCount = j.candles.length;
      allCandles = [...allCandles, ...j.candles];
      loadedMin = Math.min(loadedMin, j.min_time);
      loadedMax = j.max_time;
      candleSeries.setData(allCandles);
      // preserve viewport (same logical indices)
      if (oldFrom !== null && oldTo !== null) {
        chart.timeScale().setVisibleLogicalRange({
          from: oldFrom,
          to: oldTo
        });
      } else {
        chart.timeScale().fitContent();
      }
    }
  } finally {
    loading = false;
  }
}

// ---------- load custom range
async function loadRange(startIso, endIso) {
  if (!startIso) return alert("Please select a start datetime");
  let start = startIso;
  let end = endIso;
  if (!end) end = "";
  const qs = new URLSearchParams({ start: start, end: end });
  const res = await fetch(`/data-range?${qs.toString()}`);
  const j = await res.json();
  allCandles = j.candles || [];
  if (!allCandles.length) {
    candleSeries.setData([]);
    alert("No data found for this range");
    return;
  }
  loadedMin = j.min_time;
  loadedMax = j.max_time;
  candleSeries.setData(allCandles);
  chart.timeScale().fitContent();
}

// ---------- boot
(async function init() {
  createChart();
  await initUI();
  await loadLatest();
})();
