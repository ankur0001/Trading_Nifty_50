// static/chart.js
// Full client-side chart + indicators (SMA20, SMA50, EMA12, Bollinger, PacePro, RSI14, MACD)
// Main chart overlays: SMA20, SMA50, EMA12, Bollinger bands, PacePro histogram
// RSI and MACD are separate stacked charts below the main chart
// Indicators update on any candle update (latest / before / after / range)

const INITIAL_LIMIT = 300;

// main chart + subcharts
let mainChart, rsiChart, macdChart;
let candleSeries;
let sma20Series, sma50Series, ema12Series;
let bbUpperSeries, bbMiddleSeries, bbLowerSeries;
let paceProSeries;
let rsiSeries;
let macdSeries, macdSignalSeries, macdHistSeries;

let allCandles = [];   // ascending by time
let loadedMin = null;
let loadedMax = null;
let loading = false;

// toggles
let opts = {
  sma20: true,
  sma50: true,
  ema12: true,
  bb: true,
  pacePro: true,
  rsi: true,
  macd: true
};

// ----------------- chart creation -----------------
function createCharts() {
  // Main chart (candles + overlays)
  mainChart = LightweightCharts.createChart(document.getElementById('chart-main'), {
    layout: { background: { color: '#ffffff' }, textColor: '#111' },
    grid: { vertLines: { color: '#f0f0f0' }, horzLines: { color: '#f7f7f7' } },
    rightPriceScale: { borderVisible: true },
    timeScale: { timeVisible: true, secondsVisible: false, rightOffset: 12, barSpacing: 20, minBarSpacing: 6 },
    handleScroll: { mouseWheel: true, pressedMouseMove: true },
    handleScale: { mouseWheel: true, pinch: true }
  });

  candleSeries = mainChart.addCandlestickSeries({
    upColor: '#26a69a', downColor: '#ef5350',
    wickUpColor: '#26a69a', wickDownColor: '#ef5350',
    borderVisible: false
  });

  // overlay lines
  sma20Series = mainChart.addLineSeries({ color: '#1E88E5', lineWidth: 2, visible: true, lastValueVisible: false });
  sma50Series = mainChart.addLineSeries({ color: '#6A1B9A', lineWidth: 2, visible: true, lastValueVisible: false });
  ema12Series = mainChart.addLineSeries({ color: '#00ACC1', lineWidth: 2, visible: true, lastValueVisible: false });

  // Bollinger bands: middle (SMA20), upper, lower (lighter color)
  bbMiddleSeries = mainChart.addLineSeries({ color: '#666', lineWidth: 1.5, lastValueVisible: false });
  bbUpperSeries = mainChart.addLineSeries({ color: '#BBB', lineWidth: 1, lastValueVisible: false });
  bbLowerSeries = mainChart.addLineSeries({ color: '#BBB', lineWidth: 1, lastValueVisible: false });

  // PacePro histogram overlay (will use per-bar color)
  paceProSeries = mainChart.addHistogramSeries({
    color: '#1976D2',
    base: 0,
    priceFormat: { type: 'volume' },
    priceScaleId: '', // overlay
    scaleMargins: { top: 0.82, bottom: 0 }
  });

  // RSI chart (0-100)
  rsiChart = LightweightCharts.createChart(document.getElementById('chart-rsi'), {
    layout: { background: { color: '#fff' }, textColor: '#111' },
    height: 160,
    rightPriceScale: { scaleMargins: { top: 0.2, bottom: 0.2 } },
    timeScale: { timeVisible: true, secondsVisible: false, rightOffset: 12, barSpacing: 20 }
  });
  rsiSeries = rsiChart.addLineSeries({ color: '#F57C00', lineWidth: 2 });

  // MACD chart (histogram + signal + macd line)
  macdChart = LightweightCharts.createChart(document.getElementById('chart-macd'), {
    layout: { background: { color: '#fff' }, textColor: '#111' },
    height: 160,
    rightPriceScale: { scaleMargins: { top: 0.2, bottom: 0.2 } },
    timeScale: { timeVisible: true, secondsVisible: false, rightOffset: 12, barSpacing: 20 }
  });
  macdHistSeries = macdChart.addHistogramSeries({ color: '#8E24AA', base: 0, priceFormat: { type: 'volume' } });
  macdSeries = macdChart.addLineSeries({ color: '#1565C0', lineWidth: 2 });
  macdSignalSeries = macdChart.addLineSeries({ color: '#D32F2F', lineWidth: 1.5 });

  // wire timeScale lazy-load detection on main chart (we only lazy load when viewing main chart)
  mainChart.timeScale().subscribeVisibleLogicalRangeChange(range => {
    if (!range || !allCandles.length) return;
    const from = Math.floor(range.from);
    const to = Math.ceil(range.to);
    const THRESH = 6;
    if (from <= THRESH && !loading && loadedMin > window.DATA_MIN_TS) loadBefore();
    if (to >= (allCandles.length - THRESH) && !loading && loadedMax < window.DATA_MAX_TS) loadAfter();
  });

  // sync time scales between charts so they scroll/zoom together
  function syncTimeScales() {
    const tsMain = mainChart.timeScale();
    const tsRsi = rsiChart.timeScale();
    const tsMacd = macdChart.timeScale();

    // when main changes, apply to other two
    tsMain.subscribeVisibleLogicalRangeChange(range => {
      if (!range) return;
      tsRsi.setVisibleLogicalRange(range);
      tsMacd.setVisibleLogicalRange(range);
    });
    // also when sub charts change, propagate back to main (optional)
    tsRsi.subscribeVisibleLogicalRangeChange(range => {
      if (!range) return;
      tsMain.setVisibleLogicalRange(range);
      tsMacd.setVisibleLogicalRange(range);
    });
    tsMacd.subscribeVisibleLogicalRangeChange(range => {
      if (!range) return;
      tsMain.setVisibleLogicalRange(range);
      tsRsi.setVisibleLogicalRange(range);
    });
  }
  syncTimeScales();
}

// ----------------- indicator calculations -----------------
function sma(values, period) {
  const out = [];
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
    if (i >= period) sum -= values[i - period];
    if (i >= period - 1) out.push(sum / period);
    else out.push(null);
  }
  return out;
}

function ema(values, period) {
  const out = [];
  const k = 2 / (period + 1);
  let prev = null;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (prev === null) {
      // first EMA value: simple SMA of first 'period' values when available
      if (i === period - 1) {
        const sum = values.slice(0, period).reduce((a, b) => a + b, 0);
        prev = sum / period;
        out.push(prev);
      } else {
        out.push(null);
      }
    } else {
      prev = v * k + prev * (1 - k);
      out.push(prev);
    }
  }
  return out;
}

function stddev(values, period) {
  const out = [];
  for (let i = 0; i < values.length; i++) {
    if (i < period - 1) { out.push(null); continue; }
    const window = values.slice(i - period + 1, i + 1);
    const mean = window.reduce((a, b) => a + b, 0) / period;
    const variance = window.reduce((s, x) => s + Math.pow(x - mean, 2), 0) / period;
    out.push(Math.sqrt(variance));
  }
  return out;
}

function calculateIndicators(candles) {
  // candles: [{time, open, high, low, close}]
  if (!candles.length) return {};

  const closes = candles.map(c => c.close);
  const opens = candles.map(c => c.open);
  const highs = candles.map(c => c.high);
  const lows = candles.map(c => c.low);

  // SMA20 & SMA50
  const sma20 = sma(closes, 20);
  const sma50 = sma(closes, 50);

  // EMA12 (and EMA26 for MACD)
  const ema12 = ema(closes, 12);
  const ema26 = ema(closes, 26);

  // Bollinger Bands (SMA20 +/- 2*std)
  const bbMid = sma20;
  const bbStd = stddev(closes, 20);
  const bbUpper = bbMid.map((m, i) => (m === null || bbStd[i] === null) ? null : m + 2 * bbStd[i]);
  const bbLower = bbMid.map((m, i) => (m === null || bbStd[i] === null) ? null : m - 2 * bbStd[i]);

  // PacePro (use close-open percent, normalized/smoothed)
  const paceRaw = [];
  for (let i = 0; i < candles.length; i++) {
    const c = candles[i];
    const val = ((c.close - c.open) / c.open) * 100; // percent
    paceRaw.push(val);
  }
  // normalize & smooth (moving avg)
  const paceNorm = normalizeAndSmooth(paceRaw, 5);

  // RSI (14)
  const rsi = calculateRSI(closes, 14);

  // MACD
  const macdLine = []; // ema12 - ema26
  for (let i = 0; i < closes.length; i++) {
    const e12 = ema12[i];
    const e26 = ema26[i];
    macdLine.push((e12 === null || e26 === null) ? null : e12 - e26);
  }
  // signal = ema of macdLine (only pass numeric values, but we need to compute EMA on series with nulls)
  const macdSignal = emaFilled(macdLine, 9);
  const macdHist = macdLine.map((v, i) => (v === null || macdSignal[i] === null) ? null : v - macdSignal[i]);

  return {
    sma20, sma50, ema12, bbUpper, bbMiddle: bbMid, bbLower,
    paceNorm,
    rsi, macdLine, macdSignal, macdHist
  };
}

// helper: normalize & smooth: scale values to -100..100 and moving average
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
    smoothed.push(count ? sum / count : 0);
  }
  return smoothed;
}

// RSI calculation (classic)
function calculateRSI(closes, period = 14) {
  const out = [];
  let gains = 0, losses = 0;
  for (let i = 0; i < closes.length; i++) {
    if (i === 0) { out.push(null); continue; }
    const change = closes[i] - closes[i - 1];
    const gain = Math.max(change, 0);
    const loss = Math.max(-change, 0);
    if (i <= period) {
      gains += gain;
      losses += loss;
      if (i === period) {
        const avgGain = gains / period;
        const avgLoss = losses / period;
        const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
        out.push(100 - (100 / (1 + rs)));
      } else {
        out.push(null);
      }
    } else {
      // Wilder smoothing
      const prevAvgGain = out.avgGain !== undefined ? out.avgGain : (gains / period);
      const prevAvgLoss = out.avgLoss !== undefined ? out.avgLoss : (losses / period);
      const avgGain = (prevAvgGain * (period - 1) + gain) / period;
      const avgLoss = (prevAvgLoss * (period - 1) + loss) / period;
      out.avgGain = avgGain;
      out.avgLoss = avgLoss;
      const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
      out.push(100 - (100 / (1 + rs)));
    }
  }
  return out;
}

// EMA for arrays with nulls: compute EMA ignoring leading nulls and propagate nulls until first numeric value
function emaFilled(series, period) {
  const out = [];
  const k = 2 / (period + 1);
  let prev = null;
  for (let i = 0; i < series.length; i++) {
    const v = series[i];
    if (v === null || v === undefined) {
      out.push(null);
      continue;
    }
    if (prev === null) {
      // compute SMA over the first 'period' available numeric values ending at i if possible
      const window = [];
      for (let j = i; j > i - period && j >= 0; j--) {
        if (series[j] !== null && series[j] !== undefined) window.unshift(series[j]);
      }
      if (window.length < period) {
        // not enough for an initial SMA -> put null until we can
        out.push(null);
        continue;
      } else {
        const smaInit = window.reduce((a, b) => a + b, 0) / window.length;
        prev = smaInit;
        out.push(prev);
        continue;
      }
    } else {
      prev = v * k + prev * (1 - k);
      out.push(prev);
    }
  }
  return out;
}

// ----------------- apply indicator data to series -----------------
function updateAllIndicatorSeries() {
  const indicators = calculateIndicators(allCandles);

  // helper to build series entries array or []
  function buildSeries(arr, keyName = 'value') {
    if (!arr || !arr.length) return [];
    const out = [];
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      if (v === null || v === undefined) continue;
      out.push({ time: allCandles[i].time, [keyName]: v });
    }
    return out;
  }

  // SMA & EMA
  if (opts.sma20) sma20Series.setData(buildSeries(indicators.sma20, 'value')); else sma20Series.setData([]);
  if (opts.sma50) sma50Series.setData(buildSeries(indicators.sma50, 'value')); else sma50Series.setData([]);
  if (opts.ema12) ema12Series.setData(buildSeries(indicators.ema12, 'value')); else ema12Series.setData([]);

  // Bollinger Bands (three lines)
  if (opts.bb) {
    bbMiddleSeries.setData(buildSeries(indicators.bbMiddle, 'value'));
    bbUpperSeries.setData(buildSeries(indicators.bbUpper, 'value'));
    bbLowerSeries.setData(buildSeries(indicators.bbLower, 'value'));
  } else {
    bbMiddleSeries.setData([]); bbUpperSeries.setData([]); bbLowerSeries.setData([]);
  }

  // PacePro histogram (per-bar color)
  if (opts.pacePro) {
    // indicators.paceNorm is an array aligned with candles
    const paceArr = indicators.paceNorm || [];
    const hist = [];
    for (let i = 0; i < paceArr.length; i++) {
      if (paceArr[i] === null || paceArr[i] === undefined) continue;
      // choose neutral colors: positive -> blue, negative -> orange, zero -> gray
      const v = paceArr[i];
      let color = '#1E88E5';
      if (v < 0) color = '#FF9800';
      else if (Math.abs(v) < 0.0001) color = '#999';
      hist.push({ time: allCandles[i].time, value: v, color });
    }
    paceProSeries.setData(hist);
  } else {
    paceProSeries.setData([]);
  }

  // RSI series (0..100)
  if (opts.rsi) {
    const rsiArr = indicators.rsi || [];
    const rsiData = [];
    for (let i = 0; i < rsiArr.length; i++) {
      const v = rsiArr[i];
      if (v === null || v === undefined) continue;
      rsiData.push({ time: allCandles[i].time, value: v });
    }
    rsiSeries.setData(rsiData);
  } else {
    rsiSeries.setData([]);
  }

  // MACD: series & signal & histogram
  if (opts.macd) {
    const macd = indicators.macdLine || [];
    const signal = indicators.macdSignal || [];
    const hist = indicators.macdHist || [];
    const macdData = [], sigData = [], histData = [];
    for (let i = 0; i < allCandles.length; i++) {
      if (macd[i] !== null && macd[i] !== undefined) macdData.push({ time: allCandles[i].time, value: macd[i] });
      if (signal[i] !== null && signal[i] !== undefined) sigData.push({ time: allCandles[i].time, value: signal[i] });
      if (hist[i] !== null && hist[i] !== undefined) histData.push({ time: allCandles[i].time, value: hist[i], color: (hist[i] >= 0 ? '#8E24AA' : '#FFB300') });
    }
    macdSeries.setData(macdData);
    macdSignalSeries.setData(sigData);
    macdHistSeries.setData(histData);
  } else {
    macdSeries.setData([]); macdSignalSeries.setData([]); macdHistSeries.setData([]);
  }
}

// ----------------- data loading (uses your APIs) -----------------
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
    updateAllIndicatorSeries();
    // fit content and sync subcharts
    mainChart.timeScale().fitContent();
  } finally { loading = false; }
}

async function loadBefore(limit = INITIAL_LIMIT) {
  if (loading || loadedMin === null || loadedMin <= window.DATA_MIN_TS) return;
  const oldRange = mainChart.timeScale().getVisibleLogicalRange();
  loading = true;
  try {
    const res = await fetch(`/data-before?time=${loadedMin}&limit=${limit}`);
    const j = await res.json();
    if (j.candles && j.candles.length) {
      const newCount = j.candles.length;
      allCandles = [...j.candles, ...allCandles];
      loadedMin = j.min_time;
      loadedMax = Math.max(loadedMax, j.max_time);
      candleSeries.setData(allCandles);
      updateAllIndicatorSeries();
      if (oldRange) {
        mainChart.timeScale().setVisibleLogicalRange({ from: oldRange.from + newCount, to: oldRange.to + newCount });
      } else {
        mainChart.timeScale().fitContent();
      }
    }
  } finally { loading = false; }
}

async function loadAfter(limit = INITIAL_LIMIT) {
  if (loading || loadedMax === null || loadedMax >= window.DATA_MAX_TS) return;
  const oldRange = mainChart.timeScale().getVisibleLogicalRange();
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
      updateAllIndicatorSeries();
      if (oldRange) {
        mainChart.timeScale().setVisibleLogicalRange({ from: oldRange.from, to: oldRange.to });
      } else {
        mainChart.timeScale().fitContent();
      }
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
    updateAllIndicatorSeries();
    alert("No data found for this range");
    return;
  }
  loadedMin = j.min_time;
  loadedMax = j.max_time;
  candleSeries.setData(allCandles);
  updateAllIndicatorSeries();
  mainChart.timeScale().fitContent();
}

// ----------------- UI init & toggles -----------------
async function initUI() {
  // data info
  const info = await fetch('/data-info').then(r => r.json());
  window.DATA_MIN_TS = info.min_ts;
  window.DATA_MAX_TS = info.max_ts;

  // date inputs
  const startEl = document.getElementById('start');
  const endEl = document.getElementById('end');
  startEl.min = info.min_str; startEl.max = info.max_str;
  endEl.min = info.min_str; endEl.max = info.max_str;
  const maxDate = new Date(info.max_ts * 1000);
  const defaultStart = new Date((info.max_ts - 24 * 3600) * 1000);
  startEl.value = toLocalInput(defaultStart);
  endEl.value = toLocalInput(maxDate);

  // range buttons
  document.getElementById('btnLatest').onclick = () => loadLatest();
  document.getElementById('btnLoad').onclick = () => loadRange(startEl.value, endEl.value);

  // toggles
  document.getElementById('toggle-sma20').addEventListener('change', e => { opts.sma20 = e.target.checked; updateAllIndicatorSeries(); });
  document.getElementById('toggle-sma50').addEventListener('change', e => { opts.sma50 = e.target.checked; updateAllIndicatorSeries(); });
  document.getElementById('toggle-ema12').addEventListener('change', e => { opts.ema12 = e.target.checked; updateAllIndicatorSeries(); });
  document.getElementById('toggle-bb').addEventListener('change', e => { opts.bb = e.target.checked; updateAllIndicatorSeries(); });
  document.getElementById('paceproToggle').addEventListener('change', e => { opts.pacePro = e.target.checked; updateAllIndicatorSeries(); });

  document.getElementById('toggle-rsi').addEventListener('change', e => {
    opts.rsi = e.target.checked;
    if (!opts.rsi) rsiSeries.setData([]); else updateAllIndicatorSeries();
  });
  document.getElementById('toggle-macd').addEventListener('change', e => {
    opts.macd = e.target.checked;
    if (!opts.macd) { macdSeries.setData([]); macdSignalSeries.setData([]); macdHistSeries.setData([]); }
    else updateAllIndicatorSeries();
  });
}

// small helper to format date to datetime-local input value
function toLocalInput(d) {
  const pad = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

// ----------------- boot -----------------
(async function init() {
  createCharts();
  await initUI();
  await loadLatest();
})();
