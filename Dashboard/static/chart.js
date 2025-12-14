// static/chart.js
// Full client-side chart + overlays + toggles + fullscreen + slide-out sidebar
// Uses endpoints: /data-info, /data-latest, /data-before, /data-after, /data-range

const INITIAL_LIMIT = 300;

// Chart instances & series
let mainChart, rsiChart, macdChart;
let candleSeries;
let sma20Series, sma50Series, ema12Series;
let bbUpperSeries, bbMiddleSeries, bbLowerSeries;
let paceProSeries;
let rsiSeries;
let macdSeries, macdSignalSeries, macdHistSeries;

// in-memory candles (ascending by time)
let allCandles = [];
let loadedMin = null;
let loadedMax = null;
let loading = false;

// ---- Replay state ----
let replayMode = false;
let replayTimer = null;
let replayIndex = 0;
let replayCandles = [];
const INDICATOR_HEIGHT = 160;
const MAIN_MIN_HEIGHT = 200;   // 🔥 critical



const panelHeights = {
  rsi: null,
  macd: null
};


// toggles/options
const opts = {
  sma20: true, sma50: true, ema12: true, bb: true, pacePro: true, rsi: true, macd: true, signals: true
};

// ---- Header UI elements ----
let rangeBtn, replayBtn;
let rangePanel, replayPanel;

// indicator colors (auto-assigned non-conflicting)
const COLORS = {
  sma20: '#1E88E5',   // blue
  sma50: '#6A1B9A',   // purple
  ema12: '#00ACC1',   // teal
  bb: '#9E9E9E',      // gray
  pacePos: '#1565C0', // dark blue
  paceNeg: '#FF9800', // orange
  rsi: '#F57C00',
  macd: '#1565C0',
  macdSignal: '#D32F2F',
  macdHistPos: '#8E24AA',
  macdHistNeg: '#FFB300'
};

// ---------- create charts & series ----------
function createCharts() {
  // main chart
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

  // overlay series
  sma20Series = mainChart.addLineSeries({ color: COLORS.sma20, lineWidth: 2, lastValueVisible: false });
  sma50Series = mainChart.addLineSeries({ color: COLORS.sma50, lineWidth: 2, lastValueVisible: false });
  ema12Series = mainChart.addLineSeries({ color: COLORS.ema12, lineWidth: 2, lastValueVisible: false });

  bbMiddleSeries = mainChart.addLineSeries({ color: COLORS.bb, lineWidth: 1.2, lastValueVisible: false });
  bbUpperSeries = mainChart.addLineSeries({ color: COLORS.bb, lineWidth: 1, lastValueVisible: false });
  bbLowerSeries = mainChart.addLineSeries({ color: COLORS.bb, lineWidth: 1, lastValueVisible: false });

  paceProSeries = mainChart.addHistogramSeries({
    color: COLORS.pacePos,
    base: 0,
    priceFormat: { type: 'volume' },
    priceScaleId: '',
    scaleMargins: { top: 0.82, bottom: 0 }
  });

  // RSI chart
  rsiChart = LightweightCharts.createChart(document.getElementById('chart-rsi'), {
    layout: { background: { color: '#fff' }, textColor: '#111' },
    rightPriceScale: { scaleMargins: { top: 0.2, bottom: 0.2 } },
    timeScale: { timeVisible: true, secondsVisible: false, rightOffset: 12, barSpacing: 20 }
  });
  rsiSeries = rsiChart.addLineSeries({ color: COLORS.rsi, lineWidth: 2 });

  // MACD chart
  macdChart = LightweightCharts.createChart(document.getElementById('chart-macd'), {
    layout: { background: { color: '#fff' }, textColor: '#111' },
    rightPriceScale: { scaleMargins: { top: 0.2, bottom: 0.2 } },
    timeScale: { timeVisible: true, secondsVisible: false, rightOffset: 12, barSpacing: 20 }
  });
  macdHistSeries = macdChart.addHistogramSeries({ color: COLORS.macdHistPos, base: 0, priceFormat: { type: 'volume' } });
  macdSeries = macdChart.addLineSeries({ color: COLORS.macd, lineWidth: 2 });
  macdSignalSeries = macdChart.addLineSeries({ color: COLORS.macdSignal, lineWidth: 1.5 });

  // sync time scales
  syncTimeScales();

  // lazy-load detection on main chart
  mainChart.timeScale().subscribeVisibleLogicalRangeChange(range => {
    if (!range || !allCandles.length) return;
    const from = Math.floor(range.from);
    const to = Math.ceil(range.to);
    const THRESH = 6;
    if (from <= THRESH && !loading && loadedMin > window.DATA_MIN_TS) loadBefore();
    if (to >= (allCandles.length - THRESH) && !loading && loadedMax < window.DATA_MAX_TS) loadAfter();
  });

  // set badge colors in sidebar
  setIndicatorBadges();
}

function syncTimeScales() {
  const tsMain = mainChart.timeScale();
  const tsRsi = rsiChart.timeScale();
  const tsMacd = macdChart.timeScale();

  tsMain.subscribeVisibleLogicalRangeChange(range => {
    if (!range) return;
    tsRsi.setVisibleLogicalRange(range);
    tsMacd.setVisibleLogicalRange(range);
  });
  tsRsi.subscribeVisibleLogicalRangeChange(range => { if (!range) return; tsMain.setVisibleLogicalRange(range); tsMacd.setVisibleLogicalRange(range); });
  tsMacd.subscribeVisibleLogicalRangeChange(range => { if (!range) return; tsMain.setVisibleLogicalRange(range); tsRsi.setVisibleLogicalRange(range); });
}

// ---------------- indicator calculations ----------------
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
      if (i === period - 1) {
        const sum = values.slice(0, period).reduce((a, b) => a + b, 0);
        prev = sum / period;
        out.push(prev);
      } else out.push(null);
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

function emaFilled(series, period) {
  const out = [];
  const k = 2 / (period + 1);
  let prev = null;
  for (let i = 0; i < series.length; i++) {
    const v = series[i];
    if (v === null || v === undefined) { out.push(null); continue; }
    if (prev === null) {
      const window = [];
      for (let j = i; j > i - period && j >= 0; j--) {
        if (series[j] !== null && series[j] !== undefined) window.unshift(series[j]);
      }
      if (window.length < period) { out.push(null); continue; }
      const smaInit = window.reduce((a, b) => a + b, 0) / window.length;
      prev = smaInit;
      out.push(prev);
    } else {
      prev = v * k + prev * (1 - k);
      out.push(prev);
    }
  }
  return out;
}

function calculateRSI(closes, period = 14) {
  const out = [];
  let gains = 0, losses = 0;
  for (let i = 0; i < closes.length; i++) {
    if (i === 0) { out.push(null); continue; }
    const change = closes[i] - closes[i - 1];
    const gain = Math.max(change, 0);
    const loss = Math.max(-change, 0);
    if (i <= period) {
      gains += gain; losses += loss;
      if (i === period) {
        const avgGain = gains / period; const avgLoss = losses / period;
        const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
        out.push(100 - (100 / (1 + rs)));
      } else out.push(null);
    } else {
      const prevAvgGain = out.avgGain !== undefined ? out.avgGain : (gains / period);
      const prevAvgLoss = out.avgLoss !== undefined ? out.avgLoss : (losses / period);
      const avgGain = (prevAvgGain * (period - 1) + gain) / period;
      const avgLoss = (prevAvgLoss * (period - 1) + loss) / period;
      out.avgGain = avgGain; out.avgLoss = avgLoss;
      const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
      out.push(100 - (100 / (1 + rs)));
    }
  }
  return out;
}

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

function calculateIndicators(candles) {
  if (!candles.length) return {};
  const closes = candles.map(c => c.close);

  const sma20 = sma(closes, 20);
  const sma50 = sma(closes, 50);
  const ema12 = ema(closes, 12);
  const ema26 = ema(closes, 26);

  const bbMid = sma20;
  const bbStd = stddev(closes, 20);
  const bbUpper = bbMid.map((m, i) => (m === null || bbStd[i] === null) ? null : m + 2 * bbStd[i]);
  const bbLower = bbMid.map((m, i) => (m === null || bbStd[i] === null) ? null : m - 2 * bbStd[i]);

  const paceRaw = [];
  for (let i = 0; i < candles.length; i++) {
    const c = candles[i];
    const val = ((c.close - c.open) / c.open) * 100;
    paceRaw.push(val);
  }
  const paceNorm = normalizeAndSmooth(paceRaw, 5);

  const rsi = calculateRSI(closes, 14);

  const macdLine = [];
  for (let i = 0; i < closes.length; i++) {
    const e12 = ema12[i]; const e26 = ema26[i];
    macdLine.push((e12 === null || e26 === null) ? null : e12 - e26);
  }
  const macdSignal = emaFilled(macdLine, 9);
  const macdHist = macdLine.map((v, i) => (v === null || macdSignal[i] === null) ? null : v - macdSignal[i]);

  return { sma20, sma50, ema12, bbUpper, bbMiddle: bbMid, bbLower, paceNorm, rsi, macdLine, macdSignal, macdHist };
}

function calculateTradeSignals(candles, indicators) {
  const markers = [];

  for (let i = 1; i < candles.length; i++) {
    const ema = indicators.ema12;
    const sma = indicators.sma20;
    const rsi = indicators.rsi;

    if (
      ema[i] && sma[i] &&
      ema[i - 1] <= sma[i - 1] &&
      ema[i] > sma[i] &&
      rsi[i] > 40
    ) {
      markers.push({
        time: candles[i].time,
        position: 'belowBar',
        color: '#2ECC71',
        shape: 'arrowUp',
        text: 'BUY'
      });
    }

    if (
      ema[i] && sma[i] &&
      ema[i - 1] >= sma[i - 1] &&
      ema[i] < sma[i] &&
      rsi[i] < 60
    ) {
      markers.push({
        time: candles[i].time,
        position: 'aboveBar',
        color: '#E74C3C',
        shape: 'arrowDown',
        text: 'SELL'
      });
    }
  }
  return markers;
}

// -------------------- apply indicator data to series --------------------
function updateAllIndicatorSeries() {
  const indicators = calculateIndicators(allCandles);

  function buildSeries(arr) {
    if (!arr || !arr.length) return [];
    const out = [];
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      if (v === null || v === undefined) continue;
      out.push({ time: allCandles[i].time, value: v });
    }
    return out;
  }

  if (opts.sma20) sma20Series.setData(buildSeries(indicators.sma20)); else sma20Series.setData([]);
  if (opts.sma50) sma50Series.setData(buildSeries(indicators.sma50)); else sma50Series.setData([]);
  if (opts.ema12) ema12Series.setData(buildSeries(indicators.ema12)); else ema12Series.setData([]);

  if (opts.bb) {
    bbMiddleSeries.setData(buildSeries(indicators.bbMiddle));
    bbUpperSeries.setData(buildSeries(indicators.bbUpper));
    bbLowerSeries.setData(buildSeries(indicators.bbLower));
  } else {
    bbMiddleSeries.setData([]); bbUpperSeries.setData([]); bbLowerSeries.setData([]);
  }

  if (opts.pacePro) {
    const paceArr = indicators.paceNorm || [];
    const hist = [];
    for (let i = 0; i < paceArr.length; i++) {
      if (paceArr[i] === null || paceArr[i] === undefined) continue;
      const v = paceArr[i];
      const color = (v < 0) ? COLORS.paceNeg : COLORS.pacePos;
      hist.push({ time: allCandles[i].time, value: v, color });
    }
    paceProSeries.setData(hist);
  } else {
    paceProSeries.setData([]);
  }

  if (opts.rsi) {
    const rsiArr = indicators.rsi || [];
    const data = [];
    for (let i = 0; i < rsiArr.length; i++) {
      const v = rsiArr[i];
      if (v === null || v === undefined) continue;
      data.push({ time: allCandles[i].time, value: v });
    }
    rsiSeries.setData(data);
  } else {
    rsiSeries.setData([]);
  }

  if (opts.macd) {
    const macd = indicators.macdLine || [];
    const sig = indicators.macdSignal || [];
    const hist = indicators.macdHist || [];
    const macdData = [], sigData = [], histData = [];
    for (let i = 0; i < allCandles.length; i++) {
      if (macd[i] !== null && macd[i] !== undefined) macdData.push({ time: allCandles[i].time, value: macd[i] });
      if (sig[i] !== null && sig[i] !== undefined) sigData.push({ time: allCandles[i].time, value: sig[i] });
      if (hist[i] !== null && hist[i] !== undefined) histData.push({ time: allCandles[i].time, value: hist[i], color: (hist[i] >= 0 ? COLORS.macdHistPos : COLORS.macdHistNeg) });
    }
    macdSeries.setData(macdData);
    macdSignalSeries.setData(sigData);
    macdHistSeries.setData(histData);
  } else {
    macdSeries.setData([]); macdSignalSeries.setData([]); macdHistSeries.setData([]);
  }

  // ----- BUY / SELL ALERTS -----
  if (opts.signals) {
    const indicators = calculateIndicators(allCandles);
    const tradeSignals = calculateTradeSignals(allCandles, indicators);
    candleSeries.setMarkers(tradeSignals);
  } else {
    candleSeries.setMarkers([]);
  }


  // after updating, resize charts to ensure proper rendering
  scheduleResizeCharts();
}

// ------------------- loading data (uses your Flask endpoints) -------------------
async function loadLatest(limit = INITIAL_LIMIT) {
  if (loading) return;
  loading = true;
  try {
    const res = await fetch(`/data-latest?limit=${limit}`);
    const j = await res.json();
    allCandles = j.candles || [];
    loadedMin = j.min_time; loadedMax = j.max_time;
    candleSeries.setData(allCandles);
    updateAllIndicatorSeries();
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
      loadedMin = j.min_time; loadedMax = Math.max(loadedMax, j.max_time);
      candleSeries.setData(allCandles);
      updateAllIndicatorSeries();
      if (oldRange) mainChart.timeScale().setVisibleLogicalRange({ from: oldRange.from + newCount, to: oldRange.to + newCount });
      else mainChart.timeScale().fitContent();
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
      loadedMin = Math.min(loadedMin, j.min_time); loadedMax = j.max_time;
      candleSeries.setData(allCandles);
      updateAllIndicatorSeries();
      if (oldRange) mainChart.timeScale().setVisibleLogicalRange({ from: oldRange.from, to: oldRange.to });
      else mainChart.timeScale().fitContent();
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
    candleSeries.setData([]); updateAllIndicatorSeries(); alert("No data found for this range"); return;
  }
  loadedMin = j.min_time; loadedMax = j.max_time;
  candleSeries.setData(allCandles);
  updateAllIndicatorSeries();
  mainChart.timeScale().fitContent();
}

// ---------------- UI init, toggles, badges, hamburger, fullscreen ----------------
function setIndicatorBadges() {
  const set = (id, color) => {
    const el = document.getElementById(id);
    if (el) el.style.background = color;
  };
  set('badge-sma20', COLORS.sma20);
  set('badge-sma50', COLORS.sma50);
  set('badge-ema12', COLORS.ema12);
  set('badge-bb', COLORS.bb);
  set('badge-pace', COLORS.pacePos);
  set('badge-rsi', COLORS.rsi);
  set('badge-macd', COLORS.macd);
}

async function initUI() {
    // ---- header elements ----
  rangeBtn = document.getElementById('rangeBtn');
  replayBtn = document.getElementById('replayBtn');
  rangePanel = document.getElementById('rangePanel');
  replayPanel = document.getElementById('replayPanel');
  const rsiContainer = document.getElementById('chart-rsi');
  const macdContainer = document.getElementById('chart-macd');


  rangeBtn.onclick = () => {
    rangePanel.classList.toggle('hidden');
    replayPanel.classList.add('hidden');
  };

  replayBtn.onclick = () => {
    replayPanel.classList.toggle('hidden');
    rangePanel.classList.add('hidden');
  };

  const info = await fetch('/data-info').then(r => r.json());
  window.DATA_MIN_TS = info.min_ts;
  window.DATA_MAX_TS = info.max_ts;

  const startEl = document.getElementById('start');
  const endEl = document.getElementById('end');
  startEl.min = info.min_str; startEl.max = info.max_str;
  endEl.min = info.min_str; endEl.max = info.max_str;
  const maxDate = new Date(info.max_ts * 1000);
  const defaultStart = new Date((info.max_ts - 24 * 3600) * 1000);
  startEl.value = toLocalInput(defaultStart); endEl.value = toLocalInput(maxDate);

  // document.getElementById('btnLatest').onclick = () => loadLatest();
  const btnLatest = document.getElementById('btnLatest');
  if (btnLatest) {
    btnLatest.onclick = () => loadLatest();
  }

  document.getElementById('btnLoad').onclick = () => {
    loadRange(startEl.value, endEl.value);

    rangeBtn.textContent =
      `Range: ${startEl.value.replace('T',' ')} → ${endEl.value.replace('T',' ')}`;
    rangeBtn.classList.add('active');
    rangePanel.classList.add('hidden');
  };


  // toggles (wire checkboxes)
  document.getElementById('toggle-sma20').addEventListener('change', e => { opts.sma20 = e.target.checked; updateAllIndicatorSeries(); });
  document.getElementById('toggle-sma50').addEventListener('change', e => { opts.sma50 = e.target.checked; updateAllIndicatorSeries(); });
  document.getElementById('toggle-ema12').addEventListener('change', e => { opts.ema12 = e.target.checked; updateAllIndicatorSeries(); });
  document.getElementById('toggle-bb').addEventListener('change', e => { opts.bb = e.target.checked; updateAllIndicatorSeries(); });
  document.getElementById('paceproToggle').addEventListener('change', e => { opts.pacePro = e.target.checked; updateAllIndicatorSeries(); });
  document.getElementById('toggle-signals').addEventListener('change', e => { opts.signals = e.target.checked; updateAllIndicatorSeries(); });
  

    document.getElementById('toggle-rsi').addEventListener('change', e => {
      const range = getCurrentVisibleRange();
      opts.rsi = e.target.checked;

      if (opts.rsi) {
        rsiContainer.classList.remove('hidden');
        rsiContainer.style.height = INDICATOR_HEIGHT + 'px';
      } else {
        rsiContainer.classList.add('hidden');
      }

      scheduleResizeCharts();

      requestAnimationFrame(() => {
        if (opts.rsi) {
          rsiChart.resize(
            rsiContainer.clientWidth,
            rsiContainer.clientHeight
          );
        }
        restoreVisibleRange(range);
      });
    });


    document.getElementById('toggle-macd').addEventListener('change', e => {
      const range = getCurrentVisibleRange();
      opts.macd = e.target.checked;

      if (opts.macd) {
        macdContainer.classList.remove('hidden');
        macdContainer.style.height = INDICATOR_HEIGHT + 'px';
      } else {
        macdContainer.classList.add('hidden');
      }

      scheduleResizeCharts();

      requestAnimationFrame(() => {
        if (opts.macd) {
          macdChart.resize(
            macdContainer.clientWidth,
            macdContainer.clientHeight
          );
        }
        restoreVisibleRange(range);
      });
    });

    // ---- Replay controls ----
  const toggleReplay = document.getElementById('toggle-replay');
  if (toggleReplay) {
    toggleReplay.addEventListener('change', e => {
      replayMode = e.target.checked;
      if (!replayMode) stopReplay();
    });
  }


  document.getElementById('btnPlay').onclick = () => {
    if (!replayMode) replayMode = true;

    const ok = prepareReplayData();
    if (!ok) return;

    startReplay(Number(document.getElementById('replaySpeed').value) || 800);
    setReplayStatus('playing');
  };

  document.getElementById('btnStop').onclick = () => {
    stopReplay();
    setReplayStatus('stopped');
  };


  // fullscreen button (only main chart fullscreen)
  const fsBtn = document.getElementById('fullscreenBtn');
  fsBtn.addEventListener('click', () => {
    const isFs = document.body.classList.toggle('fullscreen-mode');
    // when entering fullscreen, hide sidebar and subcharts; when leaving restore
    if (!isFs) {
      // restore
    } else {
      // ensure rsi/macd hidden
    }
    // when toggling fullscreen, re-calc sizes
    scheduleResizeCharts();
  });
  // when window resizes, resize charts
  window.addEventListener('resize', () => scheduleResizeCharts());
}

function setReplayStatus(state) {
  const start = document.getElementById('replayStart').value.replace('T',' ');
  replayBtn.textContent = `Replay: ${start}`;
  replayBtn.classList.add('active');

  document.getElementById('btnPlay').disabled = state === 'playing';
  document.getElementById('btnStop').disabled = state !== 'playing';
}

function getCurrentVisibleRange() {
  try {
    return mainChart.timeScale().getVisibleLogicalRange();
  } catch {
    return null;
  }
}

function restoreVisibleRange(range) {
  if (!range) return;
  try {
    mainChart.timeScale().setVisibleLogicalRange(range);
  } catch {
    // ignore
  }
}

function updateLayout() {
  const area = document.getElementById('chart-area');
  const totalHeight = area.clientHeight;

  let usedHeight = 0;
  if (opts.rsi)  usedHeight += INDICATOR_HEIGHT;
  if (opts.macd) usedHeight += INDICATOR_HEIGHT;

  let mainHeight = totalHeight - usedHeight;
  if (mainHeight < MAIN_MIN_HEIGHT) {
    mainHeight = MAIN_MIN_HEIGHT;
  }

  // DOM heights
  chartMain.style.height = mainHeight + 'px';

  if (opts.rsi) {
    chartRSI.style.height = INDICATOR_HEIGHT + 'px';
  }

  if (opts.macd) {
    chartMACD.style.height = INDICATOR_HEIGHT + 'px';
  }

  // Lightweight Charts heights
  mainChart.applyOptions({ height: mainHeight });

  if (opts.rsi) {
    rsiChart.applyOptions({ height: INDICATOR_HEIGHT });
  }

  if (opts.macd) {
    macdChart.applyOptions({ height: INDICATOR_HEIGHT });
  }
}



function prepareReplayData() {
  const replayStartInput = document.getElementById('replayStart').value;
  if (!replayStartInput) {
    alert('Please select Replay Start time');
    return false;
  }

  const replayStartTs = new Date(replayStartInput).getTime() / 1000;

  // Split existing loaded candles
  const past = [];
  const future = [];

  for (const c of allCandles) {
    if (c.time < replayStartTs) past.push(c);
    else future.push(c);
  }

  if (!future.length) {
    alert('No candles available after replay start time');
    return false;
  }

  // Reset state
  stopReplay();
  replayIndex = 0;
  replayCandles = future;
  allCandles = past;

  candleSeries.setData(allCandles);
  updateAllIndicatorSeries();
  mainChart.timeScale().fitContent();

  return true;
}

// helper to format datetime-local
function toLocalInput(d) {
  const pad = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

// ---------------- chart resizing logic ----------------
let resizeTimer = null;
function scheduleResizeCharts() {
  if (resizeTimer) clearTimeout(resizeTimer);
  resizeTimer = setTimeout(resizeCharts, 120);
}

function resizeCharts() {
  // resize each chart to its container size
  try {
    const mainContainer = document.getElementById('chart-main');
    const rsiContainer = document.getElementById('chart-rsi');
    const macdContainer = document.getElementById('chart-macd');

    const wMain = mainContainer.clientWidth || mainContainer.offsetWidth;
    const hMain = mainContainer.clientHeight || mainContainer.offsetHeight;

    if (wMain > 0 && hMain > 0) {
      mainChart.resize(wMain, hMain);
    }

    if (getComputedStyle(rsiContainer).display !== 'none') {
      rsiChart.resize(rsiContainer.clientWidth, rsiContainer.clientHeight);
    }
    if (getComputedStyle(macdContainer).display !== 'none') {
      macdChart.resize(macdContainer.clientWidth, macdContainer.clientHeight);
    }
  } catch (e) {
    // ignore
  }
}

function startReplay(speed) {
  if (replayTimer) return;

  replayTimer = setInterval(() => {
    if (replayIndex >= replayCandles.length) {
      stopReplay();
      return;
    }

    allCandles.push(replayCandles[replayIndex]);
    candleSeries.setData(allCandles);
    updateAllIndicatorSeries();

    replayIndex++;
  }, speed);
}

function stopReplay() {
  if (replayTimer) {
    clearInterval(replayTimer);
    replayTimer = null;
  }
}

// ------------------ boot ------------------
(async function init() {
  createCharts();
  await initUI();
  await loadLatest();

  // force correct size after header layout
  setTimeout(resizeCharts, 0);
})();
