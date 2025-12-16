// static/chart.js - Part 1 of 2
// Trading chart with indicators, replay mode, and timeframe switching

const INITIAL_LIMIT = 300;
const INDICATOR_HEIGHT = 160;
const MAIN_MIN_HEIGHT = 200;

// Chart instances & series
let mainChart, rsiChart, macdChart;
let candleSeries, sma20Series, sma50Series, ema12Series;
let bbUpperSeries, bbMiddleSeries, bbLowerSeries, paceProSeries;
let rsiSeries, macdSeries, macdSignalSeries, macdHistSeries;

// Data storage
let allCandles = [], raw1mData = [], currentTF = 1;
let loadedMin = null, loadedMax = null, loading = false;

// Range & Replay state
let rangeSelected = false, rangeStartTs = null, rangeEndTs = null;
let replayState = 'IDLE', replayTimer = null, replayIndex = 0, replayCandles = [], replaySpeed = 800;

// UI elements
let rangeBtn, replayBtn, rangePanel, replayPanel;

// Indicator toggles
const opts = { sma20: true, sma50: true, ema12: true, bb: true, pacePro: false, rsi: true, macd: true, signals: false };

// Colors
const COLORS = {
  sma20: '#1E88E5', sma50: '#6A1B9A', ema12: '#00ACC1', bb: '#9E9E9E',
  pacePos: '#1565C0', paceNeg: '#FF9800', rsi: '#F57C00', macd: '#1565C0',
  macdSignal: '#D32F2F', macdHistPos: '#8E24AA', macdHistNeg: '#FFB300'
};

// ==================== CHART CREATION ====================
function createCharts() {
  const chartOpts = {
    layout: { background: { color: '#ffffff' }, textColor: '#111' },
    grid: { vertLines: { color: '#f0f0f0' }, horzLines: { color: '#f7f7f7' } },
    rightPriceScale: { borderVisible: true },
    timeScale: { timeVisible: true, secondsVisible: false, rightOffset: 12, barSpacing: 20, minBarSpacing: 6 },
    handleScroll: { mouseWheel: true, pressedMouseMove: true },
    handleScale: { mouseWheel: true, pinch: true }
  };

  mainChart = LightweightCharts.createChart(document.getElementById('chart-main'), chartOpts);
  candleSeries = mainChart.addCandlestickSeries({
    upColor: '#26a69a', downColor: '#ef5350', wickUpColor: '#26a69a', wickDownColor: '#ef5350', borderVisible: false
  });

  sma20Series = mainChart.addLineSeries({ color: COLORS.sma20, lineWidth: 2, lastValueVisible: false });
  sma50Series = mainChart.addLineSeries({ color: COLORS.sma50, lineWidth: 2, lastValueVisible: false });
  ema12Series = mainChart.addLineSeries({ color: COLORS.ema12, lineWidth: 2, lastValueVisible: false });
  bbMiddleSeries = mainChart.addLineSeries({ color: COLORS.bb, lineWidth: 1.2, lastValueVisible: false });
  bbUpperSeries = mainChart.addLineSeries({ color: COLORS.bb, lineWidth: 1, lastValueVisible: false });
  bbLowerSeries = mainChart.addLineSeries({ color: COLORS.bb, lineWidth: 1, lastValueVisible: false });
  paceProSeries = mainChart.addHistogramSeries({
    color: COLORS.pacePos, base: 0, priceFormat: { type: 'volume' }, priceScaleId: '', scaleMargins: { top: 0.82, bottom: 0 }
  });

  rsiChart = LightweightCharts.createChart(document.getElementById('chart-rsi'), {
    layout: { background: { color: '#fff' }, textColor: '#111' },
    rightPriceScale: { scaleMargins: { top: 0.2, bottom: 0.2 } },
    timeScale: { timeVisible: true, secondsVisible: false, rightOffset: 12, barSpacing: 20 }
  });
  rsiSeries = rsiChart.addLineSeries({ color: COLORS.rsi, lineWidth: 2 });

  macdChart = LightweightCharts.createChart(document.getElementById('chart-macd'), {
    layout: { background: { color: '#fff' }, textColor: '#111' },
    rightPriceScale: { scaleMargins: { top: 0.2, bottom: 0.2 } },
    timeScale: { timeVisible: true, secondsVisible: false, rightOffset: 12, barSpacing: 20 }
  });
  macdHistSeries = macdChart.addHistogramSeries({ color: COLORS.macdHistPos, base: 0, priceFormat: { type: 'volume' } });
  macdSeries = macdChart.addLineSeries({ color: COLORS.macd, lineWidth: 2 });
  macdSignalSeries = macdChart.addLineSeries({ color: COLORS.macdSignal, lineWidth: 1.5 });

  syncTimeScales();
  setupLazyLoading();
  setIndicatorBadges();
}

function syncTimeScales() {
  const ts = [mainChart.timeScale(), rsiChart.timeScale(), macdChart.timeScale()];
  ts.forEach((t, i) => {
    t.subscribeVisibleLogicalRangeChange(r => {
      if (!r) return;
      ts.forEach((ot, j) => { if (i !== j) ot.setVisibleLogicalRange(r); });
    });
  });
}

function setupLazyLoading() {
  mainChart.timeScale().subscribeVisibleLogicalRangeChange(range => {
    if (!range || !allCandles.length || replayState === 'PLAYING') return;
    const from = Math.floor(range.from), to = Math.ceil(range.to), THRESH = 6;
    if (from <= THRESH && !loading && loadedMin > window.DATA_MIN_TS) loadBefore();
    if (to >= (allCandles.length - THRESH) && !loading && loadedMax < window.DATA_MAX_TS) loadAfter();
  });
}

// ==================== INDICATORS ====================
function sma(values, period) {
  const out = []; let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
    if (i >= period) sum -= values[i - period];
    out.push(i >= period - 1 ? sum / period : null);
  }
  return out;
}

function ema(values, period) {
  const out = [], k = 2 / (period + 1); let prev = null;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (prev === null) {
      if (i === period - 1) { prev = values.slice(0, period).reduce((a, b) => a + b, 0) / period; out.push(prev); }
      else out.push(null);
    } else { prev = v * k + prev * (1 - k); out.push(prev); }
  }
  return out;
}

function stddev(values, period) {
  const out = [];
  for (let i = 0; i < values.length; i++) {
    if (i < period - 1) { out.push(null); continue; }
    const w = values.slice(i - period + 1, i + 1), m = w.reduce((a, b) => a + b, 0) / period;
    out.push(Math.sqrt(w.reduce((s, x) => s + Math.pow(x - m, 2), 0) / period));
  }
  return out;
}

function emaFilled(series, period) {
  const out = [], k = 2 / (period + 1); let prev = null;
  for (let i = 0; i < series.length; i++) {
    const v = series[i];
    if (v === null || v === undefined) { out.push(null); continue; }
    if (prev === null) {
      const w = []; for (let j = i; j > i - period && j >= 0; j--) if (series[j] != null) w.unshift(series[j]);
      if (w.length < period) { out.push(null); continue; }
      prev = w.reduce((a, b) => a + b, 0) / w.length; out.push(prev);
    } else { prev = v * k + prev * (1 - k); out.push(prev); }
  }
  return out;
}

function calculateRSI(closes, period = 14) {
  const out = []; let gains = 0, losses = 0;
  for (let i = 0; i < closes.length; i++) {
    if (i === 0) { out.push(null); continue; }
    const change = closes[i] - closes[i - 1], gain = Math.max(change, 0), loss = Math.max(-change, 0);
    if (i <= period) {
      gains += gain; losses += loss;
      if (i === period) {
        const avgG = gains / period, avgL = losses / period, rs = avgL === 0 ? 100 : avgG / avgL;
        out.push(100 - (100 / (1 + rs)));
      } else out.push(null);
    } else {
      const prevG = out.avgGain ?? (gains / period), prevL = out.avgLoss ?? (losses / period);
      const avgG = (prevG * (period - 1) + gain) / period, avgL = (prevL * (period - 1) + loss) / period;
      out.avgGain = avgG; out.avgLoss = avgL;
      const rs = avgL === 0 ? 100 : avgG / avgL;
      out.push(100 - (100 / (1 + rs)));
    }
  }
  return out;
}

function calculateIndicators(candles) {
  if (!candles.length) return {};
  const closes = candles.map(c => c.close);
  const sma20Val = sma(closes, 20), sma50Val = sma(closes, 50);
  const ema12Val = ema(closes, 12), ema26 = ema(closes, 26);
  const bbStd = stddev(closes, 20);
  const bbUpper = sma20Val.map((m, i) => (m == null || bbStd[i] == null) ? null : m + 2 * bbStd[i]);
  const bbLower = sma20Val.map((m, i) => (m == null || bbStd[i] == null) ? null : m - 2 * bbStd[i]);
  const paceRaw = candles.map(c => ((c.close - c.open) / c.open) * 100);
  const absMax = Math.max(...paceRaw.map(v => Math.abs(v))) || 1;
  const normalized = paceRaw.map(v => (v / absMax) * 100);
  const paceNorm = normalized.map((_, i) => {
    let sum = 0, cnt = 0;
    for (let j = i - 4; j <= i; j++) if (j >= 0) { sum += normalized[j]; cnt++; }
    return cnt ? sum / cnt : 0;
  });
  const rsiVal = calculateRSI(closes, 14);
  const macdLine = ema12Val.map((e12, i) => (e12 == null || ema26[i] == null) ? null : e12 - ema26[i]);
  const macdSignal = emaFilled(macdLine, 9);
  const macdHist = macdLine.map((v, i) => (v == null || macdSignal[i] == null) ? null : v - macdSignal[i]);
  return { sma20: sma20Val, sma50: sma50Val, ema12: ema12Val, bbUpper, bbMiddle: sma20Val, bbLower, paceNorm, rsi: rsiVal, macdLine, macdSignal, macdHist };
}

function calculateTradeSignals(candles, indicators) {
  const markers = [];
  for (let i = 1; i < candles.length; i++) {
    const { ema12: ema, sma20: sma, rsi } = indicators;
    if (ema[i] && sma[i] && ema[i - 1] <= sma[i - 1] && ema[i] > sma[i] && rsi[i] > 40)
      markers.push({ time: candles[i].time, position: 'belowBar', color: '#2ECC71', shape: 'arrowUp', text: 'BUY' });
    if (ema[i] && sma[i] && ema[i - 1] >= sma[i - 1] && ema[i] < sma[i] && rsi[i] < 60)
      markers.push({ time: candles[i].time, position: 'aboveBar', color: '#E74C3C', shape: 'arrowDown', text: 'SELL' });
  }
  return markers;
}

// SEE PART 2 IN NEXT MESSAGE FOR REMAINING FUNCTIONS
// static/chart.js - Part 2 of 2
// APPEND THIS TO PART 1

// ==================== UPDATE SERIES ====================
function updateAllIndicatorSeries() {
  const indicators = calculateIndicators(allCandles);
  const buildSeries = arr => {
    if (!arr?.length) return [];
    return arr.map((v, i) => v != null ? { time: allCandles[i].time, value: v } : null).filter(Boolean);
  };

  sma20Series.setData(opts.sma20 ? buildSeries(indicators.sma20) : []);
  sma50Series.setData(opts.sma50 ? buildSeries(indicators.sma50) : []);
  ema12Series.setData(opts.ema12 ? buildSeries(indicators.ema12) : []);

  if (opts.bb) {
    bbMiddleSeries.setData(buildSeries(indicators.bbMiddle));
    bbUpperSeries.setData(buildSeries(indicators.bbUpper));
    bbLowerSeries.setData(buildSeries(indicators.bbLower));
  } else { bbMiddleSeries.setData([]); bbUpperSeries.setData([]); bbLowerSeries.setData([]); }

  if (opts.pacePro) {
    const hist = (indicators.paceNorm || []).map((v, i) => 
      v != null ? { time: allCandles[i].time, value: v, color: v < 0 ? COLORS.paceNeg : COLORS.pacePos } : null
    ).filter(Boolean);
    paceProSeries.setData(hist);
  } else paceProSeries.setData([]);

  rsiSeries.setData(opts.rsi ? buildSeries(indicators.rsi) : []);

  if (opts.macd) {
    macdSeries.setData(buildSeries(indicators.macdLine));
    macdSignalSeries.setData(buildSeries(indicators.macdSignal));
    const hist = (indicators.macdHist || []).map((v, i) =>
      v != null ? { time: allCandles[i].time, value: v, color: v >= 0 ? COLORS.macdHistPos : COLORS.macdHistNeg } : null
    ).filter(Boolean);
    macdHistSeries.setData(hist);
  } else { macdSeries.setData([]); macdSignalSeries.setData([]); macdHistSeries.setData([]); }

  candleSeries.setMarkers(opts.signals ? calculateTradeSignals(allCandles, indicators) : []);
  scheduleResizeCharts();
}

// ==================== DATA LOADING ====================
async function loadLatest(limit = INITIAL_LIMIT) {
  if (loading) return;
  loading = true;
  try {
    const res = await fetch(`/data-latest?limit=${limit}`);
    const j = await res.json();
    raw1mData = j.candles || [];
    allCandles = resampleCandles(raw1mData, currentTF);
    loadedMin = j.min_time; loadedMax = j.max_time;
    candleSeries.setData(allCandles);
    updateAllIndicatorSeries();
    mainChart.timeScale().fitContent();
    rangeSelected = false; rangeStartTs = null; rangeEndTs = null;
    if (replayBtn) replayBtn.disabled = true;
    if (rangeBtn) { rangeBtn.textContent = 'Range ▾'; rangeBtn.classList.remove('active'); }
  } finally { loading = false; }
}

async function loadBefore(limit = INITIAL_LIMIT) {
  if (loading || !loadedMin || loadedMin <= window.DATA_MIN_TS) return;
  const oldRange = mainChart.timeScale().getVisibleLogicalRange();
  loading = true;
  try {
    const res = await fetch(`/data-before?time=${loadedMin}&limit=${limit}`);
    const j = await res.json();
    if (j.candles?.length) {
      const newCnt = j.candles.length;
      raw1mData = [...j.candles, ...raw1mData];
      allCandles = resampleCandles(raw1mData, currentTF);
      loadedMin = j.min_time; loadedMax = Math.max(loadedMax, j.max_time);
      candleSeries.setData(allCandles);
      updateAllIndicatorSeries();
      if (oldRange) mainChart.timeScale().setVisibleLogicalRange({ from: oldRange.from + newCnt, to: oldRange.to + newCnt });
      else mainChart.timeScale().fitContent();
    }
  } finally { loading = false; }
}

async function loadAfter(limit = INITIAL_LIMIT) {
  if (loading || !loadedMax || loadedMax >= window.DATA_MAX_TS) return;
  const oldRange = mainChart.timeScale().getVisibleLogicalRange();
  loading = true;
  try {
    const res = await fetch(`/data-after?time=${loadedMax}&limit=${limit}`);
    const j = await res.json();
    if (j.candles?.length) {
      raw1mData = [...raw1mData, ...j.candles];
      allCandles = resampleCandles(raw1mData, currentTF);
      loadedMin = Math.min(loadedMin, j.min_time); loadedMax = j.max_time;
      candleSeries.setData(allCandles);
      updateAllIndicatorSeries();
      if (oldRange) mainChart.timeScale().setVisibleLogicalRange(oldRange);
      else mainChart.timeScale().fitContent();
    }
  } finally { loading = false; }
}

async function loadRange(startIso, endIso) {
  if (!startIso) return alert("Please select a start datetime");
  const res = await fetch(`/data-range?${new URLSearchParams({ start: startIso, end: endIso })}`);
  const j = await res.json();
  raw1mData = j.candles || [];
  allCandles = resampleCandles(raw1mData, currentTF);
  if (!allCandles.length) { candleSeries.setData([]); updateAllIndicatorSeries(); return alert("No data found for this range"); }
  loadedMin = j.min_time; loadedMax = j.max_time;
  candleSeries.setData(allCandles);
  updateAllIndicatorSeries();
  mainChart.timeScale().fitContent();
  rangeSelected = true; rangeStartTs = loadedMin; rangeEndTs = loadedMax;
  if (replayBtn) replayBtn.disabled = false;
}

// ==================== TIMEFRAME ====================
function resampleCandles(data, tfMinutes) {
  if (tfMinutes === 1) return data.slice();
  const tfSec = tfMinutes * 60, result = [];
  let bucket = null;
  for (const c of data) {
    const bucketTime = Math.floor(c.time / tfSec) * tfSec;
    if (!bucket || bucket.time !== bucketTime) {
      if (bucket) result.push(bucket);
      bucket = { time: bucketTime, open: c.open, high: c.high, low: c.low, close: c.close, volume: c.volume || 0 };
    } else {
      bucket.high = Math.max(bucket.high, c.high);
      bucket.low = Math.min(bucket.low, c.low);
      bucket.close = c.close;
      bucket.volume += c.volume || 0;
    }
  }
  if (bucket) result.push(bucket);
  return result;
}

function applyTimeframe(tf) {
  if (!raw1mData.length) return;
  const prevRange = mainChart.timeScale().getVisibleLogicalRange();
  currentTF = tf;
  allCandles = resampleCandles(raw1mData, tf);
  candleSeries.setData(allCandles);
  updateAllIndicatorSeries();
  if (prevRange) requestAnimationFrame(() => {
    try { mainChart.timeScale().setVisibleLogicalRange(prevRange); }
    catch { mainChart.timeScale().fitContent(); }
  });
  else mainChart.timeScale().fitContent();
  const tfBtn = document.getElementById('tfBtn');
  if (tfBtn) tfBtn.textContent = `${tf}m ▾`;
}

// ==================== REPLAY ====================
function prepareReplayData() {
  const input = document.getElementById('replayStart');
  if (!input?.value) return alert('Please select Replay Start time'), false;
  const replayStartTs = new Date(input.value).getTime() / 1000;
  if (replayStartTs < rangeStartTs || replayStartTs > rangeEndTs) 
    return alert('Replay start time must be within the loaded range'), false;
  const tfData = resampleCandles(raw1mData, currentTF);
  const past = [], future = [];
  tfData.forEach(c => (c.time < replayStartTs ? past : future).push(c));
  if (!future.length) return alert('No candles available after replay start time'), false;
  stopReplay();
  replayIndex = 0; allCandles = past; replayCandles = future;
  candleSeries.setData(allCandles);
  updateAllIndicatorSeries();
  mainChart.timeScale().fitContent();
  return true;
}

function startReplay(speed) {
  if (replayState === 'PLAYING') return;
  replaySpeed = speed || 800;
  replayState = 'PLAYING';
  replayTimer = setInterval(() => {
    if (replayIndex >= replayCandles.length) return stopReplay();
    const candle = replayCandles[replayIndex];
    allCandles.push(candle);
    candleSeries.update(candle);
    updateAllIndicatorSeries();
    replayIndex++;
    mainChart.timeScale().scrollToRealTime();
  }, replaySpeed);
  updateReplayUI('playing');
}

function pauseReplay() {
  if (replayState !== 'PLAYING') return;
  clearInterval(replayTimer);
  replayTimer = null;
  replayState = 'PAUSED';
  updateReplayUI('paused');
}

function stopReplay() {
  clearInterval(replayTimer);
  replayTimer = null;
  replayState = 'IDLE';
  replayIndex = 0;
  replayCandles = [];
  allCandles = resampleCandles(raw1mData, currentTF);
  candleSeries.setData(allCandles);
  updateAllIndicatorSeries();
  mainChart.timeScale().fitContent();
  updateReplayUI('stopped');
}

function updateReplayUI(state) {
  const input = document.getElementById('replayStart');
  if (input?.value && replayBtn) {
    replayBtn.textContent = `Replay: ${input.value.replace('T', ' ')}`;
    replayBtn.classList.add('active');
  }
  const playBtn = document.getElementById('btnPlay');
  const stopBtn = document.getElementById('btnStop');
  if (playBtn) playBtn.disabled = state === 'playing';
  if (stopBtn) stopBtn.disabled = state !== 'playing';
}

// ==================== UI INITIALIZATION ====================
function setIndicatorBadges() {
  const set = (id, color) => { const el = document.getElementById(id); if (el) el.style.background = color; };
  set('badge-sma20', COLORS.sma20); set('badge-sma50', COLORS.sma50);
  set('badge-ema12', COLORS.ema12); set('badge-bb', COLORS.bb);
  set('badge-pace', COLORS.pacePos); set('badge-rsi', COLORS.rsi);
  set('badge-macd', COLORS.macd);
}

async function initUI() {
  rangeBtn = document.getElementById('rangeBtn');
  replayBtn = document.getElementById('replayBtn');
  rangePanel = document.getElementById('rangePanel');
  replayPanel = document.getElementById('replayPanel');
  const rsiContainer = document.getElementById('chart-rsi');
  const macdContainer = document.getElementById('chart-macd');

  if (rangeBtn) rangeBtn.onclick = () => {
    rangePanel?.classList.toggle('hidden');
    replayPanel?.classList.add('hidden');
  };

  if (replayBtn) replayBtn.onclick = () => {
    replayPanel?.classList.toggle('hidden');
    rangePanel?.classList.add('hidden');
  };

  const info = await fetch('/data-info').then(r => r.json());
  window.DATA_MIN_TS = info.min_ts;
  window.DATA_MAX_TS = info.max_ts;

  const startEl = document.getElementById('start');
  const endEl = document.getElementById('end');
  if (startEl && endEl) {
    startEl.min = info.min_str; startEl.max = info.max_str;
    endEl.min = info.min_str; endEl.max = info.max_str;
    const maxDate = new Date(info.max_ts * 1000);
    const defaultStart = new Date((info.max_ts - 24 * 3600) * 1000);
    const pad = n => String(n).padStart(2, '0');
    const toLocal = d => `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
    startEl.value = toLocal(defaultStart); endEl.value = toLocal(maxDate);
  }

  const btnLatest = document.getElementById('btnLatest');
  if (btnLatest) btnLatest.onclick = () => loadLatest();

  const btnLoad = document.getElementById('btnLoad');
  if (btnLoad && startEl && endEl && rangeBtn && rangePanel) {
    btnLoad.onclick = () => {
      loadRange(startEl.value, endEl.value);
      rangeBtn.textContent = `Range: ${startEl.value.replace('T',' ')} → ${endEl.value.replace('T',' ')}`;
      rangeBtn.classList.add('active');
      rangePanel.classList.add('hidden');
    };
  }

  // Indicator toggles
  const addToggle = (id, key, callback) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('change', e => { opts[key] = e.target.checked; if (callback) callback(e); else updateAllIndicatorSeries(); });
  };

  addToggle('toggle-sma20', 'sma20');
  addToggle('toggle-sma50', 'sma50');
  addToggle('toggle-ema12', 'ema12');
  addToggle('toggle-bb', 'bb');
  addToggle('paceproToggle', 'pacePro');
  addToggle('toggle-signals', 'signals');

  addToggle('toggle-rsi', 'rsi', e => {
    const range = mainChart.timeScale().getVisibleLogicalRange();
    if (e.target.checked) {
      rsiContainer.classList.remove('hidden');
      rsiContainer.style.height = INDICATOR_HEIGHT + 'px';
    } else rsiContainer.classList.add('hidden');
    scheduleResizeCharts();
    requestAnimationFrame(() => {
      if (e.target.checked) rsiChart.resize(rsiContainer.clientWidth, rsiContainer.clientHeight);
      if (range) try { mainChart.timeScale().setVisibleLogicalRange(range); } catch {}
    });
  });

  addToggle('toggle-macd', 'macd', e => {
    const range = mainChart.timeScale().getVisibleLogicalRange();
    if (e.target.checked) {
      macdContainer.classList.remove('hidden');
      macdContainer.style.height = INDICATOR_HEIGHT + 'px';
    } else macdContainer.classList.add('hidden');
    scheduleResizeCharts();
    requestAnimationFrame(() => {
      if (e.target.checked) macdChart.resize(macdContainer.clientWidth, macdContainer.clientHeight);
      if (range) try { mainChart.timeScale().setVisibleLogicalRange(range); } catch {}
    });
  });

  // Timeframe buttons
  document.querySelectorAll('.tf-option').forEach(btn => {
    btn.addEventListener('click', () => applyTimeframe(parseInt(btn.dataset.tf, 10)));
  });

  // Replay controls
  const btnPlay = document.getElementById('btnPlay');
  if (btnPlay) btnPlay.onclick = () => {
    if (prepareReplayData()) startReplay(Number(document.getElementById('replaySpeed')?.value) || 800);
  };

  const btnStop = document.getElementById('btnStop');
  if (btnStop) btnStop.onclick = stopReplay;

  // Fullscreen
  const fsBtn = document.getElementById('fullscreenBtn');
  if (fsBtn) fsBtn.addEventListener('click', () => {
    document.body.classList.toggle('fullscreen-mode');
    scheduleResizeCharts();
  });

  window.addEventListener('resize', scheduleResizeCharts);
}

// ==================== CHART RESIZING ====================
let resizeTimer = null;
function scheduleResizeCharts() {
  if (resizeTimer) clearTimeout(resizeTimer);
  resizeTimer = setTimeout(resizeCharts, 120);
}

function resizeCharts() {
  try {
    const mainContainer = document.getElementById('chart-main');
    const rsiContainer = document.getElementById('chart-rsi');
    const macdContainer = document.getElementById('chart-macd');

    if (mainContainer) {
      const w = mainContainer.clientWidth || mainContainer.offsetWidth;
      const h = mainContainer.clientHeight || mainContainer.offsetHeight;
      if (w > 0 && h > 0) mainChart.resize(w, h);
    }

    if (rsiContainer && getComputedStyle(rsiContainer).display !== 'none') {
      rsiChart.resize(rsiContainer.clientWidth, rsiContainer.clientHeight);
    }

    if (macdContainer && getComputedStyle(macdContainer).display !== 'none') {
      macdChart.resize(macdContainer.clientWidth, macdContainer.clientHeight);
    }
  } catch (e) { console.error('Resize error:', e); }
}

// ==================== BOOT ====================
(async function init() {
  createCharts();
  await initUI();
  await loadLatest();
  setTimeout(resizeCharts, 0);
})();