// static/chart.js
// Advanced candlestick chart with technical indicators, replay, and range selection
// Uses endpoints: /data-info, /data-latest, /data-before, /data-after, /data-range

const INITIAL_LIMIT = 300;
const INDICATOR_HEIGHT = 160;
const MAIN_MIN_HEIGHT = 200;
const MARKET_OPEN_HOUR = 9;
const MARKET_OPEN_MIN = 15;
const MARKET_CLOSE_HOUR = 15;
const MARKET_CLOSE_MIN = 30;

// Chart instances & series
let mainChart, rsiChart, macdChart;
let candleSeries;
let sma20Series, sma50Series, ema12Series;
let bbUpperSeries, bbMiddleSeries, bbLowerSeries;
let paceProSeries;
let rsiSeries;
let macdSeries, macdSignalSeries, macdHistSeries;

// Data state
let allCandles = [];          // current timeframe candles (ascending by time)
let raw1mData = [];           // original 1-minute candles
let currentTF = 1;            // current timeframe in minutes
let loadedMin = null;
let loadedMax = null;
let loading = false;

// Replay state
let replayMode = false;
let replayTimer = null;
let replayIndex = 0;
let replayCandles = [];
let replayState = 'IDLE';     // IDLE | PREPARING | PLAYING | PAUSED
let replayData = [];
let rangeSelected = false;
let rangeStartTs = null;
let rangeEndTs = null;
let replayStartInputEl = null;
let btnPlay = null;
let btnPause = null;
let replaySpeed = 800;
let replayTimeIndicator = null;
let cachedIndicators = null;  // Cache for incremental updates

// UI elements
let rangeBtn, replayBtn;
let rangePanel, replayPanel;

// Indicator colors
const COLORS = {
  sma20: '#1E88E5',
  sma50: '#6A1B9A',
  ema12: '#00ACC1',
  bb: '#9E9E9E',
  pacePos: '#1565C0',
  paceNeg: '#FF9800',
  rsi: '#F57C00',
  macd: '#1565C0',
  macdSignal: '#D32F2F',
  macdHistPos: '#8E24AA',
  macdHistNeg: '#FFB300'
};

// Indicator toggles
const opts = {
  sma20: true,
  sma50: true,
  ema12: true,
  bb: true,
  pacePro: false,
  rsi: true,
  macd: true,
  signals: false
};

// Chart resizing
let resizeTimer = null;

// ========== CHART CREATION ==========
function createCharts() {
  // Main chart
  mainChart = LightweightCharts.createChart(document.getElementById('chart-main'), {
    layout: { background: { color: '#ffffff' }, textColor: '#111' },
    grid: { vertLines: { color: '#f0f0f0' }, horzLines: { color: '#f7f7f7' } },
    rightPriceScale: { borderVisible: true },
    timeScale: { 
      timeVisible: true, 
      secondsVisible: false, 
      rightOffset: 12, 
      barSpacing: 20, 
      minBarSpacing: 6 
    },
    handleScroll: { mouseWheel: true, pressedMouseMove: true },
    handleScale: { mouseWheel: true, pinch: true }
  });

  candleSeries = mainChart.addCandlestickSeries({
    upColor: '#26a69a',
    downColor: '#ef5350',
    wickUpColor: '#26a69a',
    wickDownColor: '#ef5350',
    borderVisible: false
  });

  // Overlay series
  sma20Series = mainChart.addLineSeries({ 
    color: COLORS.sma20, 
    lineWidth: 2, 
    lastValueVisible: false 
  });
  sma50Series = mainChart.addLineSeries({ 
    color: COLORS.sma50, 
    lineWidth: 2, 
    lastValueVisible: false 
  });
  ema12Series = mainChart.addLineSeries({ 
    color: COLORS.ema12, 
    lineWidth: 2, 
    lastValueVisible: false 
  });

  bbMiddleSeries = mainChart.addLineSeries({ 
    color: COLORS.bb, 
    lineWidth: 1.2, 
    lastValueVisible: false 
  });
  bbUpperSeries = mainChart.addLineSeries({ 
    color: COLORS.bb, 
    lineWidth: 1, 
    lastValueVisible: false 
  });
  bbLowerSeries = mainChart.addLineSeries({ 
    color: COLORS.bb, 
    lineWidth: 1, 
    lastValueVisible: false 
  });

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
    timeScale: { 
      timeVisible: true, 
      secondsVisible: false, 
      rightOffset: 12, 
      barSpacing: 20 
    }
  });
  rsiSeries = rsiChart.addLineSeries({ color: COLORS.rsi, lineWidth: 2 });

  // MACD chart
  macdChart = LightweightCharts.createChart(document.getElementById('chart-macd'), {
    layout: { background: { color: '#fff' }, textColor: '#111' },
    rightPriceScale: { scaleMargins: { top: 0.2, bottom: 0.2 } },
    timeScale: { 
      timeVisible: true, 
      secondsVisible: false, 
      rightOffset: 12, 
      barSpacing: 20 
    }
  });
  macdHistSeries = macdChart.addHistogramSeries({ 
    color: COLORS.macdHistPos, 
    base: 0, 
    priceFormat: { type: 'volume' } 
  });
  macdSeries = macdChart.addLineSeries({ color: COLORS.macd, lineWidth: 2 });
  macdSignalSeries = macdChart.addLineSeries({ 
    color: COLORS.macdSignal, 
    lineWidth: 1.5 
  });

  // Sync time scales
  syncTimeScales();

  // Lazy-load detection on main chart (disabled during replay)
  mainChart.timeScale().subscribeVisibleLogicalRangeChange(range => {
    // Don't lazy load during replay - only when replay is IDLE
    // PREPARING state means we're setting up replay, so don't load either
    if (replayState !== 'IDLE') return;
    
    if (!range || !allCandles.length) return;
    const from = Math.floor(range.from);
    const to = Math.ceil(range.to);
    const THRESH = 6;
    if (from <= THRESH && !loading && loadedMin > window.DATA_MIN_TS) {
      loadBefore();
    }
    if (to >= (allCandles.length - THRESH) && !loading && loadedMax < window.DATA_MAX_TS) {
      loadAfter();
    }
  });

  // Set badge colors in sidebar
  setIndicatorBadges();

  // Prevent browser navigation gestures from interfering with chart scrolling
  preventBrowserNavigationGestures();
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

// ========== INDICATOR CALCULATIONS ==========
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
    if (i < period - 1) {
      out.push(null);
      continue;
    }
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
    if (v === null || v === undefined) {
      out.push(null);
      continue;
    }
    if (prev === null) {
      const window = [];
      for (let j = i; j > i - period && j >= 0; j--) {
        if (series[j] !== null && series[j] !== undefined) {
          window.unshift(series[j]);
        }
      }
      if (window.length < period) {
        out.push(null);
        continue;
      }
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
  let avgGain = 0;
  let avgLoss = 0;
  
  for (let i = 0; i < closes.length; i++) {
    if (i === 0) {
      out.push(null);
      continue;
    }
    
    const change = closes[i] - closes[i - 1];
    const gain = Math.max(change, 0);
    const loss = Math.max(-change, 0);
    
    if (i <= period) {
      avgGain += gain;
      avgLoss += loss;
      if (i === period) {
        avgGain /= period;
        avgLoss /= period;
        const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
        out.push(100 - (100 / (1 + rs)));
    } else {
        out.push(null);
      }
    } else {
      avgGain = (avgGain * (period - 1) + gain) / period;
      avgLoss = (avgLoss * (period - 1) + loss) / period;
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
      if (j >= 0) {
        sum += normalized[j];
        count++;
      }
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
  const bbUpper = bbMid.map((m, i) => 
    (m === null || bbStd[i] === null) ? null : m + 2 * bbStd[i]
  );
  const bbLower = bbMid.map((m, i) => 
    (m === null || bbStd[i] === null) ? null : m - 2 * bbStd[i]
  );

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
    const e12 = ema12[i];
    const e26 = ema26[i];
    macdLine.push((e12 === null || e26 === null) ? null : e12 - e26);
  }
  const macdSignal = emaFilled(macdLine, 9);
  const macdHist = macdLine.map((v, i) => 
    (v === null || macdSignal[i] === null) ? null : v - macdSignal[i]
  );

  return { 
    sma20, sma50, ema12, 
    bbUpper, bbMiddle: bbMid, bbLower, 
    paceNorm, rsi, 
    macdLine, macdSignal, macdHist 
  };
}

function calculateTradeSignals(candles, indicators) {
  const markers = [];

  for (let i = 1; i < candles.length; i++) {
    const ema = indicators.ema12;
    const sma = indicators.sma20;
    const rsi = indicators.rsi;

    // BUY signal: EMA crosses above SMA and RSI > 40
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

    // SELL signal: EMA crosses below SMA and RSI < 60
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

// ========== UPDATE INDICATOR SERIES ==========
function updateAllIndicatorSeries() {
  const indicators = calculateIndicators(allCandles);
  cachedIndicators = indicators;  // Cache for incremental updates

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

  // Overlay indicators
  if (opts.sma20) {
    sma20Series.setData(buildSeries(indicators.sma20));
  } else {
    sma20Series.setData([]);
  }
  
  if (opts.sma50) {
    sma50Series.setData(buildSeries(indicators.sma50));
  } else {
    sma50Series.setData([]);
  }
  
  if (opts.ema12) {
    ema12Series.setData(buildSeries(indicators.ema12));
  } else {
    ema12Series.setData([]);
  }

  if (opts.bb) {
    bbMiddleSeries.setData(buildSeries(indicators.bbMiddle));
    bbUpperSeries.setData(buildSeries(indicators.bbUpper));
    bbLowerSeries.setData(buildSeries(indicators.bbLower));
  } else {
    bbMiddleSeries.setData([]);
    bbUpperSeries.setData([]);
    bbLowerSeries.setData([]);
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

  // RSI
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

  // MACD
  if (opts.macd) {
    const macd = indicators.macdLine || [];
    const sig = indicators.macdSignal || [];
    const hist = indicators.macdHist || [];
    const macdData = [], sigData = [], histData = [];
    for (let i = 0; i < allCandles.length; i++) {
      if (macd[i] !== null && macd[i] !== undefined) {
        macdData.push({ time: allCandles[i].time, value: macd[i] });
      }
      if (sig[i] !== null && sig[i] !== undefined) {
        sigData.push({ time: allCandles[i].time, value: sig[i] });
      }
      if (hist[i] !== null && hist[i] !== undefined) {
        histData.push({ 
          time: allCandles[i].time, 
          value: hist[i], 
          color: (hist[i] >= 0 ? COLORS.macdHistPos : COLORS.macdHistNeg) 
        });
      }
    }
    macdSeries.setData(macdData);
    macdSignalSeries.setData(sigData);
    macdHistSeries.setData(histData);
  } else {
    macdSeries.setData([]);
    macdSignalSeries.setData([]);
    macdHistSeries.setData([]);
  }

  // Trade signals
  if (opts.signals) {
    const tradeSignals = calculateTradeSignals(allCandles, indicators);
    candleSeries.setMarkers(tradeSignals);
  } else {
    candleSeries.setMarkers([]);
  }

  // Resize charts after updating
  scheduleResizeCharts();
}

// ========== DATA LOADING ==========
async function loadLatest(limit = INITIAL_LIMIT) {
  if (loading) return;
  loading = true;
  try {
    const res = await fetch(`/data-latest?limit=${limit}`);
    const j = await res.json();

    raw1mData = j.candles || [];
    raw1mData.sort((a, b) => a.time - b.time);

    allCandles = resampleCandles(raw1mData, currentTF);
    loadedMin = j.min_time;
    loadedMax = j.max_time;

    candleSeries.setData(allCandles);
    updateAllIndicatorSeries();
    mainChart.timeScale().fitContent();

    // Clear range selection
    rangeSelected = false;
    rangeStartTs = null;
    rangeEndTs = null;

    if (replayBtn) replayBtn.disabled = true;
    if (btnPlay) btnPlay.disabled = true;
    if (replayStartInputEl) {
    replayStartInputEl.disabled = true;
    replayStartInputEl.value = '';
    replayStartInputEl.min = '';
    replayStartInputEl.max = '';
    }
  } finally {
    loading = false;
  }
}

async function loadBefore(limit = INITIAL_LIMIT) {
  // Don't load during replay or when preparing replay
  if (replayState !== 'IDLE') return;
  
  if (loading || loadedMin === null || loadedMin <= window.DATA_MIN_TS) return;
  const oldRange = mainChart.timeScale().getVisibleLogicalRange();
  loading = true;
  try {
    const res = await fetch(`/data-before?time=${loadedMin}&limit=${limit}`);
    const j = await res.json();
    if (j.candles && j.candles.length) {
      const newCount = j.candles.length;
      raw1mData = [...j.candles, ...raw1mData];
      raw1mData.sort((a, b) => a.time - b.time);
      allCandles = resampleCandles(raw1mData, currentTF);
      loadedMin = j.min_time;
      loadedMax = Math.max(loadedMax, j.max_time);
      candleSeries.setData(allCandles);
      updateAllIndicatorSeries();
      if (oldRange) {
        mainChart.timeScale().setVisibleLogicalRange({ 
          from: oldRange.from + newCount, 
          to: oldRange.to + newCount 
        });
      } else {
        mainChart.timeScale().fitContent();
      }
    }
  } finally {
    loading = false;
  }
}

async function loadAfter(limit = INITIAL_LIMIT) {
  // Don't load during replay or when preparing replay
  if (replayState !== 'IDLE') return;
  
  if (loading || loadedMax === null || loadedMax >= window.DATA_MAX_TS) return;
  const oldRange = mainChart.timeScale().getVisibleLogicalRange();
  loading = true;
  try {
    const res = await fetch(`/data-after?time=${loadedMax}&limit=${limit}`);
    const j = await res.json();
    if (j.candles && j.candles.length) {
      const newCount = j.candles.length;
      raw1mData = [...raw1mData, ...j.candles];
      raw1mData.sort((a, b) => a.time - b.time);
      allCandles = resampleCandles(raw1mData, currentTF);
      loadedMin = Math.min(loadedMin, j.min_time);
      loadedMax = j.max_time;
      candleSeries.setData(allCandles);
      updateAllIndicatorSeries();
      if (oldRange) {
        mainChart.timeScale().setVisibleLogicalRange({ 
          from: oldRange.from, 
          to: oldRange.to 
        });
      } else {
        mainChart.timeScale().fitContent();
      }
    }
  } finally {
    loading = false;
  }
}

async function loadRange(startIso, endIso) {
  const userStartTs = Math.floor(new Date(startIso).getTime() / 1000);
  const userEndTs = endIso
    ? Math.floor(new Date(endIso).getTime() / 1000)
    : loadedMax;

  if (!startIso) {
    alert("Please select a start datetime");
    return;
  }

  // Calculate expected number of candles for the range
  // For 1-minute data: ~375 minutes per trading day (09:15 to 15:30)
  // Add buffer for safety - use a high limit to ensure we get all data
  const daysDiff = Math.ceil((userEndTs - userStartTs) / (24 * 60 * 60));
  const expectedCandles = daysDiff * 400; // 400 candles per day (with buffer)
  const limit = Math.max(10000, expectedCandles); // At least 10000, or calculated amount
  
  console.log(`Loading range: ${startIso} to ${endIso}, days: ${daysDiff}, limit: ${limit}`);
  
  const qs = new URLSearchParams({ 
    start: startIso, 
    end: endIso,
    limit: limit.toString()
  });
  const res = await fetch(`/data-range?${qs.toString()}`);
  const j = await res.json();
  
  console.log(`Received ${j.candles ? j.candles.length : 0} candles from backend`);

  raw1mData = j.candles || [];
  raw1mData.sort((a, b) => a.time - b.time);

  allCandles = resampleCandles(raw1mData, currentTF);

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

  // Set range selection
  rangeSelected = true;
  rangeStartTs = userStartTs;
  rangeEndTs = userEndTs;

  // Enable replay controls
  if (replayBtn) replayBtn.disabled = false;
  if (btnPlay) btnPlay.disabled = false;

  // Set replay start input with market hours constraints
  if (replayStartInputEl) {
  const startDate = new Date(rangeStartTs * 1000);
    const endDate = new Date(rangeEndTs * 1000);
    const clamped = clampToMarketHours(startDate);
    
    // Ensure clamped date is within range
    const clampedTs = Math.floor(clamped.getTime() / 1000);
    const finalDate = clampedTs < rangeStartTs ? startDate : 
                     (clampedTs > rangeEndTs ? endDate : clamped);

  replayStartInputEl.disabled = false;
    
    // Set min/max to range boundaries
  replayStartInputEl.min = toLocalInput(startDate);
  replayStartInputEl.max = toLocalInput(endDate);
    replayStartInputEl.value = toLocalInput(finalDate);
  }
}

// ========== TIMEFRAME RESAMPLING ==========
function resampleCandles(data, tfMinutes) {
  if (!data.length) return [];
  
  // Ensure ascending order
  if (data.length > 1 && data[0].time > data[data.length - 1].time) {
    data = data.slice().sort((a, b) => a.time - b.time);
  }

  if (tfMinutes === 1) return data.slice();

  const tfSec = tfMinutes * 60;
  const result = [];
  let bucket = null;

  for (const c of data) {
    const bucketTime = Math.floor(c.time / tfSec) * tfSec;

    if (!bucket || bucket.time !== bucketTime) {
      if (bucket) result.push(bucket);

      bucket = {
        time: bucketTime,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
        volume: c.volume || 0
      };
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
  if (!raw1mData.length) {
    alert('Please load data first');
    return;
  }

  // Stop replay if it's running (timeframe change resets replay)
  if (replayState !== 'IDLE') {
    stopReplay();
  }

  const prevRange = mainChart.timeScale().getVisibleLogicalRange();

  currentTF = tf;
  allCandles = resampleCandles(raw1mData, tf);

  candleSeries.setData(allCandles);
  updateAllIndicatorSeries();

  // Update visible range to maintain user's view
  if (prevRange) {
    requestAnimationFrame(() => {
      try {
        // Try to maintain the same logical range
        const newRange = {
          from: Math.max(0, prevRange.from),
          to: Math.min(allCandles.length, prevRange.to)
        };
        mainChart.timeScale().setVisibleLogicalRange(newRange);
      } catch {
        // If that fails, fit content
        mainChart.timeScale().fitContent();
      }
    });
  } else {
    mainChart.timeScale().fitContent();
  }

  // Update button text
  const tfBtn = document.getElementById('tfBtn');
  if (tfBtn) {
    tfBtn.textContent = `${tf}m ▾`;
  }

  // Update active state in dropdown
  document.querySelectorAll('.tf-option').forEach(btn => {
    const btnTf = parseInt(btn.dataset.tf, 10);
    if (btnTf === tf) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });
}

// ========== REPLAY FUNCTIONALITY (TradingView-style) ==========
function updateIndicatorsIncremental(newCandle) {
  // Recalculate indicators for all candles (needed for accuracy)
  // But only update the last few values for performance
  const indicators = calculateIndicators(allCandles);

  // Update only the last value for each indicator (TradingView-style)
  const lastIdx = allCandles.length - 1;
  
  function updateLastValue(series, arr, enabled) {
    if (!enabled || !series || !arr || lastIdx >= arr.length) return;
    const val = arr[lastIdx];
    if (val !== null && val !== undefined) {
      series.update({ time: allCandles[lastIdx].time, value: val });
    }
  }
  
  // Update last 5 values for smoother transitions (especially for moving averages)
  function updateLastFew(series, arr, enabled, count = 5) {
    if (!enabled || !series || !arr) return;
    const start = Math.max(0, arr.length - count);
    for (let i = start; i < arr.length; i++) {
      if (i < allCandles.length && arr[i] !== null && arr[i] !== undefined) {
        series.update({ time: allCandles[i].time, value: arr[i] });
      }
    }
  }

  // Update overlay indicators (last few values for smooth transitions)
  if (opts.sma20 && sma20Series) {
    updateLastFew(sma20Series, indicators.sma20, true);
  }
  if (opts.sma50 && sma50Series) {
    updateLastFew(sma50Series, indicators.sma50, true);
  }
  if (opts.ema12 && ema12Series) {
    updateLastFew(ema12Series, indicators.ema12, true);
  }
  if (opts.bb) {
    if (bbMiddleSeries) updateLastFew(bbMiddleSeries, indicators.bbMiddle, true);
    if (bbUpperSeries) updateLastFew(bbUpperSeries, indicators.bbUpper, true);
    if (bbLowerSeries) updateLastFew(bbLowerSeries, indicators.bbLower, true);
  }
  if (opts.pacePro && paceProSeries) {
    const paceArr = indicators.paceNorm || [];
    if (paceArr.length > 0 && lastIdx < paceArr.length) {
      const v = paceArr[lastIdx];
      if (v !== null && v !== undefined) {
        paceProSeries.update({
          time: allCandles[lastIdx].time,
          value: v,
          color: v < 0 ? COLORS.paceNeg : COLORS.pacePos
        });
      }
    }
  }

  // Update RSI (last few values)
  if (opts.rsi && rsiSeries) {
    updateLastFew(rsiSeries, indicators.rsi, true);
  }

  // Update MACD (last few values)
  if (opts.macd) {
    if (macdSeries) updateLastFew(macdSeries, indicators.macdLine, true);
    if (macdSignalSeries) updateLastFew(macdSignalSeries, indicators.macdSignal, true);
    if (macdHistSeries) {
      const hist = indicators.macdHist || [];
      if (hist.length > 0 && lastIdx < hist.length) {
        const val = hist[lastIdx];
        if (val !== null && val !== undefined) {
          macdHistSeries.update({
            time: allCandles[lastIdx].time,
            value: val,
            color: val >= 0 ? COLORS.macdHistPos : COLORS.macdHistNeg
          });
        }
      }
    }
  }

  // Update trade signals (only check last candle)
  if (opts.signals && allCandles.length >= 2) {
    const lastIdx = allCandles.length - 1;
    const prevIdx = lastIdx - 1;
    const ema = indicators.ema12;
    const sma = indicators.sma20;
    const rsi = indicators.rsi;
    
    const relEmaIdx = ema.length - 1;
    const relSmaIdx = sma.length - 1;
    const relRsiIdx = rsi.length - 1;
    const relPrevEmaIdx = relEmaIdx - 1;
    const relPrevSmaIdx = relSmaIdx - 1;

    const markers = [];
    
    // Check for BUY signal
    if (ema[relEmaIdx] && sma[relSmaIdx] && ema[relPrevEmaIdx] && sma[relPrevSmaIdx] &&
        ema[relPrevEmaIdx] <= sma[relPrevSmaIdx] &&
        ema[relEmaIdx] > sma[relSmaIdx] &&
        rsi[relRsiIdx] > 40) {
      markers.push({
        time: newCandle.time,
        position: 'belowBar',
        color: '#2ECC71',
        shape: 'arrowUp',
        text: 'BUY'
      });
    }
    
    // Check for SELL signal
    if (ema[relEmaIdx] && sma[relSmaIdx] && ema[relPrevEmaIdx] && sma[relPrevSmaIdx] &&
        ema[relPrevEmaIdx] >= sma[relPrevSmaIdx] &&
        ema[relEmaIdx] < sma[relSmaIdx] &&
        rsi[relRsiIdx] < 60) {
      markers.push({
        time: newCandle.time,
        position: 'aboveBar',
        color: '#E74C3C',
        shape: 'arrowDown',
        text: 'SELL'
      });
    }
    
    if (markers.length > 0) {
      const existingMarkers = candleSeries.markers() || [];
      candleSeries.setMarkers([...existingMarkers, ...markers]);
    }
  }
}

function prepareReplayData() {
  if (!rangeSelected) {
    alert('Please select a range before replay');
    return false;
  }

  const replayStartInput = replayStartInputEl ? replayStartInputEl.value : null;
  if (!replayStartInput) {
    alert('Please select Replay Start time');
    return false;
  }

  const replayStartTs = Math.floor(new Date(replayStartInput).getTime() / 1000);
  
  // Always use raw1mData and resample to current timeframe
  // raw1mData should contain all candles from the selected range
  // This ensures we have the complete dataset to work with
  if (!raw1mData || raw1mData.length === 0) {
    alert('No data available. Please load a range first.');
    return false;
  }
  
  const tfData = resampleCandles(raw1mData, currentTF);
  
  // Split into historical (before replay start) and future (to replay)
  const historical = [];
  const future = [];

  for (const c of tfData) {
    if (c.time < replayStartTs) {
      historical.push(c);
    } else if (c.time >= replayStartTs && c.time <= rangeEndTs) {
      // Only include candles within the selected range
      future.push(c);
    }
  }
  

  // Check if replay start time is within the selected range
  // If it's within range, proceed without popup
  if (replayStartTs < rangeStartTs || replayStartTs > rangeEndTs) {
    // Replay start is outside range - this is invalid
    alert('Replay start time must be within the selected range');
    return false;
  }

  // If no future candles, it means replay start is at or very close to range end
  // This is OK - we can still show historical candles
  // Allow replay even with no future candles (it will just end immediately)
  // No popup needed - this is a valid case

  // Store replay data
  replayData = [...historical, ...future];  // Full dataset for indicators
  replayCandles = [...future];  // Candles to replay
  replayIndex = 0;  // Start from first future candle

  // Set replay state to PREPARING to prevent lazy loading interference
  replayState = 'PREPARING';

  // CRITICAL: Preserve ALL candles BEFORE replay start time
  // DO NOT clear the chart - only remove candles AFTER replay start
  // The key is: we want to show candles that are currently on the chart BEFORE replay start
  // and then add candles AFTER replay start one by one
  
  console.log('prepareReplayData: Current allCandles length:', allCandles ? allCandles.length : 0);
  console.log('prepareReplayData: Historical candles from range:', historical.length);
  console.log('prepareReplayData: Replay start timestamp:', replayStartTs, new Date(replayStartTs * 1000).toLocaleString());
  
  // Get current candles from chart (what's currently displayed)
  // This includes all candles that were loaded, not just from the range
  const currentChartCandles = allCandles || [];
  console.log('prepareReplayData: Current chart candles:', currentChartCandles.length);
  
  // CRITICAL: Filter to keep ONLY candles BEFORE replay start time
  // This preserves any candles that were already on the chart before replay start
  const preservedCandles = currentChartCandles.filter(c => c.time < replayStartTs);
  console.log('prepareReplayData: Preserved candles (before replay start):', preservedCandles.length);
  
  // Combine preserved candles with historical candles from range
  // Use Map to avoid duplicates by time
  const candlesMap = new Map();
  
  // Add preserved candles first (from current chart - may include older data outside range)
  for (const c of preservedCandles) {
    candlesMap.set(c.time, c);
  }
  
  // Add historical candles from range (ensures we have all candles from range before replay start)
  for (const c of historical) {
    if (c.time < replayStartTs) {
      candlesMap.set(c.time, c);
    }
  }
  
  // Set allCandles to combined historical candles (before replay start)
  // These will remain visible, and new candles will be added one by one
  allCandles = Array.from(candlesMap.values()).sort((a, b) => a.time - b.time);
  console.log('prepareReplayData: Final allCandles length (before replay start):', allCandles.length);
  
  // CRITICAL: Update chart to show ONLY historical candles (before replay start)
  // This removes candles after replay start, but keeps all candles before
  // IMPORTANT: Only call setData if allCandles has candles, otherwise preserve current chart
  if (allCandles.length > 0) {
    console.log('prepareReplayData: Setting chart with', allCandles.length, 'historical candles');
    // Set chart to show historical candles (before replay start time)
    // This removes candles after replay start, but keeps all candles before
    candleSeries.setData(allCandles);
    updateAllIndicatorSeries();
    
    // Fit content to show all historical candles, then scroll to end
    requestAnimationFrame(() => {
      mainChart.timeScale().fitContent();
      setTimeout(() => {
        // Scroll to show the last candle before replay start
        mainChart.timeScale().scrollToRealTime();
      }, 100);
    });
  } else {
    // No candles before replay start in combined data
    // Check if current chart has candles before replay start that we should preserve
    console.warn('prepareReplayData: No candles before replay start in combined data');
    console.log('prepareReplayData: Current chart has', currentChartCandles.length, 'candles');
    
    // Check if current chart has any candles before replay start
    const candlesBeforeInChart = currentChartCandles.filter(c => c.time < replayStartTs);
    
    if (candlesBeforeInChart.length > 0) {
      // There ARE candles before replay start in current chart - use them
      allCandles = candlesBeforeInChart;
      console.log('prepareReplayData: Found', allCandles.length, 'candles before replay start in current chart - preserving them');
      candleSeries.setData(allCandles);
      updateAllIndicatorSeries();
      
      requestAnimationFrame(() => {
        mainChart.timeScale().fitContent();
        setTimeout(() => {
          mainChart.timeScale().scrollToRealTime();
        }, 100);
      });
    } else {
      // No candles before replay start at all - this is expected when replay start = range start
      // In this case, we need to remove all candles from chart (they're all at/after replay start)
      // so they can be added back one by one during replay
      console.log('prepareReplayData: No candles before replay start');
      console.log('prepareReplayData: Current chart has', currentChartCandles.length, 'candles, all at/after replay start');
      console.log('prepareReplayData: Will clear chart so candles can be added one by one from replay start');
      
      // Set allCandles to empty - candles after replay start will be added one by one
      allCandles = [];
      
      // Clear the chart - this is necessary to remove candles after replay start
      // The replay loop will add them back one by one
      candleSeries.setData([]);
      updateAllIndicatorSeries();
      
      requestAnimationFrame(() => {
        mainChart.timeScale().fitContent();
      });
    }
  }
  
  // Clear any existing markers
  candleSeries.setMarkers([]);

  // Update time indicator
  updateReplayTimeIndicator();

  return true;
}

function updateReplayTimeIndicator() {
  if (!replayTimeIndicator) return;
  
  // If replay is not active, clear the indicator
  if (replayState === 'IDLE' || replayState === 'PREPARING') {
    if (replayState === 'PREPARING' && allCandles.length > 0) {
      // During preparation, show the last historical candle time
      const currentCandle = allCandles[allCandles.length - 1];
      if (currentCandle) {
        const date = new Date(currentCandle.time * 1000);
        const timeStr = date.toLocaleString('en-US', {
          month: 'short',
          day: 'numeric',
          year: 'numeric',
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
          hour12: false
        });
        replayTimeIndicator.textContent = `${timeStr} (Ready)`;
      }
    } else {
      replayTimeIndicator.textContent = '';
    }
    return;
  }

  // Get the current candle being displayed
  // During replay, the last candle in allCandles is the one we just added
  // But we should use the candle from replayCandles to ensure we're showing the correct replayed candle
  let currentCandle = null;
  
  if (replayState === 'PLAYING' && replayCandles.length > 0 && replayIndex < replayCandles.length) {
    // We've just added replayCandles[replayIndex], so that's the current candle
    currentCandle = replayCandles[replayIndex];
  } else if (allCandles.length > 0) {
    // Fallback to last candle in allCandles
    currentCandle = allCandles[allCandles.length - 1];
  }
  
  if (!currentCandle) {
    replayTimeIndicator.textContent = '';
    return;
  }
  
  const date = new Date(currentCandle.time * 1000);
  const timeStr = date.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false
  });
  
  // Calculate progress: how many candles we've replayed out of total to replay
  // When updateReplayTimeIndicator() is called:
  // - replayIndex is the index of the candle we just added (before increment)
  // - We've added (replayIndex + 1) candles (indices 0 through replayIndex)
  // - So progress = (replayIndex + 1) / replayCandles.length * 100
  const progress = replayCandles.length > 0 
    ? Math.round(((replayIndex + 1) / replayCandles.length) * 100) 
    : 0;
  
  replayTimeIndicator.textContent = `${timeStr} (${progress}%)`;
}

function startReplay(speed) {
  console.log('startReplay() called. Current state:', replayState, 'replayTimer:', replayTimer);
  // If already playing, do nothing
  if (replayTimer || replayState === 'PLAYING') {
    console.log('Replay already playing, returning early');
    return;
  }
  
  // If in PREPARING state from a previous failed attempt, reset to IDLE
  if (replayState === 'PREPARING') {
    console.log('Resetting from PREPARING state');
    replayState = 'IDLE';
  }
  
  // If paused, resume from where we left off
  if (replayState === 'PAUSED') {
    replayState = 'PLAYING';
    replaySpeed = speed || replaySpeed;
    
    // Update UI
    if (btnPlay) btnPlay.disabled = true;
    if (btnPause) btnPause.disabled = false;
    const btnStop = document.getElementById('btnStop');
    if (btnStop) btnStop.disabled = false;
    if (replayBtn) {
      replayBtn.textContent = 'Replay: PLAYING';
    }
    
    // Resume the timer - ensure historical candles AND indicators are still visible
    // Verify chart has all candles (historical + already replayed)
    // allCandles is the source of truth - if it has data, ensure chart displays it
    if (!allCandles || allCandles.length === 0) {
      console.warn('allCandles is empty when resuming replay');
      // Try to restore from replayData if available
      if (replayData && replayData.length > 0) {
        allCandles = [...replayData];
        candleSeries.setData(allCandles);
        updateAllIndicatorSeries();
      }
    } else {
      // Ensure chart displays current allCandles
      candleSeries.setData(allCandles);
      updateAllIndicatorSeries();
    }
    
    // Check indicators separately - verify they are correct
    // Check if RSI series has data (as a proxy for all indicators)
    if (opts.rsi && rsiSeries) {
      // We can't directly check rsiSeries.data(), so just ensure indicators are updated
      // The updateAllIndicatorSeries() call above should have handled this
    } else if (opts.macd && macdSeries) {
      // Check MACD if RSI is not enabled
      // We can't directly check macdSeries.data(), so just ensure indicators are updated
      // The updateAllIndicatorSeries() call above should have handled this
    }
    
    // Indicators should already be updated by updateAllIndicatorSeries() call above
    // Since we can't check series.data() directly, we rely on updateAllIndicatorSeries()
    // to ensure indicators are correctly set based on allCandles
    
    // Resume the timer
    replayTimer = setInterval(() => {
      if (replayIndex >= replayCandles.length) {
    stopReplay();
        return;
      }

      const newCandle = replayCandles[replayIndex];
      
      // Add candle using update() - this preserves all existing candles (historical + new)
      // allCandles should already have historical candles + previously replayed candles
      // The update() method adds the new candle while keeping all previous candles visible
      candleSeries.update(newCandle);
      allCandles.push(newCandle);
      
      // Update indicators incrementally
      updateIndicatorsIncremental(newCandle);
      
      // Smooth scroll to keep latest candle visible
      mainChart.timeScale().scrollToRealTime();
      
      // Update time indicator
      updateReplayTimeIndicator();
      
      replayIndex++;
    }, replaySpeed);
    
    return;
  }
  
  // Start fresh - prepare data first
  console.log('Preparing replay data...');
  const prepared = prepareReplayData();
  if (!prepared) {
    console.error('prepareReplayData() returned false');
    alert('Failed to prepare replay data. Please check that range and replay start time are set correctly.');
    return;
  }
  
  console.log('Replay data prepared. Historical candles:', allCandles.length, 'Future candles to replay:', replayCandles ? replayCandles.length : 0);
  
  // Verify that we have candles to replay
  if (!replayCandles || replayCandles.length === 0) {
    console.warn('No candles to replay - replay will end immediately');
    // Still allow replay to start, but it will end immediately
  }
  
  // Now that data is prepared, set state to PLAYING
  // This happens after prepareReplayData() which sets state to PREPARING
  replaySpeed = speed || 800;
  replayState = 'PLAYING';
  
  // CRITICAL: The prepareReplayData() should have already set historical candles
  // DO NOT call setData again here as it might cause flickering or clearing
  // The chart should already be showing historical candles from prepareReplayData()
  console.log('startReplay: allCandles length after prepareReplayData:', allCandles.length);
  
  // Only verify indicators are updated (prepareReplayData() should have done this)
  if (allCandles.length > 0) {
    // Indicators should already be updated by prepareReplayData(), but ensure they're correct
    // Only update if really needed to avoid unnecessary work and potential clearing
    updateAllIndicatorSeries();
  } else {
    console.warn('startReplay: allCandles is empty after prepareReplayData()');
  }
  
  // Update UI
  if (btnPlay) btnPlay.disabled = true;
  if (btnPause) btnPause.disabled = false;
  const btnStop = document.getElementById('btnStop');
  if (btnStop) btnStop.disabled = false;
  if (replayBtn) {
    replayBtn.textContent = 'Replay: PLAYING';
  replayBtn.classList.add('active');
  }

  // Start replay timer - this will add new candles while preserving historical ones
  // Indicators should already be updated by updateAllIndicatorSeries() call above
  // Since we can't check series.data() directly, we rely on updateAllIndicatorSeries()
  // to ensure indicators are correctly set based on allCandles
  
  replayTimer = setInterval(() => {
    if (!replayCandles || replayIndex >= replayCandles.length) {
      stopReplay();
      return;
    }

    const newCandle = replayCandles[replayIndex];
    if (!newCandle) {
      console.error('Invalid candle at index', replayIndex);
      replayIndex++;
      return;
    }
    
    // Add candle using update() - this preserves all existing candles (historical + new)
    // allCandles should already have historical candles, so we just update
    // The update() method adds the new candle while keeping all previous candles visible
    candleSeries.update(newCandle);
    allCandles.push(newCandle);
    
    // Update indicators incrementally
    updateIndicatorsIncremental(newCandle);
    
    // Smooth scroll to keep latest candle visible
    mainChart.timeScale().scrollToRealTime();
    
    // Update time indicator
    updateReplayTimeIndicator();
    
    replayIndex++;
  }, replaySpeed);
}

function pauseReplay() {
  if (replayState !== 'PLAYING') return;
  
  clearInterval(replayTimer);
  replayTimer = null;
  replayState = 'PAUSED';
  
  // Update UI
  if (btnPlay) btnPlay.disabled = false;
  if (btnPause) btnPause.disabled = true;
  const btnStop = document.getElementById('btnStop');
  if (btnStop) btnStop.disabled = false;
  if (replayBtn) {
    replayBtn.textContent = 'Replay: PAUSED';
  }
}

function stopReplay() {
  clearInterval(replayTimer);
  replayTimer = null;
  replayState = 'IDLE';
  cachedIndicators = null;

  // Reset to show all candles normally (fresh screen)
  // Restore full chart with all data at current timeframe
  allCandles = resampleCandles(raw1mData, currentTF);
  replayIndex = 0;
  replayCandles = [];
  replayData = [];

  // Update chart with full data (fresh screen - shows all candles normally)
  candleSeries.setData(allCandles);
  updateAllIndicatorSeries();
  mainChart.timeScale().fitContent();
  
  // Update UI
  if (btnPlay) btnPlay.disabled = false;
  if (btnPause) btnPause.disabled = true;
  const btnStop = document.getElementById('btnStop');
  if (btnStop) btnStop.disabled = false;
  if (replayBtn) {
    replayBtn.textContent = 'Replay: OFF';
    replayBtn.classList.remove('active');
  }
  if (replayTimeIndicator) {
    replayTimeIndicator.textContent = '';
  }
  
  // Clear markers
  candleSeries.setMarkers([]);
}

// ========== UTILITY FUNCTIONS ==========
function toLocalInput(date) {
  const pad = n => String(n).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function isWithinMarketHours(date) {
  const h = date.getHours();
  const m = date.getMinutes();

  const afterOpen = h > MARKET_OPEN_HOUR || (h === MARKET_OPEN_HOUR && m >= MARKET_OPEN_MIN);
  const beforeClose = h < MARKET_CLOSE_HOUR || (h === MARKET_CLOSE_HOUR && m <= MARKET_CLOSE_MIN);

  return afterOpen && beforeClose;
}

function clampToMarketHours(date) {
  const d = new Date(date);
  const originalDate = new Date(d);

  // Check if time is before market open (09:15 AM)
  if (d.getHours() < MARKET_OPEN_HOUR || 
      (d.getHours() === MARKET_OPEN_HOUR && d.getMinutes() < MARKET_OPEN_MIN)) {
    // Set to same day 09:15 AM
      d.setHours(MARKET_OPEN_HOUR, MARKET_OPEN_MIN, 0, 0);
    return d;
  }

  // Check if time is after market close (03:30 PM)
  if (d.getHours() > MARKET_CLOSE_HOUR || 
      (d.getHours() === MARKET_CLOSE_HOUR && d.getMinutes() > MARKET_CLOSE_MIN)) {
    // Set to next working day 09:15 AM
    d.setDate(d.getDate() + 1);
    
    // Skip weekends (Saturday = 6, Sunday = 0)
    while (d.getDay() === 0 || d.getDay() === 6) {
      d.setDate(d.getDate() + 1);
    }
    
    d.setHours(MARKET_OPEN_HOUR, MARKET_OPEN_MIN, 0, 0);
  return d;
}

  // Time is within market hours
  return d;
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

function setReplayStatus(state) {
  if (state === 'playing') {
    if (btnPlay) btnPlay.disabled = true;
    if (btnPause) btnPause.disabled = false;
    if (replayBtn) {
      replayBtn.textContent = 'Replay: PLAYING';
      replayBtn.classList.add('active');
    }
  } else if (state === 'paused') {
    if (btnPlay) btnPlay.disabled = false;
    if (btnPause) btnPause.disabled = true;
    if (replayBtn) {
      replayBtn.textContent = 'Replay: PAUSED';
      replayBtn.classList.add('active');
    }
  } else {
    if (btnPlay) btnPlay.disabled = false;
    if (btnPause) btnPause.disabled = true;
    if (replayBtn) {
      replayBtn.textContent = 'Replay: OFF';
      replayBtn.classList.remove('active');
    }
  }
}

// ========== CHART RESIZING ==========
function scheduleResizeCharts() {
  if (resizeTimer) clearTimeout(resizeTimer);
  resizeTimer = setTimeout(resizeCharts, 120);
}

function resizeCharts() {
  try {
    const mainContainer = document.getElementById('chart-main');
    const rsiContainer = document.getElementById('chart-rsi');
    const macdContainer = document.getElementById('chart-macd');

    if (mainContainer && mainChart) {
    const wMain = mainContainer.clientWidth || mainContainer.offsetWidth;
    const hMain = mainContainer.clientHeight || mainContainer.offsetHeight;
    if (wMain > 0 && hMain > 0) {
      mainChart.resize(wMain, hMain);
      }
    }

    if (rsiContainer && rsiChart) {
    if (getComputedStyle(rsiContainer).display !== 'none') {
      rsiChart.resize(rsiContainer.clientWidth, rsiContainer.clientHeight);
    }
    }
    
    if (macdContainer && macdChart) {
    if (getComputedStyle(macdContainer).display !== 'none') {
      macdChart.resize(macdContainer.clientWidth, macdContainer.clientHeight);
      }
    }
  } catch (e) {
    // ignore errors
  }
}

// ========== UI INITIALIZATION ==========
async function initUI() {
  // Header elements
  rangeBtn = document.getElementById('rangeBtn');
  replayBtn = document.getElementById('replayBtn');
  rangePanel = document.getElementById('rangePanel');
  replayPanel = document.getElementById('replayPanel');
  const rsiContainer = document.getElementById('chart-rsi');
  const macdContainer = document.getElementById('chart-macd');
  replayStartInputEl = document.getElementById('replayStart');
  btnPlay = document.getElementById('btnPlay');
  btnPause = document.getElementById('btnPause');
  replayTimeIndicator = document.getElementById('replayTimeIndicator');
  const rangeStartInput = document.getElementById('start');
  const rangeEndInput = document.getElementById('end');

  // Range/Replay panel toggles
  if (rangeBtn) {
  rangeBtn.onclick = () => {
      if (rangePanel) rangePanel.classList.toggle('hidden');
      if (replayPanel) replayPanel.classList.add('hidden');
  };
  }

  if (replayBtn) {
  replayBtn.onclick = () => {
      if (replayPanel) replayPanel.classList.toggle('hidden');
      if (rangePanel) rangePanel.classList.add('hidden');
  };
  }

  // Load data info
  const info = await fetch('/data-info').then(r => r.json());
  window.DATA_MIN_TS = info.min_ts;
  window.DATA_MAX_TS = info.max_ts;

  // Set date input ranges based on available data
  // Date selection is limited to data range, time is limited to market hours
  if (rangeStartInput && rangeEndInput) {
    // Store the data date boundaries (for date selection)
    const dataMinDate = new Date(info.min_ts * 1000);
    const dataMaxDate = new Date(info.max_ts * 1000);
    
    // Set date boundaries (YYYY-MM-DD format)
    const dataMinDateStr = dataMinDate.toISOString().split('T')[0];
    const dataMaxDateStr = dataMaxDate.toISOString().split('T')[0];
    
    // Set min/max dates (date portion only, time will be constrained separately)
    rangeStartInput.min = `${dataMinDateStr}T09:15`;
    rangeStartInput.max = `${dataMaxDateStr}T15:30`;
    rangeEndInput.min = `${dataMinDateStr}T09:15`;
    rangeEndInput.max = `${dataMaxDateStr}T15:30`;
    
    // Set default values (last 24 hours, clamped to market hours)
  const maxDate = new Date(info.max_ts * 1000);
  const defaultStart = new Date((info.max_ts - 24 * 3600) * 1000);
    
    // Clamp to market hours and ensure within data range
    let clampedStart = clampToMarketHours(defaultStart);
    let clampedEnd = clampToMarketHours(maxDate);
    
    // Ensure dates are within data range
    if (clampedStart < dataMinDate) {
      clampedStart = new Date(dataMinDate);
      clampedStart.setHours(MARKET_OPEN_HOUR, MARKET_OPEN_MIN, 0, 0);
    }
    if (clampedEnd > dataMaxDate) {
      clampedEnd = new Date(dataMaxDate);
      clampedEnd.setHours(MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN, 0, 0);
    }
    
    rangeStartInput.value = toLocalInput(clampedStart);
    rangeEndInput.value = toLocalInput(clampedEnd);
  }

  // Load range button
  const btnLoad = document.getElementById('btnLoad');
  if (btnLoad && rangeStartInput && rangeEndInput) {
    btnLoad.onclick = () => {
      loadRange(rangeStartInput.value, rangeEndInput.value);
      if (rangeBtn) {
        rangeBtn.textContent = `Range: ${rangeStartInput.value.replace('T', ' ')} → ${rangeEndInput.value.replace('T', ' ')}`;
    rangeBtn.classList.add('active');
      }
      if (rangePanel) rangePanel.classList.add('hidden');
    };
  }

  // Indicator toggles
  const toggleSMA20 = document.getElementById('toggle-sma20');
  if (toggleSMA20) {
    toggleSMA20.addEventListener('change', e => {
      opts.sma20 = e.target.checked;
    updateAllIndicatorSeries();
    });
  }

  const toggleSMA50 = document.getElementById('toggle-sma50');
  if (toggleSMA50) {
    toggleSMA50.addEventListener('change', e => {
      opts.sma50 = e.target.checked;
  updateAllIndicatorSeries();
    });
  }

  const toggleEMA12 = document.getElementById('toggle-ema12');
  if (toggleEMA12) {
    toggleEMA12.addEventListener('change', e => {
      opts.ema12 = e.target.checked;
      updateAllIndicatorSeries();
    });
  }

  const toggleBB = document.getElementById('toggle-bb');
  if (toggleBB) {
    toggleBB.addEventListener('change', e => {
      opts.bb = e.target.checked;
      updateAllIndicatorSeries();
    });
  }

  const paceproToggle = document.getElementById('paceproToggle');
  if (paceproToggle) {
    paceproToggle.addEventListener('change', e => {
      opts.pacePro = e.target.checked;
      updateAllIndicatorSeries();
    });
  }

  const toggleSignals = document.getElementById('toggle-signals');
  if (toggleSignals) {
    toggleSignals.addEventListener('change', e => {
      opts.signals = e.target.checked;
      updateAllIndicatorSeries();
    });
  }

  // RSI toggle
  const toggleRSI = document.getElementById('toggle-rsi');
  if (toggleRSI && rsiContainer) {
    toggleRSI.addEventListener('change', e => {
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
      rsiChart.resize(rsiContainer.clientWidth, rsiContainer.clientHeight);
        }
        restoreVisibleRange(range);
      });
    });
  }

  // MACD toggle
  const toggleMACD = document.getElementById('toggle-macd');
  if (toggleMACD && macdContainer) {
    toggleMACD.addEventListener('change', e => {
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
      macdChart.resize(macdContainer.clientWidth, macdContainer.clientHeight);
        }
        restoreVisibleRange(range);
      });
    });
  }

  // Indicators dropdown button click handler
  const indicatorsBtn = document.querySelector('.header-right .dropdown .header-btn');
  if (indicatorsBtn && indicatorsBtn.textContent.includes('Indicators')) {
    const indicatorsDropdown = indicatorsBtn.closest('.dropdown');
    if (indicatorsDropdown) {
      const indicatorsPanel = indicatorsDropdown.querySelector('.dropdown-panel');
      indicatorsBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (indicatorsPanel) {
          const computedStyle = window.getComputedStyle(indicatorsPanel);
          const isVisible = indicatorsPanel.style.display === 'block' || 
                           computedStyle.display === 'block';
          if (isVisible) {
            indicatorsPanel.style.display = 'none';
          } else {
            indicatorsPanel.style.display = 'block';
          }
        }
      });
    }
  }

  // Timeframe dropdown button click handler
  const tfBtn = document.getElementById('tfBtn');
  if (tfBtn) {
    const tfDropdown = tfBtn.closest('.dropdown');
    if (tfDropdown) {
      const tfPanel = tfDropdown.querySelector('.dropdown-panel');
      tfBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (tfPanel) {
          const isVisible = tfPanel.style.display === 'block' || 
                           window.getComputedStyle(tfPanel).display === 'block';
          tfPanel.style.display = isVisible ? 'none' : 'block';
        }
      });
    }
  }

  // Timeframe option buttons
    document.querySelectorAll('.tf-option').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
        const tf = parseInt(btn.dataset.tf, 10);
        applyTimeframe(tf);
      // Close dropdown after selection
      const dropdown = btn.closest('.dropdown');
      if (dropdown) {
        const panel = dropdown.querySelector('.dropdown-panel');
        if (panel) panel.style.display = 'none';
      }
      });
    });

  // Set initial active timeframe (1m)
  const initialTfBtn = document.querySelector('.tf-option[data-tf="1"]');
  if (initialTfBtn) {
    initialTfBtn.classList.add('active');
  }

  // Close dropdowns when clicking outside
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.dropdown')) {
      document.querySelectorAll('.dropdown-panel').forEach(panel => {
        // Close panels that were opened via click (have inline display style)
        // Remove the inline style to allow CSS hover to work again
        if (panel.style.display === 'block') {
          panel.style.display = '';
        }
      });
    }
  });

  // Replay controls with market hours validation
  if (replayStartInputEl) {
    const setReplayTimeConstraints = () => {
      if (replayStartInputEl.value && rangeSelected) {
        const date = new Date(replayStartInputEl.value);
        const dateStr = date.toISOString().split('T')[0];
        
        // Set min/max based on range
        const rangeStartDate = new Date(rangeStartTs * 1000);
        const rangeEndDate = new Date(rangeEndTs * 1000);
        
        // If date is within range, allow market hours
        if (date >= rangeStartDate && date <= rangeEndDate) {
          replayStartInputEl.min = `${dateStr}T09:15`;
          replayStartInputEl.max = `${dateStr}T15:30`;
        } else {
          // Use range boundaries
          replayStartInputEl.min = toLocalInput(rangeStartDate);
          replayStartInputEl.max = toLocalInput(rangeEndDate);
        }
      }
    };

    replayStartInputEl.addEventListener('input', () => {
      setReplayTimeConstraints();
      
      if (!rangeSelected || !replayStartInputEl.value) {
        if (btnPlay) btnPlay.disabled = true;
        return;
      }

      const ts = Math.floor(new Date(replayStartInputEl.value).getTime() / 1000);
      if (btnPlay) {
        btnPlay.disabled = ts < rangeStartTs || ts > rangeEndTs;
      }
    });
    
    replayStartInputEl.addEventListener('change', () => {
      if (replayStartInputEl.value) {
        const d = clampToMarketHours(new Date(replayStartInputEl.value));
        
        // Ensure it's within the selected range
        const ts = Math.floor(d.getTime() / 1000);
        if (ts < rangeStartTs) {
          d.setTime(rangeStartTs * 1000);
        } else if (ts > rangeEndTs) {
          d.setTime(rangeEndTs * 1000);
        }
        
        replayStartInputEl.value = toLocalInput(d);
        setReplayTimeConstraints();
        
        // Update play button state
        if (btnPlay) {
          const finalTs = Math.floor(d.getTime() / 1000);
          btnPlay.disabled = finalTs < rangeStartTs || finalTs > rangeEndTs;
        }
      }
    });
  }

  // Market hours validation for range inputs
  // Date selection is constrained by data range, time is constrained to market hours
  if (rangeStartInput) {
    const dataMinDate = new Date(window.DATA_MIN_TS * 1000);
    const dataMaxDate = new Date(window.DATA_MAX_TS * 1000);
    const dataMinDateStr = dataMinDate.toISOString().split('T')[0];
    const dataMaxDateStr = dataMaxDate.toISOString().split('T')[0];

    // Update constraints: date from data range, time from market hours
    const updateConstraints = (input) => {
      if (input.value) {
        const date = new Date(input.value);
        const dateStr = date.toISOString().split('T')[0];
        
        // Date must be within data range
        // Time must be within market hours (09:15 to 15:30)
        if (dateStr === dataMinDateStr) {
          // On minimum data date, start from market open
          input.min = `${dataMinDateStr}T09:15`;
  } else {
          input.min = `${dataMinDateStr}T09:15`;
        }
        
        if (dateStr === dataMaxDateStr) {
          // On maximum data date, end at market close
          input.max = `${dataMaxDateStr}T15:30`;
        } else {
          input.max = `${dataMaxDateStr}T15:30`;
        }
      }
    };

    rangeStartInput.addEventListener('input', () => {
      updateConstraints(rangeStartInput);
    });

    rangeStartInput.addEventListener('change', () => {
      if (rangeStartInput.value) {
        const inputDate = new Date(rangeStartInput.value);
        
        // Ensure date is within data range
        if (inputDate < dataMinDate) {
          inputDate.setTime(dataMinDate.getTime());
          inputDate.setHours(MARKET_OPEN_HOUR, MARKET_OPEN_MIN, 0, 0);
        } else if (inputDate > dataMaxDate) {
          inputDate.setTime(dataMaxDate.getTime());
          inputDate.setHours(MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN, 0, 0);
        }
        
        // Clamp time to market hours
        const d = clampToMarketHours(inputDate);
        
        // Ensure it's still within data range after clamping
        if (d < dataMinDate) {
          d.setTime(dataMinDate.getTime());
        } else if (d > dataMaxDate) {
          d.setTime(dataMaxDate.getTime());
        }
        
        rangeStartInput.value = toLocalInput(d);
        updateConstraints(rangeStartInput);
        
        // Ensure end date is after start date
        if (rangeEndInput && rangeEndInput.value) {
          const endDate = new Date(rangeEndInput.value);
          if (endDate < d) {
            endDate.setTime(d.getTime() + 6 * 60 * 60 * 1000); // Add 6 hours
            const clampedEnd = clampToMarketHours(endDate);
            if (clampedEnd > dataMaxDate) {
              clampedEnd.setTime(dataMaxDate.getTime());
            }
            rangeEndInput.value = toLocalInput(clampedEnd);
            updateConstraints(rangeEndInput);
          }
        }
      }
    });
  }

  if (rangeEndInput) {
    const dataMinDate = new Date(window.DATA_MIN_TS * 1000);
    const dataMaxDate = new Date(window.DATA_MAX_TS * 1000);
    const dataMinDateStr = dataMinDate.toISOString().split('T')[0];
    const dataMaxDateStr = dataMaxDate.toISOString().split('T')[0];

    const updateConstraints = (input) => {
      if (input.value) {
        // Date must be within data range, time within market hours
        input.min = `${dataMinDateStr}T09:15`;
        input.max = `${dataMaxDateStr}T15:30`;
      }
    };

    rangeEndInput.addEventListener('input', () => {
      updateConstraints(rangeEndInput);
    });

    rangeEndInput.addEventListener('change', () => {
      if (rangeEndInput.value) {
        const inputDate = new Date(rangeEndInput.value);
        
        // Ensure date is within data range
        if (inputDate < dataMinDate) {
          inputDate.setTime(dataMinDate.getTime());
          inputDate.setHours(MARKET_OPEN_HOUR, MARKET_OPEN_MIN, 0, 0);
        } else if (inputDate > dataMaxDate) {
          inputDate.setTime(dataMaxDate.getTime());
          inputDate.setHours(MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN, 0, 0);
        }
        
        // Clamp time to market hours
        const d = clampToMarketHours(inputDate);
        
        // Ensure it's still within data range after clamping
        if (d < dataMinDate) {
          d.setTime(dataMinDate.getTime());
        } else if (d > dataMaxDate) {
          d.setTime(dataMaxDate.getTime());
        }
        
        rangeEndInput.value = toLocalInput(d);
        updateConstraints(rangeEndInput);
        
        // Ensure end date is after start date
        if (rangeStartInput && rangeStartInput.value) {
          const startDate = new Date(rangeStartInput.value);
          if (d < startDate) {
            startDate.setTime(d.getTime() - 6 * 60 * 60 * 1000); // Subtract 6 hours
            const clampedStart = clampToMarketHours(startDate);
            if (clampedStart < dataMinDate) {
              clampedStart.setTime(dataMinDate.getTime());
            }
            rangeStartInput.value = toLocalInput(clampedStart);
            updateConstraints(rangeStartInput);
          }
        }
      }
    });
  }

  // Replay play button
  if (btnPlay) {
    btnPlay.onclick = () => {
      try {
        const speedInput = document.getElementById('replaySpeed');
        const speed = speedInput ? Number(speedInput.value) || 800 : 800;
        console.log('Play button clicked, starting replay with speed:', speed);
        startReplay(speed);
        setReplayStatus('playing');
      } catch (error) {
        console.error('Error starting replay:', error);
        alert('Error starting replay: ' + error.message);
      }
    };
  }

  // Replay pause button
  if (btnPause) {
    btnPause.onclick = () => {
      pauseReplay();
      setReplayStatus('paused');
    };
  }

  // Replay stop button
  const btnStop = document.getElementById('btnStop');
  if (btnStop) {
    btnStop.onclick = () => {
      stopReplay();
      setReplayStatus('stopped');
    };
  }

  // Fullscreen button
  const fsBtn = document.getElementById('fullscreenBtn');
  if (fsBtn) {
    fsBtn.addEventListener('click', () => {
      document.body.classList.toggle('fullscreen-mode');
      scheduleResizeCharts();
    });
  }

  // Window resize
  window.addEventListener('resize', () => scheduleResizeCharts());
}

// ========== PREVENT BROWSER NAVIGATION GESTURES ==========
function preventBrowserNavigationGestures() {
  // Rely on CSS overscroll-behavior (set in style.css) to prevent browser navigation
  // This is the cleanest solution - it prevents navigation without blocking chart scrolling
  
  // The CSS properties overscroll-behavior-x: contain and overscroll-behavior-y: contain
  // prevent the browser from navigating when scrolling reaches the boundaries of the chart area
  
  // No JavaScript needed - CSS handles it all!
  // This allows the chart library to receive all wheel events and handle scrolling naturally
}

// ========== INITIALIZATION ==========
(async function init() {
  createCharts();
  await initUI();
  await loadLatest();
  setTimeout(resizeCharts, 0);
})();
