// ========== CONSTANTS & CONFIGURATION ==========
const INITIAL_LIMIT = 300;
const INDICATOR_HEIGHT = 160;
const MAIN_MIN_HEIGHT = 200;
const MARKET_OPEN_HOUR = 9;
const MARKET_OPEN_MIN = 15;
const MARKET_CLOSE_HOUR = 15;
const MARKET_CLOSE_MIN = 30;

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

// ========== CHART INSTANCES ==========
let mainChart, rsiChart, macdChart;
let candleSeries, sma20Series, sma50Series, ema12Series;
let bbUpperSeries, bbMiddleSeries, bbLowerSeries, paceProSeries;
let rsiSeries, macdSeries, macdSignalSeries, macdHistSeries;

// ========== DATA STATE ==========
let raw1mData = [];           // Original 1-minute candles from selected range
let allCandles = [];          // Current timeframe candles displayed on chart
let currentTF = 1;            // Current timeframe in minutes
let loadedMin = null;         // Min timestamp of loaded data
let loadedMax = null;         // Max timestamp of loaded data
let loading = false;          // Flag to prevent concurrent loads

// ========== REPLAY STATE ==========
let replayMode = false;
let replayTimer = null;
let replayIndex = 0;
let replayState = 'IDLE';     // IDLE | PREPARING | PLAYING | PAUSED
let replayStartTs = null;     // Replay start timestamp
let replayEndTs = null;       // Replay end timestamp
let replayCandles = [];       // Candles to replay (after replay start)
let replayHistorical = [];    // Candles before replay start (keep displayed)
let rangeSelected = false;
let rangeStartTs = null;      // Selected range start
let rangeEndTs = null;        // Selected range end
let replaySpeed = 800;

// ========== UI ELEMENTS ==========
let rangeBtn, replayBtn, rangePanel, replayPanel;
let replayStartInputEl, btnPlay, btnPause, replayTimeIndicator;
let resizeTimer = null;

// ========== DATA LOADING FUNCTIONS ==========
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

    // Reset range selection
    rangeSelected = false;
    rangeStartTs = null;
    rangeEndTs = null;

    // Disable replay buttons until range is selected
    if (replayBtn) replayBtn.disabled = true;
    if (btnPlay) btnPlay.disabled = true;
    if (replayStartInputEl) {
      replayStartInputEl.disabled = true;
      replayStartInputEl.value = '';
    }
  } finally {
    loading = false;
  }
}

async function loadBefore(limit = INITIAL_LIMIT) {
  // Don't load during replay
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
  // Don't load during replay
  if (replayState !== 'IDLE') return;
  if (loading || loadedMax === null || loadedMax >= window.DATA_MAX_TS) return;

  const oldRange = mainChart.timeScale().getVisibleLogicalRange();
  loading = true;
  try {
    const res = await fetch(`/data-after?time=${loadedMax}&limit=${limit}`);
    const j = await res.json();
    if (j.candles && j.candles.length) {
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

  // Calculate expected number of candles
  const daysDiff = Math.ceil((userEndTs - userStartTs) / (24 * 60 * 60));
  const expectedCandles = daysDiff * 400;
  const limit = Math.max(10000, expectedCandles);

  const qs = new URLSearchParams({
    start: startIso,
    end: endIso,
    limit: limit.toString()
  });
  const res = await fetch(`/data-range?${qs.toString()}`);
  const j = await res.json();

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
  if (btnPlay) btnPlay.disabled = true; // Stay disabled until replay time is set
  if (replayStartInputEl) {
    const startDate = new Date(rangeStartTs * 1000);
    const endDate = new Date(rangeEndTs * 1000);
    const clamped = clampToMarketHours(startDate);
    const clampedTs = Math.floor(clamped.getTime() / 1000);
    const finalDate = clampedTs < rangeStartTs ? startDate : (clampedTs > rangeEndTs ? endDate : clamped);

    replayStartInputEl.disabled = false;
    replayStartInputEl.min = toLocalInput(startDate);
    replayStartInputEl.max = toLocalInput(endDate);
    replayStartInputEl.value = toLocalInput(finalDate);
  }
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
  
      // BUY signal
      if (ema[i] && sma[i] && ema[i - 1] <= sma[i - 1] && ema[i] > sma[i] && rsi[i] > 40) {
        markers.push({
          time: candles[i].time,
          position: 'belowBar',
          color: '#2ECC71',
          shape: 'arrowUp',
          text: 'BUY'
        });
      }
  
      // SELL signal
      if (ema[i] && sma[i] && ema[i - 1] >= sma[i - 1] && ema[i] < sma[i] && rsi[i] < 60) {
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
  
  // ========== UPDATE ALL INDICATOR SERIES ==========
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
  
    // Update overlay indicators
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
  
    scheduleResizeCharts();
  }

  // ========== TIMEFRAME RESAMPLING ==========
function resampleCandles(data, tfMinutes) {
  if (!data.length) return [];

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

  // Stop replay if running
  if (replayState !== 'IDLE') {
    stopReplay();
  }

  const prevRange = mainChart.timeScale().getVisibleLogicalRange();

  currentTF = tf;
  allCandles = resampleCandles(raw1mData, tf);

  candleSeries.setData(allCandles);
  updateAllIndicatorSeries();

  if (prevRange) {
    requestAnimationFrame(() => {
      try {
        const newRange = {
          from: Math.max(0, prevRange.from),
          to: Math.min(allCandles.length, prevRange.to)
        };
        mainChart.timeScale().setVisibleLogicalRange(newRange);
      } catch {
        mainChart.timeScale().fitContent();
      }
    });
  } else {
    mainChart.timeScale().fitContent();
  }

  const tfBtn = document.getElementById('tfBtn');
  if (tfBtn) {
    tfBtn.textContent = `${tf}m ▾`;
  }

  document.querySelectorAll('.tf-option').forEach(btn => {
    const btnTf = parseInt(btn.dataset.tf, 10);
    if (btnTf === tf) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });
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

  if (d.getHours() < MARKET_OPEN_HOUR ||
      (d.getHours() === MARKET_OPEN_HOUR && d.getMinutes() < MARKET_OPEN_MIN)) {
    d.setHours(MARKET_OPEN_HOUR, MARKET_OPEN_MIN, 0, 0);
    return d;
  }

  if (d.getHours() > MARKET_CLOSE_HOUR ||
      (d.getHours() === MARKET_CLOSE_HOUR && d.getMinutes() > MARKET_CLOSE_MIN)) {
    d.setDate(d.getDate() + 1);

    while (d.getDay() === 0 || d.getDay() === 6) {
      d.setDate(d.getDate() + 1);
    }

    d.setHours(MARKET_OPEN_HOUR, MARKET_OPEN_MIN, 0, 0);
    return d;
  }

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

// ========== REPLAY FUNCTIONALITY (FIXED) ==========
// KEY FIX: Keep historical candles ALWAYS visible, only add new ones during replay

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
  
    if (!raw1mData || raw1mData.length === 0) {
      alert('No data available. Please load a range first.');
      return false;
    }
  
    // Check if replay start time is within selected range
    if (replayStartTs < rangeStartTs || replayStartTs > rangeEndTs) {
      alert('Replay start time must be within the selected range');
      return false;
    }
  
    // Resample to current timeframe
    const tfData = resampleCandles(raw1mData, currentTF);
  
    // Split into historical (before replay start) and future (to replay)
    replayHistorical = [];
    replayCandles = [];
  
    for (const c of tfData) {
      if (c.time < replayStartTs) {
        replayHistorical.push(c);
      } else if (c.time >= replayStartTs && c.time <= rangeEndTs) {
        replayCandles.push(c);
      }
    }
  
    // FIX: Set allCandles to ONLY historical candles
    // These will stay on screen and new candles will be ADDED to them
    allCandles = [...replayHistorical];
  
    // Update chart with historical candles
    candleSeries.setData(allCandles);
    updateAllIndicatorSeries();
  
    // Clear markers
    candleSeries.setMarkers([]);
  
    // Update time indicator
    updateReplayTimeIndicator();
  
    replayIndex = 0;
    replayState = 'PREPARING';
  
    return true;
  }
  
  function updateReplayTimeIndicator() {
    if (!replayTimeIndicator) return;
  
    if (replayState === 'IDLE') {
      replayTimeIndicator.textContent = '';
      return;
    }
  
    if (replayState === 'PREPARING') {
      if (allCandles.length > 0) {
        const currentCandle = allCandles[allCandles.length - 1];
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
      return;
    }
  
    if (replayState === 'PLAYING' || replayState === 'PAUSED') {
      const currentCandle = allCandles.length > 0 ? allCandles[allCandles.length - 1] : null;
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
  
      const progress = replayCandles.length > 0
        ? Math.round((replayIndex / replayCandles.length) * 100)
        : 0;
  
      replayTimeIndicator.textContent = `${timeStr} (${progress}%)`;
    }
  }
  
  function updateIndicatorsIncremental(newCandle) {
    // Recalculate all indicators (needed for accuracy)
    const indicators = calculateIndicators(allCandles);
    const lastIdx = allCandles.length - 1;
  
    // Update last few values for smooth transitions
    function updateLastFew(series, arr, enabled, count = 5) {
      if (!enabled || !series || !arr) return;
      const start = Math.max(0, arr.length - count);
      for (let i = start; i < arr.length; i++) {
        if (i < allCandles.length && arr[i] !== null && arr[i] !== undefined) {
          series.update({ time: allCandles[i].time, value: arr[i] });
        }
      }
    }
  
    // Update overlay indicators
    if (opts.sma20 && sma20Series) updateLastFew(sma20Series, indicators.sma20, true);
    if (opts.sma50 && sma50Series) updateLastFew(sma50Series, indicators.sma50, true);
    if (opts.ema12 && ema12Series) updateLastFew(ema12Series, indicators.ema12, true);
  
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
  
    // Update RSI
    if (opts.rsi && rsiSeries) {
      updateLastFew(rsiSeries, indicators.rsi, true);
    }
  
    // Update MACD
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
  
    // Update trade signals
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
  
  function startReplay(speed) {
    // If already playing, do nothing
    if (replayTimer || replayState === 'PLAYING') {
      return;
    }
  
    // If paused, resume
    if (replayState === 'PAUSED') {
      replayState = 'PLAYING';
      replaySpeed = speed || replaySpeed;
  
      if (btnPlay) btnPlay.disabled = true;
      if (btnPause) btnPause.disabled = false;
      const btnStop = document.getElementById('btnStop');
      if (btnStop) btnStop.disabled = false;
      if (replayBtn) {
        replayBtn.textContent = 'Replay: PLAYING';
      }
  
      // Ensure chart displays current allCandles (historical + already replayed)
      candleSeries.setData(allCandles);
      updateAllIndicatorSeries();
  
      replayTimer = setInterval(() => {
        if (replayIndex >= replayCandles.length) {
          stopReplay();
          return;
        }
  
        const newCandle = replayCandles[replayIndex];
  
        // CRITICAL FIX: Use update() to ADD the candle, not replace
        candleSeries.update(newCandle);
        allCandles.push(newCandle);
  
        updateIndicatorsIncremental(newCandle);
        mainChart.timeScale().scrollToRealTime();
        updateReplayTimeIndicator();
  
        replayIndex++;
      }, replaySpeed);
  
      return;
    }
  
    // Start fresh - prepare data
    const prepared = prepareReplayData();
    if (!prepared) {
      alert('Failed to prepare replay data. Please check that range and replay start time are set correctly.');
      return;
    }
  
    if (!replayCandles || replayCandles.length === 0) {
      console.warn('No candles to replay - replay will end immediately');
    }
  
    replaySpeed = speed || 800;
    replayState = 'PLAYING';
  
    // Ensure chart shows historical candles and all indicators are updated
    candleSeries.setData(allCandles);
    updateAllIndicatorSeries();
  
    if (btnPlay) btnPlay.disabled = true;
    if (btnPause) btnPause.disabled = false;
    const btnStop = document.getElementById('btnStop');
    if (btnStop) btnStop.disabled = false;
    if (replayBtn) {
      replayBtn.textContent = 'Replay: PLAYING';
      replayBtn.classList.add('active');
    }
  
    // Start replay timer
    replayTimer = setInterval(() => {
      if (!replayCandles || replayIndex >= replayCandles.length) {
        stopReplay();
        return;
      }
  
      const newCandle = replayCandles[replayIndex];
      if (!newCandle) {
        replayIndex++;
        return;
      }
  
      // CRITICAL FIX: Use update() to ADD the candle, not replace
      // This preserves all historical candles and indicators
      candleSeries.update(newCandle);
      allCandles.push(newCandle);
  
      updateIndicatorsIncremental(newCandle);
      mainChart.timeScale().scrollToRealTime();
      updateReplayTimeIndicator();
  
      replayIndex++;
    }, replaySpeed);
  }
  
  function pauseReplay() {
    if (replayState !== 'PLAYING') return;
  
    clearInterval(replayTimer);
    replayTimer = null;
    replayState = 'PAUSED';
  
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
  
    // Reset to show all candles normally (full dataset at current timeframe)
    allCandles = resampleCandles(raw1mData, currentTF);
    replayIndex = 0;
    replayCandles = [];
    replayHistorical = [];
  
    // Update chart with full data
    candleSeries.setData(allCandles);
    updateAllIndicatorSeries();
    mainChart.timeScale().fitContent();
  
    if (btnPlay) btnPlay.disabled = false;
    if (btnPause) btnPause.disabled = true;
    const btnStop = document.getElementById('btnStop');
    if (btnStop) btnStop.disabled = true;
    if (replayBtn) {
      replayBtn.textContent = 'Replay: OFF';
      replayBtn.classList.remove('active');
    }
    if (replayTimeIndicator) {
      replayTimeIndicator.textContent = '';
    }
  
    candleSeries.setMarkers([]);
  }

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
  
    syncTimeScales();
  
    // Lazy-load detection
    mainChart.timeScale().subscribeVisibleLogicalRangeChange(range => {
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
  
  // ========== UI INITIALIZATION ==========
  async function initUI() {
    // Get UI elements
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
  
    // Panel toggles
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
  
    // Set date input ranges
    if (rangeStartInput && rangeEndInput) {
      const dataMinDate = new Date(info.min_ts * 1000);
      const dataMaxDate = new Date(info.max_ts * 1000);
  
      const dataMinDateStr = dataMinDate.toISOString().split('T')[0];
      const dataMaxDateStr = dataMaxDate.toISOString().split('T')[0];
  
      rangeStartInput.min = `${dataMinDateStr}T09:15`;
      rangeStartInput.max = `${dataMaxDateStr}T15:30`;
      rangeEndInput.min = `${dataMinDateStr}T09:15`;
      rangeEndInput.max = `${dataMaxDateStr}T15:30`;
  
      const maxDate = new Date(info.max_ts * 1000);
      const defaultStart = new Date((info.max_ts - 24 * 3600) * 1000);
  
      let clampedStart = clampToMarketHours(defaultStart);
      let clampedEnd = clampToMarketHours(maxDate);
  
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
    ['sma20', 'sma50', 'ema12', 'bb'].forEach(ind => {
      const toggle = document.getElementById(`toggle-${ind}`);
      if (toggle) {
        toggle.addEventListener('change', e => {
          opts[ind] = e.target.checked;
          updateAllIndicatorSeries();
        });
      }
    });
  
    // Pace Pro toggle
    const paceproToggle = document.getElementById('paceproToggle');
    if (paceproToggle) {
      paceproToggle.addEventListener('change', e => {
        opts.pacePro = e.target.checked;
        updateAllIndicatorSeries();
      });
    }
  
    // Trade signals toggle
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
  
    // Timeframe buttons
    document.querySelectorAll('.tf-option').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const tf = parseInt(btn.dataset.tf, 10);
        applyTimeframe(tf);
        const dropdown = btn.closest('.dropdown');
        if (dropdown) {
          const panel = dropdown.querySelector('.dropdown-panel');
          if (panel) panel.style.display = 'none';
        }
      });
    });
  
    // Set initial active timeframe
    const initialTfBtn = document.querySelector('.tf-option[data-tf="1"]');
    if (initialTfBtn) {
      initialTfBtn.classList.add('active');
    }
  
    // Close dropdowns when clicking outside
    document.addEventListener('click', (e) => {
      if (!e.target.closest('.dropdown')) {
        document.querySelectorAll('.dropdown-panel').forEach(panel => {
          if (panel.style.display === 'block') {
            panel.style.display = '';
          }
        });
      }
    });
  
    // Replay start input with validation
    if (replayStartInputEl) {
      replayStartInputEl.addEventListener('change', () => {
        if (replayStartInputEl.value) {
          const d = clampToMarketHours(new Date(replayStartInputEl.value));
  
          const ts = Math.floor(d.getTime() / 1000);
          if (ts < rangeStartTs) {
            d.setTime(rangeStartTs * 1000);
          } else if (ts > rangeEndTs) {
            d.setTime(rangeEndTs * 1000);
          }
  
          replayStartInputEl.value = toLocalInput(d);
  
          if (btnPlay) {
            const finalTs = Math.floor(d.getTime() / 1000);
            btnPlay.disabled = finalTs < rangeStartTs || finalTs > rangeEndTs;
          }
        }
      });
    }
  
    // Replay buttons
    if (btnPlay) {
      btnPlay.onclick = () => {
        try {
          const speedInput = document.getElementById('replaySpeed');
          const speed = speedInput ? Number(speedInput.value) || 800 : 800;
          startReplay(speed);
          setReplayStatus('playing');
        } catch (error) {
          console.error('Error starting replay:', error);
          alert('Error starting replay: ' + error.message);
        }
      };
    }
  
    if (btnPause) {
      btnPause.onclick = () => {
        pauseReplay();
        setReplayStatus('paused');
      };
    }
  
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
  
  // ========== INITIALIZATION ==========
  (async function init() {
    createCharts();
    await initUI();
    await loadLatest();
    setTimeout(resizeCharts, 0);
  })();

  