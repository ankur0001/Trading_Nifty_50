# app.py
import os
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__, template_folder="templates", static_folder="static")

# CONFIG - change CSV_PATH if needed
CSV_PATH = os.environ.get("CSV_PATH", "../archive/NIFTY 50_minute.csv")
DATETIME_COL = "date"
OHLCV_COLS = ["open", "high", "low", "close", "volume"]
INITIAL_LIMIT = 300

# Load CSV
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found at: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# Parse and validate date column
df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], errors="coerce")
df = df.dropna(subset=[DATETIME_COL])

# Ensure OHLCV numeric and drop NaNs
for c in OHLCV_COLS:
    if c not in df.columns:
        raise RuntimeError(f"CSV missing required column: {c}")
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=OHLCV_COLS)

# Sort ascending and compute integer-second timestamp safely
df = df.sort_values(DATETIME_COL).reset_index(drop=True)
df["ts"] = df[DATETIME_COL].view("int64") // 1_000_000_000

MIN_TS = int(df["ts"].min())
MAX_TS = int(df["ts"].max())
MIN_STR = df[DATETIME_COL].min().strftime("%Y-%m-%dT%H:%M")
MAX_STR = df[DATETIME_COL].max().strftime("%Y-%m-%dT%H:%M")


def rows_to_candles(rows: pd.DataFrame):
    """Return list of candles sorted ascending (by ts)."""
    candles = []
    for _, r in rows.iterrows():
        candles.append({
            "time": int(r["ts"]),
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
        })
    return candles


# ---------- frontend route ----------
@app.route("/")
def index():
    return render_template("index.html", initial_limit=INITIAL_LIMIT)


# ---------- API ----------
@app.route("/data-info")
def data_info():
    return jsonify({
        "min_ts": MIN_TS,
        "max_ts": MAX_TS,
        "min_str": MIN_STR,
        "max_str": MAX_STR
    })


@app.route("/data-latest")
def data_latest():
    limit = int(request.args.get("limit", INITIAL_LIMIT))
    rows = df.tail(limit)
    candles = rows_to_candles(rows)
    min_time = int(rows["ts"].min()) if len(rows) else MIN_TS
    max_time = int(rows["ts"].max()) if len(rows) else MAX_TS
    return jsonify({
        "candles": candles,
        "min_time": min_time,
        "max_time": max_time
    })


@app.route("/data-range")
def data_range():
    # Accept ISO strings or unix secs for start/end
    s = request.args.get("start")
    e = request.args.get("end", None)
    limit = int(request.args.get("limit", INITIAL_LIMIT))

    if not s:
        return jsonify({"candles": [], "min_time": MIN_TS, "max_time": MAX_TS})

    # parse start
    try:
        if s.replace('.', '', 1).isdigit():
            start_dt = pd.to_datetime(float(s), unit="s")
        else:
            start_dt = pd.to_datetime(s)
    except Exception:
        start_dt = pd.to_datetime(s, errors="coerce")
    if pd.isna(start_dt):
        return jsonify({"candles": [], "min_time": MIN_TS, "max_time": MAX_TS})

    # parse end or default +24h
    if e:
        try:
            if e.replace('.', '', 1).isdigit():
                end_dt = pd.to_datetime(float(e), unit="s")
            else:
                end_dt = pd.to_datetime(e)
        except Exception:
            end_dt = pd.to_datetime(e, errors="coerce")
        if pd.isna(end_dt):
            end_dt = start_dt + pd.Timedelta(hours=24)
    else:
        end_dt = start_dt + pd.Timedelta(hours=24)

    rows = df[(df[DATETIME_COL] >= start_dt) & (df[DATETIME_COL] <= end_dt)].head(limit)
    candles = rows_to_candles(rows)
    return jsonify({
        "candles": candles,
        "min_time": int(start_dt.timestamp()),
        "max_time": int(end_dt.timestamp())
    })


@app.route("/data-before")
def data_before():
    t = request.args.get("time")
    limit = int(request.args.get("limit", INITIAL_LIMIT))
    if not t:
        return jsonify({"candles": [], "min_time": MIN_TS, "max_time": MAX_TS})
    try:
        if t.replace('.', '', 1).isdigit():
            t_int = int(float(t))
        else:
            t_int = int(pd.to_datetime(t).timestamp())
    except Exception:
        return jsonify({"candles": [], "min_time": MIN_TS, "max_time": MAX_TS})

    rows = df[df["ts"] < t_int].tail(limit)
    candles = rows_to_candles(rows)
    if len(rows):
        min_time = int(rows["ts"].min())
        max_time = int(rows["ts"].max())
    else:
        min_time = t_int
        max_time = t_int
    return jsonify({"candles": candles, "min_time": min_time, "max_time": max_time})


@app.route("/data-after")
def data_after():
    t = request.args.get("time")
    limit = int(request.args.get("limit", INITIAL_LIMIT))
    if not t:
        return jsonify({"candles": [], "min_time": MIN_TS, "max_time": MAX_TS})
    try:
        if t.replace('.', '', 1).isdigit():
            t_int = int(float(t))
        else:
            t_int = int(pd.to_datetime(t).timestamp())
    except Exception:
        return jsonify({"candles": [], "min_time": MIN_TS, "max_time": MAX_TS})

    rows = df[df["ts"] > t_int].head(limit)
    candles = rows_to_candles(rows)
    if len(rows):
        min_time = int(rows["ts"].min())
        max_time = int(rows["ts"].max())
    else:
        min_time = t_int
        max_time = t_int
    return jsonify({"candles": candles, "min_time": min_time, "max_time": max_time})


if __name__ == "__main__":
    print(f"Serving CSV: {CSV_PATH}")
    print(f"Date range: {pd.to_datetime(MIN_TS, unit='s')} -> {pd.to_datetime(MAX_TS, unit='s')}")
    app.run(debug=True, host="0.0.0.0", port=8050)
