import pandas as pd
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

# -------------------------
# HTML template
# -------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>TradingView Style Stock Chart</title>
    <script src="https://unpkg.com/lightweight-charts@4.0.1/dist/lightweight-charts.standalone.production.js"></script>
</head>
<body style="margin:0; background-color:#fff;">
    <div style="padding:10px; color:#000; font-family:Arial, Helvetica, sans-serif; text-align:center;">
        <h2 style="margin:0;">TradingView-Style Chart</h2>
    </div>
    <div id="chart" style="height: 80vh;"></div>
    <div id="volume" style="height: 15vh;"></div>

    <script>
        // Main price chart
        let chart = LightweightCharts.createChart(document.getElementById("chart"), {
            layout: {
                background: { color: '#fff' },
                textColor: '#000',
                fontFamily: 'Arial, Helvetica, sans-serif',
                fontSize: 12,
            },
            grid: {
                vertLines: { color: '#e0e0e0' },
                horzLines: { color: '#e0e0e0' }
            },
            crosshair: {
                mode: LightweightCharts.CrosshairMode.Normal,
            },
            timeScale: {
                borderColor: '#cccccc',
                timeVisible: true,
                secondsVisible: false,
                // Enable drag scroll and mouse wheel zoom:
                handleScroll: true,
                handleScale: true,
            },
            rightPriceScale: {
                borderColor: '#cccccc',
            }
        });

        let candleSeries = chart.addCandlestickSeries({
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderVisible: false,
            wickUpColor: '#26a69a',
            wickDownColor: '#ef5350',
        });

        // Volume chart
        let volumeChart = LightweightCharts.createChart(document.getElementById("volume"), {
            layout: {
                background: { color: '#fff' },
                textColor: '#000',
                fontFamily: 'Arial, Helvetica, sans-serif',
                fontSize: 12,
            },
            grid: {
                vertLines: { color: '#e0e0e0' },
                horzLines: { color: '#e0e0e0' }
            },
            timeScale: {
                borderColor: '#cccccc',
                visible: true,
                handleScroll: true,
                handleScale: true,
            },
            rightPriceScale: {
                visible: false
            }
        });

        let volumeSeries = volumeChart.addHistogramSeries({
            color: '#26a69a',
            priceFormat: { type: 'volume' },
            priceScaleId: '',
            scaleMargins: {
                top: 0.2,
                bottom: 0,
            },
        });

        // Link the time scales for scrollbar sync
        chart.timeScale().subscribeVisibleTimeRangeChange((range) => {
            volumeChart.timeScale().setVisibleRange(range);
        });
        volumeChart.timeScale().subscribeVisibleTimeRangeChange((range) => {
            chart.timeScale().setVisibleRange(range);
        });

        // Load chart data
        function loadData() {
            fetch("/data")
                .then(response => response.json())
                .then(data => {
                    candleSeries.setData(data.candles);
                    volumeSeries.setData(data.volume);

                    if (data.candles.length > 0) {
                        const last = data.candles[data.candles.length - 1];
                        candleSeries.createPriceLine({
                            price: last.close,
                            color: last.close >= last.open ? '#26a69a' : '#ef5350',
                            lineWidth: 2,
                            lineStyle: LightweightCharts.LineStyle.Solid,
                            axisLabelVisible: true,
                            title: 'Last'
                        });
                    }

                    chart.timeScale().fitContent();
                    volumeChart.timeScale().fitContent();
                });
        }

        loadData();
    </script>
</body>
</html>
"""

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/data")
def get_data():
    # df = pd.read_csv("nifty50_intraday.csv")
    df = pd.read_csv("../archive/NIFTY 50_minute.csv")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    candles = []
    volume = []

    for _, row in df.iterrows():
        time_val = int(row["date"].timestamp())
        if time_val <= 0:
            continue

        candles.append({
            "time": time_val,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"])
        })
        volume.append({
            "time": time_val,
            "value": float(row["volume"]),
            "color": '#26a69a' if row["close"] >= row["open"] else '#ef5350'
        })

    return jsonify({"candles": candles, "volume": volume})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
