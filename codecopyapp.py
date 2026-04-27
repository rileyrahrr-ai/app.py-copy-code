import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Final Project Dashboard", layout="wide")


# =====================================================================
# SAFE DOWNLOAD FUNCTION (protects against yfinance rate limits)
# =====================================================================
def safe_download(ticker, period="6mo", interval="1d"):
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            threads=False
        )
        if df is None or df.empty:
            return None
        return df
    except:
        return None


# =====================================================================
# Emoji helper functions (Trend, RSI, Volatility)
# =====================================================================
def trend_emoji(tr):
    if tr == "Strong Uptrend":
        return "🟢📈 Strong Uptrend"
    elif tr == "Strong Downtrend":
        return "🔴📉 Strong Downtrend"
    else:
        return "🟡〰️ Mixed Trend"


def rsi_emoji(sig):
    if "Overbought" in sig:
        return "🔴🔥 Overbought (Sell Signal)"
    elif "Oversold" in sig:
        return "🟢💎 Oversold (Buy Signal)"
    else:
        return "🟡 Neutral"


def vol_emoji(level):
    if level == "High":
        return "🔴 High Volatility"
    elif level == "Medium":
        return "🟡 Medium Volatility"
    else:
        return "🟢 Low Volatility"


# =====================================================================
# CSS: Chart borders (green/yellow)
# =====================================================================
st.markdown("""
<style>
.chart-box-green {
    border: 5px solid #00cc44;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 25px;
}
.chart-box-yellow {
    border: 5px solid #ffeb3b;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)


# =====================================================================
# SIDEBAR INPUTS + BASIC VALIDATION
# =====================================================================
st.sidebar.title("Input Settings")

stock = st.sidebar.text_input("Stock Ticker:", "AAPL").upper()

portfolio_raw = st.sidebar.text_input(
    "Portfolio (Enter 5 tickers):",
    "AAPL MSFT AMZN GOOG NVDA"
)
portfolio = portfolio_raw.upper().split()

# Validate tickers (letters only, max length 6)
invalid = [t for t in [stock] + portfolio if not t.isalpha() or len(t) > 6]
if invalid:
    st.sidebar.error(f"⚠ Invalid tickers: {', '.join(invalid)}")


# =====================================================================
# MAIN TITLE
# =====================================================================
st.title("📊 Final Project Financial Dashboard")


# =====================================================================
# PART 1 — INDIVIDUAL STOCK ANALYSIS
# =====================================================================
st.header("Part 1: Individual Stock Analysis")

raw = yf.download(stock, period="6mo", interval="1d", progress=False)

if raw is None or raw.empty:
    st.error(
        f"❌ Unable to retrieve stock data for **{stock}**.\n"
        "Yahoo Finance may be rate-limiting your request. Please try again shortly."
    )
else:
    # Handle MultiIndex: ("Close", "AAPL")
    if isinstance(raw.columns, pd.MultiIndex):
        close_candidates = [col for col in raw.columns if col[0] == "Close"]
        if close_candidates:
            close_series = raw[close_candidates[0]]
        else:
            st.error("❌ No usable 'Close' price found.")
            st.stop()
    else:
        if "Close" not in raw.columns:
            st.error("❌ No 'Close' column in the downloaded data.")
            st.stop()
        close_series = raw["Close"]

    close_series = close_series.astype(float)

    data = pd.DataFrame({"Close": close_series})
    data["20MA"] = data["Close"].rolling(20).mean()
    data["50MA"] = data["Close"].rolling(50).mean()

    if len(data) < 60:
        st.error("⚠ Not enough data (need at least 60 days).")
        st.stop()

    price = float(data["Close"].iloc[-1])
    ma20 = float(data["20MA"].iloc[-1]) if not pd.isna(data["20MA"].iloc[-1]) else None
    ma50 = float(data["50MA"].iloc[-1]) if not pd.isna(data["50MA"].iloc[-1]) else None

    if ma20 is None or ma50 is None:
        trend = "Mixed Trend"
    else:
        if price > ma20 and ma20 > ma50:
            trend = "Strong Uptrend"
        elif price < ma20 and ma20 < ma50:
            trend = "Strong Downtrend"
        else:
            trend = "Mixed Trend"

    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))
    rsi_val = float(data["RSI"].iloc[-1]) if not pd.isna(data["RSI"].iloc[-1]) else None

    if rsi_val is None:
        rsi_sig = "Neutral"
    elif rsi_val > 70:
        rsi_sig = "Overbought (Sell Signal)"
    elif rsi_val < 30:
        rsi_sig = "Oversold (Buy Signal)"
    else:
        rsi_sig = "Neutral"

    data["Return"] = data["Close"].pct_change()
    vol_vals = data["Return"].rolling(20).std() * np.sqrt(252)
    vol_val = float(vol_vals.iloc[-1]) if not pd.isna(vol_vals.iloc[-1]) else None

    if vol_val is None:
        vol_class = "Medium"
    elif vol_val > 0.40:
        vol_class = "High"
    elif vol_val >= 0.25:
        vol_class = "Medium"
    else:
        vol_class = "Low"

    st.subheader(f"Results for {stock}")
    st.write(f"**Trend:** {trend_emoji(trend)}")
    st.write(f"**RSI:** {rsi_emoji(rsi_sig)} — {rsi_val:.2f}" if rsi_val else "RSI unavailable")
    st.write(f"**Volatility:** {vol_emoji(vol_class)} — {vol_val:.2%}" if vol_val else "Volatility unavailable")

    st.markdown('<div class="chart-box-green">', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data["Close"], label="Close", color="white")
    ax.plot(data["20MA"], label="20MA", color="cyan")
    ax.plot(data["50MA"], label="50MA", color="magenta")
    ax.legend()
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# PART 2 — PORTFOLIO PERFORMANCE
# =====================================================================
st.header("Part 2: Portfolio Performance Dashboard")

if len(portfolio) != 5:
    st.error("❌ Please enter exactly 5 tickers for the portfolio.")
else:
    prices = yf.download(portfolio, period="1y", progress=False)

    if prices is None or prices.empty:
        st.error("❌ Unable to load portfolio data. Try again later.")
        st.stop()

    # Handle MultiIndex price columns
    if isinstance(prices.columns, pd.MultiIndex):
        if ("Adj Close", portfolio[0]) in prices.columns:
            prices_used = prices["Adj Close"]
        elif ("Close", portfolio[0]) in prices.columns:
            prices_used = prices["Close"]
        else:
            st.error("❌ Missing price columns in portfolio data.")
            st.stop()
    else:
        if "Adj Close" in prices.columns:
            prices_used = prices["Adj Close"]
        elif "Close" in prices.columns:
            prices_used = prices["Close"]
        else:
            st.error("❌ No valid price columns found.")
            st.stop()

    bench = safe_download("SPY", period="1y")
    if bench is None:
        st.error("❌ SPY benchmark failed to load.")
        st.stop()

    if "Adj Close" in bench.columns:
        bench_used = bench["Adj Close"]
    elif "Close" in bench.columns:
        bench_used = bench["Close"]
    else:
        st.error("❌ SPY missing price columns.")
        st.stop()

    returns = prices_used.pct_change().dropna()
    weights = np.array([1/5] * 5)

    portfolio_returns = (returns * weights).sum(axis=1).astype(float)

    raw_bench_returns = bench_used.pct_change().dropna()
    if isinstance(raw_bench_returns, pd.DataFrame):
        benchmark_returns = raw_bench_returns.iloc[:, 0]
    else:
        benchmark_returns = raw_bench_returns

    benchmark_returns = benchmark_returns.astype(float)

    total_return = float((1 + portfolio_returns).prod() - 1)
    bench_return = float((1 + benchmark_returns).prod() - 1)
    outperf = float(total_return - bench_return)
    vol = float(portfolio_returns.std() * np.sqrt(252))
    sharpe = float((portfolio_returns.mean() * 252) /
                   (portfolio_returns.std() * np.sqrt(252)))

    st.subheader("Portfolio Metrics")
    st.write(f"**Total Return:** 🟦 {total_return:.2%}")
    st.write(f"**SPY Return:** 🟧 {bench_return:.2%}")
    st.write(f"**Outperformance:** {'🟢+' if outperf > 0 else '🔴'} {outperf:.2%}")
    st.write(f"**Volatility:** {vol_emoji('High' if vol > 0.40 else 'Medium' if vol >= 0.25 else 'Low')} — {vol:.2%}")
    st.write(f"**Sharpe Ratio:** ⭐ {sharpe:.2f}")

    st.markdown('<div class="chart-box-yellow">', unsafe_allow_html=True)
    cumulative = pd.DataFrame({
        "Portfolio": (1 + portfolio_returns).cumprod(),
        "SPY": (1 + benchmark_returns).cumprod()
    })
    st.line_chart(cumulative)
    st.markdown("</div>", unsafe_allow_html=True)
