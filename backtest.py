import os
import json
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

# ==========================================
# 1. CONFIGURATION
# ==========================================
UNIVERSE = {
    "NIFTYBEES.NS": "Nifty 50 (Large Cap)",
    "JUNIORBEES.NS": "Nifty Next 50",
    "MID150BEES.NS": "Nifty Midcap 150",
    "HDFCSML250.NS": "Nifty Smallcap 250",
    "BANKBEES.NS": "Nifty Bank",
    "PSUBNKBEES.NS": "Nifty PSU Bank",
    "ITBEES.NS": "Nifty IT",
    "PHARMABEES.NS": "Nifty Pharma",
    "AUTOBEES.NS": "Nifty Auto",
    "METALIETF.NS": "Nifty Metal",
    "FMCGIETF.NS": "Nifty FMCG",
    "OILIETF.NS": "Nifty Oil & Gas",
    "MOREALTY.NS": "Nifty Realty",
    "MODEFENCE.NS": "Nifty India Defence",
    "CPSEETF.NS": "CPSE PSU Basket",
    "MON100.NS": "Nasdaq 100 (US Tech)",
    "MASPTOP50.NS": "S&P 500 Top 50 (US)",
    "MAHKTECH.NS": "Hang Seng TECH (HK/China)",
    "GOLDBEES.NS": "Gold BeES",
    "SILVERBEES.NS": "Silver BeES",
    "LTGILTBEES.NS": "8-13 Yr G-Sec Gilt",
    "LIQUIDCASE.NS": "1D Rate Liquid ETF",
}

LOOKBACKS = [30, 60, 90, 120]
INITIAL_CAPITAL = 1_000_000.0  # ₹10,00,000 (10 Lakh)
# 0.025% per side (0.05% round-trip): covers 0.001% ETF STT, 0.015% stamp duty, NSE/SEBI/GST, DP & slippage
COST_PER_SIDE = 0.00025
DOWNLOAD_START = "2019-06-01"   # Buffer prior to Jan 2020 so 120d lookback is ready on Day 1
BACKTEST_START = "2020-01-01"
BENCHMARK_TICKER = "NIFTYBEES.NS"

STRATEGIES = [
    {"id": "top3_30d", "name": "Top 3 • 30-Day Rebalance", "top_n": 3, "rebalance_days": 30},
    {"id": "top3_90d", "name": "Top 3 • 90-Day Rebalance", "top_n": 3, "rebalance_days": 90},
    {"id": "top5_30d", "name": "Top 5 • 30-Day Rebalance", "top_n": 5, "rebalance_days": 30},
    {"id": "top5_90d", "name": "Top 5 • 90-Day Rebalance", "top_n": 5, "rebalance_days": 90},
]


# ==========================================
# 2. DATA INGESTION
# ==========================================
def fetch_prices(tickers: list[str]) -> pd.DataFrame:
    raw = yf.download(
        tickers=tickers,
        start=DOWNLOAD_START,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"].copy()
    else:
        closes = raw[["Close"]].copy()
        closes.columns = tickers

    # Drop completely empty rows (exchange holidays) and forward-fill isolated missing ticks (limit 3 days)
    closes = closes.dropna(how="all").sort_index()
    closes = closes.ffill(limit=3)
    return closes


# ==========================================
# 3. CROSS-SECTIONAL RELATIVE STRENGTH ENGINE
# ==========================================
def compute_rankings(closes: pd.DataFrame, idx: int) -> pd.DataFrame:
    """
    Ranks all ETFs that have at least max(LOOKBACKS) valid trading days up to index `idx`.
    Returns DataFrame sorted by composite cross-sectional rank (best first).
    """
    max_lb = max(LOOKBACKS)
    if idx < max_lb:
        return pd.DataFrame()

    records = []
    for ticker in closes.columns:
        series = closes[ticker].iloc[: idx + 1]
        # Require continuous valid prices over the full 120d window
        window = series.iloc[-(max_lb + 1) :]
        if window.isna().any():
            continue

        curr_price = float(series.iloc[-1])
        row = {
            "ticker": ticker.replace(".NS", ""),
            "yf_ticker": ticker,
            "sector": UNIVERSE.get(ticker, ""),
            "price": round(curr_price, 2),
        }
        rets = []
        for lb in LOOKBACKS:
            past_price = float(series.iloc[-(lb + 1)])
            ret = (curr_price / past_price) - 1.0
            row[f"ret_{lb}d"] = ret
            rets.append(ret)

        row["avg_ret"] = float(np.mean(rets))
        records.append(row)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Cross-sectional rank against the rest of the universe (1 = highest return)
    for lb in LOOKBACKS:
        df[f"rank_{lb}d"] = df[f"ret_{lb}d"].rank(ascending=False, method="min")

    rank_cols = [f"rank_{lb}d" for lb in LOOKBACKS]
    df["composite_rank_score"] = df[rank_cols].mean(axis=1)
    # Sort by lowest average rank first, break ties with highest average return
    df = df.sort_values(
        by=["composite_rank_score", "avg_ret"], ascending=[True, False]
    ).reset_index(drop=True)
    df["final_rank"] = df.index + 1
    return df


# ==========================================
# 4. PORTFOLIO SIMULATION (1/N + TXN COSTS)
# ==========================================
def run_backtest(closes: pd.DataFrame, top_n: int, rebalance_days: int) -> dict:
    bt_dates = closes.index[closes.index >= pd.Timestamp(BACKTEST_START)]
    start_idx = closes.index.get_loc(bt_dates[0])

    cash = INITIAL_CAPITAL
    holdings: dict[str, float] = {}  # yf_ticker -> shares held
    total_txn_costs = 0.0
    rebalance_count = 0

    equity_curve = []
    rebalance_logs = []

    days_since_rebalance = rebalance_days  # Trigger initial allocation on Day 0

    for idx in range(start_idx, len(closes)):
        dt = closes.index[idx]
        prices_today = closes.iloc[idx]

        # Mark-to-market portfolio value before rebalancing
        holdings_val = sum(
            shares * float(prices_today[t])
            for t, shares in holdings.items()
            if pd.notna(prices_today[t])
        )
        portfolio_val = cash + holdings_val

        # Rebalance check (every `rebalance_days` trading days)
        if days_since_rebalance >= rebalance_days:
            rankings = compute_rankings(closes, idx)
            if len(rankings) >= top_n:
                target_tickers = rankings.head(top_n)["yf_ticker"].tolist()
                target_weight = 1.0 / top_n
                target_alloc = portfolio_val * target_weight

                # Calculate sells first (positions removed or overweight)
                step_cost = 0.0
                new_holdings = {}

                # Liquidate exited positions
                for t, shares in holdings.items():
                    px = float(prices_today[t])
                    curr_val = shares * px
                    if t not in target_tickers:
                        cost = curr_val * COST_PER_SIDE
                        step_cost += cost
                        cash += curr_val - cost
                    else:
                        new_holdings[t] = shares

                # Rebalance target positions to equal weight (1/N)
                # First trim overweight positions
                for t in target_tickers:
                    px = float(prices_today[t])
                    curr_val = new_holdings.get(t, 0.0) * px
                    if curr_val > target_alloc:
                        sell_val = curr_val - target_alloc
                        cost = sell_val * COST_PER_SIDE
                        step_cost += cost
                        cash += sell_val - cost
                        new_holdings[t] = target_alloc / px

                # Then buy new/underweight positions from available cash
                for t in target_tickers:
                    px = float(prices_today[t])
                    curr_val = new_holdings.get(t, 0.0) * px
                    if curr_val < target_alloc:
                        desired_buy = target_alloc - curr_val
                        # Ensure cash covers both principal and buy-side cost
                        max_affordable = cash / (1.0 + COST_PER_SIDE)
                        buy_val = min(desired_buy, max_affordable)
                        if buy_val > 1.0:
                            cost = buy_val * COST_PER_SIDE
                            step_cost += cost
                            cash -= buy_val + cost
                            new_holdings[t] = (curr_val + buy_val) / px

                holdings = new_holdings
                total_txn_costs += step_cost
                rebalance_count += 1
                days_since_rebalance = 0

                # Recompute post-rebalance equity
                holdings_val = sum(
                    shares * float(prices_today[t]) for t, shares in holdings.items()
                )
                portfolio_val = cash + holdings_val

                rebalance_logs.append(
                    {
                        "date": dt.strftime("%Y-%m-%d"),
                        "selected": [t.replace(".NS", "") for t in target_tickers],
                        "portfolio_value": round(portfolio_val, 2),
                        "txn_cost": round(step_cost, 2),
                    }
                )

        equity_curve.append({"date": dt.strftime("%Y-%m-%d"), "value": round(portfolio_val, 2)})
        days_since_rebalance += 1

    # Performance metrics
    eq_series = pd.Series(
        [x["value"] for x in equity_curve],
        index=pd.to_datetime([x["date"] for x in equity_curve]),
    )
    daily_rets = eq_series.pct_change().dropna()
    years = (eq_series.index[-1] - eq_series.index[0]).days / 365.25
    cagr = ((eq_series.iloc[-1] / INITIAL_CAPITAL) ** (1.0 / years) - 1.0) * 100
    vol = daily_rets.std() * np.sqrt(252) * 100
    sharpe = (cagr - 6.0) / vol if vol > 0 else 0.0  # Assuming 6% risk-free rate
    rolling_max = eq_series.cummax()
    drawdown = ((eq_series - rolling_max) / rolling_max) * 100
    max_dd = float(drawdown.min())

    return {
        "final_value": round(float(eq_series.iloc[-1]), 2),
        "total_return_pct": round(((eq_series.iloc[-1] / INITIAL_CAPITAL) - 1.0) * 100, 2),
        "cagr_pct": round(float(cagr), 2),
        "volatility_pct": round(float(vol), 2),
        "sharpe": round(float(sharpe), 2),
        "max_dd_pct": round(max_dd, 2),
        "total_txn_costs": round(total_txn_costs, 2),
        "rebalance_count": rebalance_count,
        "equity_curve": equity_curve,
        "drawdown_curve": [
            {"date": d.strftime("%Y-%m-%d"), "dd": round(float(v), 2)}
            for d, v in drawdown.items()
        ],
        "rebalance_logs": rebalance_logs,
    }


def compute_benchmark(closes: pd.DataFrame) -> dict:
    bt = closes[BENCHMARK_TICKER].loc[closes.index >= pd.Timestamp(BACKTEST_START)].dropna()
    norm = (bt / bt.iloc[0]) * INITIAL_CAPITAL
    daily_rets = norm.pct_change().dropna()
    years = (norm.index[-1] - norm.index[0]).days / 365.25
    cagr = ((norm.iloc[-1] / INITIAL_CAPITAL) ** (1.0 / years) - 1.0) * 100
    vol = daily_rets.std() * np.sqrt(252) * 100
    sharpe = (cagr - 6.0) / vol if vol > 0 else 0.0
    drawdown = ((norm - norm.cummax()) / norm.cummax()) * 100

    return {
        "final_value": round(float(norm.iloc[-1]), 2),
        "total_return_pct": round(((norm.iloc[-1] / INITIAL_CAPITAL) - 1.0) * 100, 2),
        "cagr_pct": round(float(cagr), 2),
        "volatility_pct": round(float(vol), 2),
        "sharpe": round(float(sharpe), 2),
        "max_dd_pct": round(float(drawdown.min()), 2),
        "equity_curve": [
            {"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 2)}
            for d, v in norm.items()
        ],
        "drawdown_curve": [
            {"date": d.strftime("%Y-%m-%d"), "dd": round(float(v), 2)}
            for d, v in drawdown.items()
        ],
    }


# ==========================================
# 5. HTML DASHBOARD GENERATOR (GITHUB PAGES)
# ==========================================
def generate_html(results: dict, benchmark: dict, latest_rankings: pd.DataFrame, as_of_date: str):
    os.makedirs("docs", exist_ok=True)
    ist = timezone(timedelta(hours=5, minutes=30))
    run_ts = datetime.now(ist).strftime("%d %b %Y, %I:%M %p IST")

    payload = {
        "strategies": results,
        "benchmark": benchmark,
        "latest_rankings": latest_rankings.to_dict(orient="records"),
        "as_of_date": as_of_date,
        "run_ts": run_ts,
        "initial_capital": INITIAL_CAPITAL,
    }

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>NSE ETF Relative Strength Momentum Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg: #0f172a; --card: #1e293b; --border: #334155;
      --text: #f1f5f9; --muted: #94a3b8; --accent: #38bdf8;
      --green: #34d399; --red: #f87171; --amber: #fbbf24;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: system-ui, -apple-system, sans-serif; }}
    body {{ background: var(--bg); color: var(--text); padding: 24px; line-height: 1.5; }}
    .container {{ max-width: 1360px; margin: 0 auto; }}
    header {{ display: flex; flex-wrap: wrap; justify-content: space-between; align-items: baseline; margin-bottom: 24px; border-bottom: 1px solid var(--border); padding-bottom: 16px; }}
    h1 {{ font-size: 1.5rem; font-weight: 700; }}
    .meta {{ color: var(--muted); font-size: 0.875rem; }}
    .grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr)); gap: 20px; margin-bottom: 24px; }}
    .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 24px; }}
    .card h2 {{ font-size: 1.1rem; margin-bottom: 14px; color: var(--accent); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; }}
    th, td {{ padding: 10px 12px; text-align: right; border-bottom: 1px solid var(--border); }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    th {{ color: var(--muted); font-weight: 600; background: rgba(15, 23, 42, 0.4); }}
    tr:hover {{ background: rgba(56, 189, 248, 0.06); }}
    .pos {{ color: var(--green); font-weight: 600; }}
    .neg {{ color: var(--red); font-weight: 600; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.75rem; font-weight: 700; }}
    .badge-top3 {{ background: rgba(52, 211, 153, 0.18); color: var(--green); border: 1px solid rgba(52, 211, 153, 0.4); }}
    .badge-top5 {{ background: rgba(56, 189, 248, 0.18); color: var(--accent); border: 1px solid rgba(56, 189, 248, 0.4); }}
    .controls {{ display: flex; gap: 12px; margin-bottom: 14px; align-items: center; }}
    select {{ background: var(--bg); color: var(--text); border: 1px solid var(--border); padding: 6px 12px; border-radius: 8px; font-size: 0.875rem; }}
    .chart-box {{ position: relative; height: 340px; width: 100%; }}
  </style>
</head>
<body>
<div class="container">
  <header>
    <div>
      <h1>NSE ETF Cross-Sectional Relative Strength Rotation</h1>
      <div class="meta">Lookbacks: 30d / 60d / 90d / 120d • Equal Weight (1/N) • ₹10,00,000 Start • Includes Delivery Txn Costs (0.05% RT)</div>
    </div>
    <div class="meta">Market Data As Of: <strong>{as_of_date}</strong> • Updated: {run_ts}</div>
  </header>

  <div class="card">
    <h2>1. Backtest Performance Summary (Jan 2020 – {as_of_date})</h2>
    <div style="overflow-x:auto;">
      <table id="summaryTable">
        <thead>
          <tr>
            <th>Strategy</th>
            <th>Portfolio Size</th>
            <th>Final Value (₹)</th>
            <th>Total Return</th>
            <th>CAGR</th>
            <th>Sharpe</th>
            <th>Max Drawdown</th>
            <th>Rebalances</th>
            <th>Total Txn Cost (₹)</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <h2>2. Equity Curve (₹10 Lakh Base)</h2>
      <div class="chart-box"><canvas id="equityChart"></canvas></div>
    </div>
    <div class="card">
      <h2>3. Underwater Drawdown (%)</h2>
      <div class="chart-box"><canvas id="ddChart"></canvas></div>
    </div>
  </div>

  <div class="card">
    <h2>4. Live Cross-Sectional Relative Strength Rankings (As of {as_of_date})</h2>
    <div style="overflow-x:auto;">
      <table id="rankingTable">
        <thead>
          <tr>
            <th>Rank</th>
            <th>ETF Ticker</th>
            <th>Category / Sector</th>
            <th>Close (₹)</th>
            <th>30d Ret (Rank)</th>
            <th>60d Ret (Rank)</th>
            <th>90d Ret (Rank)</th>
            <th>120d Ret (Rank)</th>
            <th>Avg Rank Score</th>
            <th>Signal</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <div class="card">
    <h2>5. Historical Rebalance Log</h2>
    <div class="controls">
      <label for="strategySelect">Select Strategy:</label>
      <select id="strategySelect"></select>
    </div>
    <div style="overflow-x:auto; max-height: 420px;">
      <table id="logTable">
        <thead>
          <tr>
            <th>Rebalance Date</th>
            <th>Selected Equal-Weight ETFs (1/N)</th>
            <th>Portfolio Value (₹)</th>
            <th>Txn Cost Paid (₹)</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
</div>

<script>
const DATA = {json.dumps(payload)};
const COLORS = {{
  top3_30d: "#34d399",
  top3_90d: "#38bdf8",
  top5_30d: "#fbbf24",
  top5_90d: "#c084fc",
  benchmark: "#94a3b8"
}};

const fmtINR = (n) => "₹" + Number(n).toLocaleString("en-IN", {{ maximumFractionDigits: 0 }});
const fmtPct = (n) => (n >= 0 ? "+" : "") + Number(n).toFixed(2) + "%";

// 1. Populate Summary Table
const summaryBody = document.querySelector("#summaryTable tbody");
Object.entries(DATA.strategies).forEach(([id, s]) => {{
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td><strong>${{s.name}}</strong></td>
    <td>1/${{s.top_n}} (${{(100/s.top_n).toFixed(1)}}% each)</td>
    <td><strong>${{fmtINR(s.final_value)}}</strong></td>
    <td class="${{s.total_return_pct >= 0 ? 'pos' : 'neg'}}">${{fmtPct(s.total_return_pct)}}</td>
    <td class="${{s.cagr_pct >= 0 ? 'pos' : 'neg'}}">${{fmtPct(s.cagr_pct)}}</td>
    <td>${{s.sharpe.toFixed(2)}}</td>
    <td class="neg">${{s.max_dd_pct.toFixed(2)}}%</td>
    <td>${{s.rebalance_count}}</td>
    <td>${{fmtINR(s.total_txn_costs)}}</td>
  `;
  summaryBody.appendChild(tr);
}});

// Add Benchmark Row
const b = DATA.benchmark;
const bRow = document.createElement("tr");
bRow.innerHTML = `
  <td><em>Benchmark: NIFTYBEES (Buy & Hold)</em></td>
  <td>100%</td>
  <td>${{fmtINR(b.final_value)}}</td>
  <td class="${{b.total_return_pct >= 0 ? 'pos' : 'neg'}}">${{fmtPct(b.total_return_pct)}}</td>
  <td class="${{b.cagr_pct >= 0 ? 'pos' : 'neg'}}">${{fmtPct(b.cagr_pct)}}</td>
  <td>${{b.sharpe.toFixed(2)}}</td>
  <td class="neg">${{b.max_dd_pct.toFixed(2)}}%</td>
  <td>0</td>
  <td>₹0</td>
`;
summaryBody.appendChild(bRow);

// 2. Render Charts
const labels = DATA.benchmark.equity_curve.map(p => p.date);
const eqDatasets = Object.entries(DATA.strategies).map(([id, s]) => ({{
  label: s.name,
  data: s.equity_curve.map(p => p.value),
  borderColor: COLORS[id],
  borderWidth: 2,
  pointRadius: 0,
  tension: 0.1
}}));
eqDatasets.push({{
  label: "NIFTYBEES Benchmark",
  data: DATA.benchmark.equity_curve.map(p => p.value),
  borderColor: COLORS.benchmark,
  borderWidth: 1.5,
  borderDash: [5, 5],
  pointRadius: 0
}});

new Chart(document.getElementById("equityChart"), {{
  type: "line",
  data: {{ labels, datasets: eqDatasets }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    interaction: {{ mode: "index", intersect: false }},
    plugins: {{ legend: {{ labels: {{ color: "#f1f5f9" }} }} }},
    scales: {{
      x: {{ ticks: {{ color: "#94a3b8", maxTicksLimit: 8 }}, grid: {{ color: "#1e293b" }} }},
      y: {{ ticks: {{ color: "#94a3b8", callback: v => "₹" + (v/100000).toFixed(1) + "L" }}, grid: {{ color: "#334155" }} }}
    }}
  }}
}});

const ddDatasets = Object.entries(DATA.strategies).map(([id, s]) => ({{
  label: s.name,
  data: s.drawdown_curve.map(p => p.dd),
  borderColor: COLORS[id],
  borderWidth: 1.5,
  pointRadius: 0
}}));
new Chart(document.getElementById("ddChart"), {{
  type: "line",
  data: {{ labels, datasets: ddDatasets }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    interaction: {{ mode: "index", intersect: false }},
    plugins: {{ legend: {{ labels: {{ color: "#f1f5f9" }} }} }},
    scales: {{
      x: {{ ticks: {{ color: "#94a3b8", maxTicksLimit: 8 }}, grid: {{ color: "#1e293b" }} }},
      y: {{ ticks: {{ color: "#94a3b8", callback: v => v + "%" }}, grid: {{ color: "#334155" }} }}
    }}
  }}
}});

// 3. Populate Live Ranking Table
const rankBody = document.querySelector("#rankingTable tbody");
DATA.latest_rankings.forEach(r => {{
  const cell = (ret, rk) => `<span class="${{ret >= 0 ? 'pos' : 'neg'}}">${{(ret*100).toFixed(1)}}%</span> <span style="color:#94a3b8">(#${{rk}})</span>`;
  let badge = "";
  if (r.final_rank <= 3) badge = `<span class="badge badge-top3">TOP 3 & TOP 5</span>`;
  else if (r.final_rank <= 5) badge = `<span class="badge badge-top5">TOP 5</span>`;

  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td><strong>#${{r.final_rank}}</strong></td>
    <td><strong>${{r.ticker}}</strong></td>
    <td>${{r.sector}}</td>
    <td>₹${{r.price.toFixed(2)}}</td>
    <td>${{cell(r.ret_30d, r.rank_30d)}}</td>
    <td>${{cell(r.ret_60d, r.rank_60d)}}</td>
    <td>${{cell(r.ret_90d, r.rank_90d)}}</td>
    <td>${{cell(r.ret_120d, r.rank_120d)}}</td>
    <td><strong>${{r.composite_rank_score.toFixed(2)}}</strong></td>
    <td>${{badge}}</td>
  `;
  rankBody.appendChild(tr);
}});

// 4. Rebalance Logs Selector
const select = document.getElementById("strategySelect");
const logBody = document.querySelector("#logTable tbody");
Object.entries(DATA.strategies).forEach(([id, s]) => {{
  const opt = document.createElement("option");
  opt.value = id; opt.textContent = s.name;
  select.appendChild(opt);
}});

function renderLogs(id) {{
  logBody.innerHTML = "";
  const logs = [...DATA.strategies[id].rebalance_logs].reverse();
  logs.forEach(l => {{
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${{l.date}}</td>
      <td><strong>${{l.selected.join(", ")}}</strong></td>
      <td>${{fmtINR(l.portfolio_value)}}</td>
      <td>${{fmtINR(l.txn_cost)}}</td>
    `;
    logBody.appendChild(tr);
  }});
}}
select.addEventListener("change", (e) => renderLogs(e.target.value));
renderLogs("top3_30d");
</script>
</body>
</html>
"""
    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(html_template)


def main():
    tickers = list(UNIVERSE.keys())
    closes = fetch_prices(tickers)

    results = {}
    for cfg in STRATEGIES:
        res = run_backtest(closes, top_n=cfg["top_n"], rebalance_days=cfg["rebalance_days"])
        res["name"] = cfg["name"]
        res["top_n"] = cfg["top_n"]
        results[cfg["id"]] = res

    benchmark = compute_benchmark(closes)
    latest_idx = len(closes) - 1
    latest_rankings = compute_rankings(closes, latest_idx)
    as_of_date = closes.index[latest_idx].strftime("%Y-%m-%d")

    generate_html(results, benchmark, latest_rankings, as_of_date)
    print(f"Generated docs/index.html successfully as of {as_of_date}.")


if __name__ == "__main__":
    main()
