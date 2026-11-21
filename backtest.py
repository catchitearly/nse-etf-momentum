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

LOOKBACKS = [30, 60, 75, 90, 120]
REBALANCE_PERIODS = [30, 90]
TOP_N_OPTIONS = [3, 5]

MAX_LOOKBACK = max(LOOKBACKS)
INITIAL_CAPITAL = 1_000_000.0  # ₹10,00,000 (10 Lakh)
COST_PER_SIDE = 0.00025        # 0.025% per side = 0.05% round-trip
DOWNLOAD_START = "2019-06-01"
BACKTEST_START = "2020-01-01"
BENCHMARK_TICKER = "NIFTYBEES.NS"
ROLLING_WINDOW = 252           # ~1 trading year


# ==========================================
# 2. DATA & LOOKBACK MATRIX
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

    closes = closes.dropna(how="all").sort_index().ffill(limit=3)
    return closes


def precompute_returns(closes: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Precomputes lookback return matrices for all lookback windows."""
    return {lb: closes.pct_change(periods=lb) for lb in LOOKBACKS}


def get_top_tickers(
    closes: pd.DataFrame,
    ret_matrices: dict[int, pd.DataFrame],
    idx: int,
    lookback: int,
    top_n: int,
) -> list[str]:
    if idx < MAX_LOOKBACK:
        return []

    ret_row = ret_matrices[lookback].iloc[idx]
    eligible = []

    for ticker in closes.columns:
        # Ensure continuous valid data across the full max lookback window up to idx
        window = closes[ticker].iloc[idx - MAX_LOOKBACK : idx + 1]
        if window.isna().any():
            continue
        r_val = ret_row[ticker]
        if pd.notna(r_val):
            eligible.append((ticker, float(r_val)))

    if len(eligible) < top_n:
        return []

    eligible.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in eligible[:top_n]]


# ==========================================
# 3. BACKTEST + ROLLING & YEARLY CONSISTENCY
# ==========================================
def evaluate_combination(
    closes: pd.DataFrame,
    ret_matrices: dict[int, pd.DataFrame],
    bench_eq: pd.Series,
    lookback: int,
    rebalance_days: int,
    top_n: int,
) -> dict:
    bt_dates = closes.index[closes.index >= pd.Timestamp(BACKTEST_START)]
    start_idx = closes.index.get_loc(bt_dates[0])

    cash = INITIAL_CAPITAL
    holdings: dict[str, float] = {}
    total_txn_costs = 0.0
    rebalance_count = 0
    days_since_rebalance = rebalance_days

    eq_points = []
    dates_list = []

    for idx in range(start_idx, len(closes)):
        dt = closes.index[idx]
        prices_today = closes.iloc[idx]

        holdings_val = sum(
            shares * float(prices_today[t])
            for t, shares in holdings.items()
            if pd.notna(prices_today[t])
        )
        portfolio_val = cash + holdings_val

        if days_since_rebalance >= rebalance_days:
            targets = get_top_tickers(closes, ret_matrices, idx, lookback, top_n)
            if len(targets) == top_n:
                target_alloc = portfolio_val / top_n
                step_cost = 0.0
                new_holdings = {}

                # Sell exited ETFs
                for t, shares in holdings.items():
                    px = float(prices_today[t])
                    val = shares * px
                    if t not in targets:
                        cost = val * COST_PER_SIDE
                        step_cost += cost
                        cash += val - cost
                    else:
                        new_holdings[t] = shares

                # Trim overweight ETFs
                for t in targets:
                    px = float(prices_today[t])
                    val = new_holdings.get(t, 0.0) * px
                    if val > target_alloc:
                        sell_val = val - target_alloc
                        cost = sell_val * COST_PER_SIDE
                        step_cost += cost
                        cash += sell_val - cost
                        new_holdings[t] = target_alloc / px

                # Buy new / underweight ETFs
                for t in targets:
                    px = float(prices_today[t])
                    val = new_holdings.get(t, 0.0) * px
                    if val < target_alloc:
                        desired = target_alloc - val
                        affordable = cash / (1.0 + COST_PER_SIDE)
                        buy_val = min(desired, affordable)
                        if buy_val > 1.0:
                            cost = buy_val * COST_PER_SIDE
                            step_cost += cost
                            cash -= buy_val + cost
                            new_holdings[t] = (val + buy_val) / px

                holdings = new_holdings
                total_txn_costs += step_cost
                rebalance_count += 1
                days_since_rebalance = 0

                holdings_val = sum(
                    shares * float(prices_today[t]) for t, shares in holdings.items()
                )
                portfolio_val = cash + holdings_val

        eq_points.append(portfolio_val)
        dates_list.append(dt)
        days_since_rebalance += 1

    eq = pd.Series(eq_points, index=pd.DatetimeIndex(dates_list))
    daily_ret = eq.pct_change().dropna()

    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = ((eq.iloc[-1] / INITIAL_CAPITAL) ** (1.0 / years) - 1.0) * 100.0
    ann_vol = daily_ret.std() * np.sqrt(252) * 100.0
    sharpe = (cagr - 6.0) / ann_vol if ann_vol > 0 else 0.0

    dd_series = ((eq - eq.cummax()) / eq.cummax()) * 100.0
    max_dd = float(dd_series.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    # Rolling 1-year (252d) return & Sharpe consistency
    roll_ret = (eq / eq.shift(ROLLING_WINDOW) - 1.0).dropna() * 100.0
    roll_mean = daily_ret.rolling(ROLLING_WINDOW).mean() * 252
    roll_std = daily_ret.rolling(ROLLING_WINDOW).std() * np.sqrt(252)
    roll_sharpe = ((roll_mean - 0.06) / roll_std.replace(0, np.nan)).dropna()

    bench_aligned = bench_eq.reindex(eq.index)
    bench_roll_ret = (bench_aligned / bench_aligned.shift(ROLLING_WINDOW) - 1.0).dropna() * 100.0
    common_idx = roll_ret.index.intersection(bench_roll_ret.index)
    beat_bench_pct = (
        float((roll_ret.loc[common_idx] > bench_roll_ret.loc[common_idx]).mean() * 100.0)
        if len(common_idx) > 0
        else 0.0
    )

    roll_ret_std = float(roll_ret.std()) if len(roll_ret) > 1 else 0.0
    roll_sharpe_std = float(roll_sharpe.std()) if len(roll_sharpe) > 1 else 0.0
    pos_roll_1y_pct = float((roll_ret > 0).mean() * 100.0) if len(roll_ret) > 0 else 0.0
    worst_roll_1y = float(roll_ret.min()) if len(roll_ret) > 0 else 0.0
    median_roll_1y = float(roll_ret.median()) if len(roll_ret) > 0 else 0.0
    median_roll_sharpe = float(roll_sharpe.median()) if len(roll_sharpe) > 0 else 0.0

    # Calendar-year breakdown
    yearly = []
    for yr, grp in eq.groupby(eq.index.year):
        yr_ret = ((grp.iloc[-1] / grp.iloc[0]) - 1.0) * 100.0
        yr_dd = float((((grp - grp.cummax()) / grp.cummax()) * 100.0).min())
        yr_dret = grp.pct_change().dropna()
        yr_vol = yr_dret.std() * np.sqrt(252) * 100.0
        yr_sharpe = ((yr_ret - 6.0) / yr_vol) if yr_vol > 0 else 0.0
        yearly.append(
            {
                "year": int(yr),
                "ret_pct": round(float(yr_ret), 2),
                "max_dd_pct": round(yr_dd, 2),
                "sharpe": round(float(yr_sharpe), 2),
            }
        )

    yr_rets = [y["ret_pct"] for y in yearly]
    yr_ret_std = float(np.std(yr_rets)) if len(yr_rets) > 1 else 0.0

    combo_id = f"lb{lookback}_rb{rebalance_days}_top{top_n}"
    label = f"{lookback}d LB • {rebalance_days}d Rebal • Top {top_n}"

    return {
        "id": combo_id,
        "label": label,
        "lookback": lookback,
        "rebalance_days": rebalance_days,
        "top_n": top_n,
        "final_value": round(float(eq.iloc[-1]), 2),
        "cagr_pct": round(float(cagr), 2),
        "ann_vol_pct": round(float(ann_vol), 2),
        "sharpe": round(float(sharpe), 2),
        "max_dd_pct": round(max_dd, 2),
        "calmar": round(float(calmar), 2),
        "median_roll_1y_pct": round(median_roll_1y, 2),
        "worst_roll_1y_pct": round(worst_roll_1y, 2),
        "roll_1y_std_pct": round(roll_ret_std, 2),
        "yr_ret_std_pct": round(yr_ret_std, 2),
        "median_roll_sharpe": round(median_roll_sharpe, 2),
        "roll_sharpe_std": round(roll_sharpe_std, 2),
        "pos_roll_1y_pct": round(pos_roll_1y_pct, 1),
        "beat_bench_pct": round(beat_bench_pct, 1),
        "rebalance_count": rebalance_count,
        "total_txn_costs": round(total_txn_costs, 2),
        "yearly": yearly,
        "equity_curve": [
            {"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 2)}
            for d, v in eq.items()
        ],
        "drawdown_curve": [
            {"date": d.strftime("%Y-%m-%d"), "dd": round(float(v), 2)}
            for d, v in dd_series.items()
        ],
        "rolling_1y_curve": [
            {"date": d.strftime("%Y-%m-%d"), "ret": round(float(v), 2)}
            for d, v in roll_ret.items()
        ],
    }


def compute_benchmark_series(closes: pd.DataFrame) -> tuple[pd.Series, dict]:
    bt = closes[BENCHMARK_TICKER].loc[closes.index >= pd.Timestamp(BACKTEST_START)].dropna()
    eq = (bt / bt.iloc[0]) * INITIAL_CAPITAL
    dret = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = ((eq.iloc[-1] / INITIAL_CAPITAL) ** (1.0 / years) - 1.0) * 100.0
    vol = dret.std() * np.sqrt(252) * 100.0
    sharpe = (cagr - 6.0) / vol if vol > 0 else 0.0
    dd = ((eq - eq.cummax()) / eq.cummax()) * 100.0
    roll_ret = (eq / eq.shift(ROLLING_WINDOW) - 1.0).dropna() * 100.0

    yearly = []
    for yr, grp in eq.groupby(eq.index.year):
        yr_ret = ((grp.iloc[-1] / grp.iloc[0]) - 1.0) * 100.0
        yr_dd = float((((grp - grp.cummax()) / grp.cummax()) * 100.0).min())
        yearly.append({"year": int(yr), "ret_pct": round(float(yr_ret), 2), "max_dd_pct": round(yr_dd, 2)})

    return eq, {
        "cagr_pct": round(float(cagr), 2),
        "sharpe": round(float(sharpe), 2),
        "max_dd_pct": round(float(dd.min()), 2),
        "yearly": yearly,
        "equity_curve": [{"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 2)} for d, v in eq.items()],
        "drawdown_curve": [{"date": d.strftime("%Y-%m-%d"), "dd": round(float(v), 2)} for d, v in dd.items()],
        "rolling_1y_curve": [{"date": d.strftime("%Y-%m-%d"), "ret": round(float(v), 2)} for d, v in roll_ret.items()],
    }


def score_consistency_and_smoothness(combos: list[dict]) -> list[dict]:
    """
    Computes a normalized composite score (0-100) rewarding:
    - Higher CAGR & Calmar ratio
    - Lower Max Drawdown
    - Lower rolling 1Y return & rolling Sharpe volatility (consistency)
    - Higher % of rolling 1Y windows beating NIFTYBEES
    """
    def norm_higher(vals):
        arr = np.array(vals, dtype=float)
        span = arr.max() - arr.min()
        return ((arr - arr.min()) / span * 100.0) if span > 0 else np.full_like(arr, 50.0)

    def norm_lower(vals):
        arr = np.array(vals, dtype=float)
        span = arr.max() - arr.min()
        return ((arr.max() - arr) / span * 100.0) if span > 0 else np.full_like(arr, 50.0)

    s_cagr = norm_higher([c["cagr_pct"] for c in combos])
    s_calmar = norm_higher([c["calmar"] for c in combos])
    s_low_dd = norm_lower([abs(c["max_dd_pct"]) for c in combos])
    s_low_roll_std = norm_lower([c["roll_1y_std_pct"] for c in combos])
    s_low_sh_std = norm_lower([c["roll_sharpe_std"] for c in combos])
    s_beat = norm_higher([c["beat_bench_pct"] for c in combos])

    scores = (
        0.20 * s_cagr
        + 0.20 * s_calmar
        + 0.20 * s_low_dd
        + 0.15 * s_low_roll_std
        + 0.15 * s_low_sh_std
        + 0.10 * s_beat
    )

    for c, sc in zip(combos, scores):
        c["smooth_score"] = round(float(sc), 1)

    combos.sort(key=lambda x: x["smooth_score"], reverse=True)
    for rank, c in enumerate(combos, start=1):
        c["smooth_rank"] = rank
    return combos


# ==========================================
# 4. HTML DASHBOARD GENERATOR
# ==========================================
def generate_html(combos: list[dict], benchmark: dict, as_of_date: str):
    os.makedirs("docs", exist_ok=True)
    ist = timezone(timedelta(hours=5, minutes=30))
    run_ts = datetime.now(ist).strftime("%d %b %Y, %I:%M %p IST")

    payload = {
        "combos": combos,
        "benchmark": benchmark,
        "as_of_date": as_of_date,
        "run_ts": run_ts,
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>NSE ETF Momentum — Lookback, Rebalance & Top-N Stability Lab</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg: #0f172a; --card: #1e293b; --border: #334155;
      --text: #f1f5f9; --muted: #94a3b8; --accent: #38bdf8;
      --green: #34d399; --red: #f87171; --amber: #fbbf24;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: system-ui, -apple-system, sans-serif; }}
    body {{ background: var(--bg); color: var(--text); padding: 24px; line-height: 1.5; }}
    .container {{ max-width: 1420px; margin: 0 auto; }}
    header {{ display: flex; flex-wrap: wrap; justify-content: space-between; align-items: baseline; margin-bottom: 24px; border-bottom: 1px solid var(--border); padding-bottom: 16px; }}
    h1 {{ font-size: 1.45rem; font-weight: 700; }}
    .meta {{ color: var(--muted); font-size: 0.85rem; }}
    .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 22px; }}
    .card h2 {{ font-size: 1.05rem; margin-bottom: 12px; color: var(--accent); }}
    .grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(580px, 1fr)); gap: 20px; margin-bottom: 22px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
    th, td {{ padding: 8px 10px; text-align: right; border-bottom: 1px solid var(--border); white-space: nowrap; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    th {{ color: var(--muted); font-weight: 600; background: rgba(15, 23, 42, 0.45); }}
    tr:hover {{ background: rgba(56, 189, 248, 0.07); }}
    .pos {{ color: var(--green); font-weight: 600; }}
    .neg {{ color: var(--red); font-weight: 600; }}
    .score-pill {{ background: rgba(52, 211, 153, 0.16); color: var(--green); padding: 2px 8px; border-radius: 999px; font-weight: 700; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 14px; align-items: center; }}
    select {{ background: var(--bg); color: var(--text); border: 1px solid var(--border); padding: 6px 12px; border-radius: 8px; font-size: 0.85rem; }}
    .chart-box {{ position: relative; height: 320px; width: 100%; }}
  </style>
</head>
<body>
<div class="container">
  <header>
    <div>
      <h1>NSE ETF Rotation: Consistency, Smoothness &amp; Drawdown Matrix</h1>
      <div class="meta">Lookbacks: 30, 60, 75, 90, 120d • Rebalance: 30d &amp; 90d • Top 3 &amp; Top 5 (1/N Equal Weight) • Txn Cost: 0.05% RT</div>
    </div>
    <div class="meta">Data As Of: <strong>{as_of_date}</strong> • Updated: {run_ts}</div>
  </header>

  <div class="card">
    <h2>1. All 20 Combinations Ranked by Smoothness &amp; Consistency Score</h2>
    <div style="overflow-x:auto;">
      <table id="matrixTable">
        <thead>
          <tr>
            <th>#</th>
            <th>Combination</th>
            <th>Smooth Score</th>
            <th>CAGR</th>
            <th>Max DD</th>
            <th>Calmar</th>
            <th>Sharpe</th>
            <th>Med Roll 1Y</th>
            <th>Worst Roll 1Y</th>
            <th>Roll 1Y Std</th>
            <th>Roll Sharpe Std</th>
            <th>Beat NIFTYBEES</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <div class="controls">
    <label for="comboSelect"><strong>Inspect Combination:</strong></label>
    <select id="comboSelect"></select>
  </div>

  <div class="grid-2">
    <div class="card">
      <h2>2. Equity Curve vs. NIFTYBEES (₹10L Start)</h2>
      <div class="chart-box"><canvas id="eqChart"></canvas></div>
    </div>
    <div class="card">
      <h2>3. Rolling 252-Day Return (%) &amp; Drawdown (%)</h2>
      <div class="chart-box"><canvas id="rollChart"></canvas></div>
    </div>
  </div>

  <div class="card">
    <h2 id="yearlyTitle">4. Calendar-Year Return &amp; Drawdown Consistency</h2>
    <div style="overflow-x:auto;">
      <table id="yearlyTable">
        <thead>
          <tr>
            <th>Year</th>
            <th>Strategy Return</th>
            <th>Strategy Max DD</th>
            <th>Strategy Sharpe</th>
            <th>NIFTYBEES Return</th>
            <th>NIFTYBEES Max DD</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
</div>

<script>
const DATA = {json.dumps(payload)};
const fmtPct = (n) => (n >= 0 ? "+" : "") + Number(n).toFixed(2) + "%";

// 1. Matrix Table
const mBody = document.querySelector("#matrixTable tbody");
DATA.combos.forEach(c => {{
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td><strong>#${{c.smooth_rank}}</strong></td>
    <td><strong>${{c.label}}</strong></td>
    <td><span class="score-pill">${{c.smooth_score.toFixed(1)}}</span></td>
    <td class="${{c.cagr_pct >= 0 ? 'pos' : 'neg'}}">${{fmtPct(c.cagr_pct)}}</td>
    <td class="neg">${{c.max_dd_pct.toFixed(2)}}%</td>
    <td>${{c.calmar.toFixed(2)}}</td>
    <td>${{c.sharpe.toFixed(2)}}</td>
    <td class="${{c.median_roll_1y_pct >= 0 ? 'pos' : 'neg'}}">${{fmtPct(c.median_roll_1y_pct)}}</td>
    <td class="${{c.worst_roll_1y_pct >= 0 ? 'pos' : 'neg'}}">${{fmtPct(c.worst_roll_1y_pct)}}</td>
    <td>${{c.roll_1y_std_pct.toFixed(2)}}%</td>
    <td>${{c.roll_sharpe_std.toFixed(2)}}</td>
    <td>${{c.beat_bench_pct.toFixed(1)}}%</td>
  `;
  mBody.appendChild(tr);
}});

// Dropdown
const sel = document.getElementById("comboSelect");
DATA.combos.forEach(c => {{
  const opt = document.createElement("option");
  opt.value = c.id;
  opt.textContent = `#${{c.smooth_rank}} — ${{c.label}} (Score ${{c.smooth_score.toFixed(1)}} | CAGR ${{c.cagr_pct}}% | MaxDD ${{c.max_dd_pct}}%)`;
  sel.appendChild(opt);
}});

let eqChartObj = null;
let rollChartObj = null;

function renderComboDetail(comboId) {{
  const c = DATA.combos.find(x => x.id === comboId) || DATA.combos[0];
  const b = DATA.benchmark;

  const dates = c.equity_curve.map(p => p.date);
  if (eqChartObj) eqChartObj.destroy();
  eqChartObj = new Chart(document.getElementById("eqChart"), {{
    type: "line",
    data: {{
      labels: dates,
      datasets: [
        {{ label: c.label, data: c.equity_curve.map(p => p.value), borderColor: "#34d399", borderWidth: 2, pointRadius: 0 }},
        {{ label: "NIFTYBEES", data: b.equity_curve.map(p => p.value), borderColor: "#94a3b8", borderWidth: 1.5, borderDash: [5,5], pointRadius: 0 }}
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ labels: {{ color: "#f1f5f9" }} }} }},
      scales: {{
        x: {{ ticks: {{ color: "#94a3b8", maxTicksLimit: 8 }}, grid: {{ color: "#1e293b" }} }},
        y: {{ ticks: {{ color: "#94a3b8", callback: v => "₹" + (v/100000).toFixed(1) + "L" }}, grid: {{ color: "#334155" }} }}
      }}
    }}
  }});

  const rollMap = Object.fromEntries(c.rolling_1y_curve.map(p => [p.date, p.ret]));
  const ddMap = Object.fromEntries(c.drawdown_curve.map(p => [p.date, p.dd]));
  if (rollChartObj) rollChartObj.destroy();
  rollChartObj = new Chart(document.getElementById("rollChart"), {{
    type: "line",
    data: {{
      labels: dates,
      datasets: [
        {{ label: "Rolling 1Y Return (%)", data: dates.map(d => rollMap[d] ?? null), borderColor: "#38bdf8", borderWidth: 1.8, pointRadius: 0 }},
        {{ label: "Drawdown (%)", data: dates.map(d => ddMap[d] ?? null), borderColor: "#f87171", borderWidth: 1.5, pointRadius: 0 }}
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ labels: {{ color: "#f1f5f9" }} }} }},
      scales: {{
        x: {{ ticks: {{ color: "#94a3b8", maxTicksLimit: 8 }}, grid: {{ color: "#1e293b" }} }},
        y: {{ ticks: {{ color: "#94a3b8", callback: v => v + "%" }}, grid: {{ color: "#334155" }} }}
      }}
    }}
  }});

  document.getElementById("yearlyTitle").textContent = `4. Calendar-Year Return & Drawdown Consistency — ${{c.label}}`;
  const yBody = document.querySelector("#yearlyTable tbody");
  yBody.innerHTML = "";
  const bByYr = Object.fromEntries(b.yearly.map(y => [y.year, y]));
  c.yearly.forEach(y => {{
    const by = bByYr[y.year] || {{ ret_pct: 0, max_dd_pct: 0 }};
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${{y.year}}</strong></td>
      <td class="${{y.ret_pct >= 0 ? 'pos' : 'neg'}}">${{fmtPct(y.ret_pct)}}</td>
      <td class="neg">${{y.max_dd_pct.toFixed(2)}}%</td>
      <td>${{y.sharpe.toFixed(2)}}</td>
      <td class="${{by.ret_pct >= 0 ? 'pos' : 'neg'}}">${{fmtPct(by.ret_pct)}}</td>
      <td class="neg">${{by.max_dd_pct.toFixed(2)}}%</td>
    `;
    yBody.appendChild(tr);
  }});
}}

sel.addEventListener("change", e => renderComboDetail(e.target.value));
renderComboDetail(DATA.combos[0].id);
</script>
</body>
</html>
"""
    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(html)


def main():
    closes = fetch_prices(list(UNIVERSE.keys()))
    ret_matrices = precompute_returns(closes)
    bench_eq, benchmark = compute_benchmark_series(closes)

    combos = []
    for lb in LOOKBACKS:
        for rb in REBALANCE_PERIODS:
            for top_n in TOP_N_OPTIONS:
                combos.append(evaluate_combination(closes, ret_matrices, bench_eq, lb, rb, top_n))

    scored = score_consistency_and_smoothness(combos)
    as_of_date = closes.index[-1].strftime("%Y-%m-%d")
    generate_html(scored, benchmark, as_of_date)
    print(f"Evaluated {len(scored)} combinations as of {as_of_date}.")


if __name__ == "__main__":
    main()
