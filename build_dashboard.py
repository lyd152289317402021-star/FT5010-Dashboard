"""
Run backtest and generate a standalone HTML dashboard.
Usage:  python build_dashboard.py
Output: index.html (open in browser or deploy to GitHub Pages)
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

try:
    import yfinance as yf
except ImportError:
    raise ImportError("pip install yfinance")

# ─────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────
PAIRS = ["EURUSD=X", "GBPUSD=X", "JPY=X", "CHF=X", "AUDUSD=X", "CAD=X", "NZDUSD=X"]
PAIR_NAMES = {
    "EURUSD=X": "EUR/USD", "GBPUSD=X": "GBP/USD", "JPY=X": "USD/JPY",
    "CHF=X": "USD/CHF", "AUDUSD=X": "AUD/USD", "CAD=X": "USD/CAD", "NZDUSD=X": "NZD/USD",
}
USD_BASE_PAIRS = {"JPY=X", "CHF=X", "CAD=X"}
HOURS_PER_YEAR_FX = 6240

FAST_WINDOW = 20; SLOW_WINDOW = 50; TREND_WINDOW = 200
RSI_WINDOW = 14; ADX_WINDOW = 14; ADX_THRESHOLD = 25
RSI_LONG_THRESH = 55; RSI_SHORT_THRESH = 45
RSI_EXIT_LONG = 75; RSI_EXIT_SHORT = 25
SESSION_START_UTC = 7; SESSION_END_UTC = 17
INITIAL_CAPITAL = 10_000; RISK_PER_TRADE = 0.01
SPREAD_PIPS = 2; ATR_STOP_MULT = 2.5; TRAIL_ATR_MULT = 2.0
LEVERAGE = 50; RISK_FREE_RATE = 0.04; TRAIN_RATIO = 0.70
LOOKBACK_DAYS = 700

# ─────────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────────
@dataclass
class Trade:
    pair: str; direction: int; entry_time: pd.Timestamp; entry_price: float
    exit_time: pd.Timestamp; exit_price: float; units: float
    pnl_pips: float = 0.0; pnl_usd: float = 0.0; exit_reason: str = ""

@dataclass
class Position:
    pair: str; direction: int; entry_time: pd.Timestamp; entry_price: float
    units: float; stop_loss: float; atr_at_entry: float; trailing_stop: float = 0.0

@dataclass
class Portfolio:
    cash: float
    positions: Dict[str, Optional[Position]] = field(default_factory=dict)
    trade_log: List[Trade] = field(default_factory=list)
    equity_curve: List[Tuple] = field(default_factory=list)
    def equity(self, prices):
        total = self.cash
        for pair, pos in self.positions.items():
            if pos is not None and pair in prices:
                total += _unrealised_pnl(pos, prices[pair])
        return total

# ─────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────
def _pip_size(pair): return 0.01 if "JPY" in pair.upper() else 0.0001
def _spread_cost(pair, direction, spread_pips): return direction * spread_pips * _pip_size(pair)

def _pnl_usd(pair, pnl_pips, pip, units, exit_price):
    raw = pnl_pips * pip * units
    return raw / exit_price if pair in USD_BASE_PAIRS and exit_price > 0 else raw

def _unrealised_pnl(pos, current_price):
    pip = _pip_size(pos.pair)
    pp = (current_price - pos.entry_price) * pos.direction / pip
    return _pnl_usd(pos.pair, pp, pip, pos.units, current_price)

def _compute_units(equity, risk_frac, atr, stop_pips, pip, leverage, price, pair):
    risk_usd = equity * risk_frac; stop_dist = stop_pips * pip
    if stop_dist == 0: return 0
    units = risk_usd / stop_dist
    if pair in USD_BASE_PAIRS: units *= price
    return min(units, (equity * leverage) / price)

# ─────────────────────────────────────────────────────────────────
#  INDICATORS
# ─────────────────────────────────────────────────────────────────
def compute_rsi(s, w=14):
    d = s.diff(); g = d.clip(lower=0); l = (-d).clip(lower=0)
    ag = g.ewm(com=w-1, min_periods=w).mean(); al = l.ewm(com=w-1, min_periods=w).mean()
    return 100 - 100 / (1 + ag / al.replace(0, np.nan))

def compute_atr(df, w=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=w, min_periods=w).mean()

def compute_adx(df, w=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    um = h - h.shift(1); dm = l.shift(1) - l
    pdm = np.where((um > dm) & (um > 0), um, 0)
    mdm = np.where((dm > um) & (dm > 0), dm, 0)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr_s = tr.ewm(com=w-1, min_periods=w).mean()
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(com=w-1, min_periods=w).mean() / atr_s
    mdi = 100 * pd.Series(mdm, index=df.index).ewm(com=w-1, min_periods=w).mean() / atr_s
    dx = (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan) * 100
    return dx.ewm(com=w-1, min_periods=w).mean()

def add_indicators(df):
    df = df.copy(); c = df["Close"]
    df["ema_fast"] = c.ewm(span=FAST_WINDOW, min_periods=FAST_WINDOW).mean()
    df["ema_slow"] = c.ewm(span=SLOW_WINDOW, min_periods=SLOW_WINDOW).mean()
    df["ema_trend"] = c.ewm(span=TREND_WINDOW, min_periods=TREND_WINDOW).mean()
    df["rsi"] = compute_rsi(c, RSI_WINDOW); df["atr"] = compute_atr(df); df["adx"] = compute_adx(df, ADX_WINDOW)
    df["regime"] = np.where(df["ema_fast"] > df["ema_slow"], 1, -1)
    df["crossover"] = df["regime"].diff().fillna(0).ne(0)
    df["in_session"] = (df.index.hour >= SESSION_START_UTC) & (df.index.hour < SESSION_END_UTC) if hasattr(df.index, "hour") else True
    tl = c > df["ema_trend"]; ts_ = c < df["ema_trend"]
    rl = df["rsi"] >= RSI_LONG_THRESH; rs = df["rsi"] <= RSI_SHORT_THRESH
    ao = df["adx"] >= ADX_THRESHOLD; so = df["in_session"]
    lo = tl & rl & ao & so; sho = ts_ & rs & ao & so
    df["signal_update"] = np.nan
    df.loc[df["crossover"] & (df["regime"]==1) & lo, "signal_update"] = 1.0
    df.loc[df["crossover"] & (df["regime"]==-1) & sho, "signal_update"] = -1.0
    uc = df["crossover"] & ~(((df["regime"]==1)&lo)|((df["regime"]==-1)&sho))
    df.loc[uc, "signal_update"] = 0.0
    df.loc[df["signal_update"].isna() & (df["rsi"]>=RSI_EXIT_LONG), "signal_update"] = 0.0
    df.loc[df["signal_update"].isna() & (df["rsi"]<=RSI_EXIT_SHORT), "signal_update"] = 0.0
    df["signal"] = df["signal_update"].ffill().fillna(0).astype(int)
    df.dropna(subset=["ema_fast","ema_slow","ema_trend","rsi","atr","adx"], inplace=True)
    return df

# ─────────────────────────────────────────────────────────────────
#  BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────
def run_backtest(data):
    pf = Portfolio(cash=INITIAL_CAPITAL, positions={p: None for p in data})
    all_times = sorted(set(ts for df in data.values() for ts in df.index))
    for ts in all_times:
        cp = {p: df.at[ts,"Close"] for p, df in data.items() if ts in df.index}
        ce = pf.equity(cp)
        for pair, df in data.items():
            if ts not in df.index: continue
            row = df.loc[ts]; price = row["Close"]; atr = row["atr"]
            pip = _pip_size(pair); ap = atr/pip; sp = ap*ATR_STOP_MULT; ns = int(row["signal"])
            pos = pf.positions[pair]
            if pos is not None:
                hs = (pos.direction==1 and row["Low"]<=pos.stop_loss) or (pos.direction==-1 and row["High"]>=pos.stop_loss)
                if hs:
                    ep = pos.stop_loss; pp = (ep-pos.entry_price)*pos.direction/pip
                    pu = _pnl_usd(pair,pp,pip,pos.units,ep); pf.cash += pu
                    pf.trade_log.append(Trade(pair=pair,direction=pos.direction,entry_time=pos.entry_time,entry_price=pos.entry_price,exit_time=ts,exit_price=ep,units=pos.units,pnl_pips=pp,pnl_usd=pu,exit_reason="stop_loss"))
                    pf.positions[pair] = None; pos = None
            pos = pf.positions[pair]
            if pos is not None:
                td = atr*TRAIL_ATR_MULT
                if pos.direction==1:
                    nt = price-td
                    if nt>pos.trailing_stop: pos.trailing_stop=nt
                    if pos.trailing_stop>pos.stop_loss: pos.stop_loss=pos.trailing_stop
                else:
                    nt = price+td
                    if pos.trailing_stop==0 or nt<pos.trailing_stop: pos.trailing_stop=nt
                    if pos.trailing_stop<pos.stop_loss: pos.stop_loss=pos.trailing_stop
            cd = pos.direction if pos else 0
            if ns != cd:
                if pos is not None:
                    ep = price-_spread_cost(pair,pos.direction,SPREAD_PIPS)
                    pp = (ep-pos.entry_price)*pos.direction/pip; pu = _pnl_usd(pair,pp,pip,pos.units,ep); pf.cash += pu
                    pf.trade_log.append(Trade(pair=pair,direction=pos.direction,entry_time=pos.entry_time,entry_price=pos.entry_price,exit_time=ts,exit_price=ep,units=pos.units,pnl_pips=pp,pnl_usd=pu,exit_reason="signal"))
                    pf.positions[pair] = None
                if ns != 0 and ce > 0:
                    ep = price+_spread_cost(pair,ns,SPREAD_PIPS); sl = ep-ns*sp*pip
                    u = _compute_units(ce,RISK_PER_TRADE,atr,sp,pip,LEVERAGE,ep,pair)
                    if u > 0:
                        it = ep-ns*atr*TRAIL_ATR_MULT
                        pf.positions[pair] = Position(pair=pair,direction=ns,entry_time=ts,entry_price=ep,units=u,stop_loss=sl,atr_at_entry=atr,trailing_stop=it)
        pf.equity_curve.append((ts, pf.equity(cp)))
    for pair, pos in pf.positions.items():
        if pos is not None and pair in data:
            df = data[pair]; lt = df.index[-1]; price = df.at[lt,"Close"]; pip = _pip_size(pair)
            pp = (price-pos.entry_price)*pos.direction/pip; pu = _pnl_usd(pair,pp,pip,pos.units,price); pf.cash += pu
            pf.trade_log.append(Trade(pair=pair,direction=pos.direction,entry_time=pos.entry_time,entry_price=pos.entry_price,exit_time=lt,exit_price=price,units=pos.units,pnl_pips=pp,pnl_usd=pu,exit_reason="end_of_data"))
            pf.positions[pair] = None
    return pf

# ─────────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────────
def compute_metrics(pf, label):
    eq_df = pd.DataFrame(pf.equity_curve, columns=["time","equity"]).set_index("time").sort_index()
    if eq_df.empty or not pf.trade_log: return {}
    eq = eq_df["equity"]; ret = eq.pct_change().dropna()
    n = len(eq); ny = n/HOURS_PER_YEAR_FX if n>0 else 1
    tr = (eq.iloc[-1]/eq.iloc[0])-1; ar = (1+tr)**(1/max(ny,0.1))-1
    av = ret.std()*np.sqrt(HOURS_PER_YEAR_FX); sh = (ar-RISK_FREE_RATE)/av if av!=0 else 0
    rm = eq.cummax(); dd = (eq-rm)/rm; md = dd.min(); cal = ar/abs(md) if md!=0 else 0
    pnls = [t.pnl_usd for t in pf.trade_log]; wins = [p for p in pnls if p>0]; losses = [p for p in pnls if p<0]
    wr = len(wins)/len(pnls) if pnls else 0
    gp = sum(wins); gl = abs(sum(losses)); pf_ = gp/gl if gl>0 else float("inf")
    return {"label":label,"total_return":round(tr*100,2),"ann_return":round(ar*100,2),
            "ann_vol":round(av*100,2),"sharpe":round(sh,3),"max_dd":round(md*100,2),
            "calmar":round(cal,3),"win_rate":round(wr*100,2),"profit_factor":round(pf_,3),
            "avg_win":round(np.mean(wins),2) if wins else 0,"avg_loss":round(np.mean(losses),2) if losses else 0,
            "total_trades":len(pnls),"avg_pnl":round(np.mean(pnls),2) if pnls else 0,
            "final_equity":round(eq.iloc[-1],2)}

def per_pair_metrics(pf):
    rows = []
    for pair in PAIRS:
        pt = [t for t in pf.trade_log if t.pair==pair]
        if not pt: continue
        pnls = [t.pnl_usd for t in pt]; wins = [p for p in pnls if p>0]
        rows.append({"pair":PAIR_NAMES.get(pair,pair),"trades":len(pnls),
                      "win_rate":round(len(wins)/len(pnls)*100,1),"total_pnl":round(sum(pnls),2),
                      "avg_pnl":round(np.mean(pnls),2),"best":round(max(pnls),2),"worst":round(min(pnls),2)})
    return sorted(rows, key=lambda x: x["total_pnl"], reverse=True)

# ─────────────────────────────────────────────────────────────────
#  MAIN — FETCH, BACKTEST, BUILD HTML
# ─────────────────────────────────────────────────────────────────
def main():
    today = datetime.today()
    start = (today - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")

    print(f"Fetching data {start} → {end} ...")
    raw = {}
    for pair in PAIRS:
        try:
            df = yf.download(pair, start=start, end=end, interval="1h", progress=False)
            if df.empty: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            for col in ["Open","High","Low","Close"]: df[col] = pd.to_numeric(df[col], errors="coerce")
            df.dropna(subset=["Close"], inplace=True)
            if not df.empty: raw[pair] = df; print(f"  {PAIR_NAMES[pair]:10s} {len(df):,} bars")
        except Exception as e:
            print(f"  {pair} failed: {e}")

    print("Computing indicators ...")
    train_data, test_data = {}, {}
    for pair, df in raw.items():
        full = add_indicators(df)
        si = int(len(full)*TRAIN_RATIO)
        train_data[pair] = full.iloc[:si]; test_data[pair] = full.iloc[si:]

    print("Running backtest (train) ..."); pf_train = run_backtest(train_data)
    print("Running backtest (test) ...");  pf_test  = run_backtest(test_data)

    m_train = compute_metrics(pf_train, "Train"); m_test = compute_metrics(pf_test, "Test")
    pp_train = per_pair_metrics(pf_train); pp_test = per_pair_metrics(pf_test)

    # Equity curves (downsample for performance)
    def downsample_eq(pf, n=800):
        eq = [(t.isoformat(), round(v,2)) for t,v in pf.equity_curve]
        step = max(1, len(eq)//n)
        return eq[::step] + [eq[-1]]

    eq_train = downsample_eq(pf_train); eq_test = downsample_eq(pf_test)

    # Trade log
    trades_json = []
    for t in pf_test.trade_log:
        trades_json.append({"pair":PAIR_NAMES.get(t.pair,t.pair),"dir":"Long" if t.direction==1 else "Short",
                            "entry_time":t.entry_time.strftime("%Y-%m-%d %H:%M"),"entry_price":round(t.entry_price,5),
                            "exit_time":t.exit_time.strftime("%Y-%m-%d %H:%M"),"exit_price":round(t.exit_price,5),
                            "pnl_pips":round(t.pnl_pips,1),"pnl_usd":round(t.pnl_usd,2),"reason":t.exit_reason})

    # PnL series for cumulative chart
    pnls_test = [round(t.pnl_usd,2) for t in pf_test.trade_log]

    # Exit reasons
    reasons = {}
    for t in pf_test.trade_log:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

    # Benchmark
    eq_s = pd.DataFrame(pf_test.equity_curve, columns=["time","equity"]).set_index("time")["equity"].sort_index()
    bm_pair = "EURUSD=X" if "EURUSD=X" in test_data else next(iter(test_data))
    bm_p = test_data[bm_pair]["Close"].reindex(eq_s.index).ffill().dropna()
    if bm_pair in USD_BASE_PAIRS: bm_p = 1.0/bm_p
    bh = (bm_p/bm_p.iloc[0]*INITIAL_CAPITAL)
    ci = eq_s.index.intersection(bh.index)
    eq_s = eq_s.reindex(ci).ffill(); bh = bh.reindex(ci).ffill()
    rh = (1+RISK_FREE_RATE)**(1/HOURS_PER_YEAR_FX)-1
    rf = pd.Series(INITIAL_CAPITAL*(1+rh)**np.arange(len(ci)), index=ci)

    ns = (eq_s/eq_s.iloc[0]*100); nb = (bh/bh.iloc[0]*100); nr = (rf/rf.iloc[0]*100)
    step = max(1, len(ci)//800)
    bm_labels = [d.strftime("%Y-%m-%d") for d in ci[::step]]
    bm_strat = [round(v,2) for v in ns.values[::step]]
    bm_bh = [round(v,2) for v in nb.values[::step]]
    bm_rf = [round(v,2) for v in nr.values[::step]]

    # Monthly returns
    meq = pd.DataFrame(pf_test.equity_curve, columns=["time","equity"]).set_index("time")["equity"].sort_index()
    monthly = meq.resample("ME").last().pct_change().dropna()
    monthly_labels = [d.strftime("%Y-%m") for d in monthly.index]
    monthly_vals = [round(v*100,2) for v in monthly.values]

    # Build data payload
    data_payload = json.dumps({
        "m_train": m_train, "m_test": m_test,
        "pp_train": pp_train, "pp_test": pp_test,
        "eq_train": eq_train, "eq_test": eq_test,
        "trades": trades_json, "pnls": pnls_test,
        "reasons": reasons,
        "bm_labels": bm_labels, "bm_strat": bm_strat,
        "bm_bh": bm_bh, "bm_rf": bm_rf,
        "bm_pair_name": PAIR_NAMES.get(bm_pair, bm_pair),
        "monthly_labels": monthly_labels, "monthly_vals": monthly_vals,
        "config": {
            "pairs": [PAIR_NAMES[p] for p in PAIRS],
            "start": start, "end": end,
            "fast": FAST_WINDOW, "slow": SLOW_WINDOW, "trend": TREND_WINDOW,
            "rsi": RSI_WINDOW, "adx_thresh": ADX_THRESHOLD,
            "capital": INITIAL_CAPITAL, "risk_pct": RISK_PER_TRADE*100,
            "atr_stop": ATR_STOP_MULT, "trail": TRAIL_ATR_MULT,
        },
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })

    html = build_html(data_payload)
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n✅  Dashboard saved to index.html")
    print(f"    Open in browser or deploy to GitHub Pages.")


# ─────────────────────────────────────────────────────────────────
#  HTML TEMPLATE
# ─────────────────────────────────────────────────────────────────
def build_html(data_json: str) -> str:
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FT5010 Forex Strategy Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
:root {{
  --bg: #0f172a; --surface: #1e293b; --surface2: #334155;
  --text: #f1f5f9; --text2: #94a3b8; --accent: #38bdf8;
  --green: #22c55e; --red: #ef4444; --yellow: #eab308; --cyan: #06b6d4;
  --border: #475569; --radius: 10px;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,-apple-system,sans-serif; line-height:1.6; }}
.container {{ max-width:1400px; margin:0 auto; padding:20px; }}

/* Header */
header {{ text-align:center; padding:30px 0 10px; }}
header h1 {{ font-size:1.8rem; font-weight:700; background:linear-gradient(135deg,var(--accent),var(--cyan)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
header p {{ color:var(--text2); font-size:0.9rem; margin-top:4px; }}

/* KPI Cards */
.kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; margin:24px 0; }}
.kpi {{ background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:18px; text-align:center; transition:transform .15s; }}
.kpi:hover {{ transform:translateY(-2px); }}
.kpi .label {{ font-size:0.75rem; color:var(--text2); text-transform:uppercase; letter-spacing:0.5px; }}
.kpi .value {{ font-size:1.6rem; font-weight:700; margin-top:4px; }}
.kpi .value.positive {{ color:var(--green); }}
.kpi .value.negative {{ color:var(--red); }}

/* Tabs */
.tabs {{ display:flex; gap:4px; margin:28px 0 16px; border-bottom:2px solid var(--surface2); padding-bottom:0; flex-wrap:wrap; }}
.tab {{ padding:10px 22px; cursor:pointer; color:var(--text2); font-size:0.9rem; font-weight:500;
        border-bottom:2px solid transparent; margin-bottom:-2px; transition:all .15s; user-select:none; }}
.tab:hover {{ color:var(--text); }}
.tab.active {{ color:var(--accent); border-bottom-color:var(--accent); }}
.tab-content {{ display:none; }}
.tab-content.active {{ display:block; }}

/* Chart containers */
.chart-row {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:16px; }}
.chart-row.full {{ grid-template-columns:1fr; }}
.chart-box {{ background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:18px; }}
.chart-box h3 {{ font-size:0.85rem; color:var(--text2); margin-bottom:12px; font-weight:500; }}
.chart-box canvas {{ width:100%!important; }}

/* Table */
.table-wrap {{ overflow-x:auto; }}
table {{ width:100%; border-collapse:collapse; font-size:0.82rem; }}
th {{ background:var(--surface2); color:var(--text2); padding:10px 12px; text-align:left; font-weight:500;
      position:sticky; top:0; text-transform:uppercase; font-size:0.7rem; letter-spacing:0.5px; }}
td {{ padding:9px 12px; border-bottom:1px solid var(--border); }}
tr:hover td {{ background:rgba(56,189,248,0.04); }}
.pnl-pos {{ color:var(--green); font-weight:600; }}
.pnl-neg {{ color:var(--red); font-weight:600; }}

/* Comparison table */
.comp-table {{ margin-top:12px; }}
.comp-table td:first-child {{ color:var(--text2); font-weight:500; }}
.comp-table td {{ text-align:center; }}
.comp-table th {{ text-align:center; }}

/* Config info */
.config-bar {{ display:flex; flex-wrap:wrap; gap:16px; padding:14px 20px; background:var(--surface); border-radius:var(--radius);
               border:1px solid var(--border); margin-bottom:16px; font-size:0.8rem; color:var(--text2); }}
.config-bar span {{ white-space:nowrap; }}
.config-bar strong {{ color:var(--text); }}

/* Footer */
footer {{ text-align:center; padding:30px 0 20px; color:var(--text2); font-size:0.75rem; }}

@media (max-width:768px) {{
  .chart-row {{ grid-template-columns:1fr; }}
  .kpi-grid {{ grid-template-columns:repeat(3,1fr); }}
}}
</style>
</head>
<body>
<div class="container">

<header>
  <h1>FT5010 Forex Momentum Strategy</h1>
  <p>Dual EMA Crossover + RSI / ADX Filter &mdash; Event-Based Backtester</p>
</header>

<div class="config-bar" id="configBar"></div>

<!-- KPI -->
<div class="kpi-grid" id="kpiGrid"></div>

<!-- Tabs -->
<div class="tabs" id="tabBar">
  <div class="tab active" data-tab="overview">Overview</div>
  <div class="tab" data-tab="train">Train Set</div>
  <div class="tab" data-tab="test">Test Set</div>
  <div class="tab" data-tab="benchmark">Benchmark</div>
  <div class="tab" data-tab="trades">Trade Log</div>
</div>

<!-- Overview -->
<div class="tab-content active" id="tc-overview">
  <div class="chart-row">
    <div class="chart-box"><h3>Train vs Test Metrics</h3><div class="table-wrap" id="compTable"></div></div>
    <div class="chart-box"><h3>Exit Reasons (Test)</h3><canvas id="chartPie"></canvas></div>
  </div>
  <div class="chart-row">
    <div class="chart-box"><h3>Monthly Returns — Test (%)</h3><canvas id="chartMonthly"></canvas></div>
    <div class="chart-box"><h3>Per-Trade PnL — Test</h3><canvas id="chartPnlBar"></canvas></div>
  </div>
</div>

<!-- Train -->
<div class="tab-content" id="tc-train">
  <div class="chart-row full"><div class="chart-box"><h3>Equity Curve — Train</h3><canvas id="chartEqTrain"></canvas></div></div>
  <div class="chart-row">
    <div class="chart-box"><h3>PnL by Pair — Train</h3><canvas id="chartPairTrain"></canvas></div>
    <div class="chart-box"><h3>Per-Pair Breakdown — Train</h3><div class="table-wrap" id="pairTableTrain"></div></div>
  </div>
</div>

<!-- Test -->
<div class="tab-content" id="tc-test">
  <div class="chart-row full"><div class="chart-box"><h3>Equity Curve — Test</h3><canvas id="chartEqTest"></canvas></div></div>
  <div class="chart-row">
    <div class="chart-box"><h3>PnL by Pair — Test</h3><canvas id="chartPairTest"></canvas></div>
    <div class="chart-box"><h3>Per-Pair Breakdown — Test</h3><div class="table-wrap" id="pairTableTest"></div></div>
  </div>
  <div class="chart-row full"><div class="chart-box"><h3>Cumulative PnL — Test</h3><canvas id="chartCumPnl"></canvas></div></div>
</div>

<!-- Benchmark -->
<div class="tab-content" id="tc-benchmark">
  <div class="chart-row full"><div class="chart-box"><h3>Strategy vs Benchmarks (Normalised to 100)</h3><canvas id="chartBenchmark"></canvas></div></div>
</div>

<!-- Trade Log -->
<div class="tab-content" id="tc-trades">
  <div class="chart-box">
    <h3>Trade Log — Test Set <span id="tradeCount" style="color:var(--accent)"></span></h3>
    <div style="margin-bottom:12px;display:flex;gap:10px;flex-wrap:wrap;">
      <input type="text" id="tradeSearch" placeholder="Search pair..." style="background:var(--surface2);border:1px solid var(--border);color:var(--text);padding:6px 12px;border-radius:6px;font-size:0.82rem;width:200px;">
      <select id="tradeDirFilter" style="background:var(--surface2);border:1px solid var(--border);color:var(--text);padding:6px 12px;border-radius:6px;font-size:0.82rem;">
        <option value="">All Directions</option><option value="Long">Long</option><option value="Short">Short</option>
      </select>
      <select id="tradeReasonFilter" style="background:var(--surface2);border:1px solid var(--border);color:var(--text);padding:6px 12px;border-radius:6px;font-size:0.82rem;">
        <option value="">All Reasons</option>
      </select>
    </div>
    <div class="table-wrap"><table id="tradeTable"><thead><tr>
      <th>Pair</th><th>Dir</th><th>Entry Time</th><th>Entry Price</th>
      <th>Exit Time</th><th>Exit Price</th><th>PnL (pips)</th><th>PnL (USD)</th><th>Reason</th>
    </tr></thead><tbody></tbody></table></div>
  </div>
</div>

<footer>
  Generated <span id="genTime"></span> &mdash; FT5010 Final-Term Project
</footer>

</div>

<script>
const D = {data_json};

// ── Helpers ──
const fmt = (v,d=2) => v==null?'—':Number(v).toFixed(d);
const cls = v => v>0?'positive':v<0?'negative':'';
const clsP = v => v>0?'pnl-pos':v<0?'pnl-neg':'';

const chartDefs = {{ responsive:true, maintainAspectRatio:false }};
const gridColor = 'rgba(148,163,184,0.1)';
const chartOpts = (h=280) => ({{
  responsive:true, maintainAspectRatio:false,
  plugins:{{ legend:{{ labels:{{ color:'#94a3b8', font:{{size:11}} }} }} }},
  scales:{{ x:{{ ticks:{{color:'#94a3b8',maxTicksLimit:12}}, grid:{{color:gridColor}} }},
            y:{{ ticks:{{color:'#94a3b8'}}, grid:{{color:gridColor}} }} }}
}});

// ── Config bar ──
const c = D.config;
document.getElementById('configBar').innerHTML = [
  `<span>Period: <strong>${{c.start}} → ${{c.end}}</strong></span>`,
  `<span>EMA: <strong>${{c.fast}}/${{c.slow}}</strong> Trend: <strong>${{c.trend}}</strong></span>`,
  `<span>RSI: <strong>${{c.rsi}}</strong> ADX&gt;<strong>${{c.adx_thresh}}</strong></span>`,
  `<span>Capital: <strong>$${{c.capital.toLocaleString()}}</strong></span>`,
  `<span>Risk: <strong>${{c.risk_pct}}%</strong>/trade</span>`,
  `<span>Stop: <strong>${{c.atr_stop}}</strong>&times;ATR Trail: <strong>${{c.trail}}</strong>&times;ATR</span>`,
].join('');

// ── KPI ──
const m = D.m_test;
const kpis = [
  ['Total Return', `${{m.total_return}}%`, m.total_return],
  ['Sharpe Ratio', `${{m.sharpe}}`, m.sharpe],
  ['Max Drawdown', `${{m.max_dd}}%`, m.max_dd],
  ['Win Rate', `${{m.win_rate}}%`, m.win_rate-50],
  ['Profit Factor', `${{m.profit_factor}}`, m.profit_factor-1],
  ['Calmar Ratio', `${{m.calmar}}`, m.calmar],
  ['Total Trades', `${{m.total_trades}}`, 0],
  ['Final Equity', `$${{m.final_equity.toLocaleString()}}`, m.final_equity-c.capital],
];
document.getElementById('kpiGrid').innerHTML = kpis.map(([l,v,s])=>
  `<div class="kpi"><div class="label">${{l}}</div><div class="value ${{cls(s)}}">${{v}}</div></div>`).join('');

// ── Tabs ──
document.querySelectorAll('.tab').forEach(tab => {{
  tab.addEventListener('click', () => {{
    document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t=>t.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('tc-'+tab.dataset.tab).classList.add('active');
  }});
}});

// ── Comparison table ──
const metrics_keys = ['total_return','ann_return','ann_vol','sharpe','max_dd','calmar','win_rate','profit_factor','avg_win','avg_loss','total_trades','avg_pnl','final_equity'];
const metrics_labels = ['Total Return (%)','Ann. Return (%)','Ann. Volatility (%)','Sharpe Ratio','Max Drawdown (%)','Calmar Ratio','Win Rate (%)','Profit Factor','Avg Win (USD)','Avg Loss (USD)','Total Trades','Avg PnL/Trade','Final Equity (USD)'];
let compHtml = '<table class="comp-table"><thead><tr><th></th><th>Train</th><th>Test</th></tr></thead><tbody>';
metrics_keys.forEach((k,i) => {{
  const tv = D.m_train[k], sv = D.m_test[k];
  compHtml += `<tr><td>${{metrics_labels[i]}}</td><td class="${{clsP(tv)}}">${{fmt(tv)}}</td><td class="${{clsP(sv)}}">${{fmt(sv)}}</td></tr>`;
}});
compHtml += '</tbody></table>';
document.getElementById('compTable').innerHTML = compHtml;

// ── Charts ──
function setCanvasHeight(id, h) {{ document.getElementById(id).parentElement.style.height = h+'px'; }}

// Pie — exit reasons
setCanvasHeight('chartPie', 260);
new Chart(document.getElementById('chartPie'), {{
  type:'doughnut',
  data:{{ labels:Object.keys(D.reasons), datasets:[{{ data:Object.values(D.reasons),
    backgroundColor:['#22c55e','#ef4444','#eab308','#06b6d4','#8b5cf6'] }}] }},
  options:{{ responsive:true, maintainAspectRatio:false, plugins:{{ legend:{{ position:'bottom', labels:{{color:'#94a3b8'}} }} }} }}
}});

// Monthly returns
setCanvasHeight('chartMonthly', 280);
new Chart(document.getElementById('chartMonthly'), {{
  type:'bar',
  data:{{ labels:D.monthly_labels, datasets:[{{ data:D.monthly_vals, backgroundColor:D.monthly_vals.map(v=>v>=0?'#22c55e':'#ef4444') }}] }},
  options:{{ ...chartOpts(), plugins:{{ legend:{{display:false}} }} }}
}});

// Per-trade PnL bar
setCanvasHeight('chartPnlBar', 280);
new Chart(document.getElementById('chartPnlBar'), {{
  type:'bar',
  data:{{ labels:D.pnls.map((_,i)=>i+1), datasets:[{{ data:D.pnls, backgroundColor:D.pnls.map(v=>v>=0?'#22c55e':'#ef4444') }}] }},
  options:{{ ...chartOpts(), plugins:{{ legend:{{display:false}} }}, scales:{{ x:{{display:false}}, y:{{ticks:{{color:'#94a3b8'}},grid:{{color:gridColor}}}} }} }}
}});

// Equity curves
function plotEquity(canvasId, eqData, color) {{
  setCanvasHeight(canvasId, 320);
  const labels = eqData.map(e=>e[0].substring(0,10));
  const values = eqData.map(e=>e[1]);
  new Chart(document.getElementById(canvasId), {{
    type:'line',
    data:{{ labels, datasets:[{{ data:values, borderColor:color, borderWidth:1.5, pointRadius:0, fill:false }}] }},
    options:{{ ...chartOpts(), plugins:{{ legend:{{display:false}} }}, scales:{{ x:{{ticks:{{color:'#94a3b8',maxTicksLimit:10}},grid:{{color:gridColor}}}}, y:{{ticks:{{color:'#94a3b8'}},grid:{{color:gridColor}}}} }} }}
  }});
}}
plotEquity('chartEqTrain', D.eq_train, '#22c55e');
plotEquity('chartEqTest', D.eq_test, '#38bdf8');

// Pair PnL bars
function plotPairPnl(canvasId, ppData) {{
  setCanvasHeight(canvasId, 280);
  new Chart(document.getElementById(canvasId), {{
    type:'bar',
    data:{{ labels:ppData.map(r=>r.pair), datasets:[{{ data:ppData.map(r=>r.total_pnl),
      backgroundColor:ppData.map(r=>r.total_pnl>=0?'#22c55e':'#ef4444') }}] }},
    options:{{ ...chartOpts(), plugins:{{ legend:{{display:false}} }} }}
  }});
}}
plotPairPnl('chartPairTrain', D.pp_train);
plotPairPnl('chartPairTest', D.pp_test);

// Per-pair tables
function buildPairTable(containerId, ppData) {{
  let h = '<table><thead><tr><th>Pair</th><th>Trades</th><th>Win Rate</th><th>Total PnL</th><th>Avg PnL</th><th>Best</th><th>Worst</th></tr></thead><tbody>';
  ppData.forEach(r => {{
    h += `<tr><td>${{r.pair}}</td><td>${{r.trades}}</td><td>${{r.win_rate}}%</td>
      <td class="${{clsP(r.total_pnl)}}">${{fmt(r.total_pnl)}}</td><td class="${{clsP(r.avg_pnl)}}">${{fmt(r.avg_pnl)}}</td>
      <td class="pnl-pos">${{fmt(r.best)}}</td><td class="pnl-neg">${{fmt(r.worst)}}</td></tr>`;
  }});
  h += '</tbody></table>';
  document.getElementById(containerId).innerHTML = h;
}}
buildPairTable('pairTableTrain', D.pp_train);
buildPairTable('pairTableTest', D.pp_test);

// Cumulative PnL
setCanvasHeight('chartCumPnl', 280);
const cumPnl = []; let cumSum = 0;
D.pnls.forEach(v => {{ cumSum += v; cumPnl.push(Math.round(cumSum*100)/100); }});
new Chart(document.getElementById('chartCumPnl'), {{
  type:'line',
  data:{{ labels:cumPnl.map((_,i)=>i+1), datasets:[{{ data:cumPnl, borderColor:'#06b6d4', borderWidth:1.5, pointRadius:0, fill:true, backgroundColor:'rgba(6,182,212,0.1)' }}] }},
  options:{{ ...chartOpts(), plugins:{{ legend:{{display:false}} }} }}
}});

// Benchmark
setCanvasHeight('chartBenchmark', 360);
new Chart(document.getElementById('chartBenchmark'), {{
  type:'line',
  data:{{ labels:D.bm_labels, datasets:[
    {{ label:'Strategy', data:D.bm_strat, borderColor:'#22c55e', borderWidth:2, pointRadius:0, fill:false }},
    {{ label:D.bm_pair_name+' B&H', data:D.bm_bh, borderColor:'#eab308', borderWidth:1.5, pointRadius:0, borderDash:[6,3], fill:false }},
    {{ label:'Risk-Free', data:D.bm_rf, borderColor:'#06b6d4', borderWidth:1.5, pointRadius:0, borderDash:[2,2], fill:false }},
  ] }},
  options:{{ ...chartOpts(), plugins:{{ legend:{{ labels:{{color:'#94a3b8'}} }} }} }}
}});

// ── Trade log ──
const tbody = document.querySelector('#tradeTable tbody');
const searchEl = document.getElementById('tradeSearch');
const dirEl = document.getElementById('tradeDirFilter');
const reasonEl = document.getElementById('tradeReasonFilter');
const countEl = document.getElementById('tradeCount');

// Populate reason filter
const reasonSet = [...new Set(D.trades.map(t=>t.reason))];
reasonSet.forEach(r => {{ const o = document.createElement('option'); o.value=r; o.textContent=r; reasonEl.appendChild(o); }});

function renderTrades() {{
  const sf = searchEl.value.toLowerCase();
  const df = dirEl.value;
  const rf = reasonEl.value;
  const filtered = D.trades.filter(t =>
    (!sf || t.pair.toLowerCase().includes(sf)) &&
    (!df || t.dir === df) &&
    (!rf || t.reason === rf)
  );
  countEl.textContent = `(${{filtered.length}} / ${{D.trades.length}})`;
  tbody.innerHTML = filtered.map(t =>
    `<tr><td>${{t.pair}}</td><td>${{t.dir}}</td><td>${{t.entry_time}}</td><td>${{t.entry_price}}</td>
     <td>${{t.exit_time}}</td><td>${{t.exit_price}}</td><td>${{fmt(t.pnl_pips,1)}}</td>
     <td class="${{clsP(t.pnl_usd)}}">${{fmt(t.pnl_usd)}}</td><td>${{t.reason}}</td></tr>`
  ).join('');
}}
searchEl.addEventListener('input', renderTrades);
dirEl.addEventListener('change', renderTrades);
reasonEl.addEventListener('change', renderTrades);
renderTrades();

document.getElementById('genTime').textContent = D.generated;
</script>
</body>
</html>''';


if __name__ == "__main__":
    main()
