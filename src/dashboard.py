#!/usr/bin/env python3
"""Local read-only dashboard for live_quoter.py.

Run this in a second terminal while live_quoter.py is running:
    python dashboard.py
then open http://127.0.0.1:8787 in a browser. It only reads the parquet files
live_quoter.py flushes to live_data/ (every FLUSH_INTERVAL_SEC) — it never
touches the trading process, so it's safe to start/stop/refresh independently.
"""
import glob
import http.server
import json
import socketserver
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

LIVE_DATA_DIR = Path(__file__).resolve().parent.parent / "live_data"
PORT = 8787
MAX_POINTS = 400  # downsample series sent to the browser so long runs stay light
RUN_GAP_SEC = 300  # a gap this large between flushed files means a new run started

# Each flush writes a brand-new, immutable file (quotes_<flush-epoch>.parquet), so
# already-loaded files never need re-reading — only new ones get appended to the cache,
# keyed by run_start so switching between historical runs doesn't discard other runs' data.
_cache = {}  # run_start_epoch -> {"loaded": set(), "df": DataFrame}


def _epoch_of(path):
    return int(Path(path).stem.split("_")[-1])


def _pid_of(path):
    """Writer PID embedded in the filename (quotes_<pid>_<epoch>.parquet), or None for older
    files (quotes_<epoch>.parquet) that predate the collision fix mentioned in _run_id below."""
    parts = Path(path).stem.split("_")
    return parts[1] if len(parts) == 3 else None


# capture_ratio never changes for a given file (one process, one --capture-ratio, one flush),
# so this is safe to cache forever, same spirit as _cache below for full dataframes.
_capture_ratio_cache = {}


def _capture_ratio_of(path):
    if path in _capture_ratio_cache:
        return _capture_ratio_cache[path]
    try:
        col = pd.read_parquet(path, columns=["capture_ratio"])["capture_ratio"]
        value = float(col.iloc[-1]) if len(col) else None
    except Exception:
        # Either an older file that predates this column, or (rarely) this exact file caught
        # mid-flush-write by live_quoter.py. Can't tell which from here, so don't cache the
        # failure — if it's the latter, the next poll (after the write finishes) reads it fine.
        return None
    _capture_ratio_cache[path] = value
    return value


def _all_runs():
    """All runs found on disk, oldest first. Files are first split by capture_ratio, then
    chained within each ratio wherever consecutive (same-ratio) files share a PID and are
    <= RUN_GAP_SEC apart. The ratio split matters because running several live_quoter.py
    processes concurrently (e.g. to sweep --capture-ratio 0.2/0.5/0.7 against the same live
    market) means their flush timestamps interleave in overall time order — comparing only to
    the immediately-preceding file by timestamp would then almost always see a different ratio
    and never rejoin two flushes from the same process, splitting every single flush into its
    own 1-file "run". The PID check matters separately: a quick restart (e.g. to pick up a code
    change) can land well within RUN_GAP_SEC of the previous process's last flush, but it's a
    fresh process — new in-memory bandit windows, possibly a different reward/config version —
    so it's shown as its own run rather than silently spliced onto the old one."""
    files = sorted(glob.glob(str(LIVE_DATA_DIR / "quotes_*.parquet")), key=_epoch_of)
    if not files:
        return []

    buckets = {}
    for f in files:
        buckets.setdefault(_capture_ratio_of(f), []).append(f)

    runs = []
    for bucket_files in buckets.values():
        current = [bucket_files[0]]
        for f in bucket_files[1:]:
            same_process = _pid_of(f) == _pid_of(current[-1])
            if same_process and _epoch_of(f) - _epoch_of(current[-1]) <= RUN_GAP_SEC:
                current.append(f)
            else:
                runs.append(current)
                current = [f]
        runs.append(current)

    runs.sort(key=lambda run: _epoch_of(run[0]))
    return runs


MIN_RUN_FILES = 60  # hide short/aborted runs from the dropdown — not enough data to be worth browsing
LIVE_STALE_SEC = 90  # matches the frontend's own staleLive threshold


def list_runs():
    runs = _all_runs()
    if not runs:
        return []
    now = time.time()
    result = []
    for files in runs:
        end_ts = _epoch_of(files[-1])
        # Recency, not list position, decides "live" — concurrent sweep processes can all be
        # genuinely live at once, not just whichever happens to sort last.
        is_live = (now - end_ts) < LIVE_STALE_SEC
        if len(files) < MIN_RUN_FILES and not is_live:
            continue  # short/aborted historical run — not a live one, so safe to hide
        result.append({
            "run_id": _run_id(files),
            "start_ts": _epoch_of(files[0]),
            "end_ts": end_ts,
            "n_files": len(files),
            "is_live": is_live,
            "capture_ratio": _capture_ratio_of(files[-1]),
        })
    result.sort(key=lambda r: r["start_ts"], reverse=True)  # newest first, for the dropdown
    return result


def _run_id(files):
    # The first file's own name — unique since the flush-filename fix added the writer's PID —
    # rather than just its start epoch, which two concurrently-swept processes (e.g.
    # --capture-ratio 0.2 and 0.5 started in the same second) can otherwise share.
    return Path(files[0]).stem


def _load_run(files):
    run_id = _run_id(files)
    entry = _cache.setdefault(run_id, {"loaded": set(), "df": pd.DataFrame()})
    new_files = [f for f in files if f not in entry["loaded"]]
    if new_files:
        dfs = []
        for f in new_files:
            try:
                dfs.append(pd.read_parquet(f))
            except Exception as e:
                # live_quoter.py writes atomically (temp file + os.replace) as of the fix for
                # the concurrent-process filename collision that produced these, so a file that
                # fails to parse now is permanently corrupt garbage, not a write-in-progress —
                # safe to mark "loaded" (i.e. give up on it) rather than retry every request.
                print(f"skipping unreadable {f}: {e}", file=sys.stderr)
            entry["loaded"].add(f)
        if dfs:
            new_df = pd.concat(dfs, ignore_index=True)
            entry["df"] = pd.concat([entry["df"], new_df], ignore_index=True) if len(entry["df"]) else new_df
    return entry["df"].sort_values("timestamp").reset_index(drop=True)


def build_status(run_param=None):
    runs = _all_runs()
    if not runs:
        return {"error": "no data yet — is live_quoter.py running?"}

    if run_param is None:
        files = runs[-1]
    else:
        matches = [r for r in runs if _run_id(r) == run_param]
        if not matches:
            return {"error": "that run wasn't found (files may have been deleted)"}
        files = matches[0]

    is_latest = files is runs[-1]
    df = _load_run(files)
    if df.empty:
        return {"error": "no data yet — is live_quoter.py running?"}

    # Older runs predate columns added later in the project (momentum, unwind_stage) —
    # degrade gracefully instead of crashing when browsing that history.
    def col(name, row=None):
        if name not in df.columns:
            return None
        val = (row if row is not None else df)[name]
        return val if row is None else (val if pd.notna(val) else None)

    def to_list(series):
        return [None if pd.isna(v) else float(v) for v in series]

    spread = df["target_ask"] - df["target_bid"]
    ext_spread = df["extended_ask"] - df["extended_bid"]
    fill_diff = df["inventory"].diff().fillna(0)
    buy_fills = int((fill_diff > 0).sum())
    sell_fills = int((fill_diff < 0).sum())
    latest = df.iloc[-1]

    # Exact fill markers (not downsampled — fills are rare enough that sending every one
    # is cheap, and downsampling the tick series would otherwise silently drop most of them).
    fill_markers = []
    for idx, d in fill_diff[fill_diff != 0].items():
        row = df.loc[idx]
        fill_markers.append({
            "t": int(row["timestamp"]),
            "price": float(row["target_bid"] if d > 0 else row["target_ask"]),
            "side": "buy" if d > 0 else "sell",
        })

    step = max(1, len(df) // MAX_POINTS)
    s = df.iloc[::step]
    s_spread = spread.iloc[::step]
    s_ext_spread = ext_spread.iloc[::step]

    momentum = col("momentum", latest)
    unwind_stage = col("unwind_stage", latest)
    capture_ratio = col("capture_ratio", latest)

    return {
        "run_start_ts": int(df["timestamp"].iloc[0] / 1000),
        "latest_ts": int(latest["timestamp"]),
        "is_latest": is_latest,
        "updated_at": time.time(),
        "rows": len(df),
        "mode": str(latest["mode"]),
        "capture_ratio": float(capture_ratio) if capture_ratio is not None else None,
        "latest": {
            "fair_price": float(latest["fair_price"]),
            "target_bid": float(latest["target_bid"]),
            "target_ask": float(latest["target_ask"]),
            "extended_bid": float(latest["extended_bid"]) if pd.notna(latest["extended_bid"]) else None,
            "extended_ask": float(latest["extended_ask"]) if pd.notna(latest["extended_ask"]) else None,
            "spread": float(spread.iloc[-1]),
            "extended_spread": float(ext_spread.iloc[-1]) if pd.notna(ext_spread.iloc[-1]) else None,
            "sigma": float(latest["sigma_dollar"]) if pd.notna(latest["sigma_dollar"]) else None,
            "momentum": float(momentum) if momentum is not None else None,
            "inventory": float(latest["inventory"]),
            "pnl": float(latest["pnl"]),
            "unwind_stage": str(unwind_stage) if unwind_stage is not None else "n/a",
        },
        "fills": {
            "buy": buy_fills,
            "sell": sell_fills,
            "total": buy_fills + sell_fills,
        },
        "pnl_min": float(df["pnl"].min()),
        "pnl_max": float(df["pnl"].max()),
        "inventory_min": float(df["inventory"].min()),
        "inventory_max": float(df["inventory"].max()),
        "fill_markers": fill_markers,
        "series": {
            "t": s["timestamp"].tolist(),
            "pnl": to_list(s["pnl"]),
            "spread": to_list(s_spread),
            "extended_spread": to_list(s_ext_spread),
            "sigma": to_list(s["sigma_dollar"]),
            "inventory": to_list(s["inventory"]),
            "fair_price": to_list(s["fair_price"]),
            "edge_bid": to_list(s["edge_bid"]) if "edge_bid" in s.columns else [],
            "edge_ask": to_list(s["edge_ask"]) if "edge_ask" in s.columns else [],
        },
    }


DASHBOARD_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>live_quoter dashboard</title>
<style>
  :root {
    --bg: #0b0e14; --panel: #131722; --border: #232838; --text: #e6e9f0;
    --muted: #7b8496; --green: #3fb950; --red: #f85149; --blue: #58a6ff;
    --amber: #d29922; --purple: #bc8cff;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }
  .titlebar { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; }
  h1 { font-size: 18px; font-weight: 600; margin: 0 0 4px; }
  .sub { color: var(--muted); font-size: 13px; margin-bottom: 20px; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
  .dot.live { background: var(--green); box-shadow: 0 0 6px var(--green); animation: pulse 1.5s infinite; }
  .dot.stale { background: var(--red); }
  .dot.historical { background: var(--muted); }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
  select#runPicker {
    background: var(--panel); color: var(--text); border: 1px solid var(--border);
    border-radius: 8px; padding: 6px 10px; font-size: 13px; max-width: 360px;
  }
  .cards {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px; margin-bottom: 24px;
  }
  .card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 14px 16px;
  }
  .card .label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
  .card .value { font-size: 22px; font-weight: 600; margin-top: 4px; font-variant-numeric: tabular-nums; }
  .card .extra { font-size: 12px; color: var(--muted); margin-top: 2px; }
  .pos { color: var(--green); } .neg { color: var(--red); }
  .badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 13px; font-weight: 600; }
  .badge.none { background: rgba(63,185,80,0.15); color: var(--green); }
  .badge.warn { background: rgba(210,153,34,0.15); color: var(--amber); }
  .badge.force { background: rgba(248,81,73,0.15); color: var(--red); }
  .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .chart-box {
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px;
  }
  .chart-box .title {
    font-size: 13px; color: var(--muted); margin-bottom: 8px;
    display: flex; justify-content: space-between; align-items: center;
  }
  .legend { font-size: 11px; display: flex; gap: 10px; }
  .legend span { display: inline-flex; align-items: center; gap: 4px; }
  .legend .sw { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
  canvas { width: 100%; height: 160px; display: block; }
  .empty { color: var(--muted); padding: 60px; text-align: center; }
  @media (max-width: 800px) { .charts { grid-template-columns: 1fr; } }
</style>
</head>
<body>
  <div class="titlebar">
    <h1><span id="statusDot" class="dot stale"></span>live_quoter dashboard</h1>
    <select id="runPicker"><option>loading runs…</option></select>
  </div>
  <div class="sub" id="subline">waiting for data…</div>

  <div id="content" style="display:none">
    <div class="cards">
      <div class="card"><div class="label">Inventory</div><div class="value" id="cInventory">–</div><div class="extra" id="cInventoryRange"></div></div>
      <div class="card"><div class="label">PnL</div><div class="value" id="cPnl">–</div><div class="extra" id="cPnlRange"></div></div>
      <div class="card"><div class="label">Spread (ours / extended's)</div><div class="value" id="cSpread">–</div><div class="extra" id="cSpreadRatio"></div></div>
      <div class="card"><div class="label">Sigma</div><div class="value" id="cSigma">–</div></div>
      <div class="card"><div class="label">Fills (buy / sell)</div><div class="value" id="cFills">–</div></div>
      <div class="card"><div class="label">Unwind stage</div><div class="value"><span class="badge none" id="cStage">–</span></div></div>
    </div>

    <div class="charts">
      <div class="chart-box">
        <div class="title">
          <span>Fair price with fills ($)</span>
          <span class="legend">
            <span><span class="sw" style="background:#3fb950"></span>buy</span>
            <span><span class="sw" style="background:#f85149"></span>sell</span>
          </span>
        </div>
        <canvas id="chartPrice"></canvas>
      </div>
      <div class="chart-box">
        <div class="title">
          <span>Edge: distance from Extended's touch ($)</span>
          <span class="legend">
            <span><span class="sw" style="background:#58a6ff"></span>bid</span>
            <span><span class="sw" style="background:#bc8cff"></span>ask</span>
          </span>
        </div>
        <canvas id="chartEdge"></canvas>
      </div>
      <div class="chart-box"><div class="title">PnL over time ($)</div><canvas id="chartPnl"></canvas></div>
      <div class="chart-box"><div class="title">Spread: ours vs Extended's ($)</div><canvas id="chartSpread"></canvas></div>
      <div class="chart-box"><div class="title">Sigma ($)</div><canvas id="chartSigma"></canvas></div>
      <div class="chart-box"><div class="title">Inventory (BTC)</div><canvas id="chartInventory"></canvas></div>
    </div>
  </div>
  <div class="empty" id="emptyState">No run detected yet. Start live_quoter.py — this page updates automatically.</div>

<script>
const REFRESH_MS = 5000;
const COLORS = { pnl: '#58a6ff', spread: '#58a6ff', extSpread: '#7b8496', sigma: '#d29922', inv: '#bc8cff' };

function fmtUsd(x, digits=2) {
  if (x === null || x === undefined) return '–';
  const sign = x < 0 ? '-' : '';
  return sign + '$' + Math.abs(x).toFixed(digits);
}
function fmtBtc(x) { return (x === null || x === undefined) ? '–' : x.toFixed(4); }
function fmtElapsed(sec) {
  const h = Math.floor(sec/3600), m = Math.floor((sec%3600)/60), s = Math.floor(sec%60);
  return h > 0 ? `${h}h ${m}m ${s}s` : (m > 0 ? `${m}m ${s}s` : `${s}s`);
}

function drawChart(canvas, series, color, opts={}) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const w = rect.width, h = rect.height, pad = 6;
  ctx.clearRect(0, 0, w, h);
  if (!series.length) return;

  const vals = series.filter(v => v !== null && v !== undefined);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (opts.zeroLine) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  if (lo === hi) { lo -= 1; hi += 1; }
  const range = hi - lo;
  const x = i => pad + (i / (series.length - 1 || 1)) * (w - 2*pad);
  const y = v => h - pad - ((v - lo) / range) * (h - 2*pad);

  if (opts.zeroLine && lo < 0 && hi > 0) {
    ctx.strokeStyle = '#2a3040'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, y(0)); ctx.lineTo(w-pad, y(0)); ctx.stroke();
  }

  ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.beginPath();
  series.forEach((v, i) => {
    if (v === null || v === undefined) return;
    i === 0 ? ctx.moveTo(x(i), y(v)) : ctx.lineTo(x(i), y(v));
  });
  ctx.stroke();

  ctx.fillStyle = 'rgba(255,255,255,0.35)'; ctx.font = '10px sans-serif';
  ctx.fillText(hi.toFixed(2), 2, 10);
  ctx.fillText(lo.toFixed(2), 2, h - 2);
}

function drawDualLineChart(canvas, seriesA, seriesB, colorA, colorB, opts={}) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d'); ctx.scale(dpr, dpr);
  const w = rect.width, h = rect.height, pad = 6;
  ctx.clearRect(0, 0, w, h);
  if (!seriesA.length) return;
  const all = seriesA.concat(seriesB).filter(v => v !== null && v !== undefined);
  let lo = Math.min(opts.zeroLine ? 0 : Infinity, ...all), hi = Math.max(opts.zeroLine ? 0 : -Infinity, ...all);
  if (lo === hi) { lo -= 1; hi += 1; }
  const range = hi - lo;
  const x = i => pad + (i / (seriesA.length - 1 || 1)) * (w - 2*pad);
  const y = v => h - pad - ((v - lo) / range) * (h - 2*pad);
  if (opts.zeroLine && lo < 0 && hi > 0) {
    ctx.strokeStyle = '#2a3040'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, y(0)); ctx.lineTo(w-pad, y(0)); ctx.stroke();
  }
  const line = (series, color) => {
    ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.beginPath();
    series.forEach((v, i) => { if (v==null) return; i===0?ctx.moveTo(x(i),y(v)):ctx.lineTo(x(i),y(v)); });
    ctx.stroke();
  };
  line(seriesB, colorB);
  line(seriesA, colorA);
  ctx.fillStyle = 'rgba(255,255,255,0.35)'; ctx.font = '10px sans-serif';
  ctx.fillText(hi.toFixed(2), 2, 10);
  ctx.fillText(lo.toFixed(2), 2, h - 2);
}

function drawPriceChart(canvas, times, prices, fills, tMin, tMax) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d'); ctx.scale(dpr, dpr);
  const w = rect.width, h = rect.height, pad = 6;
  ctx.clearRect(0, 0, w, h);
  if (!prices.length) return;
  const vals = prices.filter(v => v !== null && v !== undefined);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (lo === hi) { lo -= 1; hi += 1; }
  const range = hi - lo;
  const span = (tMax - tMin) || 1;
  const x = t => pad + ((t - tMin) / span) * (w - 2*pad);
  const y = v => h - pad - ((v - lo) / range) * (h - 2*pad);

  ctx.strokeStyle = COLORS.pnl; ctx.lineWidth = 1.6; ctx.beginPath();
  times.forEach((t, i) => {
    const v = prices[i]; if (v == null) return;
    i === 0 ? ctx.moveTo(x(t), y(v)) : ctx.lineTo(x(t), y(v));
  });
  ctx.stroke();

  fills.forEach(f => {
    if (f.price < lo || f.price > hi) return;
    ctx.fillStyle = f.side === 'buy' ? '#3fb950' : '#f85149';
    ctx.beginPath();
    ctx.arc(x(f.t), y(f.price), 2.5, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.fillStyle = 'rgba(255,255,255,0.35)'; ctx.font = '10px sans-serif';
  ctx.fillText(hi.toFixed(2), 2, 10);
  ctx.fillText(lo.toFixed(2), 2, h - 2);
}

function fmtWhen(ts) {
  return new Date(ts * 1000).toLocaleString(undefined, {
    month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit'
  });
}

async function loadRuns() {
  const picker = document.getElementById('runPicker');
  const prevValue = picker.value;
  let runs;
  try {
    runs = await (await fetch('/runs')).json();
  } catch (e) {
    return;
  }
  if (!runs.length) return;
  picker.innerHTML = '';
  runs.forEach((r, i) => {
    const opt = document.createElement('option');
    opt.value = r.run_id;
    const ratioTag = (r.capture_ratio !== null && r.capture_ratio !== undefined) ? ` [ratio ${r.capture_ratio}]` : '';
    opt.textContent = (r.is_live ? '● Live — ' : '') + fmtWhen(r.start_ts) + ' – ' + fmtWhen(r.end_ts) +
      ` (${r.n_files} files)` + ratioTag;
    picker.appendChild(opt);
  });
  // keep whatever the user had selected, unless this is the very first load
  if (prevValue && [...picker.options].some(o => o.value === prevValue)) {
    picker.value = prevValue;
  }
}

async function tick() {
  const picker = document.getElementById('runPicker');
  const selected = picker.value;
  const url = selected ? `/data?run=${selected}` : '/data';
  let data;
  try {
    const res = await fetch(url);
    data = await res.json();
  } catch (e) {
    document.getElementById('statusDot').className = 'dot stale';
    return;
  }
  if (data.error) {
    document.getElementById('emptyState').style.display = 'block';
    document.getElementById('emptyState').textContent = data.error;
    document.getElementById('content').style.display = 'none';
    document.getElementById('statusDot').className = 'dot stale';
    return;
  }
  document.getElementById('emptyState').style.display = 'none';
  document.getElementById('content').style.display = 'block';

  const lastTs = data.series.t.length ? data.series.t[data.series.t.length - 1] : null;
  const staleLive = data.is_latest && lastTs && (Date.now() - lastTs) > 90000;
  document.getElementById('statusDot').className =
    'dot ' + (!data.is_latest ? 'historical' : (staleLive ? 'stale' : 'live'));

  const durationSec = lastTs ? Math.floor(lastTs / 1000 - data.run_start_ts) : 0;
  const ratioSuffix = (data.capture_ratio !== null && data.capture_ratio !== undefined)
    ? ` · capture ratio ${data.capture_ratio}` : '';
  document.getElementById('subline').textContent = (data.is_latest
    ? `${data.mode} · running ${fmtElapsed(durationSec)} · ${data.rows.toLocaleString()} ticks`
    : `${data.mode} · historical run, lasted ${fmtElapsed(durationSec)} · ${data.rows.toLocaleString()} ticks`) + ratioSuffix;

  const l = data.latest;
  const invEl = document.getElementById('cInventory');
  invEl.textContent = fmtBtc(l.inventory);
  invEl.className = 'value ' + (l.inventory > 0 ? 'pos' : (l.inventory < 0 ? 'neg' : ''));
  document.getElementById('cInventoryRange').textContent =
    `range ${fmtBtc(data.inventory_min)} to ${fmtBtc(data.inventory_max)}`;

  const pnlEl = document.getElementById('cPnl');
  pnlEl.textContent = fmtUsd(l.pnl);
  pnlEl.className = 'value ' + (l.pnl > 0 ? 'pos' : (l.pnl < 0 ? 'neg' : ''));
  document.getElementById('cPnlRange').textContent =
    `range ${fmtUsd(data.pnl_min)} to ${fmtUsd(data.pnl_max)}`;

  document.getElementById('cSpread').textContent =
    fmtUsd(l.spread) + ' / ' + fmtUsd(l.extended_spread);
  document.getElementById('cSpreadRatio').textContent =
    l.extended_spread ? (l.spread / l.extended_spread).toFixed(2) + 'x wider' : '';

  document.getElementById('cSigma').textContent = l.sigma !== null ? fmtUsd(l.sigma) : 'warming up…';
  document.getElementById('cFills').textContent = `${data.fills.buy} / ${data.fills.sell}`;

  const stageEl = document.getElementById('cStage');
  stageEl.textContent = l.unwind_stage;
  stageEl.className = 'badge ' + l.unwind_stage;

  const tMin = data.run_start_ts * 1000, tMax = data.latest_ts;
  drawPriceChart(document.getElementById('chartPrice'), data.series.t, data.series.fair_price, data.fill_markers, tMin, tMax);
  drawDualLineChart(document.getElementById('chartEdge'), data.series.edge_bid, data.series.edge_ask, COLORS.spread, COLORS.inv, {zeroLine: true});
  drawChart(document.getElementById('chartPnl'), data.series.pnl, COLORS.pnl, {zeroLine: true});
  drawDualLineChart(document.getElementById('chartSpread'), data.series.spread, data.series.extended_spread, COLORS.spread, COLORS.extSpread);
  drawChart(document.getElementById('chartSigma'), data.series.sigma, COLORS.sigma);
  drawChart(document.getElementById('chartInventory'), data.series.inventory, COLORS.inv, {zeroLine: true});
}

document.getElementById('runPicker').addEventListener('change', tick);

async function init() {
  await loadRuns();
  await tick();
}
init();
setInterval(tick, REFRESH_MS);
setInterval(loadRuns, 15000);
</script>
</body>
</html>
"""


class Handler(http.server.BaseHTTPRequestHandler):
    def _send_json(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/data":
            run_param = parse_qs(parsed.query).get("run", [None])[0]
            self._send_json(build_status(run_param))
        elif parsed.path == "/runs":
            self._send_json(list_runs())
        else:
            body = DASHBOARD_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *args):
        pass  # keep the terminal quiet; errors still raise normally


if __name__ == "__main__":
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        print(f"Dashboard running at http://127.0.0.1:{PORT}  (reads {LIVE_DATA_DIR}, read-only)")
        httpd.serve_forever()
