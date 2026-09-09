# High-Frequency Market Making for BTC-USD Perpetuals

An empirical research project and event-driven implementation of a BTC-USD perpetual market-making system. The strategy quotes around **Extended's own order book**, uses **Binance BTCUSDT as a short-horizon lead/momentum signal**, and applies an infinite-horizon **Avellaneda-Stoikov** inventory-risk framework.

> **[Read the full research paper](high_frequency_market_making_for_btc_usd_perpetuals.pdf)**

## Project overview

The project evolved from exploratory market-making notebooks into a live, asynchronous quoting engine. The final design deliberately avoids treating the persistent Binance/Extended price difference as free arbitrage: the cross-venue gap was found to be substantially affected by the USDT/USD basis. Extended therefore provides the tradeable reference price, while Binance is used only as a faster directional signal for adverse-selection defence.

### Core features

- **Extended microprice reference** — size-weighted top-of-book microprice is the primary fair-value input.
- **Binance lead signal** — fast/slow EMA momentum from BTCUSDT leans the reservation price without replacing the venue being quoted.
- **USDT/USD basis correction** — USDC/USDT is polled to convert the Binance signal into approximately comparable USD terms.
- **Avellaneda-Stoikov quoting** — inventory-aware reservation price and spread control.
- **Robust realized volatility** — per-observation time normalization, a minimum-dt floor and MAD estimation reduce sensitivity to WebSocket bursts and outliers.
- **Adaptive parameters** — a regime-aware UCB1 contextual bandit selects gamma/kappa combinations across volatility and trading-volume regimes.
- **Adaptive reward shaping** — a second UCB1 bandit evaluates fill-bonus/inventory-penalty coefficients using raw PnL.
- **Inventory circuit breaker** — progressively moves the exit quote toward the touch and can force a taker exit when inventory remains stuck.
- **Execution-aware paper simulation** — models network/cancel latency, partial fills, queue-position capture ratios and maker/taker fees against real Extended public trade prints.
- **Testnet execution path** — optional authenticated mode manages resting orders on Extended testnet; mainnet live trading is deliberately blocked by default.
- **Persistent telemetry** — quote ticks are written to Parquet for subsequent analysis.
- **Read-only dashboard** — monitors PnL, inventory, fills, spread, volatility, quote edge and historical runs without touching the trading process.

## Empirical results

The accompanying research evaluates the system on **3.5M+ live quote ticks** collected over a ten-day test corpus. A highlighted **21.1-hour continuous session** contains roughly **1.25M quote ticks and 5,883 simulated fills**.

In that session, the strategy quoted an average spread of approximately **$0.74**, versus roughly **$1.01** for Extended's native top of book. These are **paper-execution results**, not claims of realized mainnet trading profit. The simulator explicitly incorporates an 8.5 ms network-latency assumption, queue/capture-ratio sensitivity and maker/taker fee treatment.

The paper also discusses failure modes and model revisions rather than presenting only favourable runs, including volatility-estimation failures, structurally uncompetitive cross-venue quoting, inventory drawdowns and the limitations of fitting a smooth arrival-intensity model to touch-dominated fills.

## Repository structure

```text
hft-market-making-btc-usd/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── .env.example
├── src/
│   ├── live_quoter.py
│   └── dashboard.py
├── notebooks/
│   ├── 01_market_making_research.ipynb
│   └── 02_hft_latency_analysis.ipynb
├── paper/
│   └── hft_market_making.tex
└── high_frequency_market_making_for_btc_usd_perpetuals.pdf
```

## Research progression

The notebooks document the development path rather than only the final model:

1. collect and inspect market/order-book data;
2. test fixed-spread market making;
3. identify inventory accumulation and introduce inventory skew;
4. estimate volatility and order-arrival behaviour;
5. implement and calibrate Avellaneda-Stoikov quoting;
6. study latency sensitivity and event-driven execution;
7. move to Extended-native fair value with Binance as a lead signal;
8. add execution realism, circuit breakers and adaptive parameter selection.

## Installation

Python 3.13 was used for the current environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The x10 SDK currently requires `websockets<14`, so the compatible version is pinned in `requirements.txt`.

## Running the quoter

The default mode is **dry-run** and uses public market data only:

```bash
python src/live_quoter.py
```

To test queue-position sensitivity:

```bash
python src/live_quoter.py --capture-ratio 0.5
```

Quote telemetry is periodically flushed to `live_data/` as Parquet files.

### Dashboard

In a second terminal:

```bash
python src/dashboard.py
```

Then open `http://127.0.0.1:8787`.

## Testnet mode

Authenticated execution is intended for **Extended testnet only**. Copy the example environment file and supply your own credentials locally:

```bash
cp .env.example .env
```

Never commit `.env` or API/private keys.

The code deliberately refuses `--live` execution unless the configured Extended environment is TESTNET.

## Important limitations

This repository is a research/portfolio project, not a production trading system or investment recommendation. Paper fills cannot perfectly reconstruct exchange queue priority because Extended does not expose a public L3 queue. Capture ratio, latency and some risk parameters are therefore sensitivity assumptions. Historical or simulated PnL does not imply future performance.

## Paper

The full methodology, empirical results, failure analysis and model evolution are documented in:

**High-Frequency Market Making for BTC-USD Perpetuals**

[Open the PDF](high_frequency_market_making_for_btc_usd_perpetuals.pdf)

## Author

**Deniz Gursu**
