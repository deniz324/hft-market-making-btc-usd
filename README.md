# High-Frequency Market Making for BTC-USD Perpetuals

> **[📄 Read the Full Research Paper (PDF)](paper.pdf)**

An empirical research project and implementation of an automated, cross-venue high-frequency market-making (HFT) system for BTC-USD perpetuals on Extended DEX, utilizing an infinite-horizon Avellaneda–Stoikov (A-S) inventory-risk framework.

---

## 📌 Key Highlights & Architecture

- **Cross-Venue Mechanism:** Quotes around Extended DEX's size-weighted microprice while utilizing Binance BTCUSDT purely as a short-horizon lead-lag momentum signal.
- **USDT/USD Basis Correction:** Dynamically polls USDC/USDT to correct for currency basis disconnects and avoid false arbitrage quoting.
- **Robust Realized Volatility ($\sigma$):** $dt$-normalized, Median Absolute Deviation (MAD) estimator designed to handle microsecond WebSocket delivery bursts and quantity-only book updates.
- **Adaptive Parameter Calibration:** Formulates a 2-level UCB1 Contextual Bandit (Reinforcement Learning) to dynamically select risk aversion ($\gamma$) and fill intensity ($\kappa$) across drifting market volatility/volume regimes.
- **Circuit Breaker:** Multi-stage inventory override (passive join $\to$ taker cross) preventing stuck positions caused by structural cross-venue spreads.

---

## 📊 Empirical Results Summary

Evaluated across **3.5M+ live quote ticks** (10-day test corpus) and a **21.1-hour continuous session** (~1.25M ticks, 5,883 simulated fills):
- **Spread Competitiveness:** Quoted an average spread of **$0.74**, approximately **27% tighter** than Extended's native top-of-book ($1.01).
- **Execution Realism:** Paper-fill simulator explicitly models 8.5ms network latency, queue position capture ratios ($0.2$ to $1.0$), and live maker/taker fee tiers (-1.3 bps rebate / 2.25 bps taker).
- **PnL Dynamics:** Demonstrated steady PnL accumulation through normal regimes with honest reporting on intra-session directional order-flow drawdowns.
