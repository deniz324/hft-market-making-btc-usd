"""
Extended market-maker: quotes around Extended's OWN book, using Binance only as a
fast leading signal, plus Avellaneda-Stoikov for spread/inventory management.

This is a deliberate redesign from the original "quote Binance's price on Extended"
version. That approach anchored quotes near Binance's price even when it persistently
differed from Extended's own tradeable book by tens of dollars — meaning our orders
were frequently uncompetitive or aggressively marketable relative to the market we
were actually quoting on. Verified on real data: the resulting "edge" was mostly a
USDT/USD currency artifact, and what little remained was smaller than round-trip
trading fees. See conversation history / commit log for the full analysis.

This version instead uses Extended's own microprice as the primary reference (so
quotes are inherently centered on the market being traded), and uses Binance purely
as a short-horizon momentum signal (fast-EMA minus slow-EMA) to lean quotes slightly
in the direction Binance appears to be leading — the classical reason to reference a
faster/more-liquid market in market making: adverse-selection defense, not arbitrage.

Default (no flags): dry-run. Streams public market data only, no credentials
needed, only logs/records what it WOULD quote on Extended, and paper-fills
against Extended's real trade prints (see process_public_trade and
TRADE_CAPTURE_RATIO / --capture-ratio) to track a simulated PnL.

--live: places and manages real resting orders on Extended TESTNET using the
credentials in python_mm/.env. Refuses to run --live against MAINNET; change
EXTENDED_ENV below yourself if you ever mean to do that on purpose. In this
mode PnL is read directly from Extended's own position stream (realised +
unrealised) instead of the paper-fill simulation.

Every quote tick is buffered and flushed to ../live_data/quotes_<epoch>.parquet
every FLUSH_INTERVAL_SEC, same pattern as the book_*/trades_*.parquet captures
in market making.ipynb, so a long run leaves something to actually analyze.

Setup:
    pip install -r requirements.txt
    cp .env.example .env   # fill in X10_API_KEY / X10_PUBLIC_KEY / X10_PRIVATE_KEY / X10_VAULT_ID

Extended API reference: https://api.docs.extended.exchange/
Extended Python SDK:    https://github.com/x10xchange/python_sdk/tree/starknet
"""

import argparse
import asyncio
import json
import logging
import os
import ssl
import time
from collections import deque
from decimal import Decimal
from pathlib import Path
from signal import SIGINT, SIGTERM

import certifi

# Must run before any websockets.connect() call (including the x10 SDK's internal
# one for the Extended stream) — macOS Python builds often can't find the system CA
# store, which surfaces as "self-signed certificate in certificate chain" or
# "unable to get local issuer certificate" on an otherwise-valid TLS handshake.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import aiohttp
import numpy as np
import pandas as pd
import websockets
from dotenv import load_dotenv

from x10.clients.rest import RestApiClient
from x10.clients.stream import StreamClient
from x10.config import get_config_by_name
from x10.core.env_config import EnvConfig
from x10.core.stark_account import StarkPerpetualAccount
from x10.models.order import OrderSide

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("live_quoter")

BINANCE_SYMBOL = "btcusdt"
EXTENDED_MARKET = "BTC-USD"
# Binance's BTCUSDT is denominated in USDT; Extended's balance/BTC-USD is denominated in
# (effectively) real USD. USDT doesn't sit exactly at $1 — it's been trading ~7bps below
# real USD — so comparing the two raw numbers silently mixes currencies. We approximate
# real-USD-per-USDT via Binance's own USDC/USDT spot pair, treating USDC as a USD proxy
# (it sits much closer to parity than USDT does). Polled via REST, not streamed: this
# basis moves far slower than BTC's price, so 30s freshness is more than enough.
USDC_USDT_URL = "https://api.binance.com/api/v3/ticker/bookTicker?symbol=USDCUSDT"
USDT_RATE_POLL_INTERVAL_SEC = 30
# Public orderbook data needs no credentials on either environment, so this is safe to
# leave on MAINNET for dry-run viewing. --live refuses to run unless this is "TESTNET" —
# change it yourself, deliberately, the day you actually mean to trade real funds.
EXTENDED_ENV = "MAINNET"

# GAMMA=0.09 was validated in mm_hft.ipynb's sweep against a different (finite-horizon,
# Binance-anchored) formulation than the one actually running now, and it was never revisited
# after the sigma estimator got fixed. Once sigma started reporting real values (mean ~6.57,
# instead of the ~0.08 the broken MAD estimator was producing), gamma*sigma^2 alone hit ~$3.88
# — found, by pooling every fill across all of 2026-07-22/23's runs, to already exceed the
# entire distance range where fills happen at all (fill rate is ~15-25% inside Extended's own
# touch, and indistinguishable from zero anywhere $0.50+ behind it — no kappa value fixes that,
# since kappa only adds an extra term on top of gamma*sigma^2). Dropped to 0.02 so the inventory
# floor (~$0.86 at that sigma) sits inside the empirically-fillable range instead of past it —
# a reasoned starting point, not a fitted one; needs a real run to confirm.
#
# 2026-07-27: dropped further (0.02 -> 0.005) specifically for backtesting/data-gathering —
# explicitly requested looser, easier-to-fill quoting to generate more fill volume for
# analysis, not a claim this is a realistic production risk setting. Combined with the
# KAPPA_DOLLAR bump below, this puts the modeled spread at roughly $0.5-1.2 across the sigma
# range actually observed (3.8-12.5) — at or tighter than Extended's own ~$1 touch, so expect
# frequent fills and correspondingly less real inventory-risk protection.
GAMMA = 0.005
# Order-arrival intensity. The original 0.181661 was calibrated in mm_hft.ipynb as
# 1 / mean(|trade_price - mid|) over BINANCE's trade flow (avg distance ~$5.50) — on a deep
# market, so it implied a wide realistic trading distance and, through the formula, a spread
# nowhere near competitive with Extended's own ~$1 market (confirmed live: produced a stable
# but 9x-too-wide $9 spread that never got filled once). This value instead assumes trades on
# Extended happen within roughly its own spread (~$1 average distance from mid) — still a guess,
# not measured from Extended's real fills (we don't have that data yet), but a much more
# plausible one given the market it's actually quoting on. A regression fit against real fill
# data (2026-07-23) gave kappa≈0.51, but the underlying fill-vs-distance shape looks like a
# cliff at the touch, not the smooth exponential decay the model assumes, so that fit isn't
# trustworthy enough to use — left as-is pending better data.
#
# 2026-07-27: bumped 1.0 -> 5.0 alongside the GAMMA drop above, same backtesting rationale —
# higher kappa shrinks the (2/gamma)*ln(1+gamma/kappa) term, tightening the spread further to
# make fills easier to come by.
#
# 2026-07-28: GAMMA and KAPPA_DOLLAR are now just the seed values used before the bandit below
# has picked anything for the current regime — see GammaKappaBandit. Note the direction here:
# *higher* kappa and *lower* gamma both narrow the spread (see the formula in
# avellaneda_stoikov_quotes), so "more fills in a quiet market" means the bandit should push
# kappa up and gamma down when quiet, not the reverse — it's easy to get this backwards since
# kappa is usually read as a "how cautious" dial, but here it's the opposite of gamma's role.
KAPPA_DOLLAR = 5.0
# Floor for the per-sample dt used in realized_sigma_dollar()'s normalization — see the function's
# docstring. Confirmed via a live trace of Extended's raw feed: successive WS messages can arrive
# with microsecond gaps, and normalizing by an unfloored near-zero dt turns an ordinary few-cent
# tick into an apparent enormous instantaneous move.
MIN_TICK_DT_SEC = 0.005
VOL_WINDOW = 200  # samples used for the rolling realized-vol estimate (now on Extended's own ticks)
EMA_ALPHA = 0.3  # smoothing on Extended's own microprice (the quoting reference)
BINANCE_EMA_ALPHA_FAST = 0.3  # matches EMA_ALPHA; used only for the momentum signal now
BINANCE_EMA_ALPHA_SLOW = 0.05
# Fraction of Binance's fast-minus-slow EMA gap (a simple momentum/lead signal) to lean the
# reservation price by. Direct dollar pass-through, deliberately NOT routed through gamma*sigma^2
# the way inventory skew is — that's exactly the mechanism we found was ~500,000x too weak to
# matter. Uncalibrated first guess, same caveat as every other constant marked as such here.
MOMENTUM_WEIGHT = 0.3
MAX_FAIR_PRICE_STALENESS_SEC = 1.0
QUOTE_INTERVAL_SEC = 0.05

# --- live-order-management-only settings ---
# 2026-07-27: widened 0.01 -> 1.0 BTC for backtesting — explicitly requested so the strategy
# has room to run without constantly pinning at the cap, given fills should now come much
# more often (see GAMMA/KAPPA_DOLLAR above). Also means real PnL swings will be much larger
# in dollar terms than every prior run in this file's history — expect that, not a bug.
MAX_INVENTORY = Decimal("1.0")  # BTC; stop adding to a side once inventory breaches this
REPRICE_THRESHOLD_BPS = Decimal("2")  # don't cancel/replace unless target moved this far
MIN_REQUOTE_INTERVAL_SEC = 0.5  # throttle order churn independent of the quote-log cadence
# Hard sanity bound, independent of the sigma fix above: a real run produced a target_ask of
# $79,773 against a true price of ~$66,600 from one bad sigma sample. Whatever the exact cause,
# no quote should ever get this far from fair value — skip the tick entirely if it does.
MAX_QUOTE_DEVIATION_PCT = 0.02

# --- stuck-inventory circuit breaker ---
# A-S's inventory skew moves price by inventory*gamma*sigma^2 — measured from a real run at
# roughly $0.0001 at max position, against typical Binance-vs-Extended gaps of $10s. That's off
# by ~5-6 orders of magnitude, so the skew can never bridge a structural (non-noise) gap on its
# own, and a position can sit stuck indefinitely. These thresholds are uncalibrated first guesses,
# same caveat as MAX_INVENTORY — flag if you want them set more deliberately.
STUCK_WARN_SEC = 300  # stuck this long: quote the exit side at Extended's own top-of-book (still maker)
STUCK_FORCE_SEC = 1200  # stuck this long: cross the spread deliberately to guarantee an exit (taker)

# --- dry-run paper-fill settings (ignored once --live is on real position/PnL) ---
# 2026-07-27: bumped 0.001 -> 0.01 BTC (10x) for backtesting, alongside the MAX_INVENTORY
# widening above — same rationale, more size per fill for a richer dataset.
PAPER_ORDER_SIZE = Decimal("0.01")  # BTC; dry-run has no market info, so this isn't
# rounded to Extended's real min_order_size the way --live's order_size is.

# Negotiated rate confirmed 2026-07-24: zero maker cost, straight 1.3bps rebate — we get paid
# on every maker fill, not charged. Negative bps below makes process_public_trade credit that
# rebate on top of the notional instead of deducting a cost. Taker fee (only paid on the
# circuit breaker's forced-cross exits, see process_public_trade) confirmed at 2.25bps.
MAKER_FEE_BPS = Decimal("-1.3")
TAKER_FEE_BPS = Decimal("2.25")

# --- latency simulation (dry-run paper fills only; --live's orders already cross a real
# network so there's nothing to simulate there) ---
# Without this, a decided quote was treated as resting and fillable the instant the quote loop
# computed it, and a decided reprice/cancel was treated as having removed the old price the same
# instant — neither is true of a real order, which has to travel to the exchange first. Applied
# as network/cancel round-trip estimates for this setup; not yet measured against Extended
# specifically.
NETWORK_LATENCY_SEC = 0.0085  # time from deciding a quote to it actually resting on the book
CANCEL_LATENCY_SEC = 0.004  # time from deciding to replace a quote to the old one leaving the book

# --- queue-position simulation (dry-run paper fills only) ---
# Not every real trade that crosses our resting price actually reaches us: other resting
# orders at the same price (queue priority) and faster participants both get served first.
# We don't have Extended's per-order queue (no public L3 feed), so instead of modeling the
# queue directly, we treat this as a swept sensitivity parameter: only TRADE_CAPTURE_RATIO of
# each qualifying public trade's quantity is assumed to reach our order. Set via --capture-ratio
# (e.g. 0.2/0.5/0.7); defaults to 1.0 (every crossing trade fully fills us) to preserve the old
# behavior when the flag isn't passed. Overwritten from args in main() — the module-level
# default only matters if main() never runs (e.g. importing this file directly).
TRADE_CAPTURE_RATIO = Decimal("1.0")

# --- adaptive gamma/kappa (contextual bandit) ---
# GAMMA/KAPPA_DOLLAR above are frozen guesses picked (and re-picked) by hand. This replaces them
# with a value chosen per market "regime" via a UCB1 bandit over a small candidate grid — full
# reinforcement learning would be overkill and riskier here: only 2 parameters, a slowly-shifting
# regime, and we'd rather have something whose choices are cheap to audit than a trained policy
# that's opaque about why it picked what it picked.
GAMMA_CANDIDATES = [0.002, 0.005, 0.01]  # trimmed from [0.002, 0.005, 0.01, 0.02] — 0.02 was the
# single worst-performing value across every kappa it was paired with in bandit_state.json, and
# cutting it plus two kappa candidates below took the grid from 20 to 9 arms/regime so the bandit
# can actually finish exploring (and start exploiting) within a realistic session length.
KAPPA_CANDIDATES = [1.0, 5.0, 20.0]  # trimmed from [1.0, 3.0, 5.0, 10.0, 20.0] — kept low/mid/high
# rather than the adjacent in-between values, same low/mid/high shape as the regime buckets.
BANDIT_EVAL_WINDOW_SEC = 120  # how long a chosen (gamma, kappa) is held before it's re-scored
# Reward = PnL delta over the window + this bonus per fill. Pure PnL alone would already be a
# valid reward (the maker rebate per fill is baked into it), but early on — with KAPPA_DOLLAR
# itself flagged above as "a guess, not measured from real fills" — we want the bandit to still
# favor generating fill data even while PnL signal is thin and noisy. Uncalibrated first guess,
# same caveat as every other constant marked as such in this file.
BANDIT_FILL_BONUS_USD = 0.05  # initial default only — overwritten by the coef bandit below once
# its first window closes, same relationship GAMMA/KAPPA_DOLLAR have to the gamma/kappa bandit.
BANDIT_UCB_C = 2.0  # exploration strength in the UCB1 score; higher = slower to commit to a winner
# Penalizes reward by inventory left resting at window-close (not the window's peak — PnL is
# already mark-to-market, so inventory that spiked and was unwound within the window already
# shows up in pnl_delta; what's missing is the risk handed off to whatever arm runs next),
# scaled by sigma since the same leftover position is riskier when the market's moving more.
# Initial default only, same as BANDIT_FILL_BONUS_USD above — see coef bandit.
BANDIT_INVENTORY_PENALTY_USD = 1.0
# Sigma and recent-trade-volume are each bucketed low/mid/high against their own trailing
# percentiles (not fixed thresholds) — the point is to adapt as the market's normal range shifts,
# not to encode today's specific sigma range as a permanent assumption.
REGIME_HISTORY = 500
TRADE_VOLUME_WINDOW_SEC = 60
# Bandit's learned (regime, gamma, kappa) -> reward stats, so a restart resumes instead of going
# back to pure exploration. Not the trading state itself, so a stale/corrupt file is safe to
# delete — it just costs some re-exploration, not correctness.
BANDIT_STATE_PATH = Path(__file__).resolve().parent / "bandit_state.json"

# --- adaptive reward shaping (meta-bandit over the gamma/kappa bandit's own reward coefficients) ---
# BANDIT_FILL_BONUS_USD/BANDIT_INVENTORY_PENALTY_USD are themselves guesses, same as every other
# constant flagged as such in this file. Rather than hand-pick them, a second, independent UCB1
# bandit picks a (fill_bonus, inventory_penalty) pair globally (no regime split — these shape the
# reward signal itself, not the quotes, so there's no a priori reason to expect them to be
# regime-dependent the way gamma/kappa are) and scores it by RAW pnl_delta over a much longer
# window. Deliberately NOT the shaped reward those coefficients define — judging a coefficient
# choice by a metric it controls would be circular; only real PnL tells you if the shaping helped.
FILL_BONUS_CANDIDATES = [0.0, 0.05, 0.15]
INVENTORY_PENALTY_CANDIDATES = [0.0, 0.5, 1.0, 2.0]
# Long relative to BANDIT_EVAL_WINDOW_SEC on purpose: a coefficient pair only shows up in the
# gamma/kappa bandit's behavior over many of its 120s windows, so judging it any faster would
# mostly be measuring within-window noise rather than the coefficients' actual effect.
COEF_EVAL_WINDOW_SEC = 1800  # 30 min
COEF_STATE_PATH = Path(__file__).resolve().parent / "coef_bandit_state.json"

# --- recording ---
LIVE_DATA_DIR = Path(__file__).resolve().parent.parent / "live_data"
FLUSH_INTERVAL_SEC = 60  # matches the book_*/trades_*.parquet capture cadence in market making.ipynb


class PaperRestingOrder:
    """One side (bid or ask) of the dry-run paper book, with NETWORK_LATENCY_SEC/
    CANCEL_LATENCY_SEC applied. `reprice()` is called once per quote_loop tick with the newly
    decided target; `fill()` is called from process_public_trade() for every real trade print,
    since a trade can cross our price faster than the quote loop's own tick rate — that's the
    whole point of simulating sub-tick latency in the first place.

    Tracks `remaining` size at the live price (and at the just-superseded one, during the
    cancel/replace handoff) so a single trade print can only ever consume up to PAPER_ORDER_SIZE
    per resting order, same as a real order would be exhausted rather than infinitely refillable."""

    def __init__(self):
        self.price = None
        self.live_ts = None  # None, or a future timestamp, until the order is genuinely resting
        self.is_taker = False
        self.remaining = Decimal("0")  # size still available to fill at `price`
        self.prev_price = None  # price this one superseded, still resting until its cancel lands
        self.prev_expire_ts = None
        self.prev_is_taker = False
        self.prev_remaining = Decimal("0")

    def reprice(self, new_price, is_taker, now):
        if new_price == self.price and is_taker == self.is_taker:
            return  # target unchanged, nothing to replace
        if self.live_ts is not None and now >= self.live_ts and self.price is not None and self.remaining > 0:
            self.prev_price = self.price
            self.prev_is_taker = self.is_taker
            self.prev_expire_ts = now + CANCEL_LATENCY_SEC
            self.prev_remaining = self.remaining
        self.price = new_price
        self.is_taker = is_taker
        self.live_ts = now + NETWORK_LATENCY_SEC
        self.remaining = PAPER_ORDER_SIZE

    def fill(self, price_crosses, capture_qty, now):
        """price_crosses(price) -> bool decides whether a given resting price would be hit by
        this trade. Consumes up to capture_qty from whichever of the superseded/current price
        is live and crossed (checking the superseded one first, since it's the one about to
        expire). Returns (fill_price, fill_qty, is_taker) or None if nothing was fillable."""
        if self.prev_expire_ts is not None and now >= self.prev_expire_ts:
            self.prev_price = self.prev_expire_ts = None
            self.prev_remaining = Decimal("0")

        if self.prev_price is not None and self.prev_remaining > 0 and price_crosses(self.prev_price):
            qty = min(self.prev_remaining, capture_qty)
            self.prev_remaining -= qty
            return self.prev_price, qty, self.prev_is_taker

        if self.live_ts is not None and now >= self.live_ts and self.remaining > 0 and price_crosses(self.price):
            qty = min(self.remaining, capture_qty)
            self.remaining -= qty
            return self.price, qty, self.is_taker

        return None


class GammaKappaBandit:
    """UCB1 contextual bandit over GAMMA_CANDIDATES x KAPPA_CANDIDATES, one independent set of
    arm statistics per market regime string (see current_regime()). `select()` always tries an
    arm this regime has never seen before trying to rank tried ones by UCB score, so a brand new
    regime starts by sweeping the whole grid rather than committing early on one data point.

    Persisted to BANDIT_STATE_PATH as {"regime|gamma|kappa": {"n": count, "mean": running mean
    reward}} — plain JSON on purpose, so it's easy to inspect/reset by hand if it looks wrong."""

    def __init__(self):
        self.stats = {}
        self._load()

    def _load(self):
        if BANDIT_STATE_PATH.exists():
            try:
                self.stats = json.loads(BANDIT_STATE_PATH.read_text())
            except Exception as e:
                log.warning("Bandit state load failed (%s), starting fresh", e)
                self.stats = {}

    def save(self):
        try:
            BANDIT_STATE_PATH.write_text(json.dumps(self.stats))
        except Exception as e:
            log.warning("Bandit state save failed (%s)", e)

    @staticmethod
    def _key(regime, gamma, kappa):
        return f"{regime}|{gamma}|{kappa}"

    def select(self, regime):
        total_n = sum(
            self.stats.get(self._key(regime, g, k), {}).get("n", 0)
            for g in GAMMA_CANDIDATES for k in KAPPA_CANDIDATES
        )
        best_score, best_arm = None, (GAMMA_CANDIDATES[0], KAPPA_CANDIDATES[0])
        for g in GAMMA_CANDIDATES:
            for k in KAPPA_CANDIDATES:
                entry = self.stats.get(self._key(regime, g, k))
                if entry is None or entry["n"] == 0:
                    return g, k  # untried in this regime — always worth a first look
                ucb = entry["mean"] + BANDIT_UCB_C * (np.log(max(total_n, 1)) / entry["n"]) ** 0.5
                if best_score is None or ucb > best_score:
                    best_score, best_arm = ucb, (g, k)
        return best_arm

    def update(self, regime, gamma, kappa, reward):
        entry = self.stats.setdefault(self._key(regime, gamma, kappa), {"n": 0, "mean": 0.0})
        entry["n"] += 1
        entry["mean"] += (reward - entry["mean"]) / entry["n"]


class RewardCoefBandit:
    """UCB1 bandit over FILL_BONUS_CANDIDATES x INVENTORY_PENALTY_CANDIDATES — see the module
    comment above those. One global arm set, no regime split. Same UCB1/persistence shape as
    GammaKappaBandit, just over a different, longer-lived pair of knobs."""

    def __init__(self):
        self.stats = {}
        self._load()

    def _load(self):
        if COEF_STATE_PATH.exists():
            try:
                self.stats = json.loads(COEF_STATE_PATH.read_text())
            except Exception as e:
                log.warning("Coef bandit state load failed (%s), starting fresh", e)
                self.stats = {}

    def save(self):
        try:
            COEF_STATE_PATH.write_text(json.dumps(self.stats))
        except Exception as e:
            log.warning("Coef bandit state save failed (%s)", e)

    @staticmethod
    def _key(fill_bonus, inv_penalty):
        return f"{fill_bonus}|{inv_penalty}"

    def select(self):
        total_n = sum(
            self.stats.get(self._key(f, p), {}).get("n", 0)
            for f in FILL_BONUS_CANDIDATES for p in INVENTORY_PENALTY_CANDIDATES
        )
        best_score, best_arm = None, (FILL_BONUS_CANDIDATES[0], INVENTORY_PENALTY_CANDIDATES[0])
        for f in FILL_BONUS_CANDIDATES:
            for p in INVENTORY_PENALTY_CANDIDATES:
                entry = self.stats.get(self._key(f, p))
                if entry is None or entry["n"] == 0:
                    return f, p  # untried — always worth a first look
                ucb = entry["mean"] + BANDIT_UCB_C * (np.log(max(total_n, 1)) / entry["n"]) ** 0.5
                if best_score is None or ucb > best_score:
                    best_score, best_arm = ucb, (f, p)
        return best_arm

    def update(self, fill_bonus, inv_penalty, reward):
        entry = self.stats.setdefault(self._key(fill_bonus, inv_penalty), {"n": 0, "mean": 0.0})
        entry["n"] += 1
        entry["mean"] += (reward - entry["mean"]) / entry["n"]


class SharedState:
    def __init__(self):
        self.live_mode = False  # set in main(); gates process_public_trade (dry-run only)
        self.usdt_usd_rate = None  # real USD per USDT; used to convert Binance's momentum signal

        # Binance: momentum/lead signal only, no longer the quoting reference
        self.binance_price = None  # fast EMA, real-USD terms
        self.binance_price_slow = None  # slow EMA, same units — the gap between the two is momentum
        self.binance_price_ts = None

        # Extended: now the primary quoting reference (fair_price) and the sigma input
        self.fair_price = None  # EMA of Extended's own microprice
        self.fair_price_ts = None
        self.last_mid = None
        self.last_mid_ts = None
        self.price_diffs = deque(maxlen=VOL_WINDOW)  # dollar diffs between raw Extended mid ticks
        self.tick_dts = deque(maxlen=VOL_WINDOW)  # seconds between those same ticks
        self.extended_bid = None
        self.extended_ask = None
        self.inventory = Decimal("0")  # synced from Extended position stream in --live mode
        self.bid_order_id = None
        self.bid_order_price = None
        self.ask_order_id = None
        self.ask_order_price = None
        self.last_requote_ts = 0.0

        # last time |inventory| decreased (or hit 0); drives the stuck-inventory circuit breaker
        self.last_unwind_ts = time.time()

        # dry-run paper-fill PnL (unused once --live; real PnL comes from Extended instead)
        self.paper_cash = Decimal("0")
        self.paper_bid = PaperRestingOrder()
        self.paper_ask = PaperRestingOrder()

        # --live PnL, read straight from Extended's own position stream
        self.realised_pnl = Decimal("0")
        self.unrealised_pnl = Decimal("0")

        # buffered rows flushed to live_data/quotes_<epoch>.parquet every FLUSH_INTERVAL_SEC
        self.record_buffer = []
        self.last_flush_ts = time.time()

        # adaptive gamma/kappa — see GammaKappaBandit and bandit_step()
        self.fill_count = 0  # running total, both dry-run and --live; drives the bandit's reward
        self.trade_volume_window = deque()  # (timestamp, qty) pairs, pruned to TRADE_VOLUME_WINDOW_SEC
        self.sigma_history = deque(maxlen=REGIME_HISTORY)
        self.volume_history = deque(maxlen=REGIME_HISTORY)
        self.bandit = GammaKappaBandit()
        self.current_gamma = GAMMA  # overwritten by the bandit once the first window closes
        self.current_kappa = KAPPA_DOLLAR
        self.bandit_regime = None  # regime the currently-active arm was selected for
        self.bandit_window_start_ts = time.time()
        self.bandit_window_start_fills = 0
        self.bandit_window_start_pnl = 0.0

        # meta-bandit over the gamma/kappa bandit's own reward-shaping coefficients — see
        # RewardCoefBandit. No regime dependency, so (unlike current_gamma/current_kappa above)
        # it can pick immediately rather than waiting for the first window to close.
        self.coef_bandit = RewardCoefBandit()
        self.current_fill_bonus, self.current_inv_penalty = self.coef_bandit.select()
        self.coef_window_start_ts = time.time()
        self.coef_window_start_pnl = 0.0


state = SharedState()


def microprice(bid_px, bid_qty, ask_px, ask_qty):
    return (bid_px * ask_qty + ask_px * bid_qty) / (bid_qty + ask_qty)


def on_binance_book_ticker(data):
    """Binance is now a momentum/lead signal only — see module docstring. Maintains a fast and
    slow EMA of Binance's real-USD price; their gap (binance_momentum(), below) is what actually
    gets used, as a small lean on the reservation price, not as the quoting reference itself."""
    if state.usdt_usd_rate is None:
        return  # don't touch anything with an unconverted USDT number, even once

    bid_px, bid_qty = float(data["b"]), float(data["B"])
    ask_px, ask_qty = float(data["a"]), float(data["A"])
    raw = microprice(bid_px, bid_qty, ask_px, ask_qty) * state.usdt_usd_rate

    state.binance_price = raw if state.binance_price is None else (
        BINANCE_EMA_ALPHA_FAST * raw + (1 - BINANCE_EMA_ALPHA_FAST) * state.binance_price
    )
    state.binance_price_slow = raw if state.binance_price_slow is None else (
        BINANCE_EMA_ALPHA_SLOW * raw + (1 - BINANCE_EMA_ALPHA_SLOW) * state.binance_price_slow
    )
    state.binance_price_ts = time.time()


def binance_momentum():
    """Dollar bias to lean the reservation price by: positive means Binance's fast EMA is
    running above its slow EMA (recent upward lead), negative means the reverse. 0 if Binance
    data is missing or stale — momentum is a nice-to-have signal, not something worth halting
    quoting over the way a stale *primary* reference (Extended's own book) would be."""
    if state.binance_price is None or state.binance_price_slow is None:
        return 0.0
    if time.time() - state.binance_price_ts > MAX_FAIR_PRICE_STALENESS_SEC:
        return 0.0
    return MOMENTUM_WEIGHT * (state.binance_price - state.binance_price_slow)


def recent_trade_volume():
    """Sum of public-trade quantity over the trailing TRADE_VOLUME_WINDOW_SEC — the "how much is
    actually happening right now" half of the bandit's regime signal, alongside sigma."""
    cutoff = time.time() - TRADE_VOLUME_WINDOW_SEC
    while state.trade_volume_window and state.trade_volume_window[0][0] < cutoff:
        state.trade_volume_window.popleft()
    return sum(qty for _, qty in state.trade_volume_window)


def _bucket(value, history):
    """low/mid/high against the trailing distribution in `history`, not a fixed threshold — so
    this keeps adapting if the market's normal sigma/volume range drifts over a long deployment."""
    if len(history) < 20:
        return "mid"  # not enough history yet to bucket meaningfully
    lo, hi = np.percentile(history, [33, 67])
    if value <= lo:
        return "low"
    if value >= hi:
        return "high"
    return "mid"


def current_regime(sigma):
    """Market regime string for the bandit: sigma level x recent trade activity, each bucketed
    low/mid/high. Also the place trailing history for both gets appended, so call this once per
    quote_loop tick (not just when the bandit is about to act) to keep the buckets current."""
    state.sigma_history.append(sigma)
    volume = recent_trade_volume()
    state.volume_history.append(volume)
    return f"{_bucket(sigma, state.sigma_history)}_sigma__{_bucket(volume, state.volume_history)}_vol"


def bandit_step(sigma, pnl):
    """Called once per quote_loop tick. Updates the regime history every tick; only actually
    scores the outgoing arm and picks a new one every BANDIT_EVAL_WINDOW_SEC, since gamma/kappa
    need to be held steady for a while to observe their effect on fills/PnL."""
    regime = current_regime(sigma)
    now = time.time()

    if state.bandit_regime is None:
        state.bandit_regime = regime
        state.current_gamma, state.current_kappa = state.bandit.select(regime)
        state.bandit_window_start_ts = now
        state.bandit_window_start_fills = state.fill_count
        state.bandit_window_start_pnl = pnl
        log.info("Bandit: initial pick for regime=%s -> gamma=%s kappa=%s", regime, state.current_gamma, state.current_kappa)
        return

    if now - state.bandit_window_start_ts < BANDIT_EVAL_WINDOW_SEC:
        return

    fills = state.fill_count - state.bandit_window_start_fills
    pnl_delta = pnl - state.bandit_window_start_pnl
    # Inventory left resting at window-close, not the window's peak — see BANDIT_INVENTORY_PENALTY_USD
    # above for why. Scaled by sigma so the same leftover position costs more in a choppier market.
    inv_penalty = state.current_inv_penalty * abs(float(state.inventory)) * sigma
    reward = pnl_delta + state.current_fill_bonus * fills - inv_penalty
    state.bandit.update(state.bandit_regime, state.current_gamma, state.current_kappa, reward)
    log.info(
        "Bandit: regime=%s gamma=%s kappa=%s -> reward=%.4f (fills=%d pnl_delta=%.4f inv_penalty=%.4f)",
        state.bandit_regime, state.current_gamma, state.current_kappa, reward, fills, pnl_delta, inv_penalty,
    )

    state.current_gamma, state.current_kappa = state.bandit.select(regime)
    state.bandit_regime = regime
    state.bandit_window_start_ts = now
    state.bandit_window_start_fills = state.fill_count
    state.bandit_window_start_pnl = pnl
    state.bandit.save()
    log.info("Bandit: next pick for regime=%s -> gamma=%s kappa=%s", regime, state.current_gamma, state.current_kappa)


def coef_bandit_step(pnl):
    """Called once per quote_loop tick, alongside bandit_step(). Re-scores the outgoing
    (fill_bonus, inventory_penalty) pair every COEF_EVAL_WINDOW_SEC using raw pnl_delta — never
    the shaped reward those coefficients themselves define, see RewardCoefBandit for why."""
    now = time.time()
    if now - state.coef_window_start_ts < COEF_EVAL_WINDOW_SEC:
        return

    reward = pnl - state.coef_window_start_pnl
    state.coef_bandit.update(state.current_fill_bonus, state.current_inv_penalty, reward)
    log.info(
        "CoefBandit: fill_bonus=%s inv_penalty=%s -> reward=%.4f",
        state.current_fill_bonus, state.current_inv_penalty, reward,
    )

    state.current_fill_bonus, state.current_inv_penalty = state.coef_bandit.select()
    state.coef_window_start_ts = now
    state.coef_window_start_pnl = pnl
    state.coef_bandit.save()
    log.info("CoefBandit: next pick -> fill_bonus=%s inv_penalty=%s", state.current_fill_bonus, state.current_inv_penalty)


def on_extended_book_update(bid_px, bid_qty, ask_px, ask_qty):
    """Extended's own book is now the primary quoting reference and the sigma input — see
    module docstring for why this changed from the original Binance-anchored design."""
    raw = microprice(bid_px, bid_qty, ask_px, ask_qty)
    now = time.time()

    state.fair_price = raw if state.fair_price is None else (
        EMA_ALPHA * raw + (1 - EMA_ALPHA) * state.fair_price
    )

    # Fifth failure of the sigma-estimator saga: even after filtering exact-zero diffs (see
    # realized_sigma_dollar's comment), 67% of ticks still showed a sub-$0.001 "move" because
    # the qty-weighted microprice wobbles on every size-only book update even when neither the
    # bid nor ask PRICE LEVEL actually changed. That cloud of near-zero noise dragged the
    # MAD-based sigma right back down to ~0.087 — same as the pre-filter bug. Fix: feed sigma
    # off the plain price-level mid, which only moves when the touch itself moves. fair_price
    # (used for quoting) keeps using the qty-weighted microprice — that wobble is a legitimate,
    # if small, part of fair value; it's just not volatility.
    #
    # The price-level mid only actually moves on ~1.4% of raw ticks (confirmed from a real
    # run). Appending every tick and filtering zeros at compute time (like the fourth fix did)
    # would mean a 200-sample rolling window holds only ~3 real moves — under the 20-sample
    # minimum realized_sigma_dollar() requires, so it would return None almost permanently and
    # the quote loop would just stop quoting (`continue`s forever on `sigma is None`). Fix:
    # skip the no-op ticks at the append site, so the window holds the last 200 *real* price
    # moves, not the last 200 messages. dt is then correctly "time since the last real move"
    # instead of "time since the last message", which is also the more honest denominator.
    book_mid = (bid_px + ask_px) / 2
    if book_mid != state.last_mid:
        if state.last_mid is not None:
            dt = now - state.last_mid_ts
            if dt > 0:
                state.price_diffs.append(book_mid - state.last_mid)
                state.tick_dts.append(dt)
        state.last_mid = book_mid
        state.last_mid_ts = now
    state.fair_price_ts = now


def realized_sigma_dollar():
    # mm_hft.ipynb's cell 0 computed std(diffs) / sqrt(mean dt) over one batch where every
    # snapshot was ~100ms apart, so pooling was fine. Live ticks aren't evenly spaced (network
    # jitter, reconnects), so a single sample with an outsized dt/diff — e.g. one big real
    # price move during a several-second reconnect gap — got pooled against everyone else's
    # tiny average dt and blew sigma up by ~600x for the next ~200 ticks (confirmed from a real
    # run: target_ask briefly hit $79,773 against a true price of ~$66,600). Fix: normalize each
    # sample by its own dt before taking std, so one long-interval sample can't dominate.
    #
    # Second, different failure of the same shape, found by tracing Extended's raw feed directly:
    # its WebSocket sometimes delivers messages in bursts with MICROSECOND gaps (0.00002s, not the
    # ~50ms we assumed), so a real few-cent price diff divided by sqrt(near-zero) explodes instead
    # of shrinking. Confirmed from a real run: sigma pinned at 149 while the book was provably
    # static, and target quotes widened to $2000+ from fair — nothing could ever fill that wide.
    # Fix: floor dt at MIN_TICK_DT_SEC before normalizing, so a burst can't be mistaken for an
    # instantaneous, therefore infinitely-fast, price move.
    # Third failure of the same shape, found from the 2026-07-22 40-minute run: even with
    # the dt floor, a couple of genuinely fast-but-real ticks (two book updates a few ms
    # apart, each a real $0.50-$1 move) get amplified by 1/sqrt(dt) into normalized values
    # 10-15x the typical one, and plain std() lets a couple of those dominate a 200-sample
    # window (sigma spiked to ~30 against a ~2 baseline, widening quotes to $50 right while
    # already stuck unwinding). Fix: use MAD (median absolute deviation), scaled to be
    # std-equivalent under normality, which a handful of outliers can't dominate the way a
    # raw std can.
    #
    # Fourth and fifth failures, found from the runs right after the MAD fix shipped: over half
    # of Extended's consecutive book ticks carried no real price information — first found as
    # exact-zero microprice diffs (quantity-only updates), then as sub-$0.001 microprice wobble
    # from qty changes at an unchanged price level. Either way, once a majority of a 200-sample
    # window is a repeated/near-repeated value, the median — and therefore MAD, which is built
    # from the median — collapses toward it (confirmed live: sigma_dollar pinned at ~0.08 vs.
    # ~2.1 before, and target spread flatlined regardless of real volatility). Fix ended up
    # living in on_extended_book_update: only append a sample when the price-level mid actually
    # moves, so price_diffs never contains a zero/near-zero entry to begin with.
    if len(state.price_diffs) < 20:
        return None
    diffs = np.array(state.price_diffs)
    dts = np.maximum(np.array(state.tick_dts), MIN_TICK_DT_SEC)
    normalized = diffs / np.sqrt(dts)
    median = np.median(normalized)
    mad = np.median(np.abs(normalized - median))
    return float(1.4826 * mad)


def avellaneda_stoikov_quotes(fair_price, inventory, sigma_dollar, gamma, kappa_dollar, momentum_skew=0.0):
    # Infinite-horizon (stationary) approximation — no (T-t) decay term, since this
    # quotes continuously rather than winding down toward a fixed session end. Note this
    # wasn't the formulation mm_hft.ipynb validated gamma against (it used a finite,
    # decaying session horizon) — same sigma/kappa units, but the horizon treatment differs.
    # momentum_skew is a direct dollar bias (see binance_momentum()) — deliberately NOT routed
    # through gamma*sigma^2 like inventory skew is, since that's the mechanism we measured as
    # ~500,000x too weak to move price meaningfully against a real signal.
    reservation_price = fair_price - inventory * gamma * sigma_dollar**2 + momentum_skew
    spread = gamma * sigma_dollar**2 + (2 / gamma) * np.log(1 + gamma / kappa_dollar)
    return reservation_price - spread / 2, reservation_price + spread / 2


def current_pnl(live_ctx, fair_price):
    if live_ctx:
        # Authoritative: Extended's own accounting (real fills, real fees), not ours.
        return state.realised_pnl + state.unrealised_pnl
    # Paper mode: cash + mark-to-market inventory, same shape as mm_hft.ipynb's cash+inventory*mid.
    return state.paper_cash + state.inventory * Decimal(str(fair_price))


def note_inventory_change(old_inventory, new_inventory):
    """Feeds the stuck-inventory circuit breaker: the clock resets whenever a fill actually
    reduces our position (or flattens it), and keeps running otherwise."""
    if new_inventory == 0 or abs(new_inventory) < abs(old_inventory):
        state.last_unwind_ts = time.time()


def unwind_stage():
    """none / warn / force, based on how long |inventory| has gone without shrinking."""
    if state.inventory == 0:
        return "none"
    stuck_sec = time.time() - state.last_unwind_ts
    if stuck_sec >= STUCK_FORCE_SEC:
        return "force"
    if stuck_sec >= STUCK_WARN_SEC:
        return "warn"
    return "none"


def apply_unwind_override(bid, ask, stage):
    """Vanilla A-S can't bridge a structural gap (see STUCK_* comment above), so once we've
    been stuck long enough this overrides the exit side's price to actually compete with —
    or, at 'force', deliberately cross — Extended's real book instead of waiting forever.
    Returns (bid, ask, force_taker_side) where force_taker_side is None unless we should
    submit that side without post_only to guarantee an immediate fill."""
    if stage == "none" or state.extended_bid is None or state.extended_ask is None:
        return bid, ask, None

    if state.inventory > 0:  # long: exit side is the ask
        if stage == "force":
            return bid, state.extended_bid, OrderSide.SELL  # cross into their bid: guaranteed fill
        return bid, state.extended_ask, None  # join their ask: still a passive maker order
    else:  # short: exit side is the bid
        if stage == "force":
            return state.extended_ask, ask, OrderSide.BUY  # cross into their ask: guaranteed fill
        return state.extended_bid, ask, None  # join their bid: still a passive maker order


def process_public_trade(trade):
    """Dry-run only: fills against real trade prints instead of just book-crossing, so the
    fill size is bounded by what actually traded (queue position — see TRADE_CAPTURE_RATIO
    above) rather than assuming every book-crossing instantly and fully fills us.

    trade.side is Extended's convention for the taker/aggressor side of the trade — SELL means
    an aggressive sell hit the bid, BUY means an aggressive buy lifted the ask. Assumed, not yet
    confirmed against a live trace; flag if a real run's fills look mirrored.

    Applies MAKER_FEE_BPS (a negative number — we're paid a rebate, not charged) on normal
    fills, and TAKER_FEE_BPS on whichever side was resting as the stuck-inventory circuit
    breaker's force-cross, matching how manage_live_orders' post_only flag treats that side."""
    now = time.time()
    capture_qty = trade.qty * TRADE_CAPTURE_RATIO

    if trade.side == OrderSide.SELL and state.inventory < MAX_INVENTORY:
        result = state.paper_bid.fill(lambda p: trade.price <= p, capture_qty, now)
        if result:
            price, qty, is_taker = result
            fee_bps = TAKER_FEE_BPS if is_taker else MAKER_FEE_BPS
            fee = qty * price * fee_bps / Decimal("10000")
            old = state.inventory
            state.inventory += qty
            state.paper_cash -= qty * price + fee
            note_inventory_change(old, state.inventory)
            state.fill_count += 1
            log.info("PAPER FILL buy %s @ %s (of %s traded) fee=%s", qty, price, trade.qty, fee)

    if trade.side == OrderSide.BUY and state.inventory > -MAX_INVENTORY:
        result = state.paper_ask.fill(lambda p: trade.price >= p, capture_qty, now)
        if result:
            price, qty, is_taker = result
            fee_bps = TAKER_FEE_BPS if is_taker else MAKER_FEE_BPS
            fee = qty * price * fee_bps / Decimal("10000")
            old = state.inventory
            state.inventory -= qty
            state.paper_cash += qty * price - fee
            note_inventory_change(old, state.inventory)
            state.fill_count += 1
            log.info("PAPER FILL sell %s @ %s (of %s traded) fee=%s", qty, price, trade.qty, fee)


def record_snapshot(bid, ask, edge_bid, edge_ask, sigma, momentum, pnl, live_ctx, stage):
    state.record_buffer.append({
        "timestamp": int(time.time() * 1000),
        "fair_price": state.fair_price,
        "binance_price": state.binance_price,
        "momentum": momentum,
        "extended_bid": state.extended_bid,
        "extended_ask": state.extended_ask,
        "target_bid": bid,
        "target_ask": ask,
        "edge_bid": edge_bid,
        "edge_ask": edge_ask,
        "sigma_dollar": sigma,
        "usdt_usd_rate": state.usdt_usd_rate,
        "inventory": float(state.inventory),
        "pnl": float(pnl),
        "unwind_stage": stage,
        "mode": "live" if live_ctx else "dry_run",
        "capture_ratio": float(TRADE_CAPTURE_RATIO),
        "gamma": state.current_gamma,
        "kappa": state.current_kappa,
        "regime": state.bandit_regime,
        "fill_bonus": state.current_fill_bonus,
        "inv_penalty": state.current_inv_penalty,
    })


def flush_records(force=False):
    if not state.record_buffer:
        return
    if not force and time.time() - state.last_flush_ts < FLUSH_INTERVAL_SEC:
        return
    LIVE_DATA_DIR.mkdir(exist_ok=True)
    ts_tag = int(time.time())
    # PID in the filename because running several instances concurrently (e.g. sweeping
    # --capture-ratio against the same live market) means their FLUSH_INTERVAL_SEC timers stay
    # roughly in sync — two processes hitting the same wall-clock second would otherwise both
    # write to quotes_<ts_tag>.parquet and corrupt each other's file (confirmed: this produced
    # "Invalid flatbuffers message" / unreadable files during a live 3-process run).
    path = LIVE_DATA_DIR / f"quotes_{os.getpid()}_{ts_tag}.parquet"
    # Write-then-rename so dashboard.py (polling this directory independently) can never observe
    # a half-written file — os.replace is atomic on POSIX, a direct to_parquet(path) is not.
    tmp_path = path.with_suffix(".tmp")
    pd.DataFrame(state.record_buffer).to_parquet(tmp_path)
    os.replace(tmp_path, path)
    log.info("Flushed %d quote records to %s", len(state.record_buffer), path)
    state.record_buffer = []
    state.last_flush_ts = time.time()


_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


async def binance_feed():
    url = f"wss://fstream.binance.com/ws/{BINANCE_SYMBOL}@bookTicker"
    while True:
        try:
            async with websockets.connect(url, ssl=_SSL_CONTEXT) as ws:
                log.info("Connected to Binance bookTicker: %s", BINANCE_SYMBOL)
                async for raw in ws:
                    on_binance_book_ticker(json.loads(raw))
        except (websockets.ConnectionClosed, OSError) as e:
            log.warning("Binance WS dropped (%s), reconnecting in 1s...", e)
            await asyncio.sleep(1)


async def usdt_rate_feed():
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(USDC_USDT_URL, ssl=_SSL_CONTEXT, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    data = await resp.json()
                usdc_per_usdt = (float(data["bidPrice"]) + float(data["askPrice"])) / 2
                new_rate = 1 / usdc_per_usdt  # USD per USDT, treating USDC as a USD proxy
                if state.usdt_usd_rate is None:
                    log.info("USDT/USD rate initialized: %.6f", new_rate)
                state.usdt_usd_rate = new_rate
            except Exception as e:
                log.warning("USDT/USD rate poll failed (%s), keeping last known rate", e)
            await asyncio.sleep(USDT_RATE_POLL_INTERVAL_SEC)


async def extended_orderbook_feed(stream_client):
    while True:
        try:
            async with stream_client.subscribe_to_orderbooks(EXTENDED_MARKET, depth=1) as ob_stream:
                log.info("Connected to Extended orderbook stream: %s (%s)", EXTENDED_MARKET, EXTENDED_ENV)
                while True:
                    msg = await ob_stream.recv()
                    book = msg.data
                    if book is None:
                        continue
                    if book.bid:
                        state.extended_bid = float(book.bid[0].price)
                    if book.ask:
                        state.extended_ask = float(book.ask[0].price)
                    if book.bid and book.ask:
                        on_extended_book_update(
                            state.extended_bid, float(book.bid[0].qty),
                            state.extended_ask, float(book.ask[0].qty),
                        )
        except Exception as e:
            log.warning("Extended orderbook WS dropped (%s), reconnecting in 1s...", e)
            await asyncio.sleep(1)


async def extended_trades_feed(stream_client):
    """Feeds process_public_trade() (dry-run paper fills only — a no-op call in --live since
    process_public_trade() itself checks state.live_mode) with Extended's real trade prints.
    Also feeds state.trade_volume_window regardless of mode — that's the bandit's "how much is
    actually happening" regime signal, needed in --live too, not just for paper fills."""
    while True:
        try:
            async with stream_client.subscribe_to_public_trades(EXTENDED_MARKET) as trades_stream:
                log.info("Connected to Extended public trades stream: %s", EXTENDED_MARKET)
                while True:
                    msg = await trades_stream.recv()
                    if not msg.data:
                        continue
                    now = time.time()
                    for trade in msg.data:
                        state.trade_volume_window.append((now, float(trade.qty)))
                        if not state.live_mode:
                            process_public_trade(trade)
        except Exception as e:
            log.warning("Extended trades WS dropped (%s), reconnecting in 1s...", e)
            await asyncio.sleep(1)


async def extended_account_feed(stream_client, api_key):
    """Keeps state.inventory in sync with the real position on Extended, and logs fills."""
    while True:
        try:
            async with stream_client.subscribe_to_account_updates(api_key) as acc_stream:
                log.info("Connected to Extended account stream")
                while True:
                    msg = await acc_stream.recv()
                    data = msg.data
                    if data is None:
                        continue
                    if data.positions:
                        for pos in data.positions:
                            if pos.market != EXTENDED_MARKET:
                                continue
                            signed = pos.size if pos.side == "LONG" else -pos.size
                            note_inventory_change(state.inventory, signed)
                            state.inventory = signed
                            state.realised_pnl = pos.realised_pnl
                            state.unrealised_pnl = pos.unrealised_pnl
                            log.info(
                                "Position update: %s %s (side=%s) realised=%s unrealised=%s",
                                pos.market, pos.size, pos.side, pos.realised_pnl, pos.unrealised_pnl,
                            )
                    if data.trades:
                        for trade in data.trades:
                            state.fill_count += 1
                            log.info("Fill: %s %s @ %s", trade.side, trade.qty, trade.price)
        except Exception as e:
            log.warning("Extended account WS dropped (%s), reconnecting in 1s...", e)
            await asyncio.sleep(1)


async def manage_live_orders(rest_client, market, taker_fee, order_size, bid, ask, force_taker_side=None):
    """Cancel/replace resting orders on Extended to match the target bid/ask, subject to
    an inventory cap and a minimum requote interval so we don't churn the order book.
    force_taker_side (set by the stuck-inventory circuit breaker at its 'force' stage) submits
    that side without post_only, so it crosses the book and fills immediately instead of resting."""
    now = time.time()
    if now - state.last_requote_ts < MIN_REQUOTE_INTERVAL_SEC:
        return
    state.last_requote_ts = now

    bid_dec = market.trading_config.round_price(Decimal(str(bid)))
    ask_dec = market.trading_config.round_price(Decimal(str(ask)))

    want_bid = state.inventory < MAX_INVENTORY
    want_ask = state.inventory > -MAX_INVENTORY

    async def refresh_side(side, want, order_id_attr, order_price_attr, target_price):
        order_id = getattr(state, order_id_attr)
        order_price = getattr(state, order_price_attr)
        post_only = side != force_taker_side

        needs_replace = post_only is False or order_id is None or order_price is None or (
            abs(target_price - order_price) / target_price * Decimal("10000") > REPRICE_THRESHOLD_BPS
        )

        if not want:
            if order_id is not None:
                await rest_client.orders.cancel_order(order_id)
                setattr(state, order_id_attr, None)
                setattr(state, order_price_attr, None)
                log.info("%s: cancelled (inventory cap reached)", side)
            return

        if not needs_replace:
            return

        if order_id is not None:
            await rest_client.orders.cancel_order(order_id)

        try:
            response = await rest_client.place_order(
                market_name=EXTENDED_MARKET,
                amount_of_synthetic=order_size,
                price=target_price,
                side=side,
                taker_fee=taker_fee,
                post_only=post_only,
            )
        except Exception as e:
            log.warning("%s: place_order failed (%s)", side, e)
            setattr(state, order_id_attr, None)
            setattr(state, order_price_attr, None)
            return

        if response.data is None:
            log.warning("%s: order rejected (%s)", side, response.error)
            setattr(state, order_id_attr, None)
            setattr(state, order_price_attr, None)
            return

        setattr(state, order_id_attr, response.data.id)
        setattr(state, order_price_attr, target_price)
        log.info("%s: resting at %s%s", side, target_price, " (forced taker)" if not post_only else "")

    await refresh_side(OrderSide.BUY, want_bid, "bid_order_id", "bid_order_price", bid_dec)
    await refresh_side(OrderSide.SELL, want_ask, "ask_order_id", "ask_order_price", ask_dec)


async def quote_loop(live_ctx):
    while True:
        await asyncio.sleep(QUOTE_INTERVAL_SEC)

        if state.fair_price is None:
            continue
        if time.time() - state.fair_price_ts > MAX_FAIR_PRICE_STALENESS_SEC:
            log.warning("Extended book stale, skipping quote")
            if live_ctx:
                await cancel_all_orders(live_ctx["rest_client"])
            continue
        sigma = realized_sigma_dollar()
        if sigma is None or sigma == 0:
            continue

        # pnl computed here (not just later for logging) because bandit_step needs it to score
        # the outgoing arm's window before possibly switching gamma/kappa for this tick's quote.
        pnl = current_pnl(live_ctx, state.fair_price)
        bandit_step(sigma, float(pnl))
        coef_bandit_step(float(pnl))

        momentum = binance_momentum()
        bid, ask = avellaneda_stoikov_quotes(
            state.fair_price, float(state.inventory), sigma, state.current_gamma, state.current_kappa, momentum
        )

        if (
            abs(bid - state.fair_price) / state.fair_price > MAX_QUOTE_DEVIATION_PCT
            or abs(ask - state.fair_price) / state.fair_price > MAX_QUOTE_DEVIATION_PCT
        ):
            log.warning(
                "Quote too far from fair price (fair=%.2f bid=%.2f ask=%.2f sigma=%.4f) — skipping, likely a bad tick",
                state.fair_price, bid, ask, sigma,
            )
            continue

        stage = unwind_stage()
        bid, ask, force_taker_side = apply_unwind_override(bid, ask, stage)
        if stage != "none":
            log.warning(
                "Stuck at inventory=%s for %.0fs — unwind stage=%s",
                state.inventory, time.time() - state.last_unwind_ts, stage,
            )

        # The unwind override repriced only one side against Extended's live book, independently
        # of the other (still-normal) side. A real run showed those two can cross — e.g. momentum
        # had pushed the normal bid up right as the override pulled ask down to Extended's ask,
        # producing bid > ask (our own buy priced above our own sell). That's not just cosmetic:
        # the paper-fill simulator read the inflated bid as instantly marketable and fired 6 buys
        # in under a second. Never let a crossed pair reach either the simulator or a real order.
        if bid >= ask:
            log.warning(
                "Crossed quote after unwind override (bid=%.2f >= ask=%.2f, stage=%s) — skipping",
                bid, ask, stage,
            )
            continue

        ext_bid, ext_ask = state.extended_bid, state.extended_ask
        edge_bid = (ext_bid - bid) if ext_bid is not None else None
        edge_ask = (ask - ext_ask) if ext_ask is not None else None

        log.info(
            "extended_fair=%.2f  binance=%s  momentum=%.4f  book=[%s / %s]  target=[%.2f / %.2f]  "
            "edge=[%s / %s]  inventory=%s  pnl=%.4f",
            state.fair_price,
            f"{state.binance_price:.2f}" if state.binance_price is not None else None,
            momentum,
            ext_bid,
            ext_ask,
            bid,
            ask,
            f"{edge_bid:.2f}" if edge_bid is not None else None,
            f"{edge_ask:.2f}" if edge_ask is not None else None,
            state.inventory,
            pnl,
        )

        if live_ctx:
            await manage_live_orders(
                live_ctx["rest_client"], live_ctx["market"], live_ctx["taker_fee"], live_ctx["order_size"],
                bid, ask, force_taker_side,
            )
        else:
            # Dry-run has no market.trading_config to round against, so approximate Extended's
            # real tick size (observed as one decimal place on BTC-USD). Only records the new
            # target here — actual fills are checked continuously in process_public_trade
            # (real trade prints), against whatever NETWORK_LATENCY_SEC/CANCEL_LATENCY_SEC say
            # is really resting.
            now = time.time()
            state.paper_bid.reprice(Decimal(str(round(bid, 1))), force_taker_side == OrderSide.BUY, now)
            state.paper_ask.reprice(Decimal(str(round(ask, 1))), force_taker_side == OrderSide.SELL, now)

        record_snapshot(bid, ask, edge_bid, edge_ask, sigma, momentum, pnl, live_ctx, stage)
        flush_records()


async def cancel_all_orders(rest_client):
    try:
        await rest_client.orders.mass_cancel(markets=[EXTENDED_MARKET])
    except Exception as e:
        log.warning("mass_cancel failed (%s)", e)
    state.bid_order_id = state.ask_order_id = None
    state.bid_order_price = state.ask_order_price = None


async def build_live_ctx():
    if EXTENDED_ENV != "TESTNET":
        raise SystemExit(
            "--live refuses to run with EXTENDED_ENV != 'TESTNET'. "
            "Edit EXTENDED_ENV in live_quoter.py yourself if you really mean to trade MAINNET."
        )

    load_dotenv()
    env_config = EnvConfig.parse()
    env_config.validate_private_api_credentials()

    stark_account = StarkPerpetualAccount(
        api_key=env_config.api_key,
        public_key=env_config.public_key,
        private_key=env_config.private_key,
        vault=env_config.vault_id,
    )
    config = get_config_by_name(EXTENDED_ENV)
    rest_client = RestApiClient(config, stark_account)

    markets = await rest_client.info.get_markets_dict()
    market = markets[EXTENDED_MARKET]
    order_size = market.trading_config.min_order_size

    fees = await rest_client.account.get_fees(market_names=[EXTENDED_MARKET])
    taker_fee = fees.data[0].taker_fee_rate if fees.data else Decimal("0.0005")

    log.info(
        "LIVE mode on %s: market=%s order_size=%s taker_fee=%s max_inventory=%s",
        EXTENDED_ENV, EXTENDED_MARKET, order_size, taker_fee, MAX_INVENTORY,
    )

    return {
        "rest_client": rest_client,
        "market": market,
        "taker_fee": taker_fee,
        "order_size": order_size,
        "stream_client": StreamClient(api_url=config.endpoints.stream_url),
        "api_key": env_config.api_key,
    }


async def main():
    global TRADE_CAPTURE_RATIO

    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="place real orders on Extended TESTNET")
    parser.add_argument(
        "--capture-ratio", type=float, default=1.0,
        help="dry-run only: fraction (0-1) of each qualifying public trade's quantity assumed "
             "to reach our resting order, simulating queue position/latency disadvantage vs "
             "faster participants. 1.0 (default) = every crossing trade fully fills us.",
    )
    args = parser.parse_args()

    TRADE_CAPTURE_RATIO = Decimal(str(args.capture_ratio))

    live_ctx = await build_live_ctx() if args.live else None
    state.live_mode = live_ctx is not None

    market_data_stream_client = live_ctx["stream_client"] if live_ctx else StreamClient(
        api_url=get_config_by_name(EXTENDED_ENV).endpoints.stream_url
    )

    tasks = [
        asyncio.create_task(usdt_rate_feed()),
        asyncio.create_task(binance_feed()),
        asyncio.create_task(extended_orderbook_feed(market_data_stream_client)),
        asyncio.create_task(extended_trades_feed(market_data_stream_client)),
        asyncio.create_task(quote_loop(live_ctx)),
    ]
    if live_ctx:
        tasks.append(asyncio.create_task(extended_account_feed(market_data_stream_client, live_ctx["api_key"])))

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def signal_handler():
        log.info("Signal received, shutting down...")
        stop_event.set()

    loop.add_signal_handler(SIGINT, signal_handler)
    loop.add_signal_handler(SIGTERM, signal_handler)

    await stop_event.wait()

    for t in tasks:
        t.cancel()

    if live_ctx:
        log.info("Cancelling all resting orders before exit...")
        await cancel_all_orders(live_ctx["rest_client"])
        await live_ctx["rest_client"].close()

    flush_records(force=True)
    state.bandit.save()
    state.coef_bandit.save()


if __name__ == "__main__":
    asyncio.run(main())
