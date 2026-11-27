from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import json
import math
import os
import socket
import xml.etree.ElementTree as ET
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup
from google import genai
from google.genai import types

socket.setdefaulttimeout(10)

# 1. Core stocks & ETFs (always scanned)
CORE_WATCHLIST = [
    "NVDA", "AVGO", "MU", "INTC", "GOOGL",
    "META", "AAPL", "AMZN", "TSLA", "PLTR", "VUAA.L"
]

# 2. Macro & cross-sector breaking news RSS feeds
MARKET_RSS_FEEDS = [
    ("CNBC Top News", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
    ("CNBC Investing", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069"),
    ("MarketWatch Top Stories", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
]

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]

HISTORY_FILE = "docs/history.json"
MAX_HISTORY_RUNS = 15  # 3 scans/day x 5 trading days = 1 full trading week

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/128.0.0.0 Safari/537.36"
}


def clean_num(val):
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return val


def calculate_rsi(closes: list[float], period: int = 14) -> float | None:
    """Computes Wilder's 14-day Relative Strength Index from daily closes."""
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [abs(min(d, 0.0)) for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 1)


def classify_trend(price: float | None, sma_20: float | None, sma_50: float | None) -> str:
    if not price:
        return "Trend Unavailable"
    if sma_20 and sma_50:
        if price > sma_20 and price > sma_50:
            return "Strong Uptrend (>20D & >50D SMA)"
        if price > sma_50 and price <= sma_20:
            return "Pullback in Uptrend (<20D, >50D SMA)"
        if price < sma_50 and price >= sma_20:
            return "Early Rebound (>20D, <50D SMA)"
        return "Downtrend (<20D & <50D SMA)"
    if sma_20:
        return "Above 20D SMA" if price >= sma_20 else "Below 20D SMA"
    return "Insufficient History"


def detect_session() -> tuple[str, str]:
    now_et = datetime.now(ZoneInfo("America/New_York"))
    minutes = now_et.hour * 60 + now_et.minute
    if 240 <= minutes < 570:
        session = "Pre-Market (04:00-09:30 ET)"
    elif 570 <= minutes < 960:
        session = "Open Market (09:30-16:00 ET)"
    elif 960 <= minutes < 1200:
        session = "Post-Market (16:00-20:00 ET)"
    else:
        session = "Overnight / Extended Wire"
    return session, now_et.strftime("%Y-%m-%d %H:%M %Z")


def load_history() -> list[dict]:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as exc:
        print(f"[WARN] Could not read {HISTORY_FILE}: {exc}", flush=True)
        return []


def save_history(history: list[dict], session: str, timestamp_et: str, analysis: dict) -> list[dict]:
    snapshot = {
        "session": session,
        "updated_et": timestamp_et,
        "market_mood": analysis.get("market_mood"),
        "overall_sentiment_score": analysis.get("overall_sentiment_score"),
        "signals": {
            r.get("ticker"): {
                "action": r.get("action"),
                "sentiment_score": r.get("sentiment_score"),
                "catalyst": (r.get("catalyst") or "")[:140],
            }
            for r in analysis.get("recommendations", [])
            if r.get("ticker")
        },
    }
    updated = (history + [snapshot])[-MAX_HISTORY_RUNS:]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2)
    return updated


def build_trajectory_summary(history: list[dict]) -> dict:
    """Compresses up to 15 prior runs into low-token per-ticker score trails plus 2 detailed recent sessions."""
    if not history:
        return {"runs_recorded": 0, "macro_trail": [], "ticker_trails": {}, "recent_detailed_sessions": []}

    macro_trail = [
        f"{h.get('updated_et', '')} [{h.get('session', '').split(' ')[0]}]: {h.get('overall_sentiment_score', 0):+d} ({h.get('market_mood', '')})"
        for h in history
    ]

    ticker_trails: dict[str, list[str]] = {}
    for h in history:
        short_sess = (h.get("session") or "").split(" ")[0]
        for ticker, sig in (h.get("signals") or {}).items():
            score = sig.get("sentiment_score", 0)
            act = sig.get("action", "Hold").split("/")[0].strip()
            ticker_trails.setdefault(ticker, []).append(f"{short_sess}:{score:+d}({act})")

    compact_trails = {t: " -> ".join(trail[-MAX_HISTORY_RUNS:]) for t, trail in ticker_trails.items()}

    return {
        "runs_recorded": len(history),
        "macro_trail": macro_trail,
        "ticker_trails": compact_trails,
        "recent_detailed_sessions": history[-2:],
    }


def fetch_rss_items(url: str, source_name: str, limit: int = 5) -> list[dict]:
    items = []
    try:
        resp = requests.get(url, headers=HEADERS, timeout=8)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        for item in root.findall(".//item")[:limit]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            desc_raw = item.findtext("description") or ""
            desc = BeautifulSoup(desc_raw, "html.parser").get_text(" ", strip=True)[:180]
            if title:
                items.append({"source": source_name, "title": title, "summary": desc, "url": link})
    except Exception as exc:
        print(f"[WARN] RSS skipped for {source_name}: {exc}", flush=True)
    return items


def fetch_ticker_quote(ticker: str, source_group: str = "Core Watchlist", sector_hint: str | None = None) -> dict | None:
    """Fetches Pre/Open/Post-Market prices plus 20D SMA, 50D SMA, and RSI(14) from 4-month daily chart JSON."""
    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        url = f"https://{host}/v8/finance/chart/{ticker}?range=4mo&interval=1d&includePrePost=true"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=8)
            resp.raise_for_status()
            result = (resp.json().get("chart", {}).get("result") or [None])[0]
            if not result:
                continue
            meta = result.get("meta", {})
            prev_close = clean_num(meta.get("previousClose") or meta.get("chartPreviousClose"))
            reg_price = clean_num(meta.get("regularMarketPrice"))
            pre_price = clean_num(meta.get("preMarketPrice"))
            post_price = clean_num(meta.get("postMarketPrice"))

            active_price = pre_price or post_price or reg_price
            pct_change = None
            if active_price and prev_close:
                pct_change = round(((active_price - prev_close) / prev_close) * 100, 2)

            raw_closes = ((result.get("indicators", {}).get("quote") or [{}])[0]).get("close") or []
            closes = [float(c) for c in raw_closes if isinstance(c, (int, float)) and not math.isnan(c)]

            sma_20 = round(sum(closes[-20:]) / 20, 2) if len(closes) >= 20 else None
            sma_50 = round(sum(closes[-50:]) / 50, 2) if len(closes) >= 50 else None
            rsi_14 = calculate_rsi(closes, 14)
            change_5d_pct = (
                round(((closes[-1] - closes[-6]) / closes[-6]) * 100, 2)
                if len(closes) >= 6 and closes[-6]
                else None
            )

            rsi_zone = "Neutral"
            if rsi_14 is not None:
                if rsi_14 >= 70:
                    rsi_zone = "Overbought (>=70)"
                elif rsi_14 <= 30:
                    rsi_zone = "Oversold (<=30)"

            return {
                "ticker": ticker,
                "name": meta.get("shortName") or meta.get("longName") or ticker,
                "sector": sector_hint or meta.get("instrumentType") or "Equity",
                "source_group": source_group,
                "regular_price": reg_price,
                "pre_market_price": pre_price,
                "post_market_price": post_price,
                "prev_close": prev_close,
                "active_change_pct": pct_change,
                "change_5d_pct": change_5d_pct,
                "sma_20": sma_20,
                "sma_50": sma_50,
                "rsi_14": rsi_14,
                "rsi_zone": rsi_zone,
                "trend_regime": classify_trend(active_price, sma_20, sma_50),
            }
        except Exception:
            continue
    print(f"[WARN] Quote fetch skipped for {ticker}", flush=True)
    return None


def discover_market_movers(limit_per_screen: int = 5) -> list[dict]:
    """Auto-discovers top gainers and most active stocks across all sectors."""
    discovered: dict[str, dict] = {}
    screens = [
        ("Market Mover: Day Gainer", "day_gainers"),
        ("Market Mover: Most Active", "most_actives"),
    ]

    for label, screen_key in screens:
        url = f"https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?formatted=false&scrIds={screen_key}&count={limit_per_screen}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=8)
            resp.raise_for_status()
            results = resp.json().get("finance", {}).get("result") or []
            quotes = results[0].get("quotes", []) if results else []
            for q in quotes:
                sym = (q.get("symbol") or "").strip()
                if not sym or "^" in sym or "=" in sym:
                    continue
                if sym not in discovered:
                    chg = clean_num(q.get("regularMarketChangePercent"))
                    discovered[sym] = {
                        "ticker": sym,
                        "name": q.get("shortName") or q.get("longName") or sym,
                        "sector": q.get("sector") or "Broader Market",
                        "source_group": label,
                        "regular_price": clean_num(q.get("regularMarketPrice")),
                        "pre_market_price": clean_num(q.get("preMarketPrice")),
                        "post_market_price": clean_num(q.get("postMarketPrice")),
                        "prev_close": clean_num(q.get("regularMarketPreviousClose")),
                        "active_change_pct": round(float(chg), 2) if chg is not None else None,
                    }
        except Exception as exc:
            print(f"[WARN] Screener '{screen_key}' skipped: {exc}", flush=True)

    return list(discovered.values())


def collect_market_intelligence(core_tickers: list[str]) -> tuple[list[dict], list[dict]]:
    quotes = []
    headlines = []
    seen_tickers = {t.upper() for t in core_tickers}

    with ThreadPoolExecutor(max_workers=8) as pool:
        quote_futures = {pool.submit(fetch_ticker_quote, t, "Core Watchlist"): t for t in core_tickers}
        rss_futures = [
            pool.submit(fetch_rss_items, url, name, 5)
            for name, url in MARKET_RSS_FEEDS
        ]
        for t in core_tickers:
            rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}&region=US&lang=en-US"
            rss_futures.append(pool.submit(fetch_rss_items, rss_url, f"Yahoo Finance ({t})", 3))

        for fut in as_completed(quote_futures):
            q = fut.result()
            if q:
                quotes.append(q)

        for fut in as_completed(rss_futures):
            headlines.extend(fut.result())

    movers = discover_market_movers(limit_per_screen=5)
    extra_movers = []
    for mover in movers:
        sym = mover["ticker"].upper()
        if sym not in seen_tickers:
            seen_tickers.add(sym)
            extra_movers.append(mover)
        if len(extra_movers) >= 8:
            break

    if extra_movers:
        with ThreadPoolExecutor(max_workers=8) as pool:
            mover_quote_futs = {
                pool.submit(fetch_ticker_quote, m["ticker"], m["source_group"], m["sector"]): m
                for m in extra_movers
            }
            mover_rss_futs = [
                pool.submit(
                    fetch_rss_items,
                    f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={m['ticker']}&region=US&lang=en-US",
                    f"Yahoo Finance ({m['ticker']})",
                    2,
                )
                for m in extra_movers
            ]
            for fut, fallback_mover in mover_quote_futs.items():
                enriched = fut.result()
                quotes.append(enriched if enriched else fallback_mover)
            for fut in as_completed(mover_rss_futs):
                headlines.extend(fut.result())

    return quotes, headlines


def analyze_with_gemini(
    session: str,
    timestamp_et: str,
    quotes: list[dict],
    headlines: list[dict],
    history: list[dict],
) -> dict:
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY secret is empty or missing in GitHub repository settings.")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1beta"),
    )

    compact_headlines = [
        {"source": h["source"], "title": h["title"], "url": h["url"]}
        for h in headlines[:45]
    ]

    trajectory_context = build_trajectory_summary(history)

    prompt = f"""
You are a senior quantitative equity analyst. Analyze live stock quotes, multi-week technical structure (20D SMA, 50D SMA, RSI-14), 5-day sentiment trajectory, and headlines for US market session: {session} ({timestamp_et}).
Rank 8 to 10 actionable setups across BOTH Core Watchlist and Market Movers.

DECISION FRAMEWORK & ANTI-FLIP-FLOP RULES:
1. MULTI-DAY TRAJECTORY (5-DAY / 15-RUN MEMORY): Review TRAJECTORY_HISTORY below. Distinguish between steady multi-day accumulation/erosion vs. single-session noise. Keep the previous rating unless a material fundamental catalyst, earnings surprise, or >2.5% move breaking the 20D/50D SMA trend occurs.
2. TECHNICAL GUARDRAILS (20D SMA, 50D SMA & RSI-14):
   - Do NOT issue "Strong Buy / Accumulate" on stocks trading below BOTH 20D and 50D SMA unless a major bullish earnings reversal just occurred.
   - If RSI(14) >= 72 ("Overbought"), avoid chasing with "Strong Buy"—prefer "Tactical Swing Buy" (on pullback) or "Watchlist / Hold" and note overbought risk.
   - If RSI(14) <= 32 ("Oversold") above 50D SMA with stabilizing sentiment, flag the mean-reversion setup.
3. SESSION WEIGHTING:
   - Pre-Market (04:00-09:30 ET) has thin liquidity: treat small gaps without hard catalysts as preliminary.
   - Open Market (09:30-16:00 ET) tests whether early gaps hold institutional volume.
   - Post-Market (16:00-20:00 ET) reflects the official daily close + after-hours earnings: treat as the highest-conviction anchor.
4. TRANSPARENCY: Populate "prev_action", "sentiment_trail" (concise progression from ticker_trails or "First scan"), "technical_badge" (e.g. "Above 20D & 50D SMA | RSI 58"), and "change_reason".

Return ONLY valid JSON matching this schema:
{{
  "market_mood": "Bullish | Cautiously Bullish | Neutral / Mixed | Risk-Off Bearish",
  "overall_sentiment_score": <integer -100 to 100>,
  "executive_summary": "<2-3 sentence synthesis of macro drivers, 5-day sentiment direction, and technical posture>",
  "recommendations": [
    {{
      "ticker": "<TICKER>",
      "company": "<Company Name>",
      "sector": "<Sector>",
      "source_group": "Core Watchlist | Market Mover",
      "action": "Strong Buy / Accumulate | Tactical Swing Buy | Watchlist / Hold | Trim / Avoid",
      "prev_action": "<Prior session action or 'New'>",
      "sentiment_trail": "<e.g. +20 -> +45 -> +62 (5-day rising) or 'Initial baseline'>",
      "technical_badge": "<e.g. >20D & >50D SMA | RSI 61.4>",
      "change_reason": "<1 short sentence explaining continuity or why rating shifted>",
      "sentiment_score": <integer -100 to 100>,
      "confidence": <integer 1 to 100>,
      "price_snapshot": "<e.g. $142.50 (+2.4%)>",
      "catalyst": "<Specific catalyst from headlines>",
      "strategy": "<Concrete entry/position note respecting SMA & RSI levels>",
      "risk": "<Primary downside or technical risk>"
    }}
  ],
  "top_headlines": [
    {{
      "source": "<Source>",
      "ticker_tag": "<TICKER or MACRO>",
      "sentiment": "Bullish | Bearish | Neutral",
      "title": "<Headline>",
      "url": "<URL>"
    }}
  ]
}}

TRAJECTORY_HISTORY (up to 15 prior runs / 5 trading days):
{json.dumps(trajectory_context)}

QUOTES & TECHNICALS:
{json.dumps(quotes)}

HEADLINES:
{json.dumps(compact_headlines)}
"""

    last_err = None
    for model_name in GEMINI_MODELS:
        try:
            print(f"Requesting analysis from {model_name}...", flush=True)
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    client.models.generate_content,
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.15,
                    ),
                )
                response = future.result(timeout=30)

            raw_text = (response.text or "").strip()
            if not raw_text:
                raise ValueError(f"Empty response from {model_name}")
            if raw_text.startswith("```"):
                raw_text = raw_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(raw_text)
            parsed["model_used"] = model_name
            return parsed
        except Exception as exc:
            print(f"[WARN] Model {model_name} failed: {exc}", flush=True)
            last_err = exc

    raise RuntimeError(str(last_err))


def build_fallback_analysis(error_msg: str, quotes: list[dict], headlines: list[dict]) -> dict:
    recs = []
    for q in quotes[:12]:
        price = q.get("pre_market_price") or q.get("post_market_price") or q.get("regular_price")
        chg = q.get("active_change_pct")
        snap = f"${price:.2f}" if isinstance(price, (int, float)) else "N/A"
        if isinstance(chg, (int, float)):
            snap += f" ({chg:+.2f}%)"
        rsi = q.get("rsi_14")
        tech = f"{q.get('trend_regime', 'N/A')} | RSI {rsi if rsi is not None else 'N/A'}"
        recs.append({
            "ticker": q.get("ticker", ""),
            "company": q.get("name", ""),
            "sector": q.get("sector", "Market"),
            "source_group": q.get("source_group", "Watchlist"),
            "action": "Watchlist / Hold",
            "prev_action": "N/A",
            "sentiment_trail": "Fallback view",
            "technical_badge": tech,
            "change_reason": "Fallback raw quote view",
            "sentiment_score": 0,
            "confidence": 0,
            "price_snapshot": snap,
            "catalyst": "Live market quote & technicals collected; awaiting Gemini AI scoring.",
            "strategy": "Check API status banner above.",
            "risk": "AI scoring skipped on this run.",
        })

    top_news = [
        {
            "source": h.get("source", ""),
            "ticker_tag": "WIRE",
            "sentiment": "Neutral",
            "title": h.get("title", ""),
            "url": h.get("url", "#"),
        }
        for h in headlines[:18]
    ]

    return {
        "api_error": error_msg,
        "model_used": "Fallback (Raw Feed)",
        "market_mood": "Feed Active (AI Pending)",
        "overall_sentiment_score": 0,
        "executive_summary": "Live stock quotes, SMAs, RSI, and RSS headlines were collected, but Gemini AI returned an error.",
        "recommendations": recs,
        "top_headlines": top_news,
    }


def badge_color(action: str) -> str:
    lower = action.lower()
    if "strong buy" in lower or "accumulate" in lower:
        return "#10b981"
    if "swing" in lower:
        return "#3b82f6"
    if "trim" in lower or "avoid" in lower:
        return "#ef4444"
    return "#f59e0b"


def render_html(session: str, timestamp_et: str, data: dict, history: list[dict]) -> str:
    error_banner = ""
    if data.get("api_error"):
        error_banner = f"""
        <div class="err-banner">
          <strong>Gemini API Diagnostic:</strong> {html.escape(data['api_error'])}
        </div>
        """

    macro_strip = ""
    if history:
        recent_pills = []
        for h in history[-6:]:
            sess_short = (h.get("session") or "").split(" ")[0]
            sc = h.get("overall_sentiment_score", 0)
            recent_pills.append(f"<span class='hist-pill'>{html.escape(sess_short)} {sc:+d}</span>")
        macro_strip = f"<div class='macro-strip'><strong>5-Day Macro Trajectory ({len(history)}/{MAX_HISTORY_RUNS} runs):</strong> {' '.join(recent_pills)}</div>"

    recs_html = ""
    for rec in data.get("recommendations", []):
        color = badge_color(rec.get("action", ""))
        score = rec.get("sentiment_score", 0)
        source_group = rec.get("source_group", "Watchlist")
        sector = rec.get("sector", "")
        prev_action = rec.get("prev_action", "New")
        change_reason = rec.get("change_reason", "")
        tech_badge = rec.get("technical_badge", "")
        sent_trail = rec.get("sentiment_trail", "")

        tech_line = (
            f"<div class='tech-bar'><strong>Technicals:</strong> {html.escape(tech_badge)}</div>"
            if tech_badge
            else ""
        )
        delta_line = (
            f"<div class='delta'><strong>5D Trail:</strong> {html.escape(sent_trail)} &bull; <strong>Prev ({html.escape(prev_action)}):</strong> {html.escape(change_reason)}</div>"
            if (change_reason or sent_trail)
            else ""
        )

        recs_html += f"""
        <div class="card">
          <div class="card-top">
            <div>
              <span class="ticker">{html.escape(rec.get('ticker', ''))}</span>
              <span class="company">{html.escape(rec.get('company', ''))}</span>
              <div class="subtags">
                <span class="pill">{html.escape(source_group)}</span>
                <span class="pill">{html.escape(sector)}</span>
              </div>
            </div>
            <span class="badge" style="background:{color}22; color:{color}; border:1px solid {color}55;">
              {html.escape(rec.get('action', 'Hold'))}
            </span>
          </div>
          <div class="metrics">
            <span><strong>Price:</strong> {html.escape(rec.get('price_snapshot', 'N/A'))}</span>
            <span><strong>Sentiment:</strong> {score:+d}/100</span>
            <span><strong>Confidence:</strong> {rec.get('confidence', 0)}%</span>
          </div>
          {tech_line}
          {delta_line}
          <p><strong>Catalyst:</strong> {html.escape(rec.get('catalyst', ''))}</p>
          <p><strong>Action Plan:</strong> {html.escape(rec.get('strategy', ''))}</p>
          <p class="risk"><strong>Key Risk:</strong> {html.escape(rec.get('risk', ''))}</p>
        </div>
        """

    news_html = ""
    for item in data.get("top_headlines", [])[:18]:
        url = html.escape(item.get("url") or "#")
        news_html += f"""
        <li>
          <span class="tag">{html.escape(item.get('ticker_tag', 'MACRO'))}</span>
          <span class="sent">[{html.escape(item.get('sentiment', 'Neutral'))}]</span>
          <a href="{url}" target="_blank" rel="noopener">{html.escape(item.get('title', ''))}</a>
          <small>— {html.escape(item.get('source', ''))}</small>
        </li>
        """

    model_label = html.escape(data.get("model_used", "Gemini AI"))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AlphaScrape AI — Automated Market Terminal</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background:#0b0f19; color:#e2e8f0; margin:0; padding:20px; }}
    .wrap {{ max-width:1100px; margin:0 auto; }}
    header {{ display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; border-bottom:1px solid #1e293b; padding-bottom:16px; margin-bottom:20px; }}
    h1 {{ margin:0; font-size:1.5rem; color:#f8fafc; }}
    .meta {{ color:#94a3b8; font-size:0.9rem; margin-top:4px; }}
    .err-banner {{ background:#450a0a; border:1px solid #ef4444; color:#fecaca; padding:14px 16px; border-radius:8px; margin-bottom:18px; font-size:0.88rem; word-break:break-word; }}
    .banner {{ background:#111827; border:1px solid #1e293b; border-left:4px solid #3b82f6; padding:16px; border-radius:8px; margin-bottom:24px; }}
    .macro-strip {{ margin-top:10px; display:flex; flex-wrap:wrap; align-items:center; gap:6px; font-size:0.8rem; color:#94a3b8; }}
    .hist-pill {{ background:#1e293b; color:#e2e8f0; padding:2px 8px; border-radius:999px; font-size:0.75rem; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap:16px; margin-bottom:28px; }}
    .card {{ background:#111827; border:1px solid #1e293b; border-radius:10px; padding:16px; }}
    .card-top {{ display:flex; justify-content:space-between; align-items:flex-start; gap:8px; margin-bottom:10px; }}
    .ticker {{ font-size:1.2rem; font-weight:700; color:#38bdf8; margin-right:8px; }}
    .company {{ color:#94a3b8; font-size:0.85rem; }}
    .subtags {{ margin-top:4px; display:flex; flex-wrap:wrap; gap:6px; }}
    .pill {{ background:#1e293b; color:#94a3b8; font-size:0.72rem; padding:2px 7px; border-radius:4px; }}
    .badge {{ padding:4px 10px; border-radius:999px; font-size:0.78rem; font-weight:600; white-space:nowrap; }}
    .metrics {{ display:flex; flex-wrap:wrap; gap:12px; font-size:0.82rem; color:#cbd5e1; background:#0f172a; padding:8px 10px; border-radius:6px; margin-bottom:8px; }}
    .tech-bar {{ font-size:0.79rem; color:#a7f3d0; background:#064e3b66; border:1px solid #065f46; padding:5px 9px; border-radius:6px; margin-bottom:6px; }}
    .delta {{ font-size:0.79rem; color:#93c5fd; background:#172554; border:1px solid #1e3a8a; padding:6px 9px; border-radius:6px; margin-bottom:8px; }}
    p {{ font-size:0.88rem; line-height:1.45; margin:6px 0; }}
    .risk {{ color:#fca5a5; }}
    ul.wire {{ list-style:none; padding:0; background:#111827; border:1px solid #1e293b; border-radius:10px; }}
    ul.wire li {{ padding:1
