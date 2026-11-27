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

CORE_WATCHLIST = [
    "NVDA", "AVGO", "MU", "INTC", "GOOGL",
    "META", "AAPL", "AMZN", "TSLA", "PLTR", "VUAA.L"
]

MARKET_RSS_FEEDS = [
    ("CNBC Top News", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
    ("CNBC Investing", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069"),
    ("MarketWatch Top Stories", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
]

GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]
HISTORY_FILE = "docs/history.json"
MAX_HISTORY_RUNS = 15
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/128.0.0.0 Safari/537.36"}


def clean_num(val):
    return None if isinstance(val, float) and (math.isnan(val) or math.isinf(val)) else val


def calculate_rsi(closes: list[float], period: int = 14) -> float | None:
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
    return round(100.0 - (100.0 / (1.0 + (avg_gain / avg_loss))), 1)


def classify_trend(price: float | None, sma_20: float | None, sma_50: float | None) -> str:
    if not price:
        return "Trend Unavailable"
    if sma_20 and sma_50:
        if price > sma_20 and price > sma_50:
            return "Strong Uptrend (>20D & >50D SMA)"
        if price > sma_50:
            return "Pullback in Uptrend (<20D, >50D SMA)"
        if price >= sma_20:
            return "Early Rebound (>20D, <50D SMA)"
        return "Downtrend (<20D & <50D SMA)"
    if sma_20:
        return "Above 20D SMA" if price >= sma_20 else "Below 20D SMA"
    return "Insufficient History"


def detect_session() -> tuple[str, str]:
    now_et = datetime.now(ZoneInfo("America/New_York"))
    mins = now_et.hour * 60 + now_et.minute
    if 240 <= mins < 570:
        session = "Pre-Market (04:00-09:30 ET)"
    elif 570 <= mins < 960:
        session = "Open Market (09:30-16:00 ET)"
    elif 960 <= mins < 1200:
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
    except Exception:
        return []


def save_history(history: list[dict], session: str, timestamp_et: str, analysis: dict) -> list[dict]:
    snapshot = {
        "session": session,
        "updated_et": timestamp_et,
        "market_mood": analysis.get("market_mood"),
        "overall_sentiment_score": analysis.get("overall_sentiment_score"),
        "signals": {
            r["ticker"]: {
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
    if not history:
        return {"runs_recorded": 0, "macro_trail": [], "ticker_trails": {}, "recent_detailed_sessions": []}
    macro_trail = [
        f"{h.get('updated_et', '')} [{h.get('session', '').split(' ')[0]}]: {h.get('overall_sentiment_score', 0):+d} ({h.get('market_mood', '')})"
        for h in history
    ]
    trails: dict[str, list[str]] = {}
    for h in history:
        sess = (h.get("session") or "").split(" ")[0]
        for ticker, sig in (h.get("signals") or {}).items():
            act = (sig.get("action") or "Hold").split("/")[0].strip()
            trails.setdefault(ticker, []).append(f"{sess}:{sig.get('sentiment_score', 0):+d}({act})")
    return {
        "runs_recorded": len(history),
        "macro_trail": macro_trail,
        "ticker_trails": {t: " -> ".join(v[-MAX_HISTORY_RUNS:]) for t, v in trails.items()},
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
            desc = BeautifulSoup(item.findtext("description") or "", "html.parser").get_text(" ", strip=True)[:180]
            if title:
                items.append({"source": source_name, "title": title, "summary": desc, "url": link})
    except Exception as exc:
        print(f"[WARN] RSS skipped for {source_name}: {exc}", flush=True)
    return items


def fetch_ticker_quote(ticker: str, source_group: str = "Core Watchlist", sector_hint: str | None = None) -> dict | None:
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
            active = pre_price or post_price or reg_price
            pct = round(((active - prev_close) / prev_close) * 100, 2) if active and prev_close else None

            raw_closes = ((result.get("indicators", {}).get("quote") or [{}])[0]).get("close") or []
            closes = [float(c) for c in raw_closes if isinstance(c, (int, float)) and not math.isnan(c)]
            sma_20 = round(sum(closes[-20:]) / 20, 2) if len(closes) >= 20 else None
            sma_50 = round(sum(closes[-50:]) / 50, 2) if len(closes) >= 50 else None
            rsi_14 = calculate_rsi(closes, 14)
            chg_5d = round(((closes[-1] - closes[-6]) / closes[-6]) * 100, 2) if len(closes) >= 6 and closes[-6] else None

            return {
                "ticker": ticker,
                "name": meta.get("shortName") or meta.get("longName") or ticker,
                "sector": sector_hint or meta.get("instrumentType") or "Equity",
                "source_group": source_group,
                "regular_price": reg_price,
                "pre_market_price": pre_price,
                "post_market_price": post_price,
                "prev_close": prev_close,
                "active_change_pct": pct,
                "change_5d_pct": chg_5d,
                "sma_20": sma_20,
                "sma_50": sma_50,
                "rsi_14": rsi_14,
                "trend_regime": classify_trend(active, sma_20, sma_50),
            }
        except Exception:
            continue
    return None


def discover_market_movers(limit_per_screen: int = 5) -> list[dict]:
    discovered: dict[str, dict] = {}
    for label, key in [("Market Mover: Day Gainer", "day_gainers"), ("Market Mover: Most Active", "most_actives")]:
        url = f"https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?formatted=false&scrIds={key}&count={limit_per_screen}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=8)
            resp.raise_for_status()
            results = resp.json().get("finance", {}).get("result") or []
            for q in (results[0].get("quotes", []) if results else []):
                sym = (q.get("symbol") or "").strip()
                if sym and "^" not in sym and "=" not in sym and sym not in discovered:
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
            print(f"[WARN] Screener '{key}' skipped: {exc}", flush=True)
    return list(discovered.values())


def collect_market_intelligence(core_tickers: list[str]) -> tuple[list[dict], list[dict]]:
    quotes, headlines = [], []
    seen = {t.upper() for t in core_tickers}

    with ThreadPoolExecutor(max_workers=8) as pool:
        q_futs = [pool.submit(fetch_ticker_quote, t, "Core Watchlist") for t in core_tickers]
        r_futs = [pool.submit(fetch_rss_items, u, n, 5) for n, u in MARKET_RSS_FEEDS]
        for t in core_tickers:
            r_futs.append(pool.submit(fetch_rss_items, f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}&region=US&lang=en-US", f"Yahoo ({t})", 3))
        for f in as_completed(q_futs):
            if f.result():
                quotes.append(f.result())
        for f in as_completed(r_futs):
            headlines.extend(f.result())

    extra = [m for m in discover_market_movers(5) if m["ticker"].upper() not in seen][:8]
    if extra:
        with ThreadPoolExecutor(max_workers=8) as pool:
            mq_futs = {pool.submit(fetch_ticker_quote, m["ticker"], m["source_group"], m["sector"]): m for m in extra}
            mr_futs = [pool.submit(fetch_rss_items, f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={m['ticker']}&region=US&lang=en-US", f"Yahoo ({m['ticker']})", 2) for m in extra]
            for f, fb in mq_futs.items():
                quotes.append(f.result() or fb)
            for f in as_completed(mr_futs):
                headlines.extend(f.result())

    return quotes, headlines


def analyze_with_gemini(session: str, timestamp_et: str, quotes: list[dict], headlines: list[dict], history: list[dict]) -> dict:
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY secret is missing.")

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1beta"))
    compact_news = [{"source": h["source"], "title": h["title"], "url": h["url"]} for h in headlines[:45]]
    trajectory = build_trajectory_summary(history)

    prompt = f"""
You are a senior quantitative equity analyst. Analyze quotes, 20D/50D SMA, RSI(14), 5-day sentiment trajectory, and headlines for US session: {session} ({timestamp_et}).
Rank 8 to 10 setups across Core Watchlist and Market Movers.

RULES:
1. CONTINUITY: Use TRAJECTORY_HISTORY. Keep prior rating unless a material catalyst, earnings surprise, or >2.5% move breaks 20D/50D SMA.
2. TECHNICALS: Avoid "Strong Buy" below both 20D & 50D SMA (unless major earnings reversal) or when RSI(14) >= 72 (overbought).
3. SESSIONS: Pre-Market gaps are tentative; Post-Market close is the highest-conviction anchor.

Return ONLY valid JSON:
{{
  "market_mood": "Bullish | Cautiously Bullish | Neutral / Mixed | Risk-Off Bearish",
  "overall_sentiment_score": <int -100 to 100>,
  "executive_summary": "<2-3 sentence synthesis of macro, 5-day trajectory, and technical posture>",
  "recommendations": [
    {{
      "ticker": "<TICKER>",
      "company": "<Company>",
      "sector": "<Sector>",
      "source_group": "Core Watchlist | Market Mover",
      "action": "Strong Buy / Accumulate | Tactical Swing Buy | Watchlist / Hold | Trim / Avoid",
      "prev_action": "<Prior action or 'New'>",
      "sentiment_trail": "<e.g. +20 -> +45 -> +62 or 'Initial baseline'>",
      "technical_badge": "<e.g. >20D & >50D SMA | RSI 58.2>",
      "change_reason": "<1 short sentence on continuity or why rating changed>",
      "sentiment_score": <int -100 to 100>,
      "confidence": <int 1 to 100>,
      "price_snapshot": "<e.g. $142.50 (+2.4%)>",
      "catalyst": "<Catalyst from headlines>",
      "strategy": "<Entry/position note>",
      "risk": "<Primary downside/technical risk>"
    }}
  ],
  "top_headlines": [
    {{"source": "<Source>", "ticker_tag": "<TICKER or MACRO>", "sentiment": "Bullish | Bearish | Neutral", "title": "<Headline>", "url": "<URL>"}}
  ]
}}

TRAJECTORY_HISTORY: {json.dumps(trajectory)}
QUOTES: {json.dumps(quotes)}
HEADLINES: {json.dumps(compact_news)}
"""

    last_err = None
    for model_name in GEMINI_MODELS:
        try:
            print(f"Requesting analysis from {model_name}...", flush=True)
            with ThreadPoolExecutor(max_workers=1) as pool:
                resp = pool.submit(
                    client.models.generate_content,
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.15),
                ).result(timeout=30)
            text = (resp.text or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(text)
            data["model_used"] = model_name
            return data
        except Exception as exc:
            print(f"[WARN] {model_name} failed: {exc}", flush=True)
            last_err = exc
    raise RuntimeError(str(last_err))


def badge_color(action: str) -> str:
    s = action.lower()
    if "strong buy" in s or "accumulate" in s:
        return "#10b981"
    if "swing" in s:
        return "#3b82f6"
    if "trim" in s or "avoid" in s:
        return "#ef4444"
    return "#f59e0b"


def render_html(session: str, timestamp_et: str, data: dict, history: list[dict]) -> str:
    err_html = f"<div class='err'><strong>API Diagnostic:</strong> {html.escape(data['api_error'])}</div>" if data.get("api_error") else ""
    pills = "".join(
        f"<span class='pill'>{html.escape((h.get('session') or '').split(' ')[0])} {h.get('overall_sentiment_score', 0):+d}</span>"
        for h in history[-6:]
    )
    macro_strip = f"<div class='strip'><strong>5D Trajectory ({len(history)}/{MAX_HISTORY_RUNS}):</strong> {pills}</div>" if pills else ""

    cards = []
    for r in data.get("recommendations", []):
        c = badge_color(r.get("action", ""))
        cards.append(
            f"<div class='card'><div class='top'><div><span class='tk'>{html.escape(r.get('ticker',''))}</span> "
            f"<span class='co'>{html.escape(r.get('company',''))}</span><div class='tags'>"
            f"<span class='pill'>{html.escape(r.get('source_group','Watchlist'))}</span>"
            f"<span class='pill'>{html.escape(r.get('sector',''))}</span></div></div>"
            f"<span class='bdg' style='background:{c}22;color:{c};border:1px solid {c}55'>{html.escape(r.get('action','Hold'))}</span></div>"
            f"<div class='met'><span><b>Price:</b> {html.escape(r.get('price_snapshot','N/A'))}</span>"
            f"<span><b>Sent:</b> {r.get('sentiment_score',0):+d}</span><span><b>Conf:</b> {r.get('confidence',0)}%</span></div>"
            f"<div class='tech'><b>Tech:</b> {html.escape(r.get('technical_badge',''))}</div>"
            f"<div class='dlt'><b>5D Trail:</b> {html.escape(r.get('sentiment_trail',''))} &bull; <b>Prev ({html.escape(r.get('prev_action','New'))}):</b> {html.escape(r.get('change_reason',''))}</div>"
            f"<p><b>Catalyst:</b> {html.escape(r.get('catalyst',''))}</p>"
            f"<p><b>Plan:</b> {html.escape(r.get('strategy',''))}</p>"
            f"<p class='rsk'><b>Risk:</b> {html.escape(r.get('risk',''))}</p></div>"
        )

    news = "".join(
        f"<li><span class='tag'>{html.escape(i.get('ticker_tag','MACRO'))}</span> "
        f"<span class='co'>[{html.escape(i.get('sentiment','Neutral'))}]</span> "
        f"<a href='{html.escape(i.get('url') or '#')}' target='_blank' rel='noopener'>{html.escape(i.get('title',''))}</a> "
        f"<small>— {html.escape(i.get('source',''))}</small></li>"
        for i in data.get("top_headlines", [])[:18]
    )

    css = (
        "body{font-family:system-ui,-apple-system,sans-serif;background:#0b0f19;color:#e2e8f0;margin:0;padding:18px}"
        ".wrap{max-width:1100px;margin:0 auto}header{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;border-bottom:1px solid #1e293b;padding-bottom:14px;margin-bottom:18px}"
        "h1{margin:0;font-size:1.4rem}.meta,.co,small{color:#94a3b8;font-size:.85rem}"
        ".err{background:#450a0a;border:1px solid #ef4444;color:#fecaca;padding:12px;border-radius:8px;margin-bottom:16px}"
        ".ban{background:#111827;border:1px solid #1e293b;border-left:4px solid #3b82f6;padding:14px;border-radius:8px;margin-bottom:20px}"
        ".strip{margin-top:8px;display:flex;flex-wrap:wrap;gap:6px;align-items:center;font-size:.8rem;color:#94a3b8}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:14px;margin-bottom:24px}"
        ".card{background:#111827;border:1px solid #1e293b;border-radius:10px;padding:14px}"
        ".top{display:flex;justify-content:space-between;align-items:flex-start;gap:8px;margin-bottom:8px}"
        ".tk{font-size:1.15rem;font-weight:700;color:#38bdf8}.tags{display:flex;gap:5px;margin-top:4px}"
        ".pill,.tag{background:#1e293b;color:#93c5fd;font-size:.72rem;padding:2px 7px;border-radius:4px}"
        ".bdg{padding:3px 9px;border-radius:999px;font-size:.76rem;font-weight:600;white-space:nowrap}"
        ".met{display:flex;flex-wrap:wrap;gap:10px;font-size:.8rem;background:#0f172a;padding:7px 9px;border-radius:6px;margin-bottom:6px}"
        ".tech{font-size:.78rem;color:#a7f3d0;background:#064e3b66;border:1px solid #065f46;padding:5px 8px;border-radius:6px;margin-bottom:5px}"
        ".dlt{font-size:.78rem;color:#93c5fd;background:#172554;border:1px solid #1e3a8a;padding:5px 8px;border-radius:6px;margin-bottom:7px}"
        "p{font-size:.86rem;line-height:1.4;margin:5px 0}.rsk{color:#fca5a5}"
        "ul{list-style:none;padding:0;background:#111827;border:1px solid #1e293b
