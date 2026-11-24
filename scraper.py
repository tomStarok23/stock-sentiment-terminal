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

# Hard 10-second network timeout so no library can ever hang a GitHub Actions runner
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

# 3. Valid public Google AI Studio model IDs
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/128.0.0.0 Safari/537.36"
}


def clean_num(val):
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return val


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


def fetch_ticker_quote(ticker: str) -> dict | None:
    """Fetches Pre/Open/Post-Market prices directly from Yahoo Finance chart JSON."""
    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        url = f"https://{host}/v8/finance/chart/{ticker}?range=5d&interval=1d&includePrePost=true"
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

            return {
                "ticker": ticker,
                "name": meta.get("shortName") or meta.get("longName") or ticker,
                "sector": meta.get("instrumentType") or "Core Watchlist",
                "source_group": "Core Watchlist",
                "regular_price": reg_price,
                "pre_market_price": pre_price,
                "post_market_price": post_price,
                "prev_close": prev_close,
                "active_change_pct": pct_change,
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

    # Fetch all core quotes and RSS feeds in parallel (~3 seconds total)
    with ThreadPoolExecutor(max_workers=8) as pool:
        quote_futures = {pool.submit(fetch_ticker_quote, t): t for t in core_tickers}
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

    # Discover top movers across all sectors and fetch their headlines in parallel
    movers = discover_market_movers(limit_per_screen=5)
    extra_movers = []
    for mover in movers:
        sym = mover["ticker"].upper()
        if sym not in seen_tickers:
            seen_tickers.add(sym)
            quotes.append(mover)
            extra_movers.append(sym)
        if len(extra_movers) >= 8:
            break

    if extra_movers:
        with ThreadPoolExecutor(max_workers=6) as pool:
            mover_rss = [
                pool.submit(
                    fetch_rss_items,
                    f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={sym}&region=US&lang=en-US",
                    f"Yahoo Finance ({sym})",
                    2,
                )
                for sym in extra_movers
            ]
            for fut in as_completed(mover_rss):
                headlines.extend(fut.result())

    return quotes, headlines


def analyze_with_gemini(session: str, timestamp_et: str, quotes: list[dict], headlines: list[dict]) -> dict:
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

    prompt = f"""
You are a senior quantitative equity analyst. Analyze these live stock quotes and headlines for US market session: {session} ({timestamp_et}).
Rank 8 to 10 actionable setups across BOTH Core Watchlist and Market Movers.

Return ONLY valid JSON matching this schema:
{{
  "market_mood": "Bullish | Cautiously Bullish | Neutral / Mixed | Risk-Off Bearish",
  "overall_sentiment_score": <integer -100 to 100>,
  "executive_summary": "<2-3 sentence synthesis of macro and sector drivers>",
  "recommendations": [
    {{
      "ticker": "<TICKER>",
      "company": "<Company Name>",
      "sector": "<Sector>",
      "source_group": "Core Watchlist | Market Mover",
      "action": "Strong Buy / Accumulate | Tactical Swing Buy | Watchlist / Hold | Trim / Avoid",
      "sentiment_score": <integer -100 to 100>,
      "confidence": <integer 1 to 100>,
      "price_snapshot": "<e.g. $142.50 (+2.4%)>",
      "catalyst": "<Specific catalyst from headlines>",
      "strategy": "<Concrete entry/position note>",
      "risk": "<Primary downside risk>"
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

QUOTES:
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
                        temperature=0.2,
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
    """Publishes raw scraped quotes & headlines with diagnostic info if Gemini API errors."""
    recs = []
    for q in quotes[:12]:
        price = q.get("pre_market_price") or q.get("post_market_price") or q.get("regular_price")
        chg = q.get("active_change_pct")
        snap = f"${price:.2f}" if isinstance(price, (int, float)) else "N/A"
        if isinstance(chg, (int, float)):
            snap += f" ({chg:+.2f}%)"
        recs.append({
            "ticker": q.get("ticker", ""),
            "company": q.get("name", ""),
            "sector": q.get("sector", "Market"),
            "source_group": q.get("source_group", "Watchlist"),
            "action": "Watchlist / Hold",
            "sentiment_score": 0,
            "confidence": 0,
            "price_snapshot": snap,
            "catalyst": "Live market quote collected; awaiting Gemini AI scoring.",
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
        "executive_summary": "Live stock quotes and RSS headlines were scraped successfully, but Gemini AI returned an error (see red diagnostic box above).",
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


def render_html(session: str, timestamp_et: str, data: dict) -> str:
    error_banner = ""
    if data.get("api_error"):
        error_banner = f"""
        <div class="err-banner">
          <strong>Gemini API Diagnostic:</strong> {html.escape(data['api_error'])}
        </div>
        """

    recs_html = ""
    for rec in data.get("recommendations", []):
        color = badge_color(rec.get("action", ""))
        score = rec.get("sentiment_score", 0)
        source_group = rec.get("source_group", "Watchlist")
        sector = rec.get("sector", "")
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
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap:16px; margin-bottom:28px; }}
    .card {{ background:#111827; border:1px solid #1e293b; border-radius:10px; padding:16px; }}
    .card-top {{ display:flex; justify-content:space-between; align-items:flex-start; gap:8px; margin-bottom:10px; }}
    .ticker {{ font-size:1.2rem; font-weight:700; color:#38bdf8; margin-right:8px; }}
    .company {{ color:#94a3b8; font-size:0.85rem; }}
    .subtags {{ margin-top:4px; display:flex; flex-wrap:wrap; gap:6px; }}
    .pill {{ background:#1e293b; color:#94a3b8; font-size:0.72rem; padding:2px 7px; border-radius:4px; }}
    .badge {{ padding:4px 10px; border-radius:999px; font-size:0.78rem; font-weight:600; white-space:nowrap; }}
    .metrics {{ display:flex; flex-wrap:wrap; gap:12px; font-size:0.82rem; color:#cbd5e1; background:#0f172a; padding:8px 10px; border-radius:6px; margin-bottom:10px; }}
    p {{ font-size:0.88rem; line-height:1.45; margin:6px 0; }}
    .risk {{ color:#fca5a5; }}
    ul.wire {{ list-style:none; padding:0; background:#111827; border:1px solid #1e293b; border-radius:10px; }}
    ul.wire li {{ padding:12px 16px; border-bottom:1px solid #1e293b; font-size:0.88rem; }}
    ul.wire li:last-child {{ border-bottom:none; }}
    .tag {{ background:#1e293b; color:#38bdf8; padding:2px 6px; border-radius:4px; font-size:0.75rem; font-weight:600; margin-right:6px; }}
    .sent {{ color:#94a3b8; font-size:0.8rem; margin-right:6px; }}
    a {{ color:#e2e8f0; text-decoration:none; }}
    a:hover {{ text-decoration:underline; color:#38bdf8; }}
    small {{ color:#64748b; }}
    footer {{ text-align:center; color:#64748b; font-size:0.78rem; margin-top:24px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <h1>AlphaScrape AI — Automated Market Intelligence</h1>
        <div class="meta">Session: <strong>{html.escape(session)}</strong> &bull; Updated: <strong>{html.escape(timestamp_et)}</strong></div>
      </div>
      <div>
        <strong>Mood:</strong> {html.escape(data.get('market_mood', 'Neutral'))} ({data.get('overall_sentiment_score', 0):+d}/100)
      </div>
    </header>
    {error_banner}
    <div class="banner">
      <strong>Executive Synthesis:</strong> {html.escape(data.get('executive_summary', ''))}
    </div>
    <h2>Actionable Watchlist &amp; Market Mover Signals</h2>
    <div class="grid">{recs_html}</div>
    <h2>Scraped Wire Headlines</h2>
    <ul class="wire">{news_html}</ul>
    <footer>Generated automatically via GitHub Actions &amp; {model_label}. For informational research only, not financial advice.</footer>
  </div>
</body>
</html>"""


def main() -> None:
    session, timestamp_et = detect_session()
    print(f"Running {session} scan at {timestamp_et}...", flush=True)
    quotes, headlines = collect_market_intelligence(CORE_WATCHLIST)
    print(f"Collected {len(quotes)} quotes and {len(headlines)} headlines.", flush=True)

    try:
        analysis = analyze_with_gemini(session, timestamp_et, quotes, headlines)
    except Exception as exc:
        print(f"[ERROR] Gemini analysis failed: {exc}", flush=True)
        analysis = build_fallback_analysis(str(exc), quotes, headlines)

    os.makedirs("docs", exist_ok=True)
    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(render_html(session, timestamp_et, analysis))
    with open("docs/latest.json", "w", encoding="utf-8") as f:
        json.dump({"session": session, "updated_et": timestamp_et, **analysis}, f, indent=2)

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as sf:
            sf.write(f"## {session} ({timestamp_et})\n\n")
            if analysis.get("api_error"):
                sf.write(f"> **Gemini API Diagnostic:** `{analysis['api_error']}`\n\n")
            sf.write(f"**Market Mood:** {analysis.get('market_mood')} ({analysis.get('overall_sentiment_score')}/100)\n\n")
            sf.write(f"{analysis.get('executive_summary')}\n\n")
            for r in analysis.get("recommendations", []):
                sf.write(f"- **{r.get('ticker')}** [{r.get('source_group')}] — `{r.get('action')}` (Sentiment {r.get('sentiment_score')}/100): {r.get('catalyst')}\n")

    print("Dashboard successfully written to docs/index.html", flush=True)


if __name__ == "__main__":
    main()
