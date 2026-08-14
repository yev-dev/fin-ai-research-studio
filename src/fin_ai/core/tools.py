
import json
import logging
from typing import Any, Optional
logger = logging.getLogger(__name__)

import pandas as pd
from pandas import DataFrame

from fin_ai.core.exceptions import (
    MarketDataNotFoundError,
    MarketDataServiceError,
)
from fin_ai.core.service import MarketDataService


# ---------------------------------------------------------------------------
# Service instance — initialised once at import time
# ---------------------------------------------------------------------------

_market_service: MarketDataService | None = None


def _get_service() -> MarketDataService:
    """Return the (cached) market data service instance."""
    global _market_service
    if _market_service is None:
        _market_service = MarketDataService.from_environment()
        # Log creation of the market data service
        logger.info("Initialized MarketDataService from environment")
    return _market_service


# ---------------------------------------------------------------------------
# JSON serialisation helpers (unchanged — kept for the public functions)
# ---------------------------------------------------------------------------


def _to_json_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _to_json_key(value: Any) -> str | int | float | bool | None:
    normalized = _to_json_value(value)
    if isinstance(normalized, (str, int, float, bool)) or normalized is None:
        return normalized
    return str(normalized)


def _dataframe_to_records(frame: DataFrame, max_rows: int = 200) -> dict:
    if frame is None or frame.empty:
        return {"row_count": 0, "truncated": False, "records": []}

    converted = frame.copy()
    if isinstance(converted.index, pd.DatetimeIndex):
        converted.index = converted.index.strftime("%Y-%m-%d")

    records = converted.reset_index().to_dict(orient="records")
    json_records = [
        {_to_json_key(k): _to_json_value(v) for k, v in row.items()}
        for row in records[:max_rows]
    ]
    return {
        "row_count": len(records),
        "truncated": len(records) > max_rows,
        "records": json_records,
    }


def _handle_service_error(
    symbol: str,
    data_type: str,
    default_return: dict,
    exc: Exception,
) -> dict:
    """Wrap a service exception into a user-facing error dict.

    Gracefully handles both :class:`MarketDataNotFoundError` and
    :class:`MarketDataServiceError` so callers (including LLM tool-call
    dispatch) always receive a structured response.
    """
    result = dict(default_return)
    result["error"] = str(exc)
    result["error_type"] = type(exc).__name__
    return result


# ---------------------------------------------------------------------------
# Yahoo Finance tool functions
# ---------------------------------------------------------------------------


def get_stock_data(symbol: str, start_date: str, end_date: str) -> dict:
    """Retrieve stock price data for a ticker symbol within the date range."""
    logger.info("get_stock_data: symbol=%s start=%s end=%s", symbol, start_date, end_date)
    service = _get_service()
    try:
        stock_data = service.get_stock_data(symbol, start_date, end_date)
        payload = _dataframe_to_records(stock_data)
        payload.update({
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
        })
        return payload
    except (MarketDataNotFoundError, MarketDataServiceError) as exc:
        return _handle_service_error(symbol, "stock_data", {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "row_count": 0,
            "truncated": False,
            "records": [],
        }, exc)


def get_stock_info(symbol: str) -> dict:
    """Fetches and returns latest stock information."""
    logger.info("get_stock_info: symbol=%s", symbol)
    service = _get_service()
    try:
        return service.get_stock_info(symbol)
    except (MarketDataNotFoundError, MarketDataServiceError) as exc:
        return {"error": str(exc), "error_type": type(exc).__name__}


def get_company_info(symbol: str, save_path: Optional[str] = None) -> dict:
    """Fetches and returns company information as a DataFrame."""
    logger.info("get_company_info: symbol=%s save_path=%s", symbol, save_path)
    service = _get_service()
    try:
        df = service.get_company_info(symbol)
        if df.empty:
            return {
                "symbol": symbol,
                "company_info": {},
                "saved_to": None,
                "error": f"No company info available for '{symbol}'.",
            }
        company_info = df.iloc[0].to_dict()
        if save_path:
            df.to_csv(save_path)
            print(f"Company info for {symbol} saved to {save_path}")
        return {
            "symbol": symbol,
            "company_info": company_info,
            "saved_to": save_path,
        }
    except (MarketDataNotFoundError, MarketDataServiceError) as exc:
        return _handle_service_error(symbol, "company_info", {
            "symbol": symbol,
            "company_info": {},
            "saved_to": save_path,
        }, exc)


def get_stock_dividends(symbol: str, save_path: Optional[str] = None) -> dict:
    """Fetches and returns the latest dividends data as a DataFrame."""
    logger.info("get_stock_dividends: symbol=%s save_path=%s", symbol, save_path)
    service = _get_service()
    try:
        dividends = service.get_stock_dividends(symbol)
        if save_path:
            dividends.to_csv(save_path)
            print(f"Dividends for {symbol} saved to {save_path}")
        payload = _dataframe_to_records(dividends)
        payload.update({"symbol": symbol, "saved_to": save_path})
        return payload
    except (MarketDataNotFoundError, MarketDataServiceError) as exc:
        return _handle_service_error(symbol, "dividends", {
            "symbol": symbol,
            "saved_to": save_path,
            "row_count": 0,
            "truncated": False,
            "records": [],
        }, exc)


def get_income_stmt(symbol: str) -> dict:
    """Fetches and returns the latest income statement of the company as a DataFrame."""
    service = _get_service()
    try:
        income_stmt = service.get_income_stmt(symbol)
        payload = _dataframe_to_records(income_stmt)
        payload.update({"symbol": symbol})
        return payload
    except (MarketDataNotFoundError, MarketDataServiceError) as exc:
        return _handle_service_error(symbol, "income_stmt", {
            "symbol": symbol,
            "row_count": 0,
            "truncated": False,
            "records": [],
        }, exc)


def get_balance_sheet(symbol: str) -> dict:
    """Fetches and returns the latest balance sheet of the company as a DataFrame."""
    service = _get_service()
    try:
        balance_sheet = service.get_balance_sheet(symbol)
        payload = _dataframe_to_records(balance_sheet)
        payload.update({"symbol": symbol})
        return payload
    except (MarketDataNotFoundError, MarketDataServiceError) as exc:
        return _handle_service_error(symbol, "balance_sheet", {
            "symbol": symbol,
            "row_count": 0,
            "truncated": False,
            "records": [],
        }, exc)


def get_cash_flow(symbol: str) -> dict:
    """Fetches and returns the latest cash flow statement of the company as a DataFrame."""
    service = _get_service()
    try:
        cash_flow = service.get_cash_flow(symbol)
        payload = _dataframe_to_records(cash_flow)
        payload.update({"symbol": symbol})
        return payload
    except (MarketDataNotFoundError, MarketDataServiceError) as exc:
        return _handle_service_error(symbol, "cash_flow", {
            "symbol": symbol,
            "row_count": 0,
            "truncated": False,
            "records": [],
        }, exc)


def get_analyst_recommendations(symbol: str) -> dict:
    """Fetches the latest analyst recommendations and returns the most common recommendation and its count."""
    service = _get_service()
    try:
        recommendations = service.get_analyst_recommendations(symbol)
        if recommendations.empty:
            return {
                "symbol": symbol,
                "majority_recommendation": None,
                "vote_count": 0,
                "has_recommendations": False,
            }

        row_0 = recommendations.iloc[0, 1:]
        max_votes = row_0.max()
        majority_voting_result = row_0[row_0 == max_votes].index.tolist()

        return {
            "symbol": symbol,
            "majority_recommendation": majority_voting_result[0],
            "vote_count": _to_json_value(max_votes),
            "has_recommendations": True,
        }
    except (MarketDataNotFoundError, MarketDataServiceError) as exc:
        return _handle_service_error(symbol, "analyst_recommendations", {
            "symbol": symbol,
            "majority_recommendation": None,
            "vote_count": 0,
            "has_recommendations": False,
        }, exc)


def get_credit_spread(spread_id: str, start_date: str, end_date: str) -> dict:
    """Retrieve credit spread / indices time series (e.g. CDX, iTraxx) for date range.

    Parameters
    ----------
    spread_id: str
        Identifier used by the MarketDataService for the credit spread series.
    start_date, end_date: str
        ISO-format date strings (YYYY-MM-DD).
    """
    logger.info("get_credit_spread: spread_id=%s start=%s end=%s", spread_id, start_date, end_date)
    service = _get_service()
    # Prefer a dedicated service method if available
    if hasattr(service, "get_credit_spread"):
        try:
            df = service.get_credit_spread(spread_id, start_date, end_date)
            payload = _dataframe_to_records(df)
            payload.update({
                "spread_id": spread_id,
                "start_date": start_date,
                "end_date": end_date,
            })
            return payload
        except (MarketDataNotFoundError, MarketDataServiceError) as exc:
            return _handle_service_error(spread_id, "credit_spread", {
                "spread_id": spread_id,
                "start_date": start_date,
                "end_date": end_date,
                "row_count": 0,
                "truncated": False,
                "records": [],
            }, exc)

    # Fallback: map well-known credit series to an equity-like symbol or provider id
    # Common fallbacks may be configured elsewhere; for now attempt `get_stock_data`.
    try:
        # If spread_id looks like a provider symbol, try stock data endpoint
        payload = get_stock_data(spread_id, start_date, end_date)
        if payload.get("row_count", 0) > 0 and not payload.get("error"):
            payload.update({"spread_id": spread_id})
            return payload
    except Exception:
        pass

    # Final graceful failure
    return _handle_service_error(spread_id, "credit_spread", {
        "spread_id": spread_id,
        "start_date": start_date,
        "end_date": end_date,
        "row_count": 0,
        "truncated": False,
        "records": [],
    }, MarketDataNotFoundError(f"Credit spread '{spread_id}' not available"))


def get_macro_series(series_id: str, start_date: str, end_date: str) -> dict:
    """Fetch macroeconomic time series (breakevens, CPI, real yields, FX indices).

    The `series_id` is an opaque provider-specific identifier; callers should
    map friendly names (e.g. `US10Y_BREAKEVEN`, `DXY`) to provider symbols.
    """
    logger.info("get_macro_series: series_id=%s start=%s end=%s", series_id, start_date, end_date)
    service = _get_service()
    # Prefer a dedicated service method if implemented
    if hasattr(service, "get_macro_series"):
        try:
            df = service.get_macro_series(series_id, start_date, end_date)
            payload = _dataframe_to_records(df)
            payload.update({
                "series_id": series_id,
                "start_date": start_date,
                "end_date": end_date,
            })
            return payload
        except (MarketDataNotFoundError, MarketDataServiceError) as exc:
            return _handle_service_error(series_id, "macro_series", {
                "series_id": series_id,
                "start_date": start_date,
                "end_date": end_date,
                "row_count": 0,
                "truncated": False,
                "records": [],
            }, exc)

    # Fallbacks: try common endpoints
    # 1) If series_id maps to an equity/ETF symbol, use get_stock_data
    try:
        payload = get_stock_data(series_id, start_date, end_date)
        if payload.get("row_count", 0) > 0 and not payload.get("error"):
            payload.update({"series_id": series_id})
            return payload
    except Exception:
        pass

    # 2) As a last resort, try to interpret series_id as a provider-specific symbol
    # that may be available via the service generic `get_stock_data` call.
    try:
        payload = get_stock_data(series_id, start_date, end_date)
        if payload.get("row_count", 0) > 0 and not payload.get("error"):
            payload.update({"series_id": series_id})
            return payload
    except Exception:
        pass

    return _handle_service_error(series_id, "macro_series", {
        "series_id": series_id,
        "start_date": start_date,
        "end_date": end_date,
        "row_count": 0,
        "truncated": False,
        "records": [],
    }, MarketDataNotFoundError(f"Macro series '{series_id}' not available"))


# ---------------------------------------------------------------------------
# Research publishing — PDF / HTML generation + email distribution
# ---------------------------------------------------------------------------

import json as _json
import os as _os
import smtplib as _smtplib
from datetime import datetime as _datetime
from email.mime.text import MIMEText as _MIMEText
from email.mime.multipart import MIMEMultipart as _MIMEMultipart
from email.mime.base import MIMEBase as _MIMEBase
from email import encoders as _encoders
from pathlib import Path as _Path
import re as _re
import html as _html
import zlib as _zlib
import base64 as _base64

from fin_ai.config.fin_ai import PUBLISHED_RESEARCH_DIR

_OUTPUT_DIR = _Path(PUBLISHED_RESEARCH_DIR)
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ title }}</title>
<style>
  body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
         max-width: 900px; margin: 40px auto; padding: 0 20px;
         color: #1a1a2e; line-height: 1.7; background: #fafafa; }
  h1 { color: #16213e; border-bottom: 3px solid #0f3460; padding-bottom: 10px; }
  h2 { color: #0f3460; margin-top: 30px; }
  h3 { color: #533483; }
  .meta { color: #666; font-size: 0.9em; margin-bottom: 30px; }
  .disclaimer { border-top: 1px solid #ccc; margin-top: 40px; padding-top: 15px;
                font-size: 0.8em; color: #888; }
  table { border-collapse: collapse; width: 100%; margin: 15px 0; }
  th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }
  th { background: #0f3460; color: white; }
  tr:nth-child(even) { background: #f2f2f2; }
  pre { background: #1a1a2e; color: #e0e0e0; padding: 15px; border-radius: 8px;
        overflow-x: auto; }
  code { background: #eee; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }
  blockquote { border-left: 4px solid #0f3460; margin: 15px 0; padding: 10px 20px;
               background: #f0f0f5; }
  @media print { body { max-width: 100%; } }
</style>
</head>
<body>
<h1>{{ title }}</h1>
<div class="meta">
  Generated: {{ date }} | FinAI Research Publisher
</div>
{{ content }}
<div class="disclaimer">
  This research is generated by an AI-powered agent.  It does not constitute
  financial advice.  Verify all data points before making investment decisions.
</div>
</body>
</html>"""


def _embed_report_images(html: str) -> str:
    """Inline local image files referenced by ``<img>`` as base64 data URIs.

    This lets generated charts render reliably in both the HTML preview and the
    weasyprint PDF path, regardless of ``file://`` access restrictions in the
    browser or rendering engine.
    """
    if not html or "<img" not in html:
        return html

    def _inline(match: "re.Match[str]") -> str:
        src = match.group(1)
        if src.startswith("data:"):
            return match.group(0)
        path = _Path(src)
        if not path.is_file():
            return match.group(0)
        suffix = path.suffix.lower()
        if suffix not in (".png", ".jpg", ".jpeg", ".gif"):
            return match.group(0)
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
        }.get(suffix, "image/png")
        try:
            b64 = _base64.b64encode(path.read_bytes()).decode("ascii")
        except Exception:
            return match.group(0)
        return (
            f'<img src="data:{mime};base64,{b64}" '
            f'style="max-width:100%;height:auto;">'
        )

    return _re.sub(
        r'<img[^>]*?src="([^"]+\.(?:png|jpe?g|gif))"[^>]*?>',
        _inline,
        html,
        flags=_re.IGNORECASE,
    )


def _md_to_html(content: str) -> str:
    """Convert Markdown/plain/HTML content to HTML for publishing.

    Also expands ``CHART:<path>`` markers (emitted by the chart generation
    tools) into embedded images so published reports contain real charts
    rather than text placeholders.
    """
    text = (content or "").strip()
    if not text:
        return "<p><em>No report content was provided.</em></p>"

    # Expand chart markers (own-line "CHART:/abs/path.png") into markdown images.
    text = _re.sub(
        r"(?m)^CHART:\s*(\S+\.(?:png|jpe?g|gif))\s*$",
        lambda m: f"\n![chart]({m.group(1)})\n",
        text,
    )

    # If the agent already produced HTML, render it directly.
    if _re.search(r"<\s*/?\s*[a-zA-Z][^>]*>", text):
        return _embed_report_images(text)

    if any(marker in text for marker in ("#", "##", "**", "```", "- ", "* ")):
        try:
            import markdown as _mdlib
            return _embed_report_images(_mdlib.markdown(
                text,
                extensions=["tables", "fenced_code", "codehilite", "nl2br"],
            ))
        except ImportError:
            # Fallback: use markdown-it-py (already in requirements.txt)
            from markdown_it import MarkdownIt
            md = MarkdownIt("commonmark", {"breaks": True, "html": True})
            return _embed_report_images(md.render(text))

    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"<pre>{escaped}</pre>"


def _render_html(title: str, html_body: str) -> str:
    """Render full HTML page from title and body HTML."""
    from jinja2 import Template as _Template
    date_str = _datetime.now().strftime("%Y-%m-%d %H:%M")
    template = _Template(_HTML_TEMPLATE)
    return template.render(title=title, date=date_str, content=html_body)


def _content_to_plain_text(content: str) -> str:
    """Best-effort conversion of markdown/HTML/plain content to readable text.

    Used by the built-in (dependency-free) PDF writer so that a real PDF can
    always be generated, even when weasyprint and its native dependencies are
    not available.
    """
    text = (content or "").strip()
    if not text:
        return "No report content was provided."

    html = _md_to_html(text)

    # Convert block/paragraph tags and <br> into newlines before stripping tags.
    html = _re.sub(r"<br\s*/?>", "\n", html, flags=_re.IGNORECASE)
    html = _re.sub(
        r"</(p|div|h[1-6]|li|tr|pre|table|blockquote)>",
        "\n",
        html,
        flags=_re.IGNORECASE,
    )
    html = _re.sub(r"<[^>]+>", "", html)
    plain = _html.unescape(html)

    lines = []
    for ln in plain.split("\n"):
        lines.append(_re.sub(r"[ \t]+", " ", ln).strip())
    return "\n".join(lines)


def _pdf_escape(text: str) -> str:
    """Escape a string for inclusion inside a PDF string literal object."""
    s = (text or "").replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    # Strip control characters that are illegal inside a PDF string.
    return "".join(ch for ch in s if ord(ch) >= 32)


def _pdf_num(value: float) -> str:
    """Format a float for PDF coordinates (e.g. 612.0 -> '612')."""
    return f"{value:g}"


def _safe_filename(title: str) -> str:
    """Sanitize title into a safe filename prefix."""
    safe = "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).rstrip()
    return safe[:80] if safe else "research_report"


def _build_pdf_bytes(title: str, content_text: str) -> bytes:
    """Build a minimal, valid, multi-page PDF from plain text content.

    This is a fully self-contained writer (stdlib only) used as a fallback so
    that a genuine ``.pdf`` file is produced even when weasyprint — and its
    native OS dependencies (pango/cairo etc.) — are unavailable.  The output is
    deliberately simple (Helvetica text, word-wrapped, paginated) but is a real
    PDF that any PDF reader can open and print.
    """
    FONT_SIZE = 11.0
    LINE_HEIGHT = 15.0
    TITLE_SIZE = 16.0
    MARGIN = 54.0
    PAGE_W, PAGE_H = 612.0, 792.0
    CONTENT_W = PAGE_W - 2 * MARGIN
    TOP_Y = PAGE_H - MARGIN
    BOTTOM_Y = MARGIN

    def _char_w(ch: str) -> float:
        if ch == " ":
            return 0.30 * FONT_SIZE
        if ch in ".,;:!?()[]'\"`":
            return 0.28 * FONT_SIZE
        if ch in "iIlLt1":
            return 0.30 * FONT_SIZE
        if ch in "MWw@&%":
            return 0.85 * FONT_SIZE
        return 0.55 * FONT_SIZE

    def _wrap_line(line: str) -> list[str]:
        wrapped: list[str] = []
        cur = ""
        for word in line.split(" "):
            trial = (cur + " " + word).strip()
            if not cur or sum(_char_w(c) for c in trial) <= CONTENT_W:
                cur = trial
            else:
                wrapped.append(cur)
                cur = word
        if cur:
            wrapped.append(cur)
        if not wrapped:
            wrapped = [""]
        return wrapped

    # --- Paginate the content into per-page line lists --------------------
    pages: list[list[str]] = []
    page_lines: list[str] = []
    y = TOP_Y - 28.0  # reserve the title block on the first page

    for raw_line in content_text.split("\n"):
        for line in _wrap_line(raw_line):
            if y - LINE_HEIGHT < BOTTOM_Y:
                pages.append(page_lines)
                page_lines = []
                y = TOP_Y  # subsequent pages start at the top (no title block)
            page_lines.append(line)
            y -= LINE_HEIGHT
    if page_lines:
        pages.append(page_lines)
    if not pages:  # ensure at least one page exists
        pages = [[]]

    # --- Render each page as a text-drawing content stream ----------------
    def _render_stream(stream_lines: list[str], first: bool) -> bytes:
        parts: list[str] = ["BT"]
        if first:
            parts.append("0.06 0.20 0.42 rg")  # dark blue title
            parts.append(f"/F1 {TITLE_SIZE:g} Tf")
            parts.append(f"{MARGIN:g} {TOP_Y:g} Td")
            parts.append(f"({_pdf_escape(title)}) Tj")
            parts.append("0 0 0 rg")  # back to black for the body
            parts.append(f"1 -{_pdf_num(TITLE_SIZE + 12.0)} Td")
        else:
            parts.append(f"{MARGIN:g} {TOP_Y:g} Td")
        parts.append(f"/F1 {FONT_SIZE:g} Tf")
        for line in stream_lines:
            parts.append(f"({_pdf_escape(line)}) Tj")
            parts.append(f"0 -{_pdf_num(LINE_HEIGHT)} Td")
        parts.append("ET")
        stream = "\n".join(parts).encode("latin-1", "replace")
        return _zlib.compress(stream)

    n_pages = len(pages)
    page_obj_start = 4
    content_start = page_obj_start + n_pages
    kids = " ".join(f"{page_obj_start + i} 0 R" for i in range(n_pages))

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",  # 1
        (f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>").encode("latin-1"),  # 2
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",  # 3
    ]

    for i in range(n_pages):
        content_ref = content_start + i
        page = (
            f"<< /Type /Page /Parent 2 0 R "
            f"/MediaBox [0 0 {_pdf_num(PAGE_W)} {_pdf_num(PAGE_H)}] "
            f"/Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {content_ref} 0 R >>"
        ).encode("latin-1")
        objects.append(page)

    for i, stream_lines in enumerate(pages):
        compressed = _render_stream(stream_lines, first=(i == 0))
        obj = (
            f"<< /Length {len(compressed)} /Filter /FlateDecode >>\nstream\n".encode("latin-1")
            + compressed
            + b"\nendstream"
        )
        objects.append(obj)

    # --- Serialize with an xref table -------------------------------------
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for idx, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{idx} 0 obj\n".encode("latin-1")
        out += body
        out += b"\nendobj\n"

    xref_pos = len(out)
    count = len(objects) + 1
    out += f"xref\n0 {count}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode("latin-1")
    out += (
        f"trailer\n<< /Size {count} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("latin-1")
    )
    return bytes(out)


def publish_research_html(content: str, title: str = "Research Report") -> str:
    """Generate a professional HTML research report and save it locally.

    Parameters
    ----------
    content : str
        Full research content (Markdown or plain text).
    title : str
        Report title displayed in the header.
    """
    html_body = _md_to_html(content)
    safe_title = _safe_filename(title)
    html = _render_html(title, html_body)

    filename = f"{safe_title.replace(' ', '_')}_{_datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    filepath = _OUTPUT_DIR / filename
    filepath.write_text(html, encoding="utf-8")
    logger.info("Published research HTML: %s", str(filepath))

    return _json.dumps({
        "status": "published",
        "format": "html",
        "filepath": str(filepath),
        "filename": filename,
        "title": title,
    }, indent=2)


def publish_research_pdf(content: str, title: str = "Research Report") -> str:
    """Generate a PDF research report.

    Uses weasyprint for high-quality PDF output when it is available.  If
    weasyprint (or its native OS dependencies) is not installed, a real PDF is
    still produced via a built-in, dependency-free PDF writer, so the requested
    ``pdf`` format is always honoured.

    Parameters
    ----------
    content : str
        Full research content (Markdown supported).
    title : str
        Report title.
    """
    html_body = _md_to_html(content)
    safe_title = _safe_filename(title)
    html = _render_html(title, html_body)

    timestamp = _datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = safe_title.replace(" ", "_")

    fallback_reason = ""
    try:
        import contextlib as _contextlib
        import io as _io
        import logging as _logging
        # weasyprint prints a noisy, non-fatal "could not import external
        # libraries" troubleshooting message (to stderr) when its native deps
        # (libgobject / GLib etc.) are missing on macOS.  Silence that noise
        # during the import attempt — if weasyprint can't load, we transparently
        # fall back to the built-in PDF writer below.
        _wp_logger = _logging.getLogger("weasyprint")
        _wp_prev_level = _wp_logger.level
        _wp_logger.setLevel(_logging.CRITICAL)
        try:
            # weasyprint emits its "could not import external libraries" note via
            # a plain print() to STDOUT (text/ffi.py `_dlopen`) before re-raising
            # the OSError, so redirect both streams to keep the dashboard clean.
            with _contextlib.redirect_stdout(_io.StringIO()), \
                 _contextlib.redirect_stderr(_io.StringIO()):
                from weasyprint import HTML as _WHTML
        finally:
            _wp_logger.setLevel(_wp_prev_level)
            if _wp_prev_level == 0:
                _wp_logger.setLevel(_logging.ERROR)
        pdf_path = _OUTPUT_DIR / f"{prefix}_{timestamp}.pdf"
        _WHTML(string=html).write_pdf(str(pdf_path))
        return _json.dumps({
            "status": "published",
            "format": "pdf",
            "filepath": str(pdf_path),
            "filename": pdf_path.name,
            "title": title,
            "engine": "weasyprint",
        }, indent=2)
    except ImportError as exc:
        fallback_reason = f"weasyprint import failed: {exc}"
    except Exception as exc:
        # Native PDF dependencies (e.g., pango/cairo) may be unavailable.
        fallback_reason = str(exc)

    # Dependency-free fallback: produce a genuine PDF even without weasyprint.
    try:
        plain_text = _content_to_plain_text(content)
        pdf_bytes = _build_pdf_bytes(title, plain_text)
    except Exception as exc:
        fallback_reason += f" | built-in pdf writer failed: {exc}"
        # Last resort: printable HTML (clearly labelled as such, not as PDF).
        html_path = _OUTPUT_DIR / f"{prefix}_{timestamp}_printable.html"
        html_path.write_text(html, encoding="utf-8")
        logger.info("Built-in PDF failed; published printable HTML: %s", str(html_path))
        return _json.dumps({
            "status": "fallback",
            "format": "html (print-to-PDF ready)",
            "filepath": str(html_path),
            "filename": html_path.name,
            "title": title,
            "note": "Open in browser and Ctrl+P / Cmd+P to save as PDF.",
            "fallback_reason": fallback_reason,
        }, indent=2)

    pdf_path = _OUTPUT_DIR / f"{prefix}_{timestamp}_builtin.pdf"
    pdf_path.write_bytes(pdf_bytes)
    logger.info("Published research PDF (built-in writer): %s", str(pdf_path))
    return _json.dumps({
        "status": "published",
        "format": "pdf",
        "filepath": str(pdf_path),
        "filename": pdf_path.name,
        "title": title,
        "engine": "builtin (dependency-free)",
        "note": fallback_reason,
    }, indent=2)


def send_research_email(
    recipient: str,
    subject: str,
    body: str,
    attachment_path: str = "",
    smtp_host: str = "",
    smtp_port: int = 587,
    smtp_user: str = "",
    smtp_password: str = "",
) -> str:
    """Send research report via email with optional file attachment.

    SMTP credentials are read from environment variables by default:
    ``AI_RESEARCH_SMTP_HOST``, ``AI_RESEARCH_SMTP_PORT``, ``AI_RESEARCH_SMTP_USER``,
    ``AI_RESEARCH_SMTP_PASSWORD``.  Override by passing arguments directly.

    Parameters
    ----------
    recipient : str
        Email address of the recipient.
    subject : str
        Email subject line.
    body : str
        Email body (Markdown — converted to HTML automatically).
    attachment_path : str
        Optional path to a file to attach.
    smtp_host : str
        SMTP server hostname.  Default: env ``AI_RESEARCH_SMTP_HOST``.
    smtp_port : int
        SMTP port.  Default: env ``AI_RESEARCH_SMTP_PORT`` or 587.
    smtp_user : str
        SMTP username.  Default: env ``AI_RESEARCH_SMTP_USER``.
    smtp_password : str
        SMTP password.  Default: env ``AI_RESEARCH_SMTP_PASSWORD``.
    """
    host = smtp_host or _os.getenv("AI_RESEARCH_SMTP_HOST", "")
    port = smtp_port if smtp_port != 587 else int(_os.getenv("AI_RESEARCH_SMTP_PORT", "587"))
    user = smtp_user or _os.getenv("AI_RESEARCH_SMTP_USER", "")
    password = smtp_password or _os.getenv("AI_RESEARCH_SMTP_PASSWORD", "")

    if not host:
        return _json.dumps({
            "status": "not_sent",
            "error": (
                "SMTP not configured.  Set AI_RESEARCH_SMTP_HOST, AI_RESEARCH_SMTP_USER, "
                "and AI_RESEARCH_SMTP_PASSWORD environment variables."
            ),
        }, indent=2)

    html_body = _md_to_html(body)
    msg = _MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = user or "research@fin-ai.local"
    msg["To"] = recipient
    msg.attach(_MIMEText(html_body, "html", "utf-8"))

    if attachment_path and _Path(attachment_path).is_file():
        with open(attachment_path, "rb") as fh:
            part = _MIMEBase("application", "octet-stream")
            part.set_payload(fh.read())
            _encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f'attachment; filename="{_Path(attachment_path).name}"',
            )
            msg.attach(part)

    try:
        with _smtplib.SMTP(host, port, timeout=30) as server:
            server.starttls()
            if user and password:
                server.login(user, password)
            server.send_message(msg)
        return _json.dumps({
            "status": "sent",
            "recipient": recipient,
            "subject": subject,
            "attachment": attachment_path if attachment_path else None,
        }, indent=2)
    except Exception as exc:
        return _json.dumps({"status": "failed", "error": str(exc)}, indent=2)


def publish_research_report(
    content: str,
    title: str = "Research Report",
    format: str = "html",
    email: str = "",
) -> str:
    """Publish a research report — generate HTML/PDF and optionally email it.

    This is the primary publishing tool.  Chains: format → save → email.

    Parameters
    ----------
    content : str
        Full research content (Markdown format recommended).
    title : str
        Report title.
    format : str
        ``"html"`` or ``"pdf"``.
    email : str
        If provided, the report is emailed to this address after generation.
        Requires SMTP env vars to be configured.
    """
    results: dict[str, Any] = {}

    if format == "pdf":
        pub_result = _json.loads(publish_research_pdf(content, title))
    else:
        pub_result = _json.loads(publish_research_html(content, title))
    results["publish"] = pub_result

    if email.strip():
        filepath = pub_result.get("filepath", "")
        email_result = _json.loads(
            send_research_email(
                recipient=email.strip(),
                subject=f"FinAI Research: {title}",
                body=content,
                attachment_path=filepath,
            )
        )
        results["email"] = email_result

    return _json.dumps(results, indent=2)


YAHOO_FINANCE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_data",
            "description": "Retrieve historical stock prices for a ticker symbol in a date range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol, e.g., AAPL"},
                    "start_date": {"type": "string", "description": "Start date in YYYY-MM-DD format"},
                    "end_date": {"type": "string", "description": "End date in YYYY-MM-DD format"},
                },
                "required": ["symbol", "start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_info",
            "description": "Get the latest stock metadata and quote information for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol, e.g., AAPL"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_info",
            "description": "Get basic company profile fields for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol, e.g., AAPL"},
                    "save_path": {"type": "string", "description": "Optional CSV output path"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_dividends",
            "description": "Get dividend history for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol, e.g., AAPL"},
                    "save_path": {"type": "string", "description": "Optional CSV output path"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_income_stmt",
            "description": "Get the most recent income statement for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol, e.g., AAPL"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_balance_sheet",
            "description": "Get the most recent balance sheet for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol, e.g., AAPL"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cash_flow",
            "description": "Get the most recent cash-flow statement for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol, e.g., AAPL"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_analyst_recommendations",
            "description": "Get the latest analyst consensus recommendation summary for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol, e.g., AAPL"},
                },
                "required": ["symbol"],
            },
        },
    },
]


LITELLM_TOOL_FUNCTIONS = {
    "get_stock_data": get_stock_data,
    "get_stock_info": get_stock_info,
    "get_company_info": get_company_info,
    "get_stock_dividends": get_stock_dividends,
    "get_income_stmt": get_income_stmt,
    "get_balance_sheet": get_balance_sheet,
    "get_cash_flow": get_cash_flow,
    "get_analyst_recommendations": get_analyst_recommendations,
    "publish_research_html": publish_research_html,
    "publish_research_pdf": publish_research_pdf,
    "publish_research_report": publish_research_report,
    "send_research_email": send_research_email,
}


def execute_litellm_tool_call(name: str, arguments: dict[str, Any]) -> dict:
    """Execute a LiteLLM tool by name using decoded tool-call arguments."""
    func = LITELLM_TOOL_FUNCTIONS.get(name)
    if func is None:
        return {"error": f"Unsupported tool: {name}"}

    try:
        return func(**arguments)
    except TypeError as exc:
        return {"error": f"Invalid arguments for {name}: {exc}"}
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Shared utilities (extracted to avoid duplication across modules)
# ---------------------------------------------------------------------------


def extract_tool_calls(message: object) -> list[dict]:
    """Extract tool call details from an LLM response message.

    Handles both dict-style and object-style ``tool_calls`` attributes.

    Returns a list of ``{"id", "name", "arguments", "arguments_text"}`` dicts.
    """
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        return []

    extracted: list[dict] = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            fn = tc.get("function", {})
            name = fn.get("name")
            args_text = fn.get("arguments", "{}")
            call_id = tc.get("id")
        else:
            fn = getattr(tc, "function", None)
            name = getattr(fn, "name", None)
            args_text = getattr(fn, "arguments", "{}")
            call_id = getattr(tc, "id", None)
        try:
            arguments = json.loads(args_text or "{}")
        except json.JSONDecodeError:
            arguments = {}
        extracted.append({
            "id": call_id or f"call_{len(extracted)}",
            "name": name,
            "arguments": arguments,
            "arguments_text": args_text or "{}",
        })
    return extracted


def build_tool_aware_system_prompt(base_prompt: str | None = None) -> str:
    """Augment a system prompt with tool-awareness instructions."""
    tool_names = [
        tool.get("function", {}).get("name", "")
        for tool in YAHOO_FINANCE_TOOLS
        if tool.get("type") == "function"
    ]
    tool_names = [name for name in tool_names if name]
    available = ", ".join(tool_names) if tool_names else "none"
    guidance = (
        "You have access to function tools. "
        f"Available: {available}. "
        "When asked for stock data, financials, or analyst recs, call the "
        "appropriate tool instead of guessing."
    )
    base = (base_prompt or "").strip()
    if guidance in base:
        return base
    if not base:
        return guidance
    return f"{base}\n\n{guidance}"

