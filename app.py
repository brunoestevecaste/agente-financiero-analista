"""Asistente de Finanzas Personales con LangChain + Streamlit.

Requisitos:
- Configurar GOOGLE_API_KEY y TAVILY_API_KEY desde la interfaz (sidebar).
- Ejecutar con: streamlit run app.py
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any

import streamlit as st
import yfinance as yf
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from tavily import TavilyClient
except Exception:
    TavilyClient = None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_result(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _message_to_text(message: AnyMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                chunks.append(str(item["text"]))
            else:
                chunks.append(str(item))
        return "\n".join(chunks)
    return str(content)


def _get_rsi(close_series, period: int = 14) -> float | None:
    if close_series is None or len(close_series) < period + 1:
        return None
    delta = close_series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(period).mean()
    avg_loss = losses.rolling(period).mean()
    last_avg_loss = avg_loss.iloc[-1]
    if last_avg_loss == 0:
        return 100.0
    rs = avg_gain.iloc[-1] / last_avg_loss
    return _safe_float(100 - (100 / (1 + rs)))


def _safe_json_loads(payload: str) -> dict[str, Any]:
    try:
        loaded = json.loads(payload)
        if isinstance(loaded, dict):
            return loaded
        return {"value": loaded}
    except Exception:
        return {"raw_result": payload}


def _extract_json_text(payload: str) -> str:
    text = payload.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first : last + 1]
    return text


def _get_atr(high_series, low_series, close_series, period: int = 14) -> float | None:
    if high_series is None or low_series is None or close_series is None or len(close_series) < period + 1:
        return None
    prev_close = close_series.shift(1)
    tr = (high_series - low_series).to_frame("hl")
    tr["hc"] = (high_series - prev_close).abs()
    tr["lc"] = (low_series - prev_close).abs()
    true_range = tr.max(axis=1)
    atr = true_range.ewm(alpha=1 / period, adjust=False).mean()
    return _safe_float(atr.iloc[-1])


def _get_adx(high_series, low_series, close_series, period: int = 14) -> tuple[float | None, float | None, float | None]:
    if high_series is None or low_series is None or close_series is None or len(close_series) < (period * 2):
        return None, None, None

    up_move = high_series.diff()
    down_move = -low_series.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    prev_close = close_series.shift(1)
    tr = (high_series - low_series).to_frame("hl")
    tr["hc"] = (high_series - prev_close).abs()
    tr["lc"] = (low_series - prev_close).abs()
    true_range = tr.max(axis=1)
    atr = true_range.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.where(atr != 0))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.where(atr != 0))
    di_sum = (plus_di + minus_di).where((plus_di + minus_di) != 0)
    dx = ((plus_di - minus_di).abs() / di_sum) * 100
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return _safe_float(adx.iloc[-1]), _safe_float(plus_di.iloc[-1]), _safe_float(minus_di.iloc[-1])


def _days_to_time_range(days: int) -> str:
    if days <= 1:
        return "day"
    if days <= 7:
        return "week"
    if days <= 31:
        return "month"
    return "year"


def _extract_symbol_hint(text: str) -> str | None:
    cleaned = text.strip().upper()
    if not cleaned:
        return None
    pattern = re.compile(r"\b[A-Z]{1,6}(?:-[A-Z]{2,6})?\b")
    matches = pattern.findall(cleaned)
    blacklist = {"USA", "ETF", "IPO", "PER", "CON", "LAS", "LOS"}
    for match in matches:
        if match not in blacklist:
            return match
    return None


def _looks_like_asset_query(text: str) -> bool:
    sample = text.lower()
    keywords = [
        "acción",
        "acciones",
        "stock",
        "ticker",
        "cripto",
        "crypto",
        "btc",
        "eth",
        "precio",
        "analiza",
        "análisis",
        "empresa",
        "activo",
    ]
    return any(keyword in sample for keyword in keywords) or _extract_symbol_hint(text) is not None


def _simple_sentiment(text: str) -> str:
    sample = text.lower()
    positive_terms = [
        "beat",
        "growth",
        "surge",
        "upgrade",
        "profit",
        "record",
        "strong",
        "bullish",
        "outperform",
    ]
    negative_terms = [
        "miss",
        "drop",
        "downgrade",
        "loss",
        "weak",
        "lawsuit",
        "risk",
        "bearish",
        "volatility",
    ]
    pos_score = sum(sample.count(term) for term in positive_terms)
    neg_score = sum(sample.count(term) for term in negative_terms)
    if pos_score > neg_score:
        return "positivo"
    if neg_score > pos_score:
        return "negativo"
    return "neutral"


@tool
def get_market_price(symbol: str) -> str:
    """Obtiene precio actual y cambio diario aproximado de una acción o cripto con yfinance."""
    clean_symbol = symbol.upper().strip()
    ticker = yf.Ticker(clean_symbol)

    price: float | None = None
    currency = "N/A"
    previous_close: float | None = None

    try:
        fast_info = getattr(ticker, "fast_info", {}) or {}
        price = _safe_float(fast_info.get("last_price") or fast_info.get("regular_market_price"))
        currency = fast_info.get("currency") or currency
        previous_close = _safe_float(fast_info.get("previous_close"))
    except Exception:
        pass

    history = ticker.history(period="5d", interval="1d")
    if history.empty and price is None:
        return _json_result(
            {"symbol": clean_symbol, "error": f"No se encontraron datos para '{clean_symbol}'."}
        )

    if price is None and not history.empty:
        price = _safe_float(history["Close"].dropna().iloc[-1])

    if previous_close is None and not history.empty and len(history["Close"].dropna()) >= 2:
        previous_close = _safe_float(history["Close"].dropna().iloc[-2])

    day_change_pct: float | None = None
    if price is not None and previous_close not in (None, 0):
        day_change_pct = ((price - previous_close) / previous_close) * 100

    return _json_result(
        {
            "symbol": clean_symbol,
            "price": round(price, 4) if price is not None else None,
            "currency": currency,
            "previous_close": round(previous_close, 4) if previous_close is not None else None,
            "day_change_pct": round(day_change_pct, 4) if day_change_pct is not None else None,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
    )


@tool
def get_technical_analysis(symbol: str, period: str = "1y") -> str:
    """Calcula análisis técnico básico (SMA, RSI, MACD, volatilidad) de una acción o cripto."""
    clean_symbol = symbol.upper().strip()
    ticker = yf.Ticker(clean_symbol)
    history = ticker.history(period=period, interval="1d")

    if history.empty or "Close" not in history:
        return _json_result(
            {"symbol": clean_symbol, "error": "No hay histórico suficiente para análisis técnico."}
        )

    if "High" not in history or "Low" not in history:
        return _json_result(
            {"symbol": clean_symbol, "error": "No hay columnas High/Low para indicadores avanzados."}
        )

    close = history["Close"].dropna()
    high = history["High"].dropna()
    low = history["Low"].dropna()
    if len(close) < 35:
        return _json_result(
            {"symbol": clean_symbol, "error": "Datos insuficientes para calcular indicadores robustos."}
        )

    sma_20 = _safe_float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else None
    sma_50 = _safe_float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
    sma_200 = _safe_float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
    rsi_14 = _get_rsi(close, period=14)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    last_close = _safe_float(close.iloc[-1])
    last_macd = _safe_float(macd_line.iloc[-1])
    last_macd_signal = _safe_float(macd_signal.iloc[-1])
    last_macd_hist = _safe_float(macd_hist.iloc[-1])

    returns = close.pct_change().dropna()
    annualized_vol = _safe_float(returns.tail(30).std() * (252**0.5) * 100) if len(returns) >= 2 else None

    trend = "neutral"
    if last_close and sma_20 and sma_50 and last_macd is not None and last_macd_signal is not None:
        if last_close > sma_20 > sma_50 and last_macd > last_macd_signal:
            trend = "alcista"
        elif last_close < sma_20 < sma_50 and last_macd < last_macd_signal:
            trend = "bajista"

    momentum = "neutral"
    if rsi_14 is not None:
        if rsi_14 >= 70:
            momentum = "sobrecompra"
        elif rsi_14 <= 30:
            momentum = "sobreventa"

    support_20 = _safe_float(close.tail(20).min()) if len(close) >= 20 else None
    resistance_20 = _safe_float(close.tail(20).max()) if len(close) >= 20 else None

    atr_14 = _get_atr(high, low, close, period=14)
    adx_14, plus_di_14, minus_di_14 = _get_adx(high, low, close, period=14)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + (2 * bb_std)
    bb_lower = bb_mid - (2 * bb_std)
    last_bb_upper = _safe_float(bb_upper.iloc[-1]) if len(bb_upper) else None
    last_bb_lower = _safe_float(bb_lower.iloc[-1]) if len(bb_lower) else None
    last_bb_mid = _safe_float(bb_mid.iloc[-1]) if len(bb_mid) else None

    bb_width_pct: float | None = None
    if last_bb_upper is not None and last_bb_lower is not None and last_bb_mid not in (None, 0):
        bb_width_pct = ((last_bb_upper - last_bb_lower) / last_bb_mid) * 100

    bb_position_pct: float | None = None
    if (
        last_close is not None
        and last_bb_upper is not None
        and last_bb_lower is not None
        and (last_bb_upper - last_bb_lower) != 0
    ):
        bb_position_pct = ((last_close - last_bb_lower) / (last_bb_upper - last_bb_lower)) * 100

    trend_strength = "neutral"
    if adx_14 is not None:
        if adx_14 >= 25:
            trend_strength = "fuerte"
        elif adx_14 <= 18:
            trend_strength = "débil"

    risk_level = "medio"
    if annualized_vol is not None:
        if annualized_vol >= 60:
            risk_level = "alto"
        elif annualized_vol <= 25:
            risk_level = "bajo"

    return _json_result(
        {
            "symbol": clean_symbol,
            "period": period,
            "last_close": round(last_close, 4) if last_close is not None else None,
            "sma_20": round(sma_20, 4) if sma_20 is not None else None,
            "sma_50": round(sma_50, 4) if sma_50 is not None else None,
            "sma_200": round(sma_200, 4) if sma_200 is not None else None,
            "rsi_14": round(rsi_14, 4) if rsi_14 is not None else None,
            "macd": round(last_macd, 6) if last_macd is not None else None,
            "macd_signal": round(last_macd_signal, 6) if last_macd_signal is not None else None,
            "macd_histogram": round(last_macd_hist, 6) if last_macd_hist is not None else None,
            "annualized_volatility_pct": round(annualized_vol, 4) if annualized_vol is not None else None,
            "atr_14": round(atr_14, 4) if atr_14 is not None else None,
            "adx_14": round(adx_14, 4) if adx_14 is not None else None,
            "plus_di_14": round(plus_di_14, 4) if plus_di_14 is not None else None,
            "minus_di_14": round(minus_di_14, 4) if minus_di_14 is not None else None,
            "bollinger_upper": round(last_bb_upper, 4) if last_bb_upper is not None else None,
            "bollinger_lower": round(last_bb_lower, 4) if last_bb_lower is not None else None,
            "bollinger_mid": round(last_bb_mid, 4) if last_bb_mid is not None else None,
            "bollinger_width_pct": round(bb_width_pct, 4) if bb_width_pct is not None else None,
            "bollinger_position_pct": round(bb_position_pct, 4) if bb_position_pct is not None else None,
            "support_20": round(support_20, 4) if support_20 is not None else None,
            "resistance_20": round(resistance_20, 4) if resistance_20 is not None else None,
            "trend_signal": trend,
            "trend_strength": trend_strength,
            "momentum_signal": momentum,
            "risk_level": risk_level,
        }
    )


@tool
def get_recent_news_tavily(query: str, max_results: int = 5, days: int = 7) -> str:
    """Busca noticias recientes con Tavily para una empresa o activo financiero."""
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return _json_result(
            {"query": query, "error": "TAVILY_API_KEY no configurada. Añádela en la barra lateral."}
        )
    if TavilyClient is None:
        return _json_result(
            {
                "query": query,
                "error": "Falta dependencia tavily-python. Ejecuta: pip install -r requirements.txt",
            }
        )

    safe_days = max(1, min(int(days), 30))
    safe_results = max(1, min(int(max_results), 8))
    full_query = f"{query} stock or crypto market moving news, earnings, regulation, risk factors"

    client = TavilyClient(api_key=api_key)
    try:
        try:
            response = client.search(
                query=full_query,
                topic="news",
                search_depth="advanced",
                max_results=safe_results,
                days=safe_days,
                include_answer=False,
                include_raw_content=False,
            )
        except TypeError:
            response = client.search(
                query=full_query,
                topic="news",
                search_depth="advanced",
                max_results=safe_results,
                time_range=_days_to_time_range(safe_days),
                include_answer=False,
                include_raw_content=False,
            )
    except Exception as exc:
        return _json_result({"query": query, "error": f"Error consultando Tavily: {exc}"})

    normalized_results: list[dict[str, Any]] = []
    sentiment_count = {"positivo": 0, "neutral": 0, "negativo": 0}
    for item in response.get("results", [])[:safe_results]:
        snippet = item.get("content") or ""
        sentiment = _simple_sentiment(f"{item.get('title') or ''} {snippet}")
        sentiment_count[sentiment] = sentiment_count.get(sentiment, 0) + 1
        normalized_results.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "published_date": item.get("published_date"),
                "snippet": snippet,
                "source": item.get("source"),
                "sentiment": sentiment,
            }
        )

    return _json_result(
        {
            "query": query,
            "lookback_days": safe_days,
            "total_results": len(normalized_results),
            "sentiment_summary": sentiment_count,
            "results": normalized_results,
        }
    )


@tool
def get_fundamental_snapshot(symbol: str) -> str:
    """Obtiene un snapshot fundamental básico para acciones (si está disponible en yfinance)."""
    clean_symbol = symbol.upper().strip()
    ticker = yf.Ticker(clean_symbol)

    try:
        info = ticker.info or {}
    except Exception as exc:
        return _json_result(
            {"symbol": clean_symbol, "error": f"No se pudieron obtener fundamentales: {exc}"}
        )

    quote_type = str(info.get("quoteType", "N/A"))
    is_equity = quote_type.lower() in {"equity", "etf", "fund"}

    payload = {
        "symbol": clean_symbol,
        "quote_type": quote_type,
        "company_name": info.get("longName") or info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": _safe_float(info.get("marketCap")),
        "enterprise_value": _safe_float(info.get("enterpriseValue")),
        "trailing_pe": _safe_float(info.get("trailingPE")),
        "forward_pe": _safe_float(info.get("forwardPE")),
        "price_to_book": _safe_float(info.get("priceToBook")),
        "ev_to_ebitda": _safe_float(info.get("enterpriseToEbitda")),
        "profit_margins": _safe_float(info.get("profitMargins")),
        "operating_margins": _safe_float(info.get("operatingMargins")),
        "return_on_equity": _safe_float(info.get("returnOnEquity")),
        "return_on_assets": _safe_float(info.get("returnOnAssets")),
        "debt_to_equity": _safe_float(info.get("debtToEquity")),
        "current_ratio": _safe_float(info.get("currentRatio")),
        "quick_ratio": _safe_float(info.get("quickRatio")),
        "revenue_growth": _safe_float(info.get("revenueGrowth")),
        "earnings_growth": _safe_float(info.get("earningsGrowth")),
        "free_cashflow": _safe_float(info.get("freeCashflow")),
        "beta": _safe_float(info.get("beta")),
        "dividend_yield": _safe_float(info.get("dividendYield")),
        "is_equity_like": is_equity,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if payload["company_name"] is None and payload["market_cap"] is None:
        payload["warning"] = "Datos fundamentales limitados para este activo (frecuente en criptos)."

    return _json_result(payload)


def _to_iso_or_str(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


@tool
def get_earnings_calendar(symbol: str, limit: int = 4) -> str:
    """Obtiene calendario de resultados/earnings (próximos y recientes) cuando esté disponible."""
    clean_symbol = symbol.upper().strip()
    ticker = yf.Ticker(clean_symbol)

    safe_limit = max(1, min(int(limit), 8))
    payload: dict[str, Any] = {
        "symbol": clean_symbol,
        "next_earnings_date": None,
        "earnings_window_start": None,
        "earnings_window_end": None,
        "calendar_events": {},
        "earnings_dates_rows": [],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    ts_single = info.get("earningsTimestamp")
    ts_start = info.get("earningsTimestampStart")
    ts_end = info.get("earningsTimestampEnd")
    try:
        if ts_single:
            payload["next_earnings_date"] = datetime.fromtimestamp(int(ts_single), tz=timezone.utc).isoformat()
        if ts_start:
            payload["earnings_window_start"] = datetime.fromtimestamp(int(ts_start), tz=timezone.utc).isoformat()
        if ts_end:
            payload["earnings_window_end"] = datetime.fromtimestamp(int(ts_end), tz=timezone.utc).isoformat()
    except Exception:
        pass

    try:
        calendar = ticker.calendar
        if getattr(calendar, "empty", True) is False:
            for idx, row in calendar.iterrows():
                key = str(idx)
                row_values = []
                for value in row.tolist():
                    row_values.append(_to_iso_or_str(value))
                payload["calendar_events"][key] = row_values[0] if len(row_values) == 1 else row_values
    except Exception:
        pass

    try:
        earnings_dates = ticker.earnings_dates
        if getattr(earnings_dates, "empty", True) is False:
            trimmed = earnings_dates.head(safe_limit)
            for idx, row in trimmed.iterrows():
                row_payload = {"date": _to_iso_or_str(idx)}
                for col in trimmed.columns:
                    value = row[col]
                    numeric = _safe_float(value)
                    row_payload[str(col)] = numeric if numeric is not None else _to_iso_or_str(value)
                payload["earnings_dates_rows"].append(row_payload)
    except Exception:
        pass

    if (
        payload["next_earnings_date"] is None
        and not payload["calendar_events"]
        and len(payload["earnings_dates_rows"]) == 0
    ):
        payload["warning"] = "No hay calendario de resultados disponible para este activo."

    return _json_result(payload)


@tool
def calculate_savings(initial_amount: float, annual_interest_rate: float, years: float) -> str:
    """Calcula ahorro con interés compuesto anual."""
    if initial_amount < 0:
        return "Error: el monto inicial no puede ser negativo."
    if years < 0:
        return "Error: los años no pueden ser negativos."

    final_amount = initial_amount * (1 + annual_interest_rate / 100) ** years
    earned_interest = final_amount - initial_amount

    return _json_result(
        {
            "initial_amount": round(initial_amount, 2),
            "annual_interest_rate": round(annual_interest_rate, 4),
            "years": round(years, 4),
            "final_amount": round(final_amount, 2),
            "earned_interest": round(earned_interest, 2),
        }
    )


TOOLS = [
    get_market_price,
    get_technical_analysis,
    get_recent_news_tavily,
    get_fundamental_snapshot,
    get_earnings_calendar,
    calculate_savings,
]
TOOLS_BY_NAME = {tool_item.name: tool_item for tool_item in TOOLS}


class FinanceAssistant:
    """Orquestador solo con LangChain (sin LangGraph)."""

    def __init__(self, google_api_key: str, tavily_api_key: str):
        os.environ["GOOGLE_API_KEY"] = google_api_key
        os.environ["TAVILY_API_KEY"] = tavily_api_key
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
        self.analyst_llm = self.llm.bind_tools(TOOLS)

        self.analyst_prompt = (
            "Eres el Analista en un sistema de finanzas personales.\n"
            "Reglas:\n"
            "1) Si la consulta es de acciones/criptos/activo, llama get_market_price + "
            "get_technical_analysis + get_recent_news_tavily + get_fundamental_snapshot + get_earnings_calendar.\n"
            "2) Para noticias usa query con ticker o nombre y days=7, max_results=5.\n"
            "3) Para earnings usa limit=4 por defecto.\n"
            "4) Si la consulta es de ahorro/interés compuesto, llama calculate_savings.\n"
            "5) No respondas al usuario aquí: solo tool calls."
        )

        self.advisor_prompt = (
            "Eres un asesor financiero educativo y preventivo.\n"
            "Combina análisis técnico + noticias recientes (si existen).\n"
            "Incorpora fundamentales y calendario de resultados cuando estén disponibles.\n"
            "Usa el confidence_report y data_validator para modular la prudencia del consejo.\n"
            "Formato:\n"
            "1) Diagnóstico breve.\n"
            "2) Lectura técnica (tendencia, momentum, volatilidad, riesgo).\n"
            "3) Lectura fundamental (valoración, crecimiento, deuda, márgenes).\n"
            "4) Lectura de calendario de resultados/catalizadores temporales.\n"
            "5) Lectura de noticias con 2-4 fuentes (URL) y su scoring LLM.\n"
            "6) Nivel de confianza del análisis (score y etiqueta).\n"
            "7) Plan prudente con 2 acciones prácticas.\n"
            "8) Aviso de riesgo: no es asesoramiento financiero profesional.\n"
            "Si el usuario compartió su nombre antes, úsalo de forma natural."
        )

    def _build_asset_tool_calls(self, hint: str) -> list[dict[str, Any]]:
        return [
            {"name": "get_market_price", "args": {"symbol": hint}, "id": f"auto_price_{uuid.uuid4().hex[:8]}"},
            {"name": "get_technical_analysis", "args": {"symbol": hint}, "id": f"auto_ta_{uuid.uuid4().hex[:8]}"},
            {
                "name": "get_recent_news_tavily",
                "args": {"query": hint, "days": 7, "max_results": 5},
                "id": f"auto_news_{uuid.uuid4().hex[:8]}",
            },
            {"name": "get_fundamental_snapshot", "args": {"symbol": hint}, "id": f"auto_fund_{uuid.uuid4().hex[:8]}"},
            {"name": "get_earnings_calendar", "args": {"symbol": hint, "limit": 4}, "id": f"auto_cal_{uuid.uuid4().hex[:8]}"},
        ]

    def _ensure_tool_calls(
        self,
        user_query: str,
        history_messages: list[AnyMessage],
        tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        required_asset_tools = {
            "get_market_price",
            "get_technical_analysis",
            "get_recent_news_tavily",
            "get_fundamental_snapshot",
            "get_earnings_calendar",
        }

        if not tool_calls:
            if _looks_like_asset_query(user_query):
                hint = _extract_symbol_hint(user_query) or user_query
                return self._build_asset_tool_calls(hint.strip())
            return []

        called_names = {call["name"] for call in tool_calls}
        is_asset_analysis = len(called_names.intersection(required_asset_tools)) > 0
        if is_asset_analysis and "calculate_savings" not in called_names:
            hint = None
            for call in tool_calls:
                args = call.get("args", {})
                if isinstance(args, dict):
                    if args.get("symbol"):
                        hint = str(args["symbol"]).strip()
                        break
                    if args.get("query"):
                        hint = str(args["query"]).strip()
                        break
            if hint is None:
                hint = _extract_symbol_hint(user_query)
            if hint is None:
                for message in reversed(history_messages):
                    if isinstance(message, HumanMessage):
                        hint = _extract_symbol_hint(_message_to_text(message))
                        if hint:
                            break
            if hint is None:
                hint = "AAPL"

            for missing_tool in sorted(required_asset_tools - called_names):
                if missing_tool == "get_recent_news_tavily":
                    args = {"query": hint, "days": 7, "max_results": 5}
                elif missing_tool == "get_earnings_calendar":
                    args = {"symbol": hint, "limit": 4}
                else:
                    args = {"symbol": hint}
                tool_calls.append(
                    {"name": missing_tool, "args": args, "id": f"auto_{missing_tool}_{uuid.uuid4().hex[:8]}"}
                )

        return tool_calls

    def _execute_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_call_id: str,
    ) -> tuple[ToolMessage, dict[str, Any]]:
        tool_impl = TOOLS_BY_NAME.get(tool_name)
        if tool_impl is None:
            raw_result = f"Herramienta desconocida: {tool_name}"
        else:
            try:
                raw_result = tool_impl.invoke(tool_args)
            except Exception as exc:
                raw_result = f"Error ejecutando {tool_name}: {exc}"

        result_text = raw_result if isinstance(raw_result, str) else json.dumps(raw_result, ensure_ascii=False)
        parsed_result = _safe_json_loads(result_text)
        has_error = bool(parsed_result.get("error")) or result_text.lower().startswith("error")

        tool_entry = {
            "args": tool_args,
            "result": result_text,
            "parsed_result": parsed_result,
            "has_error": has_error,
            "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        tool_message = ToolMessage(content=result_text, tool_call_id=tool_call_id, name=tool_name)
        return tool_message, tool_entry

    def _run_tool_calls(self, tool_calls: list[dict[str, Any]]) -> tuple[list[ToolMessage], dict[str, Any]]:
        tool_messages: list[ToolMessage] = []
        stock_data: dict[str, Any] = {}

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            tool_message, tool_entry = self._execute_tool_call(
                tool_name=tool_name,
                tool_args=tool_args,
                tool_call_id=tool_call.get("id", f"manual_{uuid.uuid4().hex[:8]}"),
            )
            stock_data[tool_name] = tool_entry
            tool_messages.append(tool_message)

        return tool_messages, stock_data

    def _resolve_asset_hint(
        self,
        user_query: str,
        history_messages: list[AnyMessage],
        stock_data: dict[str, Any],
    ) -> str:
        hint = _extract_symbol_hint(user_query)
        if hint:
            return hint

        for tool_name in (
            "get_market_price",
            "get_technical_analysis",
            "get_recent_news_tavily",
            "get_fundamental_snapshot",
            "get_earnings_calendar",
        ):
            entry = stock_data.get(tool_name, {})
            args = entry.get("args", {})
            if isinstance(args, dict):
                if args.get("symbol"):
                    return str(args["symbol"]).strip()
                if args.get("query"):
                    return str(args["query"]).strip()

        for message in reversed(history_messages):
            if isinstance(message, HumanMessage):
                guess = _extract_symbol_hint(_message_to_text(message))
                if guess:
                    return guess

        return "AAPL"

    def _validate_asset_data(
        self,
        user_query: str,
        history_messages: list[AnyMessage],
        stock_data: dict[str, Any],
    ) -> tuple[list[ToolMessage], dict[str, Any]]:
        required_tools = [
            "get_market_price",
            "get_technical_analysis",
            "get_recent_news_tavily",
            "get_fundamental_snapshot",
            "get_earnings_calendar",
        ]
        is_asset_query = _looks_like_asset_query(user_query) or any(tool_name in stock_data for tool_name in required_tools)
        if not is_asset_query:
            stock_data["data_validator"] = {
                "status": "skipped",
                "reason": "Consulta no relacionada con activos.",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            return [], stock_data

        missing_tools = [tool_name for tool_name in required_tools if tool_name not in stock_data]
        error_tools = [tool_name for tool_name in required_tools if stock_data.get(tool_name, {}).get("has_error")]
        repair_candidates = sorted(set(missing_tools + error_tools))

        repair_messages: list[ToolMessage] = []
        repair_attempts: list[dict[str, Any]] = []
        if repair_candidates:
            hint = self._resolve_asset_hint(user_query=user_query, history_messages=history_messages, stock_data=stock_data)
            for tool_name in repair_candidates:
                if tool_name == "get_recent_news_tavily":
                    repair_args = {"query": hint, "days": 14, "max_results": 5}
                elif tool_name == "get_technical_analysis":
                    repair_args = {"symbol": hint, "period": "2y"}
                elif tool_name == "get_earnings_calendar":
                    repair_args = {"symbol": hint, "limit": 6}
                else:
                    repair_args = {"symbol": hint}

                repair_message, repair_entry = self._execute_tool_call(
                    tool_name=tool_name,
                    tool_args=repair_args,
                    tool_call_id=f"repair_{tool_name}_{uuid.uuid4().hex[:8]}",
                )
                stock_data[tool_name] = repair_entry
                repair_messages.append(repair_message)
                repair_attempts.append(
                    {
                        "tool": tool_name,
                        "args": repair_args,
                        "status": "ok" if not repair_entry["has_error"] else "failed",
                    }
                )

        missing_tools = [tool_name for tool_name in required_tools if tool_name not in stock_data]
        error_tools = [tool_name for tool_name in required_tools if stock_data.get(tool_name, {}).get("has_error")]
        ok_count = len(required_tools) - len(missing_tools) - len(error_tools)
        quality_score = max(0, min(100, int(round((ok_count / len(required_tools)) * 100))))

        stock_data["data_validator"] = {
            "status": "completed",
            "required_tools": required_tools,
            "missing_tools": missing_tools,
            "error_tools": error_tools,
            "repair_attempts": repair_attempts,
            "is_complete": len(missing_tools) == 0 and len(error_tools) == 0,
            "quality_score": quality_score,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        return repair_messages, stock_data

    def _score_news_with_llm(self, user_query: str, stock_data: dict[str, Any]) -> dict[str, Any]:
        news_entry = stock_data.get("get_recent_news_tavily", {})
        news_data = news_entry.get("parsed_result", {}) if isinstance(news_entry, dict) else {}
        if not isinstance(news_data, dict) or news_data.get("error"):
            stock_data["news_llm_scoring"] = {
                "status": "skipped",
                "reason": "No hay noticias válidas para puntuar.",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            return stock_data

        raw_results = news_data.get("results", [])
        if not isinstance(raw_results, list) or len(raw_results) == 0:
            stock_data["news_llm_scoring"] = {
                "status": "skipped",
                "reason": "La búsqueda no devolvió noticias.",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            return stock_data

        compact_items: list[dict[str, Any]] = []
        for item in raw_results[:5]:
            if not isinstance(item, dict):
                continue
            compact_items.append(
                {
                    "title": str(item.get("title", ""))[:240],
                    "source": item.get("source"),
                    "url": item.get("url"),
                    "published_date": item.get("published_date"),
                    "snippet": str(item.get("snippet", ""))[:420],
                }
            )

        if len(compact_items) == 0:
            stock_data["news_llm_scoring"] = {
                "status": "skipped",
                "reason": "No hay items de noticia utilizables.",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            return stock_data

        scoring_system_prompt = (
            "Eres un analista financiero de noticias.\n"
            "Evalúa cada noticia para el activo consultado.\n"
            "Devuelve SOLO JSON válido sin texto adicional.\n"
            "Schema exacto:\n"
            "{\n"
            '  "items": [\n'
            "    {\n"
            '      "url": "string",\n'
            '      "sentiment": "positivo|neutral|negativo",\n'
            '      "relevance_score": 0,\n'
            '      "impact_score": 0,\n'
            '      "horizon": "corto|medio|largo",\n'
            '      "reason": "string"\n'
            "    }\n"
            "  ],\n"
            '  "summary": {\n'
            '    "overall_sentiment": "positivo|neutral|negativo",\n'
            '    "overall_relevance": 0,\n'
            '    "overall_impact": 0,\n'
            '    "risk_flags": ["string"],\n'
            '    "confidence": 0\n'
            "  }\n"
            "}\n"
            "Todos los scores entre 0 y 100."
        )

        scoring_user_payload = {
            "user_query": user_query,
            "news_items": compact_items,
        }

        try:
            scoring_response = self.llm.invoke(
                [
                    SystemMessage(content=scoring_system_prompt),
                    HumanMessage(content=json.dumps(scoring_user_payload, ensure_ascii=False)),
                ]
            )
            scoring_text = _message_to_text(scoring_response)
            parsed = _safe_json_loads(_extract_json_text(scoring_text))
            if "summary" not in parsed:
                raise ValueError("Respuesta LLM sin campo summary")
            stock_data["news_llm_scoring"] = {
                "status": "completed",
                "data": parsed,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            stock_data["news_llm_scoring"] = {
                "status": "error",
                "error": f"Falló scoring LLM de noticias: {exc}",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }

        return stock_data

    def _compute_confidence_report(self, user_query: str, stock_data: dict[str, Any]) -> dict[str, Any]:
        score = 0
        factors: list[str] = []

        validator = stock_data.get("data_validator", {})
        quality_score = int(validator.get("quality_score", 0)) if isinstance(validator, dict) else 0
        score += int(round(quality_score * 0.35))
        factors.append(f"Calidad de datos: {quality_score}/100.")

        if _looks_like_asset_query(user_query):
            price_data = stock_data.get("get_market_price", {}).get("parsed_result", {})
            tech_data = stock_data.get("get_technical_analysis", {}).get("parsed_result", {})
            news_data = stock_data.get("get_recent_news_tavily", {}).get("parsed_result", {})
            fundamentals = stock_data.get("get_fundamental_snapshot", {}).get("parsed_result", {})
            earnings = stock_data.get("get_earnings_calendar", {}).get("parsed_result", {})

            if isinstance(price_data, dict) and not price_data.get("error") and price_data.get("price") is not None:
                score += 10
                factors.append("Precio de mercado disponible.")

            if isinstance(tech_data, dict) and not tech_data.get("error"):
                trend_signal = tech_data.get("trend_signal")
                adx_14 = _safe_float(tech_data.get("adx_14"))
                atr_14 = _safe_float(tech_data.get("atr_14"))
                bb_width = _safe_float(tech_data.get("bollinger_width_pct"))

                score += 10
                if trend_signal in {"alcista", "bajista"}:
                    score += 8
                    factors.append(f"Tendencia definida: {trend_signal}.")
                if adx_14 is not None:
                    if adx_14 >= 25:
                        score += 8
                        factors.append(f"Fuerza de tendencia alta (ADX {adx_14:.2f}).")
                    elif adx_14 >= 20:
                        score += 5
                        factors.append(f"Fuerza de tendencia moderada (ADX {adx_14:.2f}).")
                if atr_14 is not None:
                    score += 4
                if bb_width is not None:
                    score += 4

            if isinstance(news_data, dict) and not news_data.get("error"):
                total_results = int(news_data.get("total_results", 0) or 0)
                if total_results >= 4:
                    score += 12
                elif total_results >= 2:
                    score += 8
                elif total_results > 0:
                    score += 4
                sentiment_summary = news_data.get("sentiment_summary", {})
                if isinstance(sentiment_summary, dict):
                    counts = [int(sentiment_summary.get(key, 0) or 0) for key in ("positivo", "neutral", "negativo")]
                    total_counts = sum(counts)
                    if total_counts > 0:
                        dominant_share = max(counts) / total_counts
                        if dominant_share >= 0.6:
                            score += 5
                        else:
                            score += 3
                factors.append(f"Noticias útiles: {total_results}.")

                llm_scoring = stock_data.get("news_llm_scoring", {})
                if isinstance(llm_scoring, dict) and llm_scoring.get("status") == "completed":
                    summary = llm_scoring.get("data", {}).get("summary", {})
                    relevance = _safe_float(summary.get("overall_relevance")) if isinstance(summary, dict) else None
                    impact = _safe_float(summary.get("overall_impact")) if isinstance(summary, dict) else None
                    llm_conf = _safe_float(summary.get("confidence")) if isinstance(summary, dict) else None
                    if relevance is not None:
                        score += int(round(min(100.0, relevance) * 0.06))
                    if impact is not None:
                        score += int(round(min(100.0, impact) * 0.05))
                    if llm_conf is not None:
                        score += int(round(min(100.0, llm_conf) * 0.06))
                        factors.append(f"Scoring LLM de noticias con confianza {llm_conf:.1f}/100.")

            if isinstance(fundamentals, dict) and not fundamentals.get("error"):
                core_fields = [
                    fundamentals.get("market_cap"),
                    fundamentals.get("trailing_pe"),
                    fundamentals.get("forward_pe"),
                    fundamentals.get("debt_to_equity"),
                    fundamentals.get("revenue_growth"),
                    fundamentals.get("earnings_growth"),
                ]
                available = sum(1 for value in core_fields if value is not None)
                score += min(12, available * 2)
                factors.append(f"Cobertura fundamental: {available}/{len(core_fields)} métricas clave.")

            if isinstance(earnings, dict) and not earnings.get("error"):
                has_calendar = bool(earnings.get("next_earnings_date")) or bool(earnings.get("calendar_events"))
                has_rows = len(earnings.get("earnings_dates_rows", []) or []) > 0
                if has_calendar:
                    score += 8
                if has_rows:
                    score += 5
                if has_calendar or has_rows:
                    factors.append("Calendario de resultados disponible.")
        else:
            savings_data = stock_data.get("calculate_savings", {}).get("parsed_result", {})
            if isinstance(savings_data, dict) and not savings_data.get("error"):
                score += 70
                factors.append("Cálculo determinista de ahorro completado.")

        score = max(0, min(100, score))
        if score >= 75:
            label = "alta"
        elif score >= 50:
            label = "media"
        else:
            label = "baja"

        return {
            "score": score,
            "label": label,
            "factors": factors,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

    def invoke(self, user_query: str, history_messages: list[AnyMessage]) -> dict[str, Any]:
        analyst_messages = [SystemMessage(content=self.analyst_prompt), *history_messages, HumanMessage(user_query)]
        analyst_response = self.analyst_llm.invoke(analyst_messages)
        tool_calls = self._ensure_tool_calls(user_query, history_messages, list(analyst_response.tool_calls or []))
        tool_messages, stock_data = self._run_tool_calls(tool_calls)
        repair_messages, stock_data = self._validate_asset_data(
            user_query=user_query,
            history_messages=history_messages,
            stock_data=stock_data,
        )
        tool_messages.extend(repair_messages)
        stock_data = self._score_news_with_llm(user_query=user_query, stock_data=stock_data)
        stock_data["confidence_report"] = self._compute_confidence_report(user_query=user_query, stock_data=stock_data)

        advisor_context = json.dumps(stock_data, ensure_ascii=False)
        advisor_messages: list[AnyMessage] = [
            SystemMessage(content=f"{self.advisor_prompt}\nDatos técnicos y noticias: {advisor_context}"),
            *history_messages,
            HumanMessage(user_query),
            analyst_response,
            *tool_messages,
        ]
        advisor_response = self.llm.invoke(advisor_messages)

        return {
            "assistant_text": _message_to_text(advisor_response),
            "stock_data": stock_data,
        }


@st.cache_resource(show_spinner=False)
def get_assistant(google_api_key: str, tavily_api_key: str) -> FinanceAssistant:
    return FinanceAssistant(google_api_key=google_api_key, tavily_api_key=tavily_api_key)


def apply_custom_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #2b2d39;
            color: #f3f4f6;
        }
        h1, h2, h3, h4, h5, h6, p, label, span, div {
            color: #f3f4f6;
        }
        .stButton > button {
            background-color: #73c59b;
            color: #1f2937;
            border: none;
            border-radius: 8px;
        }
        .stButton > button:hover {
            background-color: #61b187;
            color: #111827;
        }
        .stTextInput > div > div > input {
            border: 1px solid #73c59b;
        }
        .stAlert {
            border-color: #73c59b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_session_state() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "lc_histories" not in st.session_state:
        st.session_state.lc_histories = {}
    if st.session_state.thread_id not in st.session_state.lc_histories:
        st.session_state.lc_histories[st.session_state.thread_id] = InMemoryChatMessageHistory()


def main() -> None:
    st.set_page_config(page_title="Asistente de Finanzas Personales", page_icon="💹", layout="centered")
    apply_custom_theme()

    st.title("Asistente de Finanzas Personales")
    st.caption("Arquitectura: LangChain + Gemini 2.5 Flash + yfinance + Tavily")

    with st.sidebar:
        st.subheader("Configuración")
        google_api_key = st.text_input(
            "GOOGLE_API_KEY",
            type="password",
            help="Clave de Gemini (Google AI Studio).",
        )
        tavily_api_key = st.text_input(
            "TAVILY_API_KEY",
            type="password",
            help="Clave para búsqueda de noticias recientes.",
        )
        if st.button("Nueva conversación"):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.chat_history = []
            st.rerun()

    if not google_api_key:
        st.info("Introduce tu GOOGLE_API_KEY en la barra lateral para comenzar.")
        st.stop()

    if not tavily_api_key:
        st.warning(
            "Falta TAVILY_API_KEY. El análisis de noticias no estará disponible para consultas de activos."
        )

    _init_session_state()
    assistant = get_assistant(google_api_key, tavily_api_key)
    history = st.session_state.lc_histories[st.session_state.thread_id]

    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])

    user_input = st.chat_input(
        "Ejemplo: 'Analiza AAPL con técnico + noticias' o '¿Cómo puedo ahorrar $5000 en 3 años con un interés del 4% anual?'"
    )
    if not user_input:
        return

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    result = assistant.invoke(user_query=user_input, history_messages=history.messages)
    assistant_text = result["assistant_text"]

    history.add_user_message(user_input)
    history.add_ai_message(assistant_text)

    st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})
    with st.chat_message("assistant"):
        st.markdown(assistant_text)

    stock_data = result.get("stock_data")
    if stock_data:
        confidence_report = stock_data.get("confidence_report", {})
        if isinstance(confidence_report, dict) and confidence_report.get("score") is not None:
            st.caption(
                f"Confianza del análisis: {confidence_report.get('score')}/100 "
                f"({confidence_report.get('label', 'n/a')})."
            )
        with st.expander("Ver datos técnicos y noticias extraídas"):
            st.json(stock_data)


if __name__ == "__main__":
    main()
