"""
Market router — handles all /market endpoints.
Fetches live stock, ETF, forex, and news data.
"""
from fastapi import APIRouter
from pydantic import BaseModel
from backend.tools.market_data import (
    get_stock_price,
    get_forex_rate,
    get_etf_price,
    get_financial_news
)

router = APIRouter()


# ── Request models ───────────────────────────────────────
class StockRequest(BaseModel):
    symbol: str

class ForexRequest(BaseModel):
    from_currency: str
    to_currency: str

class NewsRequest(BaseModel):
    country: str = "general"


# ── Endpoints ────────────────────────────────────────────
@router.post("/stock")
def stock_price(request: StockRequest):
    """Get live stock price for a given symbol."""
    result = get_stock_price.invoke({"symbol": request.symbol})
    return {"data": result}


@router.post("/forex")
def forex_rate(request: ForexRequest):
    """Get live exchange rate between two currencies."""
    result = get_forex_rate.invoke({
        "from_currency": request.from_currency,
        "to_currency": request.to_currency
    })
    return {"data": result}


@router.post("/etf")
def etf_price(request: StockRequest):
    """Get live ETF price for a given symbol."""
    result = get_etf_price.invoke({"symbol": request.symbol})
    return {"data": result}


@router.post("/news")
def financial_news(request: NewsRequest):
    """Get latest financial news for a given country."""
    result = get_financial_news.invoke({"country": request.country})
    return {"data": result}


@router.get("/health")
def market_health():
    return {"status": "market router ok"}