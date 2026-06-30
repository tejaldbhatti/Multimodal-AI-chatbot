"""
Market data tools using Alpha Vantage API.
Fetches live stock prices, ETF data, and forex rates.
"""
import requests
from langchain_core.tools import tool
from backend.config import ALPHA_VANTAGE_API_KEY, ALPHA_VANTAGE_BASE_URL


def _fetch_alpha_vantage(params: dict) -> dict:
    """
    Base function to call Alpha Vantage API.
    All tools use this internally.
    """
    params["apikey"] = ALPHA_VANTAGE_API_KEY
    response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


@tool
def get_stock_price(symbol: str) -> str:
    """
    Fetches the latest stock price for a given symbol.
    Use this when user asks about a specific stock price.

    Args:
        symbol: stock ticker symbol e.g. AAPL, MSFT, GOOGL, SAP (Germany)
    """
    try:
        data = _fetch_alpha_vantage({
            "function": "GLOBAL_QUOTE",
            "symbol": symbol.upper()
        })

        quote = data.get("Global Quote", {})
        if not quote:
            return f"Could not find stock data for '{symbol}'. Please check the symbol."

        price       = float(quote.get("05. price", 0))
        change      = float(quote.get("09. change", 0))
        change_pct  = quote.get("10. change percent", "0%")
        volume      = int(quote.get("06. volume", 0))
        prev_close  = float(quote.get("08. previous close", 0))
        high        = float(quote.get("03. high", 0))
        low         = float(quote.get("04. low", 0))

        direction = "📈" if change >= 0 else "📉"

        return (
            f"## {symbol.upper()} Stock Price\n\n"
            f"{direction} **Current Price:** ${price:,.2f}\n"
            f"• Change: ${change:+.2f} ({change_pct})\n"
            f"• Previous Close: ${prev_close:,.2f}\n"
            f"• Today's High: ${high:,.2f}\n"
            f"• Today's Low: ${low:,.2f}\n"
            f"• Volume: {volume:,}\n\n"
            f"⚠️ This is live market data for informational purposes only. "
            f"Not financial advice."
        )

    except Exception as e:
        return f"Error fetching stock data for '{symbol}': {str(e)}"


@tool
def get_forex_rate(from_currency: str, to_currency: str) -> str:
    """
    Fetches live exchange rate between two currencies.
    Use this when user asks about currency conversion or exchange rates.

    Args:
        from_currency: source currency code e.g. USD, EUR, INR, AUD
        to_currency: target currency code e.g. EUR, USD, GBP
    """
    try:
        data = _fetch_alpha_vantage({
            "function": "CURRENCY_EXCHANGE_RATE",
            "from_currency": from_currency.upper(),
            "to_currency": to_currency.upper()
        })

        rate_data = data.get("Realtime Currency Exchange Rate", {})
        if not rate_data:
            return f"Could not find exchange rate for {from_currency} to {to_currency}."

        rate        = float(rate_data.get("5. Exchange Rate", 0))
        last_update = rate_data.get("6. Last Refreshed", "unknown")
        from_name   = rate_data.get("2. From_Currency Name", from_currency)
        to_name     = rate_data.get("4. To_Currency Name", to_currency)

        return (
            f"## Exchange Rate: {from_currency.upper()} → {to_currency.upper()}\n\n"
            f"• **{from_name}** to **{to_name}**\n"
            f"• **Rate:** 1 {from_currency.upper()} = {rate:.4f} {to_currency.upper()}\n"
            f"• Last updated: {last_update}\n\n"
            f"⚠️ Live exchange rate for informational purposes only."
        )

    except Exception as e:
        return f"Error fetching exchange rate: {str(e)}"


@tool
def get_etf_price(symbol: str) -> str:
    """
    Fetches the latest price for an ETF.
    Use this when user asks about ETF prices or performance.

    Args:
        symbol: ETF ticker symbol e.g. SPY, QQQ, VWRL, MSCI
    """
    try:
        data = _fetch_alpha_vantage({
            "function": "GLOBAL_QUOTE",
            "symbol": symbol.upper()
        })

        quote = data.get("Global Quote", {})
        if not quote:
            return f"Could not find ETF data for '{symbol}'. Please check the symbol."

        price      = float(quote.get("05. price", 0))
        change     = float(quote.get("09. change", 0))
        change_pct = quote.get("10. change percent", "0%")
        high       = float(quote.get("03. high", 0))
        low        = float(quote.get("04. low", 0))

        direction = "📈" if change >= 0 else "📉"

        return (
            f"## {symbol.upper()} ETF Price\n\n"
            f"{direction} **Current Price:** ${price:,.2f}\n"
            f"• Change today: ${change:+.2f} ({change_pct})\n"
            f"• Today's High: ${high:,.2f}\n"
            f"• Today's Low: ${low:,.2f}\n\n"
            f"💡 Tip: For German investors, check your Freistellungsauftrag "
            f"(€1,000 free allowance) before investing in ETFs.\n\n"
            f"⚠️ Live market data for informational purposes only. Not financial advice."
        )

    except Exception as e:
        return f"Error fetching ETF data for '{symbol}': {str(e)}"


@tool
def get_financial_news(country: str = "general") -> str:
    """
    Fetches latest financial news headlines.
    Use when user asks about current market news or financial updates.

    Args:
        country: 'germany', 'usa', 'india', 'australia', or 'general'
    """
    try:
        import requests
        from backend.config import NEWS_API_KEY, NEWS_API_BASE_URL

        # Map country to news query
        country_queries = {
            "germany": "Germany finance tax investment",
            "usa": "USA finance tax investment federal reserve",
            "india": "India finance tax RBI investment",
            "australia": "Australia finance tax RBA investment",
            "general": "financial markets investment economy"
        }

        query = country_queries.get(country.lower(), country_queries["general"])

        response = requests.get(
            f"{NEWS_API_BASE_URL}/everything",
            params={
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 5,
                "apiKey": NEWS_API_KEY
            },
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        articles = data.get("articles", [])
        if not articles:
            return "No recent financial news found."

        result = f"## Latest Financial News — {country.upper()}\n\n"
        for i, article in enumerate(articles[:5], 1):
            title  = article.get("title", "No title")
            source = article.get("source", {}).get("name", "Unknown")
            url    = article.get("url", "")
            result += f"{i}. **{title}**\n   Source: {source}\n   {url}\n\n"

        return result

    except Exception as e:
        return f"Error fetching news: {str(e)}"