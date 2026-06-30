"""
Country Finance Advisor Tool.
Loads country-specific tax and investment rules
and formats them into a helpful response for the agent.
"""
from langchain_core.tools import tool
from backend.data.country_rules import COUNTRY_RULES


@tool
def get_country_financial_rules(country: str) -> str:
    """
    Returns financial rules, tax brackets, and investment
    options for a given country.
    Use this when user asks about tax or investment in their country.

    Args:
        country: one of 'germany', 'usa', 'india', 'australia'
    """
    country = country.lower().strip()

    if country not in COUNTRY_RULES:
        return (
            f"Sorry, I don't have rules for '{country}' yet. "
            f"Currently supported: Germany, USA, India, Australia."
        )

    rules = COUNTRY_RULES[country]
    currency = rules["currency"]

    # Build a readable summary
    response = f"## Financial Rules — {country.upper()}\n\n"
    response += f"**Summary:** {rules['summary']}\n\n"

    # Tax brackets
    brackets = rules.get("tax_brackets") or rules.get("tax_brackets_single") or rules.get("tax_brackets_new_regime")
    if brackets:
        response += "**Tax Brackets:**\n"
        for b in brackets:
            max_val = f"{b['max']:,}" if b['max'] else "above"
            response += f"  • {currency} {b['min']:,} – {max_val}: {int(b['rate']*100)}%\n"

    # Investment vehicles
    if rules.get("investment_vehicles"):
        response += "\n**Investment Options:**\n"
        for v in rules["investment_vehicles"]:
            response += f"  • {v}\n"

    # Retirement
    if rules.get("retirement"):
        ret = rules["retirement"]
        response += f"\n**Retirement Age:** {ret['age']}\n"
        response += "**Retirement Vehicles:** " + ", ".join(ret["vehicles"]) + "\n"

    # Freelancer info
    if rules.get("freelancer"):
        fl = rules["freelancer"]
        response += "\n**Freelancer Info:**\n"
        for k, v in fl.items():
            if not isinstance(v, dict):
                response += f"  • {k}: {v}\n"

    # Useful links
    if rules.get("useful_links"):
        response += "\n**Useful Links:**\n"
        for name, url in rules["useful_links"].items():
            response += f"  • {name}: {url}\n"

    return response


@tool
def get_tax_bracket(country: str, annual_income: float) -> str:
    """
    Calculates estimated income tax for a given country and income.
    Returns tax amount and effective tax rate.

    Args:
        country: one of 'germany', 'usa', 'india', 'australia'
        annual_income: annual income in local currency
    """
    country = country.lower().strip()

    if country not in COUNTRY_RULES:
        return f"Country '{country}' not supported yet."

    rules = COUNTRY_RULES[country]
    currency = rules["currency"]

    # Pick correct brackets
    brackets = (
        rules.get("tax_brackets") or
        rules.get("tax_brackets_single") or
        rules.get("tax_brackets_new_regime")
    )

    if not brackets:
        return "Tax bracket data not available for this country."

    # Calculate tax
    total_tax = 0.0
    breakdown = []

    for bracket in brackets:
        low = bracket["min"]
        high = bracket["max"] if bracket["max"] else float("inf")
        rate = bracket["rate"]

        if annual_income <= low:
            break

        taxable_in_bracket = min(annual_income, high) - low
        tax_in_bracket = taxable_in_bracket * rate

        if tax_in_bracket > 0:
            breakdown.append(
                f"  • {currency} {low:,}–{int(min(annual_income, high)):,} "
                f"@ {int(rate*100)}% = {currency} {tax_in_bracket:,.2f}"
            )
        total_tax += tax_in_bracket

    effective_rate = (total_tax / annual_income * 100) if annual_income > 0 else 0

    response = f"## Tax Estimate — {country.upper()}\n\n"
    response += f"**Annual Income:** {currency} {annual_income:,.2f}\n\n"
    response += "**Breakdown:**\n"
    response += "\n".join(breakdown)
    response += f"\n\n**Total Estimated Tax:** {currency} {total_tax:,.2f}"
    response += f"\n**Effective Tax Rate:** {effective_rate:.1f}%"
    response += f"\n**Take Home Pay:** {currency} {annual_income - total_tax:,.2f}"
    response += (
        "\n\n⚠️ Disclaimer: This is an estimate for educational purposes only. "
        "Please consult a licensed tax professional for your specific situation."
    )

    return response