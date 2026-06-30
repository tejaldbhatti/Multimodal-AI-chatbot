"""
Steuerklasse advisor for Germany.
Recommends the best tax class combination for couples.
"""
from langchain_core.tools import tool
from backend.tools.tax_employee import calculate_tax_from_brackets
from backend.data.country_rules import COUNTRY_RULES


@tool
def advise_steuerklasse(
    income_partner1: float,
    income_partner2: float,
) -> str:
    """
    Recommends the best Steuerklasse combination for a married
    couple in Germany based on their incomes.

    Args:
        income_partner1: annual gross income of partner 1
        income_partner2: annual gross income of partner 2
    """
    rules    = COUNTRY_RULES["germany"]
    currency = "EUR"
    brackets = rules["tax_brackets"]

    combined = income_partner1 + income_partner2

    # Determine higher and lower earner
    if income_partner1 >= income_partner2:
        higher = income_partner1
        lower  = income_partner2
        higher_label = "Partner 1"
        lower_label  = "Partner 2"
    else:
        higher = income_partner2
        lower  = income_partner1
        higher_label = "Partner 2"
        lower_label  = "Partner 1"

    income_ratio = lower / higher if higher > 0 else 0

    # ── Option 1: IV/IV ──────────────────────────────────
    tax_iv_higher = calculate_tax_from_brackets(brackets, higher)
    tax_iv_lower  = calculate_tax_from_brackets(brackets, lower)
    total_iv      = tax_iv_higher + tax_iv_lower

    # ── Option 2: Ehegattensplitting III/V ───────────────
    half_income   = combined / 2
    total_split   = calculate_tax_from_brackets(brackets, half_income) * 2

    # ── Option 3: IV/IV with Faktorverfahren ─────────────
    # More fair distribution — each pays proportional share
    factor_higher = higher / combined if combined > 0 else 0.5
    factor_lower  = lower / combined if combined > 0 else 0.5
    tax_factor    = total_split  # same total but distributed fairly
    tax_f_higher  = tax_factor * factor_higher
    tax_f_lower   = tax_factor * factor_lower

    # Best option
    best_tax = min(total_iv, total_split)
    saving   = abs(total_iv - total_split)

    if total_split < total_iv:
        recommendation = "III/V (Ehegattensplitting)"
        rec_detail = (
            f"  → {higher_label} takes Steuerklasse III\n"
            f"  → {lower_label} takes Steuerklasse V"
        )
    else:
        recommendation = "IV/IV"
        rec_detail = (
            f"  → Both partners take Steuerklasse IV\n"
            f"  → Consider IV/IV with Faktorverfahren for fairness"
        )

    # Rule of thumb explanation
    if income_ratio < 0.4:
        rule_note = (
            "Large income gap (ratio < 40%) → III/V usually better"
        )
    elif income_ratio < 0.6:
        rule_note = (
            "Medium income gap (ratio 40-60%) → compare both options"
        )
    else:
        rule_note = (
            "Similar incomes (ratio > 60%) → IV/IV usually better"
        )

    return (
        f"## Steuerklasse Advisor — GERMANY\n\n"
        f"**{higher_label} (Higher earner):** {currency} {higher:,.2f}\n"
        f"**{lower_label} (Lower earner):** {currency} {lower:,.2f}\n"
        f"**Combined Income:** {currency} {combined:,.2f}\n"
        f"**Income Ratio:** {income_ratio:.0%}\n\n"
        f"---\n\n"
        f"**Option 1 — Steuerklasse IV/IV:**\n"
        f"  • {higher_label} tax: {currency} {tax_iv_higher:,.2f}\n"
        f"  • {lower_label} tax: {currency} {tax_iv_lower:,.2f}\n"
        f"  • **Total: {currency} {total_iv:,.2f}**\n\n"
        f"**Option 2 — Ehegattensplitting III/V:**\n"
        f"  • Total: {currency} {total_split:,.2f}\n\n"
        f"**Option 3 — IV/IV with Faktorverfahren:**\n"
        f"  • {higher_label} pays: {currency} {tax_f_higher:,.2f}\n"
        f"  • {lower_label} pays: {currency} {tax_f_lower:,.2f}\n"
        f"  • Total: {currency} {tax_factor:,.2f}\n\n"
        f"---\n\n"
        f"✅ **Recommendation: {recommendation}**\n"
        f"{rec_detail}\n"
        f"💰 **Annual saving vs other option: {currency} {saving:,.2f}**\n\n"
        f"📊 Rule of thumb: {rule_note}\n\n"
        f"💡 Tip: You can change Steuerklasse once per year at your "
        f"Finanzamt. Changes take effect from the following month.\n\n"
        f"⚠️ Disclaimer: Estimate only. Consult a licensed Steuerberater."
    )