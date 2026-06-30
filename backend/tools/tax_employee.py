"""Employee tax utilities."""

"""
Employee tax calculator for Germany, USA, India, and Australia.
Calculates income tax based on employment type and Steuerklasse.
"""
from langchain_core.tools import tool
from backend.data.country_rules import COUNTRY_RULES


def calculate_tax_from_brackets(brackets: list,
                                 annual_income: float) -> float:
    """
    Helper function — calculates tax from any bracket list.
    Used by all country calculators.
    """
    total_tax = 0.0
    for bracket in brackets:
        low  = bracket["min"]
        high = bracket["max"] if bracket["max"] else float("inf")
        rate = bracket["rate"]

        if annual_income <= low:
            break

        taxable = min(annual_income, high) - low
        total_tax += taxable * rate

    return total_tax


@tool
def calculate_employee_tax(country: str,
                           annual_income: float,
                           steuerklasse: str = "I",
                           has_church_tax: bool = False) -> str:
    """
    Calculates income tax for an employed person.
    Supports Germany (all Steuerklassen), USA, India, Australia.

    Args:
        country: 'germany', 'usa', 'india', 'australia'
        annual_income: gross annual income in local currency
        steuerklasse: German tax class I/II/III/IV/V (Germany only)
        has_church_tax: whether to include church tax (Germany only)
    """
    country = country.lower().strip()

    if country not in COUNTRY_RULES:
        return f"Country '{country}' not supported."

    rules    = COUNTRY_RULES[country]
    currency = rules["currency"]

    # ── Germany ─────────────────────────────────────────
    if country == "germany":
        brackets = rules["tax_brackets"]

        # Steuerklasse III gets double free allowance (married higher earner)
        if steuerklasse == "III":
            # Apply splitting method — taxed as if income is halved then doubled
            half_income = annual_income / 2
            tax = calculate_tax_from_brackets(brackets, half_income) * 2
        else:
            tax = calculate_tax_from_brackets(brackets, annual_income)

        # Solidarity surcharge — 5.5% on income tax above threshold
        soli = 0.0
        if tax > 18130:
            soli = tax * rules["solidarity_surcharge"]

        # Church tax — 8-9% on income tax (optional)
        church = tax * 0.09 if has_church_tax else 0.0

        total_tax    = tax + soli + church
        effective    = (total_tax / annual_income * 100) if annual_income > 0 else 0
        take_home    = annual_income - total_tax
        monthly_home = take_home / 12

        return (
            f"## Employee Tax Estimate — GERMANY\n\n"
            f"**Steuerklasse:** {steuerklasse}\n"
            f"**Gross Annual Income:** {currency} {annual_income:,.2f}\n\n"
            f"**Tax Breakdown:**\n"
            f"  • Income Tax (Einkommensteuer): {currency} {tax:,.2f}\n"
            f"  • Solidarity Surcharge: {currency} {soli:,.2f}\n"
            f"  • Church Tax: {currency} {church:,.2f}\n\n"
            f"**Total Tax:** {currency} {total_tax:,.2f}\n"
            f"**Effective Tax Rate:** {effective:.1f}%\n"
            f"**Annual Take Home:** {currency} {take_home:,.2f}\n"
            f"**Monthly Take Home:** {currency} {monthly_home:,.2f}\n\n"
            f"**Steuerklasse meaning:** {rules['steuerklassen'].get(steuerklasse, 'Unknown')}\n\n"
            f"⚠️ Disclaimer: Estimate only. Actual tax depends on deductions, "
            f"allowances, and personal circumstances. Consult a Steuerberater."
        )

    # ── USA ─────────────────────────────────────────────
    elif country == "usa":
        brackets     = rules["tax_brackets_single"]
        std_ded      = rules["standard_deduction_single"]
        taxable      = max(0, annual_income - std_ded)
        tax          = calculate_tax_from_brackets(brackets, taxable)
        effective    = (tax / annual_income * 100) if annual_income > 0 else 0
        take_home    = annual_income - tax
        monthly_home = take_home / 12

        return (
            f"## Employee Tax Estimate — USA\n\n"
            f"**Gross Annual Income:** {currency} {annual_income:,.2f}\n"
            f"**Standard Deduction:** {currency} {std_ded:,.2f}\n"
            f"**Taxable Income:** {currency} {taxable:,.2f}\n\n"
            f"**Federal Income Tax:** {currency} {tax:,.2f}\n"
            f"**Effective Tax Rate:** {effective:.1f}%\n"
            f"**Annual Take Home:** {currency} {take_home:,.2f}\n"
            f"**Monthly Take Home:** {currency} {monthly_home:,.2f}\n\n"
            f"Note: This is federal tax only. State tax varies by state.\n\n"
            f"⚠️ Disclaimer: Estimate only. Consult a licensed CPA."
        )

    # ── India ────────────────────────────────────────────
    elif country == "india":
        # Compare old and new regime
        brackets_new = rules["tax_brackets_new_regime"]
        brackets_old = rules["tax_brackets_old_regime"]
        std_ded      = rules["standard_deduction"]
        sec_80c      = rules["section_80c_limit"]

        tax_new = calculate_tax_from_brackets(
            brackets_new, annual_income
        )
        taxable_old = max(0, annual_income - std_ded - sec_80c)
        tax_old     = calculate_tax_from_brackets(
            brackets_old, taxable_old
        )

        better_regime = "New Regime" if tax_new < tax_old else "Old Regime"
        saving        = abs(tax_old - tax_new)

        return (
            f"## Employee Tax Estimate — INDIA\n\n"
            f"**Gross Annual Income:** {currency} {annual_income:,.2f}\n\n"
            f"**New Tax Regime:**\n"
            f"  • Tax: {currency} {tax_new:,.2f}\n"
            f"  • Take Home: {currency} {annual_income - tax_new:,.2f}\n\n"
            f"**Old Tax Regime** (with standard deduction + 80C):\n"
            f"  • Deductions: {currency} {std_ded + sec_80c:,.2f}\n"
            f"  • Tax: {currency} {tax_old:,.2f}\n"
            f"  • Take Home: {currency} {annual_income - tax_old:,.2f}\n\n"
            f"✅ **Recommended: {better_regime}** "
            f"(saves {currency} {saving:,.2f})\n\n"
            f"⚠️ Disclaimer: Estimate only. Consult a licensed CA."
        )

    # ── Australia ────────────────────────────────────────
    elif country == "australia":
        brackets     = rules["tax_brackets"]
        medicare     = rules["medicare_levy"]
        tax          = calculate_tax_from_brackets(brackets, annual_income)
        medicare_amt = annual_income * medicare
        total_tax    = tax + medicare_amt
        effective    = (total_tax / annual_income * 100) if annual_income > 0 else 0
        take_home    = annual_income - total_tax
        monthly_home = take_home / 12

        return (
            f"## Employee Tax Estimate — AUSTRALIA\n\n"
            f"**Gross Annual Income:** {currency} {annual_income:,.2f}\n\n"
            f"**Tax Breakdown:**\n"
            f"  • Income Tax: {currency} {tax:,.2f}\n"
            f"  • Medicare Levy (2%): {currency} {medicare_amt:,.2f}\n\n"
            f"**Total Tax:** {currency} {total_tax:,.2f}\n"
            f"**Effective Tax Rate:** {effective:.1f}%\n"
            f"**Annual Take Home:** {currency} {take_home:,.2f}\n"
            f"**Monthly Take Home:** {currency} {monthly_home:,.2f}\n\n"
            f"⚠️ Disclaimer: Estimate only. Consult a registered tax agent."
        )

    return "Tax calculation not available for this country."