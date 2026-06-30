"""
Single parent tax calculator for Germany.
Calculates Steuerklasse II benefits and Entlastungsbetrag.
"""
from langchain_core.tools import tool
from backend.tools.tax_employee import calculate_tax_from_brackets
from backend.data.country_rules import COUNTRY_RULES


@tool
def calculate_single_parent_tax(
    country: str,
    annual_income: float,
    number_of_children: int = 1
) -> str:
    """
    Calculates tax for a single parent.
    For Germany: applies Steuerklasse II and Entlastungsbetrag.

    Args:
        country: 'germany', 'usa', 'india', 'australia'
        annual_income: gross annual income in local currency
        number_of_children: number of dependent children
    """
    country = country.lower().strip()

    if country not in COUNTRY_RULES:
        return f"Country '{country}' not supported."

    rules    = COUNTRY_RULES[country]
    currency = rules["currency"]

    # ── Germany — Steuerklasse II ────────────────────────
    if country == "germany":
        brackets = rules["tax_brackets"]

        # Entlastungsbetrag für Alleinerziehende
        # EUR 4,260 for first child + EUR 240 for each additional child
        entlastungsbetrag = 4260 + (number_of_children - 1) * 240

        # Kinderfreibetrag (child tax allowance)
        # EUR 6,384 per child (EUR 3,192 per parent × 2 since single parent gets both)
        kinderfreibetrag = 6384 * number_of_children

        # Taxable income after allowances
        taxable_income = max(0, annual_income - entlastungsbetrag)

        # Tax with Steuerklasse II
        tax_klasse_ii = calculate_tax_from_brackets(brackets, taxable_income)

        # Compare with Steuerklasse I (no single parent benefit)
        tax_klasse_i  = calculate_tax_from_brackets(brackets, annual_income)

        saving       = tax_klasse_i - tax_klasse_ii
        soli         = tax_klasse_ii * 0.055 if tax_klasse_ii > 18130 else 0
        total_tax    = tax_klasse_ii + soli
        take_home    = annual_income - total_tax
        monthly_home = take_home / 12

        return (
            f"## Single Parent Tax Estimate — GERMANY\n\n"
            f"**Steuerklasse:** II (Single Parent)\n"
            f"**Gross Annual Income:** {currency} {annual_income:,.2f}\n"
            f"**Number of Children:** {number_of_children}\n\n"
            f"**Single Parent Benefits:**\n"
            f"  • Entlastungsbetrag: {currency} {entlastungsbetrag:,.2f}\n"
            f"  • Kinderfreibetrag: {currency} {kinderfreibetrag:,.2f}\n"
            f"  • Taxable Income: {currency} {taxable_income:,.2f}\n\n"
            f"**Tax Breakdown:**\n"
            f"  • Income Tax (Klasse II): {currency} {tax_klasse_ii:,.2f}\n"
            f"  • Solidarity Surcharge: {currency} {soli:,.2f}\n"
            f"  • **Total Tax: {currency} {total_tax:,.2f}**\n\n"
            f"**vs Steuerklasse I** (without single parent benefit):\n"
            f"  • Tax would be: {currency} {tax_klasse_i:,.2f}\n"
            f"  • 💰 You save: {currency} {saving:,.2f} per year\n\n"
            f"**Annual Take Home:** {currency} {take_home:,.2f}\n"
            f"**Monthly Take Home:** {currency} {monthly_home:,.2f}\n\n"
            f"💡 Tip: Apply for Steuerklasse II at your local Finanzamt. "
            f"You need to confirm you live alone with your child(ren).\n\n"
            f"⚠️ Disclaimer: Estimate only. Consult a licensed Steuerberater."
        )

    # ── USA — Head of Household ──────────────────────────
    elif country == "usa":
        brackets  = rules["tax_brackets_single"]
        std_ded   = rules["standard_deduction_single"]

        # Head of Household gets higher standard deduction
        hoh_deduction = 21900  # 2024 Head of Household standard deduction
        child_tax_credit = 2000 * number_of_children  # $2,000 per child

        taxable   = max(0, annual_income - hoh_deduction)
        tax       = calculate_tax_from_brackets(brackets, taxable)
        tax_after_credit = max(0, tax - child_tax_credit)
        take_home = annual_income - tax_after_credit
        monthly   = take_home / 12

        # Compare with single filing
        tax_single = calculate_tax_from_brackets(
            brackets, max(0, annual_income - std_ded)
        )
        saving = tax_single - tax_after_credit

        return (
            f"## Single Parent Tax Estimate — USA\n\n"
            f"**Filing Status:** Head of Household\n"
            f"**Gross Annual Income:** {currency} {annual_income:,.2f}\n"
            f"**Number of Children:** {number_of_children}\n\n"
            f"**Benefits:**\n"
            f"  • HOH Standard Deduction: {currency} {hoh_deduction:,.2f}\n"
            f"  • Child Tax Credit: {currency} {child_tax_credit:,.2f}\n\n"
            f"**Tax Breakdown:**\n"
            f"  • Income Tax: {currency} {tax:,.2f}\n"
            f"  • After Child Tax Credit: {currency} {tax_after_credit:,.2f}\n\n"
            f"💰 **Saving vs Single filing: {currency} {saving:,.2f}**\n"
            f"**Annual Take Home:** {currency} {take_home:,.2f}\n"
            f"**Monthly Take Home:** {currency} {monthly:,.2f}\n\n"
            f"⚠️ Disclaimer: Estimate only. Consult a licensed CPA."
        )

    # ── India ────────────────────────────────────────────
    elif country == "india":
        brackets = rules["tax_brackets_new_regime"]
        std_ded  = rules["standard_deduction"]
        sec_80c  = rules["section_80c_limit"]

        # Children's education allowance
        edu_allowance = 100 * 12 * number_of_children  # INR 100/month per child

        taxable = max(0, annual_income - std_ded - edu_allowance)
        tax     = calculate_tax_from_brackets(brackets, taxable)
        take_home = annual_income - tax
        monthly   = take_home / 12

        return (
            f"## Single Parent Tax Estimate — INDIA\n\n"
            f"**Gross Annual Income:** {currency} {annual_income:,.2f}\n"
            f"**Number of Children:** {number_of_children}\n\n"
            f"**Deductions:**\n"
            f"  • Standard Deduction: {currency} {std_ded:,.2f}\n"
            f"  • Children Education Allowance: {currency} {edu_allowance:,.2f}\n"
            f"  • Section 80C (if invested): {currency} {sec_80c:,.2f}\n\n"
            f"**Income Tax:** {currency} {tax:,.2f}\n"
            f"**Annual Take Home:** {currency} {take_home:,.2f}\n"
            f"**Monthly Take Home:** {currency} {monthly:,.2f}\n\n"
            f"💡 Tip: Invest INR {sec_80c:,} in PPF/ELSS to save more tax.\n\n"
            f"⚠️ Disclaimer: Estimate only. Consult a licensed CA."
        )

    # ── Australia ────────────────────────────────────────
    elif country == "australia":
        brackets  = rules["tax_brackets"]
        medicare  = rules["medicare_levy"]

        # Low income tax offset for single parents
        lito = min(700, max(0, 700 - (annual_income - 37500) * 0.05))

        tax          = calculate_tax_from_brackets(brackets, annual_income)
        medicare_amt = annual_income * medicare
        total_tax    = max(0, tax + medicare_amt - lito)
        take_home    = annual_income - total_tax
        monthly      = take_home / 12

        return (
            f"## Single Parent Tax Estimate — AUSTRALIA\n\n"
            f"**Gross Annual Income:** {currency} {annual_income:,.2f}\n"
            f"**Number of Children:** {number_of_children}\n\n"
            f"**Tax Breakdown:**\n"
            f"  • Income Tax: {currency} {tax:,.2f}\n"
            f"  • Medicare Levy: {currency} {medicare_amt:,.2f}\n"
            f"  • Low Income Tax Offset: -{currency} {lito:,.2f}\n"
            f"  • **Total Tax: {currency} {total_tax:,.2f}**\n\n"
            f"**Annual Take Home:** {currency} {take_home:,.2f}\n"
            f"**Monthly Take Home:** {currency} {monthly:,.2f}\n\n"
            f"💡 Tip: You may also be eligible for Family Tax Benefit "
            f"through Centrelink.\n\n"
            f"⚠️ Disclaimer: Estimate only. Consult a registered tax agent."
        )

    return "Tax calculation not available for this country."