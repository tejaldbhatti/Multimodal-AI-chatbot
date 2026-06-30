"""
Ehegattensplitting calculator for married couples in Germany.
Compares Steuerklasse III/V vs IV/IV to find the best combination.
Also covers married couple tax for USA, India, and Australia.
"""
from langchain_core.tools import tool
from backend.tools.tax_employee import calculate_tax_from_brackets
from backend.data.country_rules import COUNTRY_RULES


@tool
def calculate_couple_tax(
    country: str,
    income_partner1: float,
    income_partner2: float,
) -> str:
    """
    Calculates tax for a married couple where both partners work.
    For Germany: compares Steuerklasse III/V vs IV/IV.
    For USA: compares single vs married filing jointly.

    Args:
        country: 'germany', 'usa', 'india', 'australia'
        income_partner1: annual gross income of partner 1
        income_partner2: annual gross income of partner 2
    """
    country = country.lower().strip()

    if country not in COUNTRY_RULES:
        return f"Country '{country}' not supported."

    rules    = COUNTRY_RULES[country]
    currency = rules["currency"]

    # ── Germany — Ehegattensplitting ────────────────────
    if country == "germany":
        brackets = rules["tax_brackets"]

        combined_income = income_partner1 + income_partner2

        # ── Option 1: Steuerklasse IV/IV ────────────────
        # Each partner taxed individually on their own income
        tax_p1_iv = calculate_tax_from_brackets(brackets, income_partner1)
        tax_p2_iv = calculate_tax_from_brackets(brackets, income_partner2)
        total_tax_iv = tax_p1_iv + tax_p2_iv

        # ── Option 2: Ehegattensplitting (III/V) ────────
        # Split combined income in half, calculate tax, then double it
        half_income   = combined_income / 2
        tax_half      = calculate_tax_from_brackets(brackets, half_income)
        total_tax_split = tax_half * 2

        # Which is better?
        saving = abs(total_tax_iv - total_tax_split)
        if total_tax_split < total_tax_iv:
            better    = "Ehegattensplitting (III/V)"
            worse     = "Steuerklasse IV/IV"
            saved_by  = better
        else:
            better    = "Steuerklasse IV/IV"
            worse     = "Ehegattensplitting (III/V)"
            saved_by  = better

        take_home_iv    = combined_income - total_tax_iv
        take_home_split = combined_income - total_tax_split

        # Solidarity surcharge
        soli_iv    = total_tax_iv * 0.055 if total_tax_iv > 18130 else 0
        soli_split = total_tax_split * 0.055 if total_tax_split > 18130 else 0

        return (
            f"## Couple Tax Estimate — GERMANY (Ehegattensplitting)\n\n"
            f"**Partner 1 Income:** {currency} {income_partner1:,.2f}\n"
            f"**Partner 2 Income:** {currency} {income_partner2:,.2f}\n"
            f"**Combined Income:** {currency} {combined_income:,.2f}\n\n"
            f"---\n\n"
            f"**Option 1 — Steuerklasse IV/IV** (each taxed individually)\n"
            f"  • Partner 1 tax: {currency} {tax_p1_iv:,.2f}\n"
            f"  • Partner 2 tax: {currency} {tax_p2_iv:,.2f}\n"
            f"  • Solidarity surcharge: {currency} {soli_iv:,.2f}\n"
            f"  • **Total tax: {currency} {total_tax_iv + soli_iv:,.2f}**\n"
            f"  • Take home: {currency} {take_home_iv:,.2f}\n\n"
            f"**Option 2 — Ehegattensplitting (III/V)** (income splitting)\n"
            f"  • Combined income split in half: {currency} {half_income:,.2f}\n"
            f"  • Tax on half × 2: {currency} {total_tax_split:,.2f}\n"
            f"  • Solidarity surcharge: {currency} {soli_split:,.2f}\n"
            f"  • **Total tax: {currency} {total_tax_split + soli_split:,.2f}**\n"
            f"  • Take home: {currency} {take_home_split:,.2f}\n\n"
            f"---\n\n"
            f"✅ **Better option: {better}**\n"
            f"💰 **Annual saving: {currency} {saving:,.2f}**\n\n"
            f"💡 Note: Ehegattensplitting saves most when income gap is large.\n"
            f"If both partners earn similarly, IV/IV is often better.\n\n"
            f"⚠️ Disclaimer: Estimate only. Consult a licensed Steuerberater."
        )

    # ── USA — Married Filing Jointly ────────────────────
    elif country == "usa":
        brackets_single  = rules["tax_brackets_single"]
        brackets_married = rules["tax_brackets_married_jointly"]
        std_ded_single   = rules["standard_deduction_single"]
        std_ded_married  = rules["standard_deduction_married"]

        combined_income = income_partner1 + income_partner2

        # Filing separately
        tax_p1 = calculate_tax_from_brackets(
            brackets_single,
            max(0, income_partner1 - std_ded_single)
        )
        tax_p2 = calculate_tax_from_brackets(
            brackets_single,
            max(0, income_partner2 - std_ded_single)
        )
        total_separate = tax_p1 + tax_p2

        # Filing jointly
        taxable_joint = max(0, combined_income - std_ded_married)
        total_joint   = calculate_tax_from_brackets(
            brackets_married, taxable_joint
        )

        saving = abs(total_separate - total_joint)
        better = "Married Filing Jointly" if total_joint < total_separate \
            else "Filing Separately"

        return (
            f"## Couple Tax Estimate — USA\n\n"
            f"**Partner 1 Income:** {currency} {income_partner1:,.2f}\n"
            f"**Partner 2 Income:** {currency} {income_partner2:,.2f}\n"
            f"**Combined Income:** {currency} {combined_income:,.2f}\n\n"
            f"**Filing Separately:**\n"
            f"  • Total tax: {currency} {total_separate:,.2f}\n"
            f"  • Take home: {currency} {combined_income - total_separate:,.2f}\n\n"
            f"**Married Filing Jointly:**\n"
            f"  • Total tax: {currency} {total_joint:,.2f}\n"
            f"  • Take home: {currency} {combined_income - total_joint:,.2f}\n\n"
            f"✅ **Better option: {better}**\n"
            f"💰 **Annual saving: {currency} {saving:,.2f}**\n\n"
            f"⚠️ Disclaimer: Estimate only. Consult a licensed CPA."
        )

    # ── India ────────────────────────────────────────────
    elif country == "india":
        # India taxes individuals separately — no joint filing
        brackets = rules["tax_brackets_new_regime"]
        std_ded  = rules["standard_deduction"]
        sec_80c  = rules["section_80c_limit"]

        tax_p1 = calculate_tax_from_brackets(
            brackets, max(0, income_partner1 - std_ded)
        )
        tax_p2 = calculate_tax_from_brackets(
            brackets, max(0, income_partner2 - std_ded)
        )
        total_tax     = tax_p1 + tax_p2
        combined      = income_partner1 + income_partner2
        take_home     = combined - total_tax

        return (
            f"## Couple Tax Estimate — INDIA\n\n"
            f"**Partner 1 Income:** {currency} {income_partner1:,.2f}\n"
            f"  • Tax: {currency} {tax_p1:,.2f}\n\n"
            f"**Partner 2 Income:** {currency} {income_partner2:,.2f}\n"
            f"  • Tax: {currency} {tax_p2:,.2f}\n\n"
            f"**Combined Income:** {currency} {combined:,.2f}\n"
            f"**Total Tax:** {currency} {total_tax:,.2f}\n"
            f"**Combined Take Home:** {currency} {take_home:,.2f}\n\n"
            f"💡 India taxes each individual separately.\n"
            f"Each partner can claim Section 80C deduction of "
            f"{currency} {sec_80c:,} independently.\n\n"
            f"⚠️ Disclaimer: Estimate only. Consult a licensed CA."
        )

    # ── Australia ────────────────────────────────────────
    elif country == "australia":
        brackets = rules["tax_brackets"]
        medicare = rules["medicare_levy"]

        tax_p1       = calculate_tax_from_brackets(brackets, income_partner1)
        tax_p2       = calculate_tax_from_brackets(brackets, income_partner2)
        medicare_p1  = income_partner1 * medicare
        medicare_p2  = income_partner2 * medicare
        total_tax    = tax_p1 + tax_p2 + medicare_p1 + medicare_p2
        combined     = income_partner1 + income_partner2
        take_home    = combined - total_tax

        return (
            f"## Couple Tax Estimate — AUSTRALIA\n\n"
            f"**Partner 1 Income:** {currency} {income_partner1:,.2f}\n"
            f"  • Income Tax: {currency} {tax_p1:,.2f}\n"
            f"  • Medicare Levy: {currency} {medicare_p1:,.2f}\n\n"
            f"**Partner 2 Income:** {currency} {income_partner2:,.2f}\n"
            f"  • Income Tax: {currency} {tax_p2:,.2f}\n"
            f"  • Medicare Levy: {currency} {medicare_p2:,.2f}\n\n"
            f"**Combined Income:** {currency} {combined:,.2f}\n"
            f"**Total Tax:** {currency} {total_tax:,.2f}\n"
            f"**Combined Take Home:** {currency} {take_home:,.2f}\n\n"
            f"💡 Australia taxes each individual separately.\n"
            f"Consider spouse superannuation contributions for tax benefits.\n\n"
            f"⚠️ Disclaimer: Estimate only. Consult a registered tax agent."
        )

    return "Tax calculation not available for this country."