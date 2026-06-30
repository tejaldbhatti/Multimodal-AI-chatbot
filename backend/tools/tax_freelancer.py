"""
Freelancer tax calculator for Germany, USA, India, and Australia.
Handles VAT, trade tax, and self-employment tax scenarios.
"""
from langchain_core.tools import tool
from backend.tools.tax_employee import calculate_tax_from_brackets
from backend.data.country_rules import COUNTRY_RULES


@tool
def calculate_freelancer_tax(
    country: str,
    annual_revenue: float,
    business_expenses: float = 0,
    is_kleinunternehmer: bool = False,
    freelancer_type: str = "freiberufler"
) -> str:
    """
    Calculates tax for a freelancer / self-employed person.
    Covers income tax, VAT, and trade tax where applicable.

    Args:
        country: 'germany', 'usa', 'india', 'australia'
        annual_revenue: total gross revenue in local currency
        business_expenses: deductible business expenses
        is_kleinunternehmer: Germany only — under EUR 22,000 revenue
        freelancer_type: 'freiberufler' or 'gewerbetreibender' (Germany only)
    """
    country = country.lower().strip()

    if country not in COUNTRY_RULES:
        return f"Country '{country}' not supported."

    rules    = COUNTRY_RULES[country]
    currency = rules["currency"]
    fl       = rules.get("freelancer", {})

    # Net profit after expenses
    net_profit = max(0, annual_revenue - business_expenses)

    # ── Germany ─────────────────────────────────────────
    if country == "germany":
        brackets = rules["tax_brackets"]

        # Income tax on net profit
        income_tax = calculate_tax_from_brackets(brackets, net_profit)

        # Solidarity surcharge
        soli = income_tax * rules["solidarity_surcharge"] if income_tax > 18130 else 0.0

        # VAT (Umsatzsteuer)
        if is_kleinunternehmer or annual_revenue < fl["kleinunternehmer_threshold"]:
            vat_collected = 0.0
            vat_note = (
                f"Kleinunternehmer — no VAT charged "
                f"(revenue under EUR {fl['kleinunternehmer_threshold']:,})"
            )
        else:
            vat_collected = annual_revenue * fl["vat_standard"]
            vat_note = (
                f"VAT collected from clients: {currency} {vat_collected:,.2f} "
                f"(19% — must be paid to Finanzamt)"
            )

        # Trade tax (Gewerbesteuer) — only for Gewerbetreibender
        gewerbe_tax = 0.0
        gewerbe_note = "No Gewerbesteuer (Freiberufler)"
        if freelancer_type == "gewerbetreibender":
            taxable_gewerbe = max(0, net_profit - fl["gewerbesteuer_free"])
            gewerbe_tax = taxable_gewerbe * 0.035 * 4  # base rate * avg multiplier
            gewerbe_note = f"Gewerbesteuer: {currency} {gewerbe_tax:,.2f}"

        total_tax    = income_tax + soli + gewerbe_tax
        effective    = (total_tax / annual_revenue * 100) if annual_revenue > 0 else 0
        take_home    = net_profit - total_tax
        monthly_home = take_home / 12

        return (
            f"## Freelancer Tax Estimate — GERMANY\n\n"
            f"**Freelancer Type:** {freelancer_type.capitalize()}\n"
            f"**Gross Revenue:** {currency} {annual_revenue:,.2f}\n"
            f"**Business Expenses:** {currency} {business_expenses:,.2f}\n"
            f"**Net Profit:** {currency} {net_profit:,.2f}\n\n"
            f"**Tax Breakdown:**\n"
            f"  • Income Tax: {currency} {income_tax:,.2f}\n"
            f"  • Solidarity Surcharge: {currency} {soli:,.2f}\n"
            f"  • {gewerbe_note}\n\n"
            f"**VAT (Umsatzsteuer):**\n"
            f"  • {vat_note}\n\n"
            f"**Total Income Tax:** {currency} {total_tax:,.2f}\n"
            f"**Effective Tax Rate:** {effective:.1f}%\n"
            f"**Annual Take Home:** {currency} {take_home:,.2f}\n"
            f"**Monthly Take Home:** {currency} {monthly_home:,.2f}\n\n"
            f"💡 Tip: Keep all receipts for business expenses — "
            f"they reduce your taxable profit directly.\n\n"
            f"⚠️ Disclaimer: Estimate only. Consult a licensed Steuerberater."
        )

    # ── USA ─────────────────────────────────────────────
    elif country == "usa":
        brackets   = rules["tax_brackets_single"]
        std_ded    = rules["standard_deduction_single"]
        se_tax     = rules["freelancer"]["self_employment_tax"]
        qbi_ded    = rules["freelancer"]["qbi_deduction"]

        # Self-employment tax (15.3%)
        se_tax_amt = net_profit * se_tax

        # QBI deduction (20% of net profit)
        qbi_deduction = net_profit * qbi_ded

        # Taxable income after deductions
        taxable = max(0, net_profit - (se_tax_amt / 2) - std_ded - qbi_deduction)
        income_tax = calculate_tax_from_brackets(brackets, taxable)

        total_tax    = income_tax + se_tax_amt
        effective    = (total_tax / annual_revenue * 100) if annual_revenue > 0 else 0
        take_home    = net_profit - total_tax
        monthly_home = take_home / 12

        return (
            f"## Freelancer Tax Estimate — USA\n\n"
            f"**Gross Revenue:** {currency} {annual_revenue:,.2f}\n"
            f"**Business Expenses:** {currency} {business_expenses:,.2f}\n"
            f"**Net Profit:** {currency} {net_profit:,.2f}\n\n"
            f"**Tax Breakdown:**\n"
            f"  • Self-Employment Tax (15.3%): {currency} {se_tax_amt:,.2f}\n"
            f"  • QBI Deduction (20%): -{currency} {qbi_deduction:,.2f}\n"
            f"  • Federal Income Tax: {currency} {income_tax:,.2f}\n\n"
            f"**Total Tax:** {currency} {total_tax:,.2f}\n"
            f"**Effective Tax Rate:** {effective:.1f}%\n"
            f"**Annual Take Home:** {currency} {take_home:,.2f}\n"
            f"**Monthly Take Home:** {currency} {monthly_home:,.2f}\n\n"
            f"💡 Tip: Pay quarterly estimated taxes to avoid penalties.\n\n"
            f"⚠️ Disclaimer: Estimate only. Consult a licensed CPA."
        )

    # ── India ────────────────────────────────────────────
    elif country == "india":
        # Presumptive taxation 44ADA — 50% of gross receipts assumed as income
        presumptive_income = annual_revenue * fl["presumptive_tax_44ADA"]
        brackets_new = rules["tax_brackets_new_regime"]
        tax = calculate_tax_from_brackets(brackets_new, presumptive_income)

        # GST check
        gst_note = ""
        if annual_revenue > fl["gst_threshold"]:
            gst = annual_revenue * fl["gst_rate_services"]
            gst_note = (
                f"\n**GST (18%):** {currency} {gst:,.2f} "
                f"(must register and pay to govt)"
            )
        else:
            gst_note = (
                f"\n**GST:** Not required "
                f"(revenue under {currency} {fl['gst_threshold']:,})"
            )

        take_home    = presumptive_income - tax
        monthly_home = take_home / 12

        return (
            f"## Freelancer Tax Estimate — INDIA\n\n"
            f"**Gross Revenue:** {currency} {annual_revenue:,.2f}\n"
            f"**Presumptive Income (44ADA — 50%):** "
            f"{currency} {presumptive_income:,.2f}\n\n"
            f"**Income Tax:** {currency} {tax:,.2f}\n"
            f"{gst_note}\n\n"
            f"**Annual Take Home:** {currency} {take_home:,.2f}\n"
            f"**Monthly Take Home:** {currency} {monthly_home:,.2f}\n\n"
            f"💡 Tip: Under 44ADA you don't need to maintain books "
            f"if income < 50% of gross receipts.\n\n"
            f"⚠️ Disclaimer: Estimate only. Consult a licensed CA."
        )

    # ── Australia ────────────────────────────────────────
    elif country == "australia":
        brackets     = rules["tax_brackets"]
        medicare     = rules["medicare_levy"]
        gst_thresh   = fl["gst_threshold"]

        income_tax   = calculate_tax_from_brackets(brackets, net_profit)
        medicare_amt = net_profit * medicare
        total_tax    = income_tax + medicare_amt
        effective    = (total_tax / annual_revenue * 100) if annual_revenue > 0 else 0
        take_home    = net_profit - total_tax
        monthly_home = take_home / 12

        # GST check
        gst_note = ""
        if annual_revenue > gst_thresh:
            gst = annual_revenue * fl["gst_rate"]
            gst_note = (
                f"\n**GST (10%):** {currency} {gst:,.2f} "
                f"(must register and pay to ATO)"
            )
        else:
            gst_note = (
                f"\n**GST:** Not required "
                f"(revenue under {currency} {gst_thresh:,})"
            )

        return (
            f"## Freelancer Tax Estimate — AUSTRALIA\n\n"
            f"**Gross Revenue:** {currency} {annual_revenue:,.2f}\n"
            f"**Business Expenses:** {currency} {business_expenses:,.2f}\n"
            f"**Net Profit:** {currency} {net_profit:,.2f}\n\n"
            f"**Tax Breakdown:**\n"
            f"  • Income Tax: {currency} {income_tax:,.2f}\n"
            f"  • Medicare Levy (2%): {currency} {medicare_amt:,.2f}\n"
            f"{gst_note}\n\n"
            f"**Total Tax:** {currency} {total_tax:,.2f}\n"
            f"**Effective Tax Rate:** {effective:.1f}%\n"
            f"**Annual Take Home:** {currency} {take_home:,.2f}\n"
            f"**Monthly Take Home:** {currency} {monthly_home:,.2f}\n\n"
            f"⚠️ Disclaimer: Estimate only. Consult a registered tax agent."
        )

    return "Tax calculation not available for this country."