"""
Steuerberater preparation assistant.
Generates a personalised document checklist for tax advisor appointments.
"""
from langchain_core.tools import tool


@tool
def generate_steuerberater_checklist(
    country: str,
    employment_type: str,
    family_situation: str,
    has_investments: bool = False,
    has_rental_income: bool = False,
) -> str:
    """
    Generates a personalised checklist of documents needed
    for a tax advisor appointment.

    Args:
        country: 'germany', 'usa', 'india', 'australia'
        employment_type: 'employed', 'freelancer', 'both'
        family_situation: 'single', 'married_one_income',
                          'married_both_income', 'single_parent'
        has_investments: whether user has stocks/ETFs/funds
        has_rental_income: whether user has rental property
    """
    country          = country.lower().strip()
    employment_type  = employment_type.lower().strip()
    family_situation = family_situation.lower().strip()

    # ── Germany ─────────────────────────────────────────
    if country == "germany":
        checklist = [
            "📋 **Personal Documents**",
            "  ☐ Personalausweis or Reisepass (ID)",
            "  ☐ Steueridentifikationsnummer (tax ID number)",
            "  ☐ IBAN of your bank account for tax refund",
            "  ☐ Last year's Steuerbescheid (tax assessment notice)",
        ]

        if employment_type in ["employed", "both"]:
            checklist += [
                "\n💼 **Employment Documents**",
                "  ☐ Lohnsteuerbescheinigung (annual wage tax certificate from employer)",
                "  ☐ All payslips (Gehaltsabrechnungen) for the year",
                "  ☐ Arbeitnehmer-Pauschbetrag receipts (work expenses > EUR 1,230)",
                "  ☐ Home office documentation if working from home",
                "  ☐ Commute distance (Entfernungspauschale — EUR 0.30/km)",
            ]

        if employment_type in ["freelancer", "both"]:
            checklist += [
                "\n🧾 **Freelancer / Self-Employment Documents**",
                "  ☐ Einnahmen-Überschuss-Rechnung (EÜR — profit/loss statement)",
                "  ☐ All client invoices issued",
                "  ☐ All business expense receipts",
                "  ☐ Business bank account statements",
                "  ☐ Umsatzsteuervoranmeldungen (VAT pre-registrations)",
                "  ☐ Equipment purchases (computer, phone, software)",
                "  ☐ Home office costs (if applicable)",
            ]

        if family_situation == "married_both_income":
            checklist += [
                "\n👫 **Marriage / Couple Documents**",
                "  ☐ Spouse's Lohnsteuerbescheinigung",
                "  ☐ Heiratsurkunde (marriage certificate) if first filing together",
                "  ☐ Current Steuerklasse for both partners",
            ]

        if family_situation == "single_parent":
            checklist += [
                "\n👶 **Single Parent Documents**",
                "  ☐ Geburtsurkunde of child(ren) (birth certificate)",
                "  ☐ Proof of Steuerklasse II status",
                "  ☐ Childcare costs receipts (Kinderbetreuungskosten)",
                "  ☐ School fees and education costs",
            ]

        if family_situation in ["married_both_income",
                                 "married_one_income", "single_parent"]:
            checklist += [
                "\n👨‍👩‍👧 **Children / Family**",
                "  ☐ Kindergeld notification letters",
                "  ☐ Kinderfreibetrag documentation",
                "  ☐ Childcare receipts (up to EUR 4,000 deductible per child)",
            ]

        if has_investments:
            checklist += [
                "\n📈 **Investment Documents**",
                "  ☐ Jahressteuerbescheinigung from your bank/broker",
                "  ☐ Freistellungsauftrag confirmation",
                "  ☐ Foreign dividend income documentation",
                "  ☐ Crypto transaction history (if applicable)",
            ]

        if has_rental_income:
            checklist += [
                "\n🏠 **Rental Income Documents**",
                "  ☐ Rental income receipts",
                "  ☐ All property expense receipts (repairs, maintenance)",
                "  ☐ Mortgage interest statements",
                "  ☐ Property management fees",
                "  ☐ Depreciation (AfA) documentation",
            ]

        checklist += [
            "\n💊 **Deductible Expenses (Sonderausgaben)**",
            "  ☐ Health insurance contributions",
            "  ☐ Pension insurance contributions",
            "  ☐ Charitable donations (Spendenbescheinigungen)",
            "  ☐ Church tax paid (Kirchensteuer)",
            "  ☐ Riester Rente contribution statements",
        ]

        checklist += [
            "\n💡 **Questions to ask your Steuerberater**",
            "  • Which deductions am I missing?",
            "  • Is my Steuerklasse optimal for my situation?",
            "  • Should I change to Ehegattensplitting?",
            "  • Can I deduct my home office costs?",
            "  • What can I do differently next year to pay less tax?",
        ]

        disclaimer = (
            "\n⚠️ Legal Note: This checklist is for preparation purposes only. "
            "It does not constitute tax advice. A licensed Steuerberater "
            "(§2 StBerG) is the only person legally authorised to give "
            "binding tax advice in Germany."
        )

    # ── USA ─────────────────────────────────────────────
    elif country == "usa":
        checklist = [
            "📋 **Personal Documents**",
            "  ☐ Social Security Number (SSN) for all family members",
            "  ☐ Last year's tax return",
            "  ☐ Bank account details for refund direct deposit",
        ]

        if employment_type in ["employed", "both"]:
            checklist += [
                "\n💼 **Employment Documents**",
                "  ☐ W-2 forms from all employers",
                "  ☐ W-2G (gambling winnings if applicable)",
            ]

        if employment_type in ["freelancer", "both"]:
            checklist += [
                "\n🧾 **Self-Employment Documents**",
                "  ☐ All 1099 forms received",
                "  ☐ Business income and expense records",
                "  ☐ Home office measurements and costs",
                "  ☐ Business mileage log",
                "  ☐ Quarterly estimated tax payment records",
            ]

        if has_investments:
            checklist += [
                "\n📈 **Investment Documents**",
                "  ☐ 1099-B (broker statements)",
                "  ☐ 1099-DIV (dividend income)",
                "  ☐ 1099-INT (interest income)",
                "  ☐ Crypto transaction records",
            ]

        checklist += [
            "\n💡 **Questions to ask your CPA**",
            "  • Should I itemize or take the standard deduction?",
            "  • Am I maximising my 401(k) and IRA contributions?",
            "  • Are there any tax credits I am missing?",
        ]

        disclaimer = (
            "\n⚠️ Note: This checklist is for preparation only. "
            "Consult a licensed CPA for advice specific to your situation."
        )

    # ── India ────────────────────────────────────────────
    elif country == "india":
        checklist = [
            "📋 **Personal Documents**",
            "  ☐ PAN card",
            "  ☐ Aadhaar card",
            "  ☐ Last year's ITR acknowledgement",
            "  ☐ Bank account statements",
        ]

        if employment_type in ["employed", "both"]:
            checklist += [
                "\n💼 **Employment Documents**",
                "  ☐ Form 16 from employer",
                "  ☐ All salary slips",
                "  ☐ HRA receipts (if claiming HRA exemption)",
            ]

        if employment_type in ["freelancer", "both"]:
            checklist += [
                "\n🧾 **Freelancer Documents**",
                "  ☐ All client invoices",
                "  ☐ Form 26AS (tax credit statement)",
                "  ☐ TDS certificates received",
                "  ☐ GST returns (if registered)",
            ]

        checklist += [
            "\n💰 **Section 80C Investment Proofs**",
            "  ☐ PPF passbook / statement",
            "  ☐ ELSS mutual fund statements",
            "  ☐ LIC premium receipts",
            "  ☐ NPS contribution statements",
            "  ☐ Home loan principal repayment certificate",
        ]

        checklist += [
            "\n💡 **Questions to ask your CA**",
            "  • Old regime or new regime — which is better for me?",
            "  • Have I maximised my Section 80C deductions?",
            "  • Am I eligible for any other deductions?",
        ]

        disclaimer = (
            "\n⚠️ Note: This checklist is for preparation only. "
            "Consult a licensed CA for advice specific to your situation."
        )

    # ── Australia ────────────────────────────────────────
    elif country == "australia":
        checklist = [
            "📋 **Personal Documents**",
            "  ☐ Tax File Number (TFN)",
            "  ☐ MyGov account access",
            "  ☐ Last year's tax return",
            "  ☐ Bank account details for refund",
        ]

        if employment_type in ["employed", "both"]:
            checklist += [
                "\n💼 **Employment Documents**",
                "  ☐ Payment summaries from all employers",
                "  ☐ Work-related expense receipts",
                "  ☐ Union fees receipts",
                "  ☐ Work uniform costs",
            ]

        if employment_type in ["freelancer", "both"]:
            checklist += [
                "\n🧾 **Sole Trader Documents**",
                "  ☐ Business income and expense records",
                "  ☐ BAS (Business Activity Statements) if GST registered",
                "  ☐ Vehicle logbook (if using car for business)",
                "  ☐ Home office expense records",
            ]

        checklist += [
            "\n🏦 **Superannuation**",
            "  ☐ Super fund statements",
            "  ☐ Voluntary contribution records",
            "  ☐ SMSF documents (if applicable)",
        ]

        checklist += [
            "\n💡 **Questions to ask your tax agent**",
            "  • What work expenses can I claim?",
            "  • Should I make extra super contributions?",
            "  • Am I eligible for any offsets or rebates?",
        ]

        disclaimer = (
            "\n⚠️ Note: This checklist is for preparation only. "
            "Consult a registered tax agent for advice specific to your situation."
        )

    else:
        return f"Country '{country}' not supported yet."

    return (
        f"## Steuerberater / Tax Advisor Preparation Checklist\n"
        f"**Country:** {country.upper()} | "
        f"**Employment:** {employment_type} | "
        f"**Family:** {family_situation}\n\n" +
        "\n".join(checklist) +
        disclaimer
    )