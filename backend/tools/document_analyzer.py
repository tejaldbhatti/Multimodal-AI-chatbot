"""
Document analyzer using GPT-4o.
Analyzes salary slips and bank statements.
"""
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from backend.config import OPENAI_API_KEY, OPENAI_MODEL_DOCS


# Use GPT-4o for document analysis — more powerful than gpt-4o-mini
llm_docs = ChatOpenAI(
    model=OPENAI_MODEL_DOCS,
    temperature=0,
    api_key=OPENAI_API_KEY
)


@tool
def analyze_salary_slip(salary_slip_text: str, country: str = "germany") -> str:
    """
    Analyzes a salary slip and extracts key financial information.
    Provides personalised advice based on the salary slip data.

    Args:
        salary_slip_text: raw text extracted from salary slip
        country: country for context (germany, usa, india, australia)
    """
    prompt = f"""You are an expert financial advisor analyzing a salary slip.

Country context: {country.upper()}

Salary slip content:
{salary_slip_text}

Please analyze this salary slip and provide:

1. **Key Figures Extracted**
   - Gross salary
   - Net salary
   - All deductions (tax, social security, health insurance etc.)
   - Any allowances or bonuses

2. **Tax Analysis**
   - Effective tax rate
   - Is the tax deduction reasonable for this income level?
   - Any observations about the tax class (if Germany)

3. **Financial Health Check**
   - What percentage is going to taxes and deductions?
   - Monthly take-home amount
   - Suggested savings amount (based on 50/30/20 rule)

4. **Personalised Recommendations**
   - What could this person do to reduce their tax burden?
   - Are there any deductions they might be missing?
   - Investment suggestions based on their income level

5. **Action Items**
   - Top 3 things this person should do right now

⚠️ Disclaimer: This analysis is for educational purposes only.
Please consult a licensed tax professional for binding advice.
"""

    response = llm_docs.invoke([
        {"role": "system", "content": "You are an expert financial advisor and tax analyst."},
        {"role": "user", "content": prompt}
    ])

    return response.content


@tool
def analyze_bank_statement(statement_text: str, country: str = "germany") -> str:
    """
    Analyzes a bank statement and provides spending breakdown
    and financial recommendations.

    Args:
        statement_text: raw text extracted from bank statement
        country: country for context
    """
    prompt = f"""You are an expert financial advisor analyzing a bank statement.

Country context: {country.upper()}

Bank statement content:
{statement_text}

Please analyze this bank statement and provide:

1. **Spending Breakdown**
   - Total income received
   - Total spending
   - Categorize spending (food, rent, transport, entertainment, etc.)
   - Savings rate

2. **Financial Health Score** (1-10)
   - Rating with explanation
   - Key strengths
   - Key concerns

3. **Budget Analysis** (50/30/20 rule)
   - Needs (50%): housing, food, transport
   - Wants (30%): entertainment, dining out
   - Savings (20%): investments, emergency fund
   - How does actual spending compare?

4. **Red Flags**
   - Any concerning spending patterns
   - Recurring unnecessary expenses
   - Overdraft or low balance warnings

5. **Personalised Action Plan**
   - Top 3 immediate changes to improve finances
   - Monthly savings target
   - Investment suggestion based on savings capacity

⚠️ Disclaimer: This analysis is for educational purposes only.
Please consult a licensed financial advisor for personalised advice.
"""

    response = llm_docs.invoke([
        {"role": "system", "content": "You are an expert financial advisor and spending analyst."},
        {"role": "user", "content": prompt}
    ])

    return response.content