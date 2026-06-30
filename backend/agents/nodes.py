"""
LangGraph nodes for the Financial Chatbot.
Each node does one specific job in the conversation flow.
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from backend.agents.state import AgentState
from backend.data.country_rules import COUNTRY_RULES
from backend.config import OPENAI_API_KEY, OPENAI_MODEL_CHAT
from backend.tools.market_data import (
    get_stock_price,
    get_forex_rate,
    get_etf_price,
    get_financial_news
)
from backend.tools.tax_employee import calculate_employee_tax
from backend.tools.tax_freelancer import calculate_freelancer_tax
from backend.tools.tax_couple import calculate_couple_tax
from backend.tools.tax_single_parent import calculate_single_parent_tax
from backend.tools.tax_class_advisor import advise_steuerklasse
from backend.tools.steuerberater_prep import generate_steuerberater_checklist
from backend.tools.document_analyzer import (
    analyze_salary_slip,
    analyze_bank_statement
)

# Initialize LLM once — reused across all nodes
llm = ChatOpenAI(
    model=OPENAI_MODEL_CHAT,
    temperature=0,
    api_key=OPENAI_API_KEY
)

# Bind market tools to LLM
# This lets the agent automatically call tools when needed
llm_with_tools = llm.bind_tools([
    # Market tools
    get_stock_price,
    get_forex_rate,
    get_etf_price,
    get_financial_news,
    # Tax tools
    calculate_employee_tax,
    calculate_freelancer_tax,
    calculate_couple_tax,
    calculate_single_parent_tax,
    advise_steuerklasse,
    generate_steuerberater_checklist,
    # Document tools
    analyze_salary_slip,
    analyze_bank_statement,
])


# ── Node 1 ──────────────────────────────────────────────
def detect_country(state: AgentState) -> AgentState:
    """
    Reads the latest message and detects which country
    the user is from. Updates state with the result.
    """
    messages = state["messages"]
    last_message = messages[-1].content.lower()

    # Simple keyword detection first
    country_keywords = {
        "germany": ["germany", "german", "deutschland", "berlin", "munich",
                    "hamburg", "euro", "steuer", "steuerklasse", "finanzamt",
                    "elster", "riester", "einkommensteuer"],
        "usa": ["usa", "united states", "america", "american", "dollar",
                "irs", "federal", "401k", "roth ira", "social security"],
        "india": ["india", "indian", "rupee", "delhi", "mumbai", "bangalore",
                  "inr", "80c", "epf", "ppf", "nps", "sebi"],
        "australia": ["australia", "australian", "aud", "sydney", "melbourne",
                      "ato", "superannuation", "medicare levy", "asx"],
    }

    detected = None
    for country, keywords in country_keywords.items():
        if any(kw in last_message for kw in keywords):
            detected = country
            break

    # If already detected before, keep it
    if not detected and state.get("country"):
        detected = state["country"]

    # If still not detected, ask LLM
    if not detected:
        response = llm.invoke([
            {
                "role": "system",
                "content": (
                    "You are a country detector. Based on the user message, "
                    "return ONLY one word: germany, usa, india, australia, or unknown. "
                    "No explanation, just the single word."
                )
            },
            {"role": "user", "content": last_message}
        ])
        detected = response.content.strip().lower()
        if detected not in ["germany", "usa", "india", "australia"]:
            detected = "unknown"

    return {**state, "country": detected}


# ── Node 2 ──────────────────────────────────────────────
def load_user_profile(state: AgentState) -> AgentState:
    """
    Checks if user profile is complete.
    If not, flags what is missing so the agent can ask.
    """
    profile = state.get("user_profile", {})

    # Set country from state if not already in profile
    if not profile.get("country") and state.get("country"):
        profile["country"] = state["country"]

    return {**state, "user_profile": profile}


# ── Node 3 ──────────────────────────────────────────────
def classify_intent(state: AgentState) -> AgentState:
    """
    Classifies what the user wants.
    Options: market, tax, investment, education, calculation, goal, unknown
    """
    last_message = state["messages"][-1].content.lower()

    intent_keywords = {
        "market": [
            "stock price", "share price", "current price", "exchange rate",
            "forex", "currency rate", "etf price", "market price",
            "how much is", "price of", "value of", "trading at",
            "apple stock", "microsoft stock", "google stock", "tesla stock",
            "eur to", "usd to", "inr to", "aud to", "dollar to", "euro to",
            "what is the price", "current rate", "live price", "today's price",
            "nvidia", "amazon", "meta stock", "sap stock"
        ],
        "tax": [
            "tax", "steuer", "steuerklasse", "freelancer", "vat",
            "umsatzsteuer", "einkommensteuer", "ehegattensplitting",
            "steuererklarung", "80c", "section 80", "ato", "irs",
            "filing", "taxable", "deduction", "refund",
            "kleinunternehmer", "gewerbesteuer", "finanzamt",
            "how much tax", "tax bracket", "tax return", "tax rate"
        ],
        "investment": [
            "should i invest", "where to invest", "best investment",
            "invest in", "portfolio", "etf", "fund", "shares",
            "crypto", "returns", "riester", "nps", "super",
            "dividend", "bonds", "index fund", "s&p", "dax", "sensex",
            "ppf", "mutual fund", "superannuation", "roth ira", "401k"
        ],
        "education": [
            "what is", "explain", "how does", "tell me about",
            "difference between", "define", "meaning of", "what are",
            "how do i", "teach me", "i want to learn", "i dont understand",
            "what does", "can you explain", "help me understand"
        ],
        "calculation": [
            "calculate", "compound", "interest",
            "savings", "budget", "afford", "estimate", "compute",
            "how many years", "monthly payment", "total amount",
            "how much will i have", "future value"
        ],
        "goal": [
            "goal", "save for", "target", "plan", "retirement",
            "house", "car", "emergency fund", "dream", "future",
            "in 5 years", "in 10 years", "financial freedom",
            "i want to save", "i want to buy"
        ],
    }

    detected_intent = "unknown"
    for intent, keywords in intent_keywords.items():
        if any(kw in last_message for kw in keywords):
            detected_intent = intent
            break

    # Tax disclaimer only for tax questions
    tax_disclaimer_needed = detected_intent == "tax"

    return {
        **state,
        "intent": detected_intent,
        "tax_disclaimer_needed": tax_disclaimer_needed
    }


# ── Node 4 ──────────────────────────────────────────────
def generate_response(state: AgentState) -> AgentState:
    """
    Generates the final response using OpenAI.
    Uses country rules, user profile, and live market tools.
    """
    country = state.get("country", "unknown")
    intent = state.get("intent", "unknown")
    profile = state.get("user_profile", {})

    # Load country rules if available
    country_context = ""
    if country in COUNTRY_RULES:
        rules = COUNTRY_RULES[country]
        country_context = (
            f"Country: {country.upper()}\n"
            f"Currency: {rules.get('currency', 'unknown')}\n"
            f"Summary: {rules.get('summary', '')}\n"
        )

        # Add tax brackets to context
        brackets = (
            rules.get("tax_brackets") or
            rules.get("tax_brackets_single") or
            rules.get("tax_brackets_new_regime")
        )
        if brackets:
            country_context += "Tax Brackets:\n"
            for b in brackets:
                max_val = f"{b['max']:,}" if b.get("max") else "and above"
                country_context += (
                    f"  - {rules['currency']} {b['min']:,} to {max_val}: "
                    f"{int(b['rate']*100)}%\n"
                )

        # Add investment vehicles
        if rules.get("investment_vehicles"):
            country_context += (
                "Investment Options: " +
                ", ".join(rules["investment_vehicles"]) + "\n"
            )

        # Add freelancer info if relevant
        if intent == "tax" and rules.get("freelancer"):
            fl = rules["freelancer"]
            country_context += "Freelancer Rules:\n"
            for k, v in fl.items():
                if not isinstance(v, dict):
                    country_context += f"  - {k}: {v}\n"

    # Build system prompt
    system_prompt = f"""You are a personalised financial advisor and tax preparation assistant.
You help users understand their financial situation and make better money decisions.

{country_context}

User Profile:
- Country: {country}
- Employment type: {profile.get('employment_type', 'not specified')}
- Family situation: {profile.get('family_situation', 'not specified')}
- Age: {profile.get('age', 'not specified')}
- Monthly income: {profile.get('monthly_income', 'not specified')}
- Risk tolerance: {profile.get('risk_tolerance', 'not specified')}

Intent detected: {intent}

Instructions:
- Give specific, personalised advice based on the user's country and profile
- Always reference the correct local tax laws and investment rules
- Use the correct currency for the user's country
- Keep answers clear, structured, and practical
- If the user's country is unknown, ask them which country they are in
- Never make up numbers — use only the tax rules provided above
- For stock prices, ETF prices, and forex rates — ALWAYS use the available tools
- Never say you do not have real-time data — you have tools for live market data
- When user asks about any stock, ETF, or currency rate — call the tool immediately
- For tax calculations — ALWAYS use the tax calculator tools
- For Steuerklasse questions — use advise_steuerklasse tool
- For Steuerberater preparation — use generate_steuerberater_checklist tool
- For freelancer tax — use calculate_freelancer_tax tool
- For married couple tax — use calculate_couple_tax tool
- For single parent tax — use calculate_single_parent_tax tool
"""

    # Add tax disclaimer instruction if needed
    if state.get("tax_disclaimer_needed"):
        system_prompt += """
IMPORTANT: You must end your response with this exact disclaimer on a new line:

⚠️ Disclaimer: This information is for educational purposes only and does not
constitute professional tax advice. Please consult a licensed Steuerberater
(Germany), CPA (USA), CA (India), or registered tax agent (Australia) for
advice specific to your situation.
"""

    # Build message history for LLM
    message_history = []
    for m in state["messages"]:
        if hasattr(m, 'type'):
            role = "assistant" if m.type == "ai" else "user"
        else:
            role = "user"
        message_history.append({"role": role, "content": m.content})

    # Generate response with tools
    response = llm_with_tools.invoke([
        {"role": "system", "content": system_prompt},
        *message_history
    ])

    # If agent called a tool, execute it and get result
    tool_result = ""
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            # Execute the right tool
            # Execute the right tool
            tools_map = {
                # Market tools
                "get_stock_price": get_stock_price,
                "get_forex_rate": get_forex_rate,
                "get_etf_price": get_etf_price,
                "get_financial_news": get_financial_news,
                # Tax tools
                "calculate_employee_tax": calculate_employee_tax,
                "calculate_freelancer_tax": calculate_freelancer_tax,
                "calculate_couple_tax": calculate_couple_tax,
                "calculate_single_parent_tax": calculate_single_parent_tax,
                "advise_steuerklasse": advise_steuerklasse,
                "generate_steuerberater_checklist": generate_steuerberater_checklist,
                # Document tools
                "analyze_salary_slip": analyze_salary_slip,
                "analyze_bank_statement": analyze_bank_statement,
            }

            if tool_name in tools_map:
                tool_result = tools_map[tool_name].invoke(tool_args)

        # If tool was called, ask LLM to summarise the result
        if tool_result:
            final_response = llm.invoke([
                {"role": "system", "content": system_prompt},
                *message_history,
                {
                    "role": "assistant",
                    "content": f"Tool result: {tool_result}"
                },
                {
                    "role": "user",
                    "content": "Based on this live data, give me a clear and helpful answer."
                }
            ])
            response = final_response

    new_message = AIMessage(content=response.content)

    return {**state, "messages": state["messages"] + [new_message]}