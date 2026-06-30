"""
Complete end-to-end test for Phase 1 + Phase 2.
Tests tax, market data, investment, and education intents.
"""
from langchain_core.messages import HumanMessage
from backend.agents.graph import financial_graph


def test_graph(user_message: str):
    print(f"\n{'='*50}")
    print(f"User: {user_message}")
    print('='*50)

    result = financial_graph.invoke({
        "messages": [HumanMessage(content=user_message)],
        "user_profile": {},
        "country": None,
        "intent": None,
        "tax_disclaimer_needed": False,
        "context": None,
    })

    print(f"Country  : {result['country']}")
    print(f"Intent   : {result['intent']}")
    print(f"\nResponse:\n{result['messages'][-1].content}")


# ── Test 1 — German freelancer tax ──────────────────────
test_graph("I live in Germany, I am a freelancer earning 60000 euros, how much tax?")

# ── Test 2 — Live stock price ────────────────────────────
test_graph("What is the current Apple stock price?")

# ── Test 3 — Live forex ──────────────────────────────────
test_graph("What is the exchange rate from EUR to USD today?")

# ── Test 4 — India investment ────────────────────────────
test_graph("I live in India, should I invest in ETFs or PPF?")

# ── Test 5 — USA tax ─────────────────────────────────────
test_graph("I live in USA and earn $80,000, what is my tax bracket?")

# ── Test 6 — Australia superannuation ───────────────────
test_graph("I live in Australia, explain superannuation to me")