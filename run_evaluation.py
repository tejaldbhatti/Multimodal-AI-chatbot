"""
RAGAs evaluation script for the Financial Chatbot.
Tests answer quality across all intents and countries.
"""
import os
import json
from datetime import datetime
from langsmith import Client
from langchain_core.messages import HumanMessage
from backend.agents.graph import financial_graph

# Initialize LangSmith client
client = Client()

# ── Test cases ───────────────────────────────────────────
TEST_CASES = [
    # Germany tax
    {
        "question": "I am a freelancer in Germany earning 60000 euros. How much tax?",
        "expected_keywords": ["11,074", "freelancer", "steuer", "disclaimer"],
        "country": "germany",
        "intent": "tax",
        "category": "Germany Tax"
    },
    {
        "question": "My husband earns 80000 and I earn 30000 in Germany. Which Steuerklasse?",
        "expected_keywords": ["III", "V", "ehegattensplitting", "2,383"],
        "country": "germany",
        "intent": "tax",
        "category": "Germany Couple Tax"
    },
    {
        "question": "I am a single parent in Germany with 2 kids earning 45000. Tax?",
        "expected_keywords": ["steuerklasse II", "entlastungsbetrag", "1,080"],
        "country": "germany",
        "intent": "tax",
        "category": "Germany Single Parent"
    },
    # USA tax
    {
        "question": "I live in USA and earn $80,000. What is my tax bracket?",
        "expected_keywords": ["22%", "47,150", "80,000"],
        "country": "usa",
        "intent": "tax",
        "category": "USA Tax"
    },
    # India tax
    {
        "question": "I live in India. Explain Section 80C to me.",
        "expected_keywords": ["80C", "1.5", "PPF", "ELSS"],
        "country": "india",
        "intent": "education",
        "category": "India Education"
    },
    # Australia
    {
        "question": "I live in Australia. Explain superannuation.",
        "expected_keywords": ["superannuation", "11%", "retirement"],
        "country": "australia",
        "intent": "education",
        "category": "Australia Education"
    },
    # Market data
    {
        "question": "What is the current Apple stock price?",
        "expected_keywords": ["AAPL", "$", "price", "volume"],
        "country": "unknown",
        "intent": "market",
        "category": "Market Data"
    },
    {
        "question": "What is the EUR to USD exchange rate today?",
        "expected_keywords": ["EUR", "USD", "rate", "1."],
        "country": "usa",
        "intent": "market",
        "category": "Forex"
    },
    # Investment
    {
        "question": "I live in India. Should I invest in ETFs or PPF?",
        "expected_keywords": ["ETF", "PPF", "80C", "risk"],
        "country": "india",
        "intent": "investment",
        "category": "India Investment"
    },
    {
        "question": "I live in Germany. What are the best ETF options?",
        "expected_keywords": ["ETF", "freistellungsauftrag", "1,000"],
        "country": "germany",
        "intent": "investment",
        "category": "Germany Investment"
    },
]


def run_single_test(test_case: dict) -> dict:
    """
    Runs a single test case through the LangGraph agent.
    Returns results with pass/fail for each keyword.
    """
    print(f"\n{'─'*50}")
    print(f"Testing: {test_case['category']}")
    print(f"Question: {test_case['question'][:60]}...")

    try:
        result = financial_graph.invoke({
            "messages": [HumanMessage(content=test_case["question"])],
            "user_profile": {},
            "country": None,
            "intent": None,
            "tax_disclaimer_needed": False,
            "context": None,
        })

        response      = result["messages"][-1].content.lower()
        country       = result.get("country", "unknown")
        intent        = result.get("intent", "unknown")

        # Check keywords
        keyword_results = {}
        for kw in test_case["expected_keywords"]:
            keyword_results[kw] = kw.lower() in response

        passed    = sum(keyword_results.values())
        total     = len(keyword_results)
        score     = passed / total if total > 0 else 0
        status    = "✅ PASS" if score >= 0.75 else "❌ FAIL"

        print(f"Country detected: {country} (expected: {test_case['country']})")
        print(f"Intent detected:  {intent} (expected: {test_case['intent']})")
        print(f"Keywords found:   {passed}/{total}")
        print(f"Score:            {score:.0%} {status}")

        return {
            "category":       test_case["category"],
            "question":       test_case["question"],
            "expected_country": test_case["country"],
            "detected_country": country,
            "expected_intent":  test_case["intent"],
            "detected_intent":  intent,
            "keyword_results":  keyword_results,
            "score":          score,
            "status":         status,
            "response_length": len(response),
        }

    except Exception as e:
        print(f"ERROR: {str(e)}")
        return {
            "category": test_case["category"],
            "question": test_case["question"],
            "score":    0,
            "status":   "❌ ERROR",
            "error":    str(e),
        }


def run_all_evaluations():
    """
    Runs all test cases and generates a report.
    """
    print("=" * 60)
    print("Financial Chatbot — RAGAs Style Evaluation")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Total test cases: {len(TEST_CASES)}")
    print("=" * 60)

    results = []
    for test_case in TEST_CASES:
        result = run_single_test(test_case)
        results.append(result)

    # ── Summary ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    total_score  = sum(r["score"] for r in results)
    avg_score    = total_score / len(results)
    passed_tests = sum(1 for r in results if r["score"] >= 0.75)

    print(f"\nTotal tests:  {len(results)}")
    print(f"Tests passed: {passed_tests}/{len(results)}")
    print(f"Average score: {avg_score:.1%}")
    print(f"Overall grade: {'✅ PASS' if avg_score >= 0.75 else '❌ NEEDS IMPROVEMENT'}")

    print("\n── Results by Category ──")
    for r in results:
        score_bar = "█" * int(r["score"] * 10) + "░" * (10 - int(r["score"] * 10))
        print(f"  {r['status']} {r['category']:<30} [{score_bar}] {r['score']:.0%}")

    # ── Save results ─────────────────────────────────────
    report = {
        "date":          datetime.now().isoformat(),
        "total_tests":   len(results),
        "passed":        passed_tests,
        "average_score": avg_score,
        "results":       results
    }

    report_path = f"evaluation/report_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Full report saved to: {report_path}")
    print("\n" + "=" * 60)

    return report


if __name__ == "__main__":
    run_all_evaluations()
