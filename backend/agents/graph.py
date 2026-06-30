"""
LangGraph StateGraph for the Financial Chatbot.
Connects all nodes into a complete conversation flow.
"""
from langgraph.graph import StateGraph, END
from backend.agents.state import AgentState
from backend.agents.nodes import (
    detect_country,
    load_user_profile,
    classify_intent,
    generate_response,
)


def route_intent(state: AgentState) -> str:
    """
    Decides which node to go to after intent classification.
    All intents currently go to generate_response.
    In Phase 3 we add document analysis routing here.
    """
    intent = state.get("intent", "unknown")

    routing = {
        "market":      "generate_response",
        "tax":         "generate_response",
        "investment":  "generate_response",
        "education":   "generate_response",
        "calculation": "generate_response",
        "goal":        "generate_response",
        "unknown":     "generate_response",
    }

    return routing.get(intent, "generate_response")


def build_graph():
    """
    Builds and compiles the LangGraph StateGraph.
    Returns a runnable graph.
    """
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("detect_country",    detect_country)
    graph.add_node("load_user_profile", load_user_profile)
    graph.add_node("classify_intent",   classify_intent)
    graph.add_node("generate_response", generate_response)

    # Define the flow
    graph.set_entry_point("detect_country")

    graph.add_edge("detect_country",    "load_user_profile")
    graph.add_edge("load_user_profile", "classify_intent")

    # Conditional routing after intent classification
    graph.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "generate_response": "generate_response",
        }
    )

    graph.add_edge("generate_response", END)

    return graph.compile()


# Build the graph once — reused across all requests
financial_graph = build_graph()