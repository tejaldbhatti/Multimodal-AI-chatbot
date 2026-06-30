"""
Chat router — handles all /chat endpoints.
Receives user messages and passes them through LangGraph.
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from langchain_core.messages import HumanMessage
from backend.agents.graph import financial_graph

router = APIRouter()


# ── Request/Response models ──────────────────────────────
class ChatRequest(BaseModel):
    message: str
    country: Optional[str] = None
    session_id: Optional[str] = None
    user_profile: Optional[dict] = {}


class ChatResponse(BaseModel):
    response: str
    country: str
    intent: str
    tax_disclaimer: bool


# ── Endpoints ────────────────────────────────────────────
@router.post("/")
def chat(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint.
    Receives a message, runs it through LangGraph, returns response.
    """
    result = financial_graph.invoke({
        "messages": [HumanMessage(content=request.message)],
        "user_profile": request.user_profile or {},
        "country": request.country,
        "intent": None,
        "tax_disclaimer_needed": False,
        "context": None,
    })

    return ChatResponse(
        response=result["messages"][-1].content,
        country=result.get("country", "unknown"),
        intent=result.get("intent", "unknown"),
        tax_disclaimer=result.get("tax_disclaimer_needed", False),
    )


@router.get("/health")
def chat_health():
    return {"status": "chat router ok"}