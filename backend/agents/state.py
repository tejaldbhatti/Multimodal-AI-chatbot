"""
State schema for the Financial Chatbot LangGraph agent.
This defines everything the agent remembers across all nodes.
"""
from typing import Annotated, Optional
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class UserProfile(TypedDict):
    """Stores the user's personal financial profile."""
    country: Optional[str]           # "germany", "usa", "india", "australia"
    employment_type: Optional[str]   # "employed", "freelancer", "both"
    family_situation: Optional[str]  # "single", "married_one_income", "married_both_income", "single_parent"
    age: Optional[int]
    monthly_income: Optional[float]
    risk_tolerance: Optional[str]    # "low", "medium", "high"


class AgentState(TypedDict):
    """
    Main state object passed between all LangGraph nodes.
    Every node reads from this and writes back to it.
    """
    messages: Annotated[list, add_messages]  # full conversation history
    user_profile: UserProfile                # who the user is
    country: Optional[str]                   # detected country
    intent: Optional[str]                    # what user wants
    tax_disclaimer_needed: bool              # True if tax advice given
    context: Optional[str]                   # extra context from tools