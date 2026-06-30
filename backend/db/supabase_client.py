"""Supabase database helpers."""

"""
Supabase client — handles all database operations.
User profiles, savings goals, and conversation history.
"""
from supabase import create_client, Client
from backend.config import SUPABASE_URL, SUPABASE_KEY
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ── User Profile Operations ──────────────────────────────

def get_user_profile(session_id: str) -> dict:
    """
    Gets user profile from Supabase.
    Returns empty dict if user not found.
    """
    try:
        result = supabase.table("user_profiles")\
            .select("*")\
            .eq("session_id", session_id)\
            .execute()

        if result.data:
            return result.data[0]
        return {}

    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        return {}


def save_user_profile(session_id: str, profile: dict) -> bool:
    """
    Saves or updates user profile in Supabase.
    Returns True if successful.
    """
    try:
        # Check if profile exists
        existing = get_user_profile(session_id)

        if existing:
            # Update existing profile
            supabase.table("user_profiles")\
                .update({**profile, "updated_at": "NOW()"})\
                .eq("session_id", session_id)\
                .execute()
        else:
            # Create new profile
            supabase.table("user_profiles")\
                .insert({"session_id": session_id, **profile})\
                .execute()

        logger.info(f"Profile saved for session: {session_id}")
        return True

    except Exception as e:
        logger.error(f"Error saving user profile: {e}")
        return False


# ── Savings Goals Operations ─────────────────────────────

def get_goals(session_id: str) -> list:
    """
    Gets all savings goals for a user.
    Returns empty list if none found.
    """
    try:
        result = supabase.table("savings_goals")\
            .select("*")\
            .eq("session_id", session_id)\
            .execute()

        return result.data or []

    except Exception as e:
        logger.error(f"Error getting goals: {e}")
        return []


def save_goal(session_id: str, goal_name: str,
              target_amount: float, target_date: str = None) -> bool:
    """
    Saves a new savings goal for a user.
    Returns True if successful.
    """
    try:
        supabase.table("savings_goals").insert({
            "session_id": session_id,
            "goal_name": goal_name,
            "target_amount": target_amount,
            "current_amount": 0,
            "target_date": target_date
        }).execute()

        logger.info(f"Goal saved: {goal_name} for session: {session_id}")
        return True

    except Exception as e:
        logger.error(f"Error saving goal: {e}")
        return False


def update_goal_progress(session_id: str,
                         goal_name: str,
                         current_amount: float) -> bool:
    """
    Updates progress on an existing savings goal.
    Returns True if successful.
    """
    try:
        supabase.table("savings_goals")\
            .update({"current_amount": current_amount,
                     "updated_at": "NOW()"})\
            .eq("session_id", session_id)\
            .eq("goal_name", goal_name)\
            .execute()

        return True

    except Exception as e:
        logger.error(f"Error updating goal: {e}")
        return False


# ── Conversation History Operations ──────────────────────

def save_message(session_id: str, role: str,
                 content: str, intent: str = None,
                 country: str = None) -> bool:
    """
    Saves a single message to conversation history.
    Returns True if successful.
    """
    try:
        supabase.table("conversation_history").insert({
            "session_id": session_id,
            "role": role,
            "content": content,
            "intent": intent,
            "country": country
        }).execute()

        return True

    except Exception as e:
        logger.error(f"Error saving message: {e}")
        return False


def get_conversation_history(session_id: str,
                             limit: int = 10) -> list:
    """
    Gets last N messages for a session.
    Returns empty list if none found.
    """
    try:
        result = supabase.table("conversation_history")\
            .select("*")\
            .eq("session_id", session_id)\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()

        # Reverse to get chronological order
        return list(reversed(result.data or []))

    except Exception as e:
        logger.error(f"Error getting history: {e}")
        return []