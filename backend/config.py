"""
Central configuration file.
All environment variables and constants live here.
Every other file imports from this — never read .env directly.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI ───────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_CHAT = "gpt-4o-mini"        # fast + cheap — for all general Q&A
OPENAI_MODEL_DOCS = "gpt-4o"             # powerful — for salary slip + bank statement
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"

# ── Pinecone ─────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = "financial-literacy-chatbot"

# ── LangSmith ────────────────────────────────────────────
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "financial-chatbot")

# ── Alpha Vantage (Phase 2) ──────────────────────────────
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

# ── NewsAPI (Phase 2) ────────────────────────────────────
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_BASE_URL = "https://newsapi.org/v2"

# ── Supabase (Phase 3) ───────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ── App settings ─────────────────────────────────────────
APP_TITLE = "Financial Literacy & Tax Advisor Chatbot"
APP_VERSION = "1.0.0"
SUPPORTED_COUNTRIES = ["germany", "usa", "india", "australia"]
MAX_CONVERSATION_HISTORY = 10

# ── Model routing logic ──────────────────────────────────
# Use this function to decide which model to use for each task
def get_model_for_task(task: str) -> str:
    """
    Returns the correct OpenAI model based on the task type.

    Args:
        task: 'chat', 'tax', 'investment', 'document', 'salary_slip', 'bank_statement'

    Returns:
        OpenAI model string
    """
    document_tasks = ["document", "salary_slip", "bank_statement", "pdf"]

    if task in document_tasks:
        return OPENAI_MODEL_DOCS      # gpt-4o — powerful for long documents
    return OPENAI_MODEL_CHAT          # gpt-4o-mini — fast and cheap for everything else


# ── Validation — warn if critical keys are missing ───────
def validate_config():
    """
    Checks all required environment variables are set.
    Prints warning for missing keys, success if all present.
    """
    required = {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "PINECONE_API_KEY": PINECONE_API_KEY,
        "PINECONE_ENVIRONMENT": PINECONE_ENVIRONMENT,
        "LANGCHAIN_API_KEY": LANGCHAIN_API_KEY,
    }

    optional = {
        "ALPHA_VANTAGE_API_KEY": ALPHA_VANTAGE_API_KEY,   # Phase 2
        "NEWS_API_KEY": NEWS_API_KEY,                       # Phase 2
        "SUPABASE_URL": SUPABASE_URL,                       # Phase 3
        "SUPABASE_KEY": SUPABASE_KEY,                       # Phase 3
    }

    # Check required keys
    missing_required = [k for k, v in required.items() if not v]
    if missing_required:
        print(f"❌ Missing REQUIRED keys: {', '.join(missing_required)}")
    else:
        print("✅ All required environment variables loaded.")

    # Check optional keys
    missing_optional = [k for k, v in optional.items() if not v]
    if missing_optional:
        print(f"⚠️  Missing optional keys (needed later): {', '.join(missing_optional)}")
    else:
        print("✅ All optional environment variables loaded.")


validate_config()