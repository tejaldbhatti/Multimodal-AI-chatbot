"""
This module initializes the core components of the financial literacy chatbot,
including OpenAI LLM, embeddings, Pinecone vector store for retrieval,
conversation memory, and a suite of specialized tools. It sets up the LangChain
agent responsible for orchestrating interactions and tool usage.
"""

import logging
import os
import re
import sys
from typing import List, Optional, Dict, Tuple, AsyncGenerator # Added AsyncGenerator

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# LangChain imports
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import Tool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage # Added SystemMessage import
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Pinecone Imports
from pinecone import Pinecone, PineconeApiException

# OpenAI specific error for authentication
from openai import AuthenticationError as OpenAIAuthenticationError
from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import APIStatusError as OpenAIAPIStatusError

# Load environment variables from .env file
load_dotenv()

# --- Configure logging ---
# Changed level to WARNING for production to reduce log overhead and improve speed.
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration Constants ---
PINECONE_INDEX_NAME = "financial-literacy-chatbot"
MEMORY_WINDOW_SIZE = 7
GPT_MODEL_NAME = "gpt-4o-mini"
GPT_TEMPERATURE = 0.7
DEFAULT_RETIREMENT_AGE = 65
COMMON_SAVINGS_TARGET_PERCENT = 0.20 # 20% for net pay savings
ESTIMATED_RETIREMENT_INCOME_MULTIPLIER = 25 # Common rule of thumb (25x desired income)
DEDUCTION_THRESHOLD_PERCENT = 30 # Threshold for high deductions in salary slip analysis

# --- Error Handling Helper ---
def exit_on_error(log_message: str, exit_message: str, exception: Optional[Exception] = None) -> None:
    """Logs an error and exits the program."""
    if exception:
        logging.error(log_message, exc_info=True)
    else:
        logging.error(log_message)
    sys.exit(exit_message)

# --- Initialize OpenAI credentials ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    exit_on_error(
        "Missing OPENAI_API_KEY in .env.",
        "OPENAI_API_KEY not set. Exiting."
    )

# --- Pinecone Configuration for Retrieval ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    exit_on_error(
        "Pinecone API key or environment not set. Please add "
        "PINECONE_API_KEY and PINECONE_ENVIRONMENT to your .env file.",
        "Exiting: Pinecone credentials missing for chatbot app."
    )

# --- Connect to Pinecone Vector Store and Initialize Embeddings ---
VECTORSTORE = None
RETRIEVER = None
EMBEDDINGS_MODEL = None

try:
    EMBEDDINGS_MODEL = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    P_CLIENT = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

    # In the chatbot backend, we assume the index already exists and is populated.
    # The data loader script is responsible for creating and populating it.
    # We just need to connect to it and get the retriever.

    # Check if index exists before trying to connect
    index_names = [index["name"] for index in P_CLIENT.list_indexes()]
    if PINECONE_INDEX_NAME not in index_names:
        exit_on_error(
            f"Pinecone index '{PINECONE_INDEX_NAME}' does not exist. "
            "Please run your data loader script first to create and populate it.",
            "Exiting: Pinecone index not found for chatbot operation."
        )

    # Initialize vectorstore and retriever if the index exists
    VECTORSTORE = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=EMBEDDINGS_MODEL
    )
    # You can specify 'k' here to limit the number of retrieved documents,
    # which can impact speed. For example:
    # RETRIEVER = VECTORSTORE.as_retriever(search_kwargs={"k": 3})
    RETRIEVER = VECTORSTORE.as_retriever()

except OpenAIAuthenticationError as err:
    exit_on_error(
        f"OpenAI Authentication Error: Your OpenAI API key is invalid or expired. "
        f"Please check your OPENAI_API_KEY in the .env file. Details: {err}",
        "Exiting: OpenAI Authentication Failed.",
        exception=err
    )
except OpenAIAPIConnectionError as err:
    exit_on_error(
        f"OpenAI API Connection Error: Could not connect to OpenAI API. "
        f"Check your internet connection or OpenAI service status. Details: {err}",
        "Exiting: OpenAI Connection Failed.",
        exception=err
    )
except OpenAIAPIStatusError as err:
    exit_on_error(
        f"OpenAI API Error (Status Code: {err.status_code}): {err.response} Details: {err}",
        "Exiting: OpenAI API Error.",
        exception=err
    )
except PineconeApiException as err:
    exit_on_error(
        f"Pinecone API Error: Check your PINECONE_API_KEY, PINECONE_ENVIRONMENT, "
        f"and index name. Details: {err}",
        "Exiting: Pinecone API Failed.",
        exception=err
    )
except Exception as err: # pylint: disable=broad-except
    # Catching broad exception here for robust initialization failure handling.
    exit_on_error(
        f"An unexpected error occurred during Pinecone/Embeddings initialization: {err}",
        "Exiting: Initialization failed.",
        exception=err
    )

# --- Initialize LLM, Memory, and Agent Prompt ---
LLM = None
try:
    LLM = ChatOpenAI(
        model=GPT_MODEL_NAME,
        temperature=GPT_TEMPERATURE,
        openai_api_key=OPENAI_API_KEY,
        streaming=True
    )
except OpenAIAuthenticationError as err:
    exit_on_error(
        f"OpenAI LLM Authentication Error: Your OpenAI API key is invalid or expired. "
        f"Please check your OPENAI_API_KEY in the .env file. Details: {err}",
        "Exiting: LLM Authentication Failed.",
        exception=err
    )
except OpenAIAPIConnectionError as err:
    exit_on_error(
        f"OpenAI LLM API Connection Error: Could not connect to OpenAI API. "
        f"Check your internet connection or OpenAI service status. Details: {err}",
        "Exiting: LLM Connection Failed.",
        exception=err
    )
except OpenAIAPIStatusError as err:
    exit_on_error(
        f"OpenAI LLM API Error (Status Code: {err.status_code}): {err.response} Details: {err}",
        "Exiting: LLM API Error.",
        exception=err
    )
except Exception as err: # pylint: disable=broad-except
    # Catching broad exception here for robust initialization failure handling.
    exit_on_error(
        f"Failed to initialize ChatOpenAI LLM: {err}",
        "Exiting: LLM could not be initialized.",
        exception=err
    )

# --- Conversation Memory ---
MEMORY = ConversationBufferWindowMemory(
    k=MEMORY_WINDOW_SIZE,
    memory_key="chat_history",
    return_messages=True,
    output_key="output"
)

# --- Agent Prompt Template ---
AGENT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=(
        "You are a highly knowledgeable and helpful financial literacy assistant. "
        "Your primary goal is to provide accurate, factual, and concise answers "
        "related to financial topics. "
        "For any factual questions, you MUST use the 'KnowledgeBaseQuery' tool "
        "to retrieve information from the knowledge base. "
        "Always prioritize and synthesize information retrieved from your tools "
        "(e.g., KnowledgeBaseQuery, compound interest calculator) for factual "
        "questions and computations. "
        "When a tool provides relevant information, you MUST use that information "
        "to formulate a direct, coherent, and concise answer to the user's "
        "question. Do NOT simply repeat tool outputs verbatim. "
        "After answering, if you used the 'KnowledgeBaseQuery' tool, explicitly "
        "state 'Source: [filename]' for each unique source document used, at the "
        "end of your complete answer. "
        "DO NOT make up information or answer outside the scope of what the "
        "tools can provide. "
        "If, after using ALL relevant tools, you genuinely cannot find enough "
        "information to confidently and completely answer the question based on "
        "the tool outputs, then, and only then, state clearly: "
        "'I cannot answer this question based on the information I have available.'"
    )),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# --- Region Settings ---
REGION_SETTINGS: Dict[str, Dict] = {
    "us": {
        "currency": "$",
        "retirement_age": 65,
        "tax_brackets": [
            (0.10, 9950), (0.12, 40525), (0.22, 86375), (0.24, 164925),
            (0.32, 209425), (0.35, 523600), (0.37, float('inf'))
        ],
        "investment_tips": (
            "In the United States, consider diversified portfolios with ETFs, "
            "mutual funds, and possibly individual stocks. Explore retirement "
            "accounts like 401(k)s and IRAs, and look into diverse US stock "
            "funds. Tech and innovation sectors are often prominent."
        ),
        "savings_recommendation": "Aim to save at least 15% of your income for retirement.",
        "countries": ["united states", "us", "usa", "america"]
    },
    "uk": {
        "currency": "£",
        "retirement_age": 66,
        "tax_brackets": [(0.20, 12570), (0.40, 50270), (0.45, float('inf'))],
        "investment_tips": (
            "In the UK, focus on stable, blue-chip companies, diversified ETFs, "
            "and consider pan-European funds. Real estate can also be a strong "
            "investment. Consider ISAs and pensions with tax relief."
        ),
        "savings_recommendation": "Try to save at least 10-15% of your income for retirement.",
        "countries": ["united kingdom", "uk", "britain", "england", "scotland",
                      "wales", "northern ireland"]
    },
    "germany": {
        "currency": "€",
        "retirement_age": 67,
        "tax_brackets": [
            (0.00, 9744), (0.14, 57918), (0.42, 274612), (0.45, float('inf'))
        ],
        "investment_tips": (
            "In Germany, consider Riester pension, ETFs, and private pension "
            "plans. Be aware of varying tax regulations."
        ),
        "savings_recommendation": "Aim to save around 10-15% of your income for retirement.",
        "countries": ["germany", "deutschland"]
    },
    "global": { # Default fallback
        "currency": "",
        "retirement_age": DEFAULT_RETIREMENT_AGE,
        "tax_brackets": [],
        "investment_tips": (
            "For general investment advice, diversification across asset classes "
            "(stocks, bonds, real estate) and geographies is key. "
            "Consider low-cost index funds or ETFs. Consult a financial advisor "
            "for personalized guidance."
        ),
        "savings_recommendation": (
            "A common guideline is to aim for 8-12 times your annual salary by "
            "retirement."
        ),
        "countries": []
    }
}

# --- Helper function to detect region from user input ---
def detect_region(text: str) -> str:
    """
    Detects the user's region based on keywords in the input text.

    Args:
        text (str): The input text from the user.

    Returns:
        str: The detected region key (e.g., "us", "uk", "germany", or "global").
    """
    text_lower = text.lower()
    for region_key, region_data in REGION_SETTINGS.items():
        if region_key != "global":
            if region_key in text_lower or \
               any(country.lower() in text_lower for country in region_data.get("countries", [])):
                return region_key
    return "global"

# --- Define Tools ---

# Pydantic Schema for KnowledgeBaseQuery
class KnowledgeBaseQuerySchema(BaseModel):
    """Schema for the KnowledgeBaseQuery tool."""
    query: str = Field(..., description="The question to ask the financial knowledge base.")

async def query_knowledge_base(query: str) -> Tuple[str, List[str]]:
    """
    Answer financial questions using the knowledge base.
    Returns a tuple: (LLM's response based on context, list of unique source filenames).
    """
    if not RETRIEVER:
        logging.warning("KnowledgeBaseQuery called but retriever is not initialized. "
                        "RAG will not work.")
        return ("Knowledge base is not available. Please ensure the Pinecone index "
                "is set up correctly.", [])
    try:
        # Retrieve documents from the vector store
        docs = RETRIEVER.invoke(query)

        retrieved_sources = [
            doc.metadata.get('source', 'No source info') for doc in docs
        ]

        context = "\n\n".join([doc.page_content for doc in docs])

        # Craft a prompt for the LLM to answer using the retrieved context
        rag_prompt_content = (
            f"Based on the following context, answer the user's question. "
            f"If the context does not contain enough information, state that "
            f"you cannot answer based on the provided information.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}"
        )

        # Use ainvoke for asynchronous LLM call
        response = await LLM.ainvoke(rag_prompt_content)

        # Return the LLM's content and the unique sources
        unique_sources = sorted(list(set(retrieved_sources)))
        return response.content, unique_sources
    except OpenAIAuthenticationError as err:
        logging.error("OpenAI Authentication Error in KnowledgeBaseQuery: %s", err, exc_info=True)
        return ("Sorry, there was an authentication issue with the knowledge base. "
                "Please check your OpenAI API key.", [])
    except OpenAIAPIConnectionError as err:
        logging.error("OpenAI API Connection Error in KnowledgeBaseQuery: %s", err, exc_info=True)
        return ("Sorry, I couldn't connect to the knowledge base. "
                "Please check your internet connection.", [])
    except OpenAIAPIStatusError as err:
        logging.error(
            "OpenAI API Error in KnowledgeBaseQuery (Status Code: %d): %s Details: %s",
            err.status_code, err.response, err, exc_info=True
        )
        return "Sorry, an OpenAI API error occurred while querying the knowledge base.", []
    except Exception as err: # pylint: disable=broad-except
        logging.error(
            "An unexpected error occurred in query_knowledge_base tool execution: %s",
            err, exc_info=True
        )
        return "Sorry, I couldn't process your knowledge base query at the moment.", []

# Pydantic Schema for SavingsRecommendation
class SavingsRecommendationSchema(BaseModel):
    """Schema for the SavingsRecommendation tool."""
    income: float = Field(..., description="The user's income.")
    spending: float = Field(..., description="The user's spending.")
    region: str = Field("global", description="The user's region (e.g., 'United States', 'UK', 'Germany').")

def recommend_savings(income: float, spending: float, region: str = "global") -> str:
    """
    Provide personalized saving recommendations based on income and spending.

    Args:
        income (float): The user's income.
        spending (float): The user's spending.
        region (str): The user's region (e.g., 'United States', 'UK', 'Germany').
                      Defaults to "global".

    Returns:
        str: A string containing personalized savings recommendations.
    """
    logging.info("Tool 'recommend_savings' called with input: %f income, %f spending, %s region",
                 income, spending, region)
    try:
        if income < 0 or spending < 0:
            return "Income and spending must be non-negative values."
        if spending > income:
            return "Your spending exceeds your income. Consider reducing expenses or " \
                   "increasing income."

        actual_region_data = REGION_SETTINGS.get(detect_region(region), REGION_SETTINGS["global"])
        currency = actual_region_data.get("currency", "$")
        savings_recommendation_text = actual_region_data.get(
            "savings_recommendation",
            "Aim to save a significant portion of your income."
        )

        savings_amount = income - spending
        if savings_amount <= 0:
            return (
                f"Based on your income ({currency}{income:,.2f}) and spending "
                f"({currency}{spending:,.2f}), you are not currently saving. "
                "Consider reviewing your expenses to find areas where you can "
                "reduce spending and start saving."
            )
        if savings_amount < income * 0.1:
            return (
                f"You are saving {currency}{savings_amount:,.2f} per period. "
                "To build a stronger financial future, aim to save at least "
                "10-20% of your income. "
                f"{savings_recommendation_text}"
            )
        return (
            f"You are saving {currency}{savings_amount:,.2f} per period, "
            f"which is a good start. Keep up the great work! "
            f"{savings_recommendation_text} Consider automating your savings."
        )
    except Exception as err: # pylint: disable=broad-except
        logging.error("Error in recommend_savings: %s", err, exc_info=True)
        return "Sorry, I couldn't process savings recommendations. " \
               "Please provide valid numbers for income and spending."

def budgeting_templates() -> str:
    """
    Provides information on various popular budgeting templates.

    Returns:
        str: A string describing different budgeting methods.
    """
    return (
        "Here are some popular budgeting templates:\n"
        "- **50/30/20 Rule:** 50% needs, 30% wants, 20% savings/debt repayment.\n"
        "- **Zero-Based Budgeting:** Every dollar is assigned a purpose.\n"
        "- **Envelope System:** Allocate cash into envelopes for different "
        "spending categories.\n"
        "- **Paycheck to Paycheck Budgeting:** Focuses on managing funds until "
        "the next paycheck.\n"
        "Which one would you like to know more about, or would you like to see "
        "a general template?"
    )

def credit_score_advice() -> str:
    """
    Gives advice on improving credit score.

    Returns:
        str: A string containing tips to improve credit score.
    """
    return (
        "Tips to improve your credit score:\n"
        "- Pay your bills on time.\n"
        "- Keep your credit utilization below 30%.\n"
        "- Avoid opening too many new accounts quickly.\n"
        "- Regularly check your credit report for errors.\n"
        "Ask me for specific credit-related questions anytime!"
    )

def investment_advice(region: str = "global") -> str:
    """
    Provide investment tips based on user's region.

    Args:
        region (str): The user's region (e.g., "United States", "Germany", "Japan").
                      Defaults to "global".

    Returns:
        str: A string containing investment advice for the detected region.
    """
    detected_region_key = detect_region(region)
    advice_text = REGION_SETTINGS.get(
        detected_region_key, REGION_SETTINGS["global"]
    )["investment_tips"]
    return advice_text

# Pydantic Schema for RetirementPlanning
class RetirementPlanningSchema(BaseModel):
    """Schema for the RetirementPlanning tool."""
    age: Optional[int] = Field(None, description="The user's current age.")
    current_savings: Optional[float] = Field(
        None, description="The user's current retirement savings."
    )
    desired_income_retirement: Optional[float] = Field(
        None, description="The user's desired annual income in retirement."
    )
    region: str = Field(
        "global", description="The user's region (e.g., 'US', 'UK', 'Germany')."
    )

def retirement_planning(
    age: Optional[int] = None,
    current_savings: Optional[float] = None,
    desired_income_retirement: Optional[float] = None,
    region: str = "global"
) -> str:
    """
    Advise on retirement age and savings.

    Can take optional parameters: age (int), current_savings (float),
    desired_income_retirement (float), and region (str).

    Args:
        age (Optional[int]): The user's current age.
        current_savings (Optional[float]): The user's current retirement savings.
        desired_income_retirement (Optional[float]): The user's desired annual
                                                      income in retirement.
        region (str): The user's region. Defaults to "global".

    Returns:
        str: A string containing retirement planning advice.
    """
    advice: List[str] = []
    detected_region_key = detect_region(region)
    actual_region_data = REGION_SETTINGS.get(detected_region_key, REGION_SETTINGS["global"])

    retirement_age_default = actual_region_data.get("retirement_age", DEFAULT_RETIREMENT_AGE)
    savings_tip = actual_region_data.get(
        "savings_recommendation",
        "Aim to save consistently for retirement."
    )
    currency = actual_region_data.get("currency", "$")

    if age is None:
        advice.append(
            "Start planning for retirement as early as possible, ideally in your "
            "20s or 30s, to maximize the benefit of compound interest."
        )
    elif age < 30:
        advice.append(
            f"At your age ({age}), you have a great head start for retirement "
            "planning. Focus on consistent contributions to a diversified portfolio."
        )
    elif age < 50:
        advice.append(
            f"At your age ({age}), it's crucial to be actively saving for retirement. "
            "Review your contributions and investment strategy regularly."
        )
    else:
        advice.append(
            f"At your age ({age}), accelerate your retirement savings. "
            "Consider catch-up contributions to retirement accounts if eligible."
        )

    advice.append(
        f"In your region ({detected_region_key.upper()}), the typical retirement age "
        f"is {retirement_age_default}."
    )
    advice.append(savings_tip)

    if current_savings is not None and desired_income_retirement is not None:
        estimated_needed = desired_income_retirement * ESTIMATED_RETIREMENT_INCOME_MULTIPLIER
        if current_savings < estimated_needed * 0.2: # 20% of goal
            advice.append(
                f"With current savings of {currency}{current_savings:,.2f} towards a goal "
                f"requiring approximately {currency}{estimated_needed:,.2f}, you may "
                "need to significantly increase your savings rate."
            )
        elif current_savings < estimated_needed * 0.5: # 50% of goal
            advice.append(
                f"With current savings of {currency}{current_savings:,.2f} towards a goal "
                f"requiring approximately {currency}{estimated_needed:,.2f}, you are on "
                "your way but consider increasing contributions to meet your desired "
                "retirement income."
            )
        else:
            advice.append(
                f"With current savings of {currency}{current_savings:,.2f} towards a goal "
                f"requiring approximately {currency}{estimated_needed:,.2f}, you are "
                "making excellent progress towards your retirement income goals!"
            )
    else:
        advice.append(
            "Provide your current savings and desired retirement income "
            "for a more personalized assessment."
        )

    return " ".join(advice)

# Pydantic Schema for CompoundInterestCalculator
class CompoundInterestCalculatorSchema(BaseModel):
    """Schema for the CompoundInterestCalculator tool."""
    principal: float = Field(..., description="The initial amount of money.")
    rate: float = Field(
        ..., description="The annual interest rate as a decimal (e.g., 0.05 for 5%)."
    )
    time: float = Field(
        ..., description="The number of years the money is invested or borrowed for."
    )
    compounds_per_period: int = Field(
        1,
        description="The number of times that interest is compounded per year "
                    "(e.g., 1 for annually, 12 for monthly)."
    )

def compound_interest_calculator(
    principal: float,
    rate: float,
    time: float,
    compounds_per_period: int
) -> str:
    """
    Calculate compound interest based on principal, annual rate (as a decimal),
    time (in years), and compounds per period.

    Args:
        principal (float): The initial amount of money.
        rate (float): The annual interest rate as a decimal (e.g., 0.05 for 5%).
        time (float): The number of years the money is invested or borrowed for.
        compounds_per_period (int): The number of times that interest is
                                     compounded per year (e.g., 1 for annually,
                                     12 for monthly).

    Returns:
        str: A formatted string showing the compound interest calculation.
    """
    logging.info("Tool 'compound_interest_calculator' called with input: "
                 "principal=%f, rate=%f, time=%f, compounds_per_period=%d",
                 principal, rate, time, compounds_per_period)
    try:
        if principal < 0 or rate < 0 or time < 0 or compounds_per_period <= 0:
            return ("All numerical inputs (principal, rate, time, compounds per period) "
                    "must be positive values.")

        amount = principal * (1 + rate / compounds_per_period)**(compounds_per_period * time)
        interest = amount - principal
        return (
            f"Compound Interest Calculation:\n"
            f"Principal: ${principal:,.2f}\n"
            f"Annual Rate: {rate*100:.2f}%\n"
            f"Compounded: {compounds_per_period} times per year\n"
            f"Duration: {time} years\n"
            f"Total Amount: ${amount:,.2f}\n"
            f"Interest Earned: ${interest:,.2f}"
        )
    except Exception as err: # pylint: disable=broad-except
        logging.error("Error in compound_interest_calculator: %s", err, exc_info=True)
        return ("Sorry, I couldn't calculate compound interest. Please provide valid numbers "
                "for principal, rate (as a decimal, e.g., 0.05 for 5%), time, and compounds "
                "per period.")

def _parse_salary_figures(salary_slip_content: str) -> Dict[str, Optional[float]]:
    """Helper to parse key financial figures from salary slip content."""
    gross_pay = None
    net_pay = None
    deductions = None

    # Helper to find a numerical value from a list of patterns
    def _find_value(content: str, patterns: List[str]) -> Optional[float]:
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1).replace(',', ''))
                    logging.info("Found value: %f using pattern: %s", value, pattern)
                    return value
                except ValueError:
                    logging.warning(
                        "Could not convert matched string '%s' to float using pattern '%s'.",
                        match.group(1), pattern
                    )
        return None

    # Regex patterns for Gross Pay, Net Pay, and Deductions
    gross_pay_patterns = [
        r'(?:Gross\s*Pay|Gross\s*Salary|Total\s*Earnings|Gross\s*Amount|Basic\s*Salary|Salary|Wages|Total\s*Pay)\s*[:\s"\n,]*[\$\€£]?\s*([\d,\.]+)',
    ]
    net_pay_patterns = [
        r'(?:Net\s*Pay|Net\s*Salary|Take\s*Home\s*Pay|Total\s*Net|Amount\s*Paid)\s*[:\s"\n,]*[\$\€£]?\s*([\d,\.]+)',
    ]
    deductions_patterns = [
        r'(?:Total\s*Deductions|Deductions\s*Total|Total\s*Withholdings|Employee\s*Deductions)\s*[:\s"\n,]*[\$\€£]?\s*([\d,\.]+)',
    ]

    gross_pay = _find_value(salary_slip_content, gross_pay_patterns)
    net_pay = _find_value(salary_slip_content, net_pay_patterns)
    deductions = _find_value(salary_slip_content, deductions_patterns)

    # If gross_pay and net_pay are found, calculate deductions from them
    if gross_pay is not None and net_pay is not None:
        if deductions is None:
            deductions = gross_pay - net_pay
            logging.info("Calculated Deductions: %f (Gross - Net)", deductions)
    return {
        "gross_pay": gross_pay,
        "net_pay": net_pay,
        "deductions": deductions
    }

def salary_slip_analysis(salary_slip_content: str) -> str:
    """
    Analyze an uploaded salary slip (provided as text content) and recommend
    financial strategies.

    This function processes the text from a salary slip (e.g., from an OCR scan
    or text input). It looks for 'Gross Pay', 'Net Pay', and 'Deductions'.

    Args:
        salary_slip_content (str): The text content of the salary slip.

    Returns:
        str: A string containing the analysis and financial recommendations.
    """
    logging.info("Attempting salary slip analysis.")
    logging.debug("Salary slip content received:\n%s...", salary_slip_content[:500])

    try:
        figures = _parse_salary_figures(salary_slip_content)
        gross_pay = figures["gross_pay"]
        net_pay = figures["net_pay"]
        deductions = figures["deductions"]

        if gross_pay is None and net_pay is None and deductions is None:
            logging.warning("No key financial figures found in salary slip content.")
            return (
                "I couldn't find key financial figures like Gross Pay, Net Pay, "
                "or Deductions in the provided salary slip content. Please ensure "
                "the text is clear and contains these details."
            )

        recommendations: List[str] = ["Based on your salary slip:"]

        if gross_pay is not None:
            recommendations.append(f"- Your Gross Pay is: ${gross_pay:,.2f}")
        if net_pay is not None:
            recommendations.append(f"- Your Net Pay is: ${net_pay:,.2f}")

        if deductions is not None:
            recommendations.append(f"- Your Total Deductions are: ${deductions:,.2f}")
            if gross_pay is not None and gross_pay > 0: # Avoid division by zero
                deduction_percentage = (deductions / gross_pay) * 100
                recommendations.append(
                    f"- This accounts for approximately {deduction_percentage:.2f}% of "
                    "your gross pay."
                )
                if deduction_percentage > DEDUCTION_THRESHOLD_PERCENT:
                    recommendations.append(
                        "- Your deductions seem quite high. You might want to review "
                        "your tax withholdings, retirement contributions, and other "
                        "benefits to ensure they align with your financial goals."
                    )

        if net_pay is not None:
            savings_target = net_pay * COMMON_SAVINGS_TARGET_PERCENT
            recommendations.append(
                f"- With a net pay of ${net_pay:,.2f}, consider aiming to save at least "
                f"{COMMON_SAVINGS_TARGET_PERCENT*100:.0f}% (${savings_target:,.2f}) "
                "of it regularly."
            )
            recommendations.append(
                "- Create a budget to track where your net pay is going and "
                "identify areas for saving."
            )

        recommendations.append("\nGeneral recommendations:")
        recommendations.append("- Build an emergency fund covering 3-6 months of essential "
                               "living expenses.")
        recommendations.append("- Explore setting up automated transfers to savings or "
                               "investment accounts.")
        recommendations.append("- Consult a financial advisor for personalized advice based "
                               "on your full financial situation.")

        return "\n".join(recommendations)

    except ValueError:
        logging.error(
            "ValueError: Could not convert extracted number to float. "
            "Check regex or input format.", exc_info=True
        )
        return ("I found some numbers but couldn't convert them to a valid numerical format "
                "(e.g., issues with commas or decimals). Please ensure numerical values are "
                "clear (e.g., 1,000.00 or 1000.00).")
    except Exception as err: # pylint: disable=broad-except
        logging.error(
            "An unexpected error occurred during salary slip analysis: %s",
            err, exc_info=True
        )
        return ("An unexpected error occurred while analyzing the salary slip. "
                "Please try again or provide content in a different format.")

# --- New Tool: Debt Repayment Strategies ---
def debt_repayment_strategies() -> str:
    """
    Provides information on various debt repayment strategies.

    Returns:
        str: A string describing different debt repayment methods.
    """
    return (
        "Here are some popular debt repayment strategies:\n"
        "- **Debt Snowball Method:** Pay off debts from smallest balance to "
        "largest. Once the smallest is paid, roll that payment into the next "
        "smallest debt.\n"
        "- **Debt Avalanche Method:** Pay off debts from highest interest rate "
        "to lowest. This method saves you the most money on interest.\n"
        "- **Debt Consolidation:** Combine multiple debts into a single, larger "
        "loan, often with a lower interest rate or more favorable terms. This "
        "can simplify payments and potentially reduce overall interest paid.\n"
        "- **Balance Transfer:** Move high-interest credit card debt to a new "
        "credit card with a lower or 0% introductory APR. Be mindful of balance "
        "transfer fees and the promotional period.\n"
        "Which strategy sounds most appealing, or would you like more details "
        "on any of them?"
    )

# --- Tool List ---
TOOLS = [
    StructuredTool(
        name="KnowledgeBaseQuery",
        func=query_knowledge_base,
        description="Answer financial questions using the knowledge base. "
                    "Always use this tool for factual financial questions.",
        args_schema=KnowledgeBaseQuerySchema,
        coroutine=query_knowledge_base
    ),
    StructuredTool(
        name="SavingsRecommendation",
        func=recommend_savings,
        description="Provide personalized saving recommendations based on "
                    "income (float) and spending (float). Optionally, provide "
                    "a region (str).",
        args_schema=SavingsRecommendationSchema
    ),
    Tool(
        name="BudgetingTemplates",
        func=budgeting_templates,
        description="Suggest popular budgeting methods."
    ),
    Tool(
        name="CreditScoreAdvice",
        func=credit_score_advice,
        description="Give advice on improving credit score."
    ),
    Tool(
        name="InvestmentAdvice",
        func=investment_advice,
        description="Provide investment tips based on user's region."
    ),
    StructuredTool(
        name="RetirementPlanning",
        func=retirement_planning,
        description="Advise on retirement age and savings. Can take optional "
                    "parameters: age (int), current_savings (float), "
                    "desired_income_retirement (float), and region (str, e.g., "
                    "'US' or 'UK').",
        args_schema=RetirementPlanningSchema
    ),
    StructuredTool(
        name="CompoundInterestCalculator",
        func=compound_interest_calculator,
        description=(
            "Calculate compound interest, total amount, or interest earned. "
            "This tool is specifically designed for precise financial "
            "calculations involving: 'principal' (initial amount as float), "
            "'rate' (annual interest rate as a decimal, e.g., 0.05 for 5%), "
            "'time' (duration in years as float), and 'compounds_per_period' "
            "(number of times interest is compounded per year as int, e.g., "
            "1 for annually, 12 for monthly). Use this tool whenever a user "
            "asks to calculate compound interest, future value, or interest "
            "earned based on these parameters."
        ),
        args_schema=CompoundInterestCalculatorSchema
    ),
    Tool(
        name="SalarySlipAnalysis",
        func=salary_slip_analysis,
        description="Analyze an uploaded salary slip (text content) for "
                    "financial figures and provide recommendations."
    ),
    Tool(
        name="DebtRepaymentStrategies",
        func=debt_repayment_strategies,
        description="Provide information on various debt repayment strategies "
                    "like debt snowball, debt avalanche, and debt consolidation."
    ),
]

# Create the agent with tools and prompt
AGENT = create_openai_tools_agent(llm=LLM, tools=TOOLS, prompt=AGENT_PROMPT)

# Create the Agent Executor
AGENT_EXECUTOR = AgentExecutor(
    agent=AGENT,
    tools=TOOLS,
    memory=MEMORY,
    verbose=False # Set to True for detailed agent logging (turn off in prod)
)

# You can now use AGENT_EXECUTOR to invoke the chatbot
# Example usage (for testing, typically this would be in an API endpoint or CLI)
async def run_chat_query(user_input: str) -> AsyncGenerator[str, None]:
    """
    Runs a query through the LangChain agent and yields responses.
    This is an asynchronous generator for streaming output.

    Args:
        user_input (str): The user's query.

    Yields:
        str: Chunks of the AI's response.
    """
    logging.info("Received user query: %s", user_input)
    # The .astream() method processes the input and yields events
    async for event in AGENT_EXECUTOR.astream(
        {"input": user_input},
        config=RunnableConfig(callbacks=[]) # Can add callbacks here for more logging
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                yield content
        elif kind == "on_tool_end":
            # For debugging, you might log tool outputs
            # logging.info("Tool Output: %s", event["data"]["output"])
            pass
        elif kind == "on_agent_end":
            # The final answer is usually in the last on_chat_model_stream event
            # or could be extracted from agent output if not streaming directly.
            pass