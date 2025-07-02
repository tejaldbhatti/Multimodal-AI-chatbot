"""
This module implements the core backend logic for the financial literacy chatbot.
It integrates LangChain agents with OpenAI models and Pinecone for RAG capabilities,
and defines various tools for financial advice, including salary slip analysis
and debt repayment strategies.
"""

import os
import logging
import re
import asyncio # Import asyncio for running async operations in sync context
from typing import List, Optional, Dict, AsyncGenerator, Tuple, Set

# Third-party imports
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pinecone import Pinecone, ServerlessSpec
from pinecone import PineconeApiException
from openai import AuthenticationError as OpenAIAuthenticationError
from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import APIStatusError as OpenAIAPIAPIStatusError

# LangChain imports
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool, StructuredTool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_pinecone import PineconeVectorStore # Added this import

# Load environment variables from .env file at the very beginning.
load_dotenv()

# --- Configure logging ---
# Set logging level to INFO for debugging purposes to see more detailed messages.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants for Configuration ---
# Changed model to gpt-4o-mini for faster responses.
# If you need gpt-4o's reasoning capabilities, consider gpt-4o-mini for speed.
OPENAI_MODEL_NAME = "gpt-4o-mini" 
OPENAI_TEMPERATURE = 0.7
MEMORY_K_VALUE = 10 # Increased memory window to 10 past conversation turns
PINECONE_INDEX_NAME = "financial-literacy-chatbot"

# --- Load OpenAI credentials ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logging.error("Missing OPENAI_API_KEY in .env. Please set this environment variable.")
    exit("OPENAI_API_KEY not set. Exiting.")

# --- Pinecone Configuration ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    logging.error("Pinecone API key or environment not set. Please add PINECONE_API_KEY and PINECONE_ENVIRONMENT to your .env file.")
    exit("Exiting: Pinecone credentials missing for chatbot app.")

# --- Global Variables for LLM, Embeddings, Vector Store, and Retriever ---
# These are initialized once at startup.
VECTORSTORE: Optional[PineconeVectorStore] = None
RETRIEVER = None
EMBEDDINGS_MODEL: Optional[OpenAIEmbeddings] = None
LLM: Optional[ChatOpenAI] = None

def initialize_rag_components():
    """
    Initializes OpenAI embeddings, Pinecone client, vector store, and retriever.
    This function handles potential errors during the setup process.
    """
    global EMBEDDINGS_MODEL, VECTORSTORE, RETRIEVER, LLM # pylint: disable=global-statement

    try:
        EMBEDDINGS_MODEL = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        
        pinecone_client = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        
        index_names = [index["name"] for index in pinecone_client.list_indexes()]
        if PINECONE_INDEX_NAME not in index_names:
            logging.error(f"Pinecone index '{PINECONE_INDEX_NAME}' does not exist. "
                          "Please run your data loader script first to create and populate it.")
            exit("Exiting: Pinecone index not found for chatbot operation.")

        VECTORSTORE = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=EMBEDDINGS_MODEL)
        RETRIEVER = VECTORSTORE.as_retriever() 
        
        LLM = ChatOpenAI(model=OPENAI_MODEL_NAME, temperature=OPENAI_TEMPERATURE, openai_api_key=OPENAI_API_KEY, streaming=True)

    except OpenAIAuthenticationError as auth_err:
        logging.error(f"OpenAI Authentication Error: Your OpenAI API key is invalid or expired. "
                      f"Please check your OPENAI_API_KEY in the .env file. Details: {auth_err}", exc_info=True)
        exit("Exiting: OpenAI Authentication Failed.")
    except OpenAIAPIConnectionError as conn_err:
        logging.error(f"OpenAI API Connection Error: Could not connect to OpenAI API. "
                      f"Check your internet connection or OpenAI service status. Details: {conn_err}", exc_info=True)
        exit("Exiting: OpenAI Connection Failed.")
    except OpenAIAPIAPIStatusError as status_err:
        logging.error(f"OpenAI API Error (Status Code: {status_err.status_code}): {status_err.response} Details: {status_err}", exc_info=True)
        exit("Exiting: OpenAI API Error.")
    except PineconeApiException as pinecone_exc:
        logging.error(f"Pinecone API Error: Check your PINECONE_API_KEY, PINECONE_ENVIRONMENT, and index name. Details: {pinecone_exc}", exc_info=True)
        exit("Exiting: Initialization failed.")
    except Exception as exc:
        logging.error(f"An unexpected error occurred during Pinecone/Embeddings/LLM initialization: {exc}", exc_info=True)
        exit("Exiting: Initialization failed.")

# Initialize components on startup.
initialize_rag_components()

# --- Conversation Memory ---
MEMORY = ConversationBufferWindowMemory(k=MEMORY_K_VALUE, memory_key="chat_history", return_messages=True, output_key="output")

# --- Agent Prompt Template ---
PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=(
        "You are a highly knowledgeable and helpful financial literacy assistant. "
        "Your primary goal is to provide accurate, factual, and concise answers related to financial topics. "
        "**For any factual questions, you MUST use the 'KnowledgeBaseQuery' tool to retrieve information from the knowledge base.** "
        "**Always prioritize and synthesize information retrieved from your tools (e.g., KnowledgeBaseQuery, compound interest calculator) for factual questions and computations.** "
        "**When a tool provides relevant information, you MUST use that information to formulate a direct, coherent, and concise answer to the user's question. Do NOT simply repeat tool outputs verbatim.** "
        "**After answering, if you used the 'KnowledgeBaseQuery' tool, explicitly state 'Source: [filename]' for each unique source document used, at the end of your complete answer.** "
        "**DO NOT make up information or answer outside the scope of what the tools can provide.** "
        "If, after using ALL relevant tools, you genuinely cannot find enough information to confidently and completely answer the question based on the tool outputs, "
        "then, and only then, state clearly: 'I cannot answer this question based on the information I have available.'"
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
        "tax_brackets": [(0.10, 9950), (0.12, 40525), (0.22, 86375), (0.24, 164925), (0.32, 209425), (0.35, 523600), (0.37, float('inf'))],
        "investment_tips": "In the United States, consider diversified portfolios with ETFs, mutual funds, and possibly individual stocks. Explore retirement accounts like 401(k)s and IRAs, and look into diverse US stock funds. Tech and innovation sectors are often prominent.",
        "savings_recommendation": "Aim to save at least 15% of your income for retirement.",
        "countries": ["united states", "us", "usa", "america"]
    },
    "uk": {
        "currency": "£",
        "retirement_age": 66,
        "tax_brackets": [(0.20, 12570), (0.40, 50270), (0.45, float('inf'))],
        "investment_tips": "In the UK, focus on stable, blue-chip companies, diversified ETFs, and consider pan-European funds. Real estate can also be a strong investment. Consider ISAs and pensions with tax relief.",
        "savings_recommendation": "Try to save at least 10-15% of your income for retirement.",
        "countries": ["united kingdom", "uk", "britain", "england", "scotland", "wales", "northern ireland"]
    },
    "germany": {
        "currency": "€",
        "retirement_age": 67,
        "tax_brackets": [ (0.00, 9744), (0.14, 57918), (0.42, 274612), (0.45, float('inf')) ],
        "investment_tips": "In Germany, consider Riester pension, ETFs, and private pension plans. Be aware of varying tax regulations.",
        "savings_recommendation": "Aim to save around 10-15% of your income for retirement.",
        "countries": ["germany", "deutschland"]
    },
    "global": { # Default fallback
        "currency": "",
        "retirement_age": 65,
        "tax_brackets": [],
        "investment_tips": "For general investment advice, diversification across asset classes (stocks, bonds, real estate) and geographies is key. Consider low-cost index funds or ETFs. Consult a financial advisor for personalized guidance.",
        "savings_recommendation": "A common guideline is to aim for 8-12 times your annual salary by retirement.",
        "countries": []
    }
}

def detect_region(text: str) -> str:
    """
    Detects the user's region based on keywords in the input text.

    Args:
        text (str): The input text from the user.

    Returns:
        str: The detected region key (e.g., "us", "uk", "germany") or "global" if no specific region is found.
    """
    text_lower = text.lower()
    for region_key, region_data in REGION_SETTINGS.items():
        if region_key != "global":
            if region_key in text_lower or any(country.lower() in text_lower for country in region_data.get("countries", [])):
                return region_key
    return "global"

# --- Define Tools ---

class KnowledgeBaseQuerySchema(BaseModel):
    """Pydantic schema for the KnowledgeBaseQuery tool."""
    query: str = Field(..., description="The question to ask the financial knowledge base.")

# Changed to an async function, to be properly awaited by the AgentExecutor
async def query_knowledge_base(query: str) -> Tuple[str, List[str]]:
    """
    Answers financial questions using the knowledge base.
    Retrieves relevant documents from Pinecone and uses the LLM to synthesize an answer.

    Args:
        query (str): The question to ask the financial knowledge base.

    Returns:
        Tuple[str, List[str]]: A tuple containing:
            - The LLM's response based on the retrieved context.
            - A list of unique source filenames used for the answer.
    """
    if not RETRIEVER:
        logging.warning("KnowledgeBaseQuery called but retriever is not initialized. RAG will not work.")
        return "Knowledge base is not available. Please ensure the Pinecone index is set up correctly.", []
    if not LLM:
        logging.warning("KnowledgeBaseQuery called but LLM is not initialized.")
        return "LLM is not available for RAG. Please check backend initialization.", []

    try:
        logging.info(f"KnowledgeBaseQuery: Attempting to retrieve documents for query: '{query}'")
        # RETRIEVER.ainvoke is async, so it must be awaited.
        docs = await RETRIEVER.ainvoke(query) 
        logging.info(f"KnowledgeBaseQuery: Retrieved {len(docs)} documents.")
        retrieved_sources = [doc.metadata.get('source', 'No source info') for doc in docs]
        
        context = "\n\n".join([doc.page_content for doc in docs])
        logging.info(f"KnowledgeBaseQuery: Context length: {len(context)} characters.")
        
        # --- REFINED RAG PROMPT CONTENT ---
        # Emphasizing the use of provided context and clear instructions for answering.
        rag_prompt_content = (
            f"You are a financial expert. Your primary goal is to answer the user's question **ONLY** using the provided context. "
            f"**You MUST extract and synthesize information from the context to form a direct, concise, and factual answer.** "
            f"**DO NOT introduce external knowledge or make assumptions.** "
            f"**If the context is insufficient to fully answer the question, you are still REQUIRED to provide any relevant information found in the context.** "
            f"**Only if the context contains absolutely no relevant information to the question, then, and only then, state: 'I cannot answer this question based on the information I have available.'**\n\n"
            f"Context:\n{context}\n\nQuestion: {query}"
        )
        
        logging.info("KnowledgeBaseQuery: Invoking LLM with RAG prompt.")
        logging.debug(f"KnowledgeBaseQuery: RAG prompt content being sent to LLM (first 500 chars): {rag_prompt_content[:500]}...") # Log first 500 chars of prompt
        
        # LLM.ainvoke is async, so it must be awaited.
        # Added a timeout to prevent indefinite hangs.
        response = await asyncio.wait_for(LLM.ainvoke(rag_prompt_content), timeout=60.0) 
        
        logging.info("KnowledgeBaseQuery: LLM invocation completed.")
        logging.debug(f"KnowledgeBaseQuery: LLM raw response content length: {len(response.content) if response.content else 0}")
        logging.debug(f"KnowledgeBaseQuery: LLM raw response content (first 500 chars): {response.content[:500] if response.content else 'None'}")
        
        unique_sources = sorted(list(set(retrieved_sources)))
        return response.content, unique_sources
    except asyncio.TimeoutError:
        logging.error("KnowledgeBaseQuery: LLM invocation timed out after 60 seconds.", exc_info=True)
        return "Sorry, the knowledge base query timed out. The AI took too long to respond.", []
    except OpenAIAuthenticationError as auth_err:
        logging.error(f"OpenAI Authentication Error in KnowledgeBaseQuery: {auth_err}", exc_info=True)
        return "Sorry, there was an authentication issue with the knowledge base. Please check your OpenAI API key.", []
    except OpenAIAPIConnectionError as conn_err:
        logging.error(f"OpenAI API Connection Error in KnowledgeBaseQuery: {conn_err}", exc_info=True)
        return "Sorry, I couldn't connect to the knowledge base. Please check your internet connection.", []
    except OpenAIAPIAPIStatusError as status_err:
        logging.error(f"OpenAI API Error in KnowledgeBaseQuery (Status Code: {status_err.status_code}): {status_err.response} Details: {status_err}", exc_info=True)
        return "Sorry, an OpenAI API error occurred while querying the knowledge base.", []
    except Exception as exc:
        logging.error(f"An unexpected error occurred in query_knowledge_base tool execution: {exc}", exc_info=True)
        return "Sorry, I couldn't process your knowledge base query at the moment.", []

class SavingsRecommendationSchema(BaseModel):
    """Pydantic schema for the SavingsRecommendation tool."""
    income: float = Field(..., description="The user's income.")
    spending: float = Field(..., description="The user's spending.")
    region: str = Field("global", description="The user's region (e.g., 'United States', 'UK', 'Germany').")

def recommend_savings(income: float, spending: float, region: str = "global") -> str:
    """
    Provides personalized saving recommendations based on income and spending.

    Args:
        income (float): The user's income.
        spending (float): The user's spending.
        region (str): The user's region (e.g., 'United States', 'UK', 'Germany').

    Returns:
        str: A string containing personalized saving recommendations.
    """
    logging.info(f"Tool 'recommend_savings' called with input: {income=}, {spending=}, {region=}")
    try:
        if income < 0 or spending < 0:
            return "Income and spending must be non-negative values."
        if spending > income:
            return "Your spending exceeds your income. Consider reducing expenses or increasing income."

        actual_region_data = REGION_SETTINGS.get(detect_region(region), REGION_SETTINGS["global"])
        currency = actual_region_data.get("currency", "$")
        savings_recommendation_text = actual_region_data.get("savings_recommendation", "Aim to save a significant portion of your income.")

        savings_amount = income - spending
        if savings_amount <= 0:
            return (f"Based on your income ({currency}{income:,.2f}) and spending ({currency}{spending:,.2f}), "
                    "you are not currently saving. Consider reviewing your expenses to find areas where you can reduce spending and start saving.")
        if savings_amount < income * 0.1:
            return (f"You are saving {currency}{savings_amount:,.2f} per period. To build a stronger financial future, "
                    f"aim to save at least 10-20% of your income. {savings_recommendation_text}")
        return (f"You are saving {currency}{savings_amount:,.2f} per period, which is a good start. "
                f"Keep up the great work! {savings_recommendation_text} Consider automating your savings.")
    except Exception as exc:
        logging.error(f"Error in recommend_savings: {exc}", exc_info=True)
        return "Sorry, I couldn't process savings recommendations. Please provide valid numbers for income and spending."

def budgeting_templates(query: str = "") -> str:
    """
    Provides information on various popular budgeting templates.

    Args:
        query (str): User's query (optional, not directly used for logic but for tool invocation).

    Returns:
        str: A string describing different budgeting templates.
    """
    _ = query # Acknowledge query parameter is not directly used to suppress pylint warning.
    return (
        "Here are some popular budgeting templates:\n"
        "- **50/30/20 Rule:** 50% needs, 30% wants, 20% savings/debt repayment.\n"
        "- **Zero-Based Budgeting:** Every dollar is assigned a purpose.\n"
        "- **Envelope System:** Allocate cash into envelopes for different spending categories.\n"
        "- **Paycheck to Paycheck Budgeting:** Focuses on managing funds until the next paycheck.\n"
        "Which one would you like to know more about, or would you like to see a general template?"
    )

def credit_score_advice(query: str = "") -> str:
    """
    Provides general advice on improving credit score.

    Args:
        query (str): User's query (optional, not directly used for logic but for tool invocation).

    Returns:
        str: A string containing tips to improve credit score.
    """
    _ = query # Acknowledge query parameter is not directly used.
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
    Provides general investment tips based on the user's region.

    Args:
        region (str): The user's region (e.g., 'United States', 'Germany', 'Japan').

    Returns:
        str: A string containing investment advice relevant to the detected region.
    """
    detected_region_key = detect_region(region)
    advice_text = REGION_SETTINGS.get(detected_region_key, REGION_SETTINGS["global"])["investment_tips"]
    return advice_text

class RetirementPlanningSchema(BaseModel):
    """Pydantic schema for the RetirementPlanning tool."""
    age: Optional[int] = Field(None, description="The user's current age.")
    current_savings: Optional[float] = Field(None, description="The user's current retirement savings.")
    desired_income_retirement: Optional[float] = Field(None, description="The user's desired annual income in retirement.")
    region: str = Field("global", description="The user's region (e.g., 'US', 'UK', 'Germany').")

def retirement_planning(age: Optional[int] = None, current_savings: Optional[float] = None, desired_income_retirement: Optional[float] = None, region: str = "global") -> str:
    """
    Advises on retirement age and savings based on provided user details and region.

    Args:
        age (Optional[int]): The user's current age.
        current_savings (Optional[float]): The user's current retirement savings.
        desired_income_retirement (Optional[float]): The user's desired annual income in retirement.
        region (str): The user's region (e.g., 'US', 'UK', 'Germany').

    Returns:
        str: A string containing personalized retirement planning advice.
    """
    advice_messages = []
    detected_region_key = detect_region(region)
    actual_region_data = REGION_SETTINGS.get(detected_region_key, REGION_SETTINGS["global"])

    retirement_age_default = actual_region_data.get("retirement_age", 65)
    savings_tip = actual_region_data.get("savings_recommendation", "Aim to save consistently for retirement.")
    currency = actual_region_data.get("currency", "$")

    if age is None:
        advice_messages.append("Start planning for retirement as early as possible, ideally in your 20s or 30s, to maximize the benefit of compound interest.")
    elif age < 30:
        advice_messages.append(f"At your age ({age}), you have a great head start for retirement planning. Focus on consistent contributions to a diversified portfolio.")
    elif age < 50:
        advice_messages.append(f"At your age ({age}), it's crucial to be actively saving for retirement. Review your contributions and investment strategy regularly.")
    else:
        advice_messages.append(f"At your age ({age}), accelerate your retirement savings. Consider catch-up contributions to retirement accounts if eligible.")

    advice_messages.append(f"In your region ({detected_region_key.upper()}), the typical retirement age is {retirement_age_default}.")
    advice_messages.append(savings_tip)

    if current_savings is not None and desired_income_retirement is not None:
        estimated_needed = desired_income_retirement * 25 # Common rule of thumb.
        if current_savings < estimated_needed * 0.2:
            advice_messages.append(f"With current savings of {currency}{current_savings:,.2f} towards a goal requiring approximately {currency}{estimated_needed:,.2f}, "
                                   "you may need to significantly increase your savings rate.")
        elif current_savings < estimated_needed * 0.5:
            advice_messages.append(f"With current savings of {currency}{current_savings:,.2f} towards a goal requiring approximately {currency}{estimated_needed:,.2f}, "
                                   "you are on your way but consider increasing contributions to meet your desired retirement income.")
        else:
            advice_messages.append(f"With current savings of {currency}{current_savings:,.2f} towards a goal requiring approximately {currency}{estimated_needed:,.2f}, "
                                   "you are making excellent progress towards your retirement income goals!")
    else:
        advice_messages.append("Provide your current savings and desired retirement income for a more personalized assessment.")

    return " ".join(advice_messages)

class CompoundInterestCalculatorSchema(BaseModel):
    """Pydantic schema for the CompoundInterestCalculator tool."""
    principal: float = Field(..., description="The initial amount of money.")
    rate: float = Field(..., description="The annual interest rate as a decimal (e.g., 0.05 for 5%).")
    time: float = Field(..., description="The number of years the money is invested or borrowed for.")
    compounds_per_period: int = Field(1, description="The number of times that interest is compounded per year (e.g., 1 for annually, 12 for monthly).")

def compound_interest_calculator(principal: float, rate: float, time: float, compounds_per_period: int) -> str:
    """
    Calculates compound interest, total amount, and interest earned.

    Args:
        principal (float): The initial amount of money.
        rate (float): The annual interest rate as a decimal (e.g., 0.05 for 5%).
        time (float): The number of years the money is invested or borrowed for.
        compounds_per_period (int): The number of times that interest is compounded per year
                                    (e.g., 1 for annually, 12 for monthly).

    Returns:
        str: A string detailing the compound interest calculation results.
    """
    logging.info(f"Tool 'compound_interest_calculator' called with input: {principal=}, {rate=}, {time=}, {compounds_per_period=}")
    try:
        if principal < 0 or rate < 0 or time < 0 or compounds_per_period <= 0:
            return "All numerical inputs (principal, rate, time, compounds per period) must be positive values."

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
    except Exception as exc:
        logging.error(f"Error in compound_interest_calculator: {exc}", exc_info=True)
        return ("Sorry, I couldn't calculate compound interest. "
                "Please provide valid numbers for principal, rate (as a decimal, e.g., 0.05 for 5%), time, and compounds per period.")

def salary_slip_analysis(salary_slip_content: str) -> str:
    """
    Analyzes an uploaded salary slip (provided as text content) and recommends financial strategies.
    This function processes the text from a salary slip (e.g., from an OCR scan or text input).
    It looks for 'Gross Pay', 'Net Pay', and 'Deductions' using robust regex patterns.

    Args:
        salary_slip_content (str): The text content extracted from a salary slip.

    Returns:
        str: A string containing the analysis of the salary slip and financial recommendations.
    """
    logging.info("Attempting salary slip analysis.")
    logging.debug(f"Salary slip content received:\n{salary_slip_content[:500]}...") # Log first 500 chars

    gross_pay = None
    net_pay = None
    deductions = None

    # Regex patterns for financial figures, robust to currency symbols, commas, newlines, and various spellings.
    # It looks for common keywords followed by optional non-numeric characters and then a number.
    # Using re.IGNORECASE for case-insensitive matching.

    # Combined Regex for Gross Pay and similar earnings.
    gross_pay_patterns = [
        r'(?:Gross\s*Pay|Gross\s*Salary|Total\s*Earnings|Gross\s*Amount|Basic\s*Salary|Salary|Wages|Total\s*Pay)\s*[:\s"\n,]*[\$\€£]?\s*([\d,\.]+)',
    ]
    for pattern in gross_pay_patterns:
        match = re.search(pattern, salary_slip_content, re.IGNORECASE)
        if match:
            gross_pay = float(match.group(1).replace(',', ''))
            logging.info(f"Found Gross Pay: {gross_pay} using pattern: {pattern}")
            break

    # Combined Regex for Net Pay and similar take-home pay.
    net_pay_patterns = [
        r'(?:Net\s*Pay|Net\s*Salary|Take\s*Home\s*Pay|Total\s*Net|Amount\s*Paid)\s*[:\s"\n,]*[\$\€£]?\s*([\d,\.]+)',
    ]
    for pattern in net_pay_patterns:
        match = re.search(pattern, salary_slip_content, re.IGNORECASE)
        if match:
            net_pay = float(match.group(1).replace(',', ''))
            logging.info(f"Found Net Pay: {net_pay} using pattern: {pattern}")
            break

    # Combined Regex for Deductions.
    deductions_patterns = [
        r'(?:Total\s*Deductions|Deductions\s*Total|Total\s*Withholdings|Employee\s*Deductions)\s*[:\s"\n,]*[\$\€£]?\s*([\d,\.]+)',
    ]
    for pattern in deductions_patterns:
        match = re.search(pattern, salary_slip_content, re.IGNORECASE)
        if match:
            deductions = float(match.group(1).replace(',', ''))
            logging.info(f"Found Deductions: {deductions} using pattern: {pattern}")
            break

    try:
        # If gross_pay and net_pay are found, calculate deductions from them if not found explicitly.
        if gross_pay is not None and net_pay is not None:
            if deductions is None:
                deductions = gross_pay - net_pay
                logging.info(f"Calculated Deductions: {deductions} (Gross - Net)")
            else:
                logging.info(f"Using explicitly found Deductions: {deductions}")
        elif gross_pay is None and net_pay is None and deductions is None:
            logging.warning("No key financial figures found in salary slip content.")
            return ("I couldn't find key financial figures like Gross Pay, Net Pay, or Deductions "
                    "in the provided salary slip content. Please ensure the text is clear and contains these details.")

        recommendations = []
        recommendations.append("Based on your salary slip:")

        if gross_pay is not None:
            recommendations.append(f"- Your Gross Pay is: ${gross_pay:,.2f}")
        if net_pay is not None:
            recommendations.append(f"- Your Net Pay is: ${net_pay:,.2f}")
        
        if deductions is not None:
            recommendations.append(f"- Your Total Deductions are: ${deductions:,.2f}")
            if gross_pay is not None and gross_pay > 0: # Avoid division by zero.
                recommendations.append(f"- This accounts for approximately {(deductions/gross_pay*100):.2f}% of your gross pay.")
                if deductions > gross_pay * 0.3:
                    recommendations.append("- Your deductions seem quite high. You might want to review your tax withholdings, "
                                           "retirement contributions, and other benefits to ensure they align with your financial goals.")

        if net_pay is not None:
            savings_target = net_pay * 0.20
            recommendations.append(f"- With a net pay of ${net_pay:,.2f}, consider aiming to save at least 20% (${savings_target:,.2f}) of it regularly.")
            recommendations.append("- Create a budget to track where your net pay is going and identify areas for saving.")

        recommendations.append("\nGeneral recommendations:")
        recommendations.append("- Build an emergency fund covering 3-6 months of essential living expenses.")
        recommendations.append("- Explore setting up automated transfers to savings or investment accounts.")
        recommendations.append("- Consult a financial advisor for personalized advice based on your full financial situation.")

        return "\n".join(recommendations)

    except ValueError:
        logging.error("ValueError: Could not convert extracted number to float. Check regex or input format.", exc_info=True)
        return ("I found some numbers but couldn't convert them to a valid numerical format "
                "(e.g., issues with commas or decimals). Please ensure numerical values are clear (e.g., 1,000.00 or 1000.00).")
    except Exception as exc:
        logging.error(f"An unexpected error occurred during salary slip analysis: {exc}", exc_info=True)
        return "An unexpected error occurred while analyzing the salary slip. Please try again or provide content in a different format."

def debt_repayment_strategies(query: str = "") -> str:
    """
    Provides information on various debt repayment strategies.

    Args:
        query (str): User's query (optional, not directly used for logic but for tool invocation).

    Returns:
        str: A string describing different debt repayment strategies.
    """
    _ = query # Acknowledge query parameter is not directly used.
    return (
        "Here are some popular debt repayment strategies:\n"
        "- **Debt Snowball Method:** Pay off debts from smallest balance to largest. Once the smallest is paid, roll that payment into the next smallest debt.\n"
        "- **Debt Avalanche Method:** Pay off debts from highest interest rate to lowest. This method saves you the most money on interest.\n"
        "- **Debt Consolidation:** Combine multiple debts into a single, larger loan, often with a lower interest rate or more favorable terms. This can simplify payments and potentially reduce overall interest paid.\n"
        "- **Balance Transfer:** Move high-interest credit card debt to a new credit card with a lower or 0% introductory APR. Be mindful of balance transfer fees and the promotional period.\n"
        "Which strategy sounds most appealing, or would you like more details on any of them?"
    )

# --- Tool List ---
TOOLS = [
    StructuredTool(
        name="KnowledgeBaseQuery",
        func=query_knowledge_base, # Still here for type hinting, but coroutine is used for execution
        description="Answer financial questions using the knowledge base. Always use this tool for factual financial questions.",
        args_schema=KnowledgeBaseQuerySchema,
        coroutine=query_knowledge_base # Explicitly specify the async function for execution
    ),
    StructuredTool(
        name="SavingsRecommendation",
        func=recommend_savings,
        description="Provide personalized saving recommendations based on income (float) and spending (float). Optionally, provide a region (str).",
        args_schema=SavingsRecommendationSchema
    ),
    Tool(name="BudgetingTemplates", func=budgeting_templates, description="Suggest popular budgeting methods."),
    Tool(name="CreditScoreAdvice", func=credit_score_advice, description="Give advice on improving credit score."),
    Tool(name="InvestmentAdvice", func=investment_advice, description="Provide investment tips based on user's region."),
    StructuredTool(
        name="RetirementPlanning",
        func=retirement_planning,
        description="Advise on retirement age and savings. Can take optional parameters: age (int), current_savings (float), desired_income_retirement (float), and region (str, e.g., 'US' or 'UK').",
        args_schema=RetirementPlanningSchema
    ),
    StructuredTool(
        name="CompoundInterestCalculator",
        func=compound_interest_calculator,
        description="Calculate compound interest, total amount, or interest earned. This tool is specifically designed for precise financial calculations involving: 'principal' (initial amount as float), 'rate' (annual interest rate as a decimal, e.g., 0.05 for 5%), 'time' (duration in years as float), and 'compounds_per_period' (number of times interest is compounded per year as int, e.g., 1 for annually, 12 for monthly). Use this tool whenever a user asks to calculate compound interest, future value, or interest earned based on these parameters.",
        args_schema=CompoundInterestCalculatorSchema
    ),
    Tool(name="SalarySlipAnalysis", func=salary_slip_analysis, description="Analyze an uploaded salary slip (text content) for financial figures and provide recommendations."),
    Tool(name="DebtRepaymentStrategies", func=debt_repayment_strategies, description="Provide information on various debt repayment strategies like debt snowball, debt avalanche, and debt consolidation."),
]

# Create the agent with tools and prompt.
AGENT = create_openai_tools_agent(llm=LLM, tools=TOOLS, prompt=PROMPT)

# Create agent executor with increased max_iterations and max_execution_time
AGENT_EXECUTOR = AgentExecutor.from_agent_and_tools(
    agent=AGENT, 
    tools=TOOLS, 
    memory=MEMORY, 
    verbose=False, # Keep this False to avoid verbose output in console
    max_iterations=50, # Increased max iterations
    max_execution_time=120 # Increased max execution time to 120 seconds
)

# --- ADD THIS LINE TEMPORARILY FOR DEBUGGING ---
print(f"DEBUG: AGENT_EXECUTOR max_iterations set to: {AGENT_EXECUTOR.max_iterations}")
print(f"DEBUG: AGENT_EXECUTOR max_execution_time set to: {AGENT_EXECUTOR.max_execution_time}")
# --- END OF TEMPORARY DEBUGGING LINES ---

async def chatbot_respond(user_input: str, uploaded_salary_slip_content: Optional[str] = None) -> AsyncGenerator[str, None]:
    """
    Asynchronously responds to user input, optionally processing an uploaded salary slip.
    Yields chunks of the response for streaming, and includes source and tool attribution.

    Args:
        user_input (str): The user's question or statement.
        uploaded_salary_slip_content (Optional[str]): Text content of an uploaded salary slip, if any.

    Yields:
        AsyncGenerator[str, None]: Chunks of the chatbot's response, including final attribution.
    """
    logging.info(f"Received user input: '{user_input}' and salary slip content provided: {uploaded_salary_slip_content is not None}")

    if uploaded_salary_slip_content:
        # For salary slip, the backend returns the full response at once, not streaming chunks.
        # We need to explicitly get the first (and only) item from the async generator.
        response_generator = chatbot_respond(user_input=user_input_text, uploaded_salary_slip_content=uploaded_salary_slip_content)
        response_from_backend = await response_generator.__anext__()
        
        yield response_from_backend
        return

    if not user_input.strip():
        yield "Please type a question or upload a salary slip."
        return

    retrieved_sources_for_display: Set[str] = set()
    tools_used_for_display: Set[str] = set()
    
    try:
        # Stream chunks from the agent executor.
        async for chunk in AGENT_EXECUTOR.astream({"input": user_input}):
            if "output" in chunk:
                # This is a text chunk from the LLM's final answer.
                yield chunk["output"]
            elif "tool_calls" in chunk:
                # Collect names of tools that were called.
                for tool_call in chunk["tool_calls"]:
                    tool_name = tool_call.get('name')
                    if tool_name:
                        tools_used_for_display.add(tool_name)
            elif "tool_outputs" in chunk:
                # Collect sources if KnowledgeBaseQuery was used.
                for tool_output in chunk["tool_outputs"]:
                    tool_name = tool_output.get('tool_name')
                    tool_output_content = tool_output.get('output')

                    # Check if tool_output_content is a tuple (response, sources)
                    if tool_name == "KnowledgeBaseQuery" and isinstance(tool_output_content, tuple) and len(tool_output_content) == 2:
                        _, sources = tool_output_content
                        retrieved_sources_for_display.update(sources)
        
        # After the main response has streamed, append the attribution.
        attribution_text = ""
        if tools_used_for_display:
            attribution_text += "\n\nTools Used: " + ", ".join(sorted(list(tools_used_for_display)))
        if retrieved_sources_for_display:
            source_text = "\n\nSources: " + ", ".join(sorted(list(retrieved_sources_for_display)))
            attribution_text += source_text
        
        if attribution_text:
            yield attribution_text # Yield attribution as a final chunk.

    except Exception as exc:
        logging.error(f"Error during agent execution: {exc}", exc_info=True)
        yield "I apologize, I encountered an error trying to answer your question. Please try again."


if __name__ == "__main__":
    print("💬 Financial Chatbot is running. Type 'exit' to quit or press Ctrl+C.")
    if not VECTORSTORE:
        print("Warning: Pinecone vector store was not initialized. Knowledge base queries may not work.")

    print("\nSetup complete. You can now ask questions.")
    print("-" * 30)
    print("Please type your question below:")
    print("Press Ctrl+C to exit gracefully.")
    print("-" * 30)

    async def main_cli_loop():
        """Main asynchronous loop for CLI interaction with the chatbot."""
        while True:
            try:
                user_input = input(">>> ")
                if user_input.lower() == "exit":
                    print("Exiting chatbot.")
                    break
            except KeyboardInterrupt:
                print("\nExiting chatbot due to Ctrl+C.")
                break
            except EOFError:
                print("\nReceived EOF. Exiting chatbot.")
                break

            try:
                # Await the async generator and print chunks.
                async for response_chunk in chatbot_respond(user_input, uploaded_salary_slip_content=None):
                    print(response_chunk, end="", flush=True) 
                print("\n") # Newline after full response.
            except Exception as exc:
                logging.error(f"Error in main CLI loop during response generation: {exc}", exc_info=True)
                print("⚠️ An unexpected error occurred. Please check logs.")
    
    asyncio.run(main_cli_loop())
