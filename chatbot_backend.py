import os
import logging
import re
import asyncio # <-- Import asyncio
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from typing import List
from typing import List, Optional, Dict, Generator # <-- Ensure 'Generator' is here
# LangChain imports
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool, StructuredTool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

# Pinecone Imports
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# --- Configure logging ---
# Changed level to WARNING for potentially faster production performance.
# You can change it to ERROR if you only want to see critical issues.
# For debugging, set back to INFO or DEBUG.
#logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load OpenAI credentials ---
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logging.error("Missing OPENAI_API_KEY in .env.")
    exit("OPENAI_API_KEY not set. Exiting.")

# --- Pinecone Configuration for Retrieval ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = "financial-literacy-chatbot"

# ADD THESE DEBUG LINES:
# These will now only show if logging level is INFO or DEBUG (not WARNING)
logging.info(f"DEBUG app.py: PINECONE_API_KEY loaded: {'*****' if PINECONE_API_KEY else 'None'}")
logging.info(f"DEBUG app.py: PINECONE_ENVIRONMENT loaded: {PINECONE_ENVIRONMENT}")
logging.info(f"DEBUG app.py: INDEX_NAME used: {INDEX_NAME}")
# END DEBUG LINES

if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    logging.error("Pinecone API key or environment not set. Please add PINECONE_API_KEY and PINECONE_ENVIRONMENT to your .env file.")
    exit("Exiting: Pinecone credentials missing for chatbot app.")

# --- Connect to Pinecone Vector Store ---
vectorstore = None
retriever = None
embeddings_model = None # Define embeddings_model here before try block

try:
    # Initialize embeddings_model here, before it's used by PineconeVectorStore
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
    logging.info("Initialized OpenAIEmbeddings model for Pinecone retrieval.")

    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    logging.info("Connected to Pinecone client.")

    index_names = [index["name"] for index in pc.list_indexes()]
    if INDEX_NAME not in index_names:
        logging.error(f"Pinecone index '{INDEX_NAME}' does not exist. Please run 'pinecone_data_loader.py' first to create and populate it.")
        exit("Exiting: Pinecone index not found.")
    
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings_model)
    retriever = vectorstore.as_retriever()
    logging.info("Pinecone vector store and retriever initialized for existing index.")

except Exception as e:
    logging.error(f"Error connecting to Pinecone index for chatbot: {e}")
    exit("Exiting: Failed to connect to Pinecone. Ensure index exists and credentials are correct.")

# --- Initialize LLM, Memory, and Agent Prompt FIRST ---
# llm = None
# try:
#     llm = ChatOpenAI(model="gpt-4o", temperature=0.7, openai_api_key=openai_api_key)
#     logging.info("ChatOpenAI LLM initialized successfully with gpt-4o.")
# except Exception as e:
#     logging.error(f"Failed to initialize ChatOpenAI LLM: {e}")
#     exit("Exiting: LLM could not be initialized.")
# --- Initialize LLM, Memory, and Agent Prompt FIRST ---
llm = None
try:
    # IMPORTANT CHANGE: Enable streaming for the LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7, openai_api_key=openai_api_key, streaming=True)
    logging.info("ChatOpenAI LLM initialized successfully with gpt-4o.")
except Exception as e:
    logging.error(f"Failed to initialize ChatOpenAI LLM: {e}")
    exit("Exiting: LLM could not be initialized.")

# --- Changed to ConversationBufferWindowMemory for controlled history ---
# k is now 3 as per your request
memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True)

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=(
        "You are a highly knowledgeable and helpful financial literacy assistant. "
        "Your primary goal is to provide accurate, factual, and concise answers related to financial topics. "
        "**Always prioritize using information retrieved from your tools, especially the 'KnowledgeBaseQuery' and any specific calculation tools (like for compound interest), for factual questions and computations.** "
        "**When a tool provides relevant information, you MUST use that information to formulate a direct answer to the user's question.** "
        "**Do NOT make up information or answer outside the scope of what the tools can provide.** "
        "If, after using ALL relevant tools, you genuinely cannot find enough information to confidently and completely answer the question based on the tool outputs, "
        "then, and only then, state clearly: 'I cannot answer this question based on the information I have available.'"
    )),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# --- Region Settings (UNCHANGED) ---
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
    "global": {
        "currency": "",
        "retirement_age": 65,
        "tax_brackets": [],
        "investment_tips": "For general investment advice, diversification across asset classes (stocks, bonds, real estate) and geographies is key. Consider low-cost index funds or ETFs. Consult a financial advisor for personalized guidance.",
        "savings_recommendation": "A common guideline is to aim for 8-12 times your annual salary by retirement.",
        "countries": []
    }
}

# --- Helper function to detect region from user input (UNCHANGED) ---
def detect_region(text: str) -> str:
    text_lower = text.lower()
    for region_key, region_data in REGION_SETTINGS.items():
        if region_key != "global":
            if region_key in text_lower or any(country.lower() in text_lower for country in region_data.get("countries", [])):
                return region_key
    return "global"

# --- Define Tools (UNCHANGED, except for async call in KnowledgeBaseQuery if you were to make llm.invoke async too) ---
import logging # Ensure logging is imported at the top of your file if it's not already

# ... (other imports and code) ...

import logging # Ensure logging is imported at the top of your file if it's not already

# ... (other imports and code) ...

def query_knowledge_base(query: str) -> str:
    """Answer financial questions using the knowledge base."""
    if not retriever:
        logging.warning("KnowledgeBaseQuery called but retriever is not initialized. RAG will not work.")
        return "Knowledge base is not available. Please ensure the Pinecone index is set up correctly."
    try:
        # Retrieve documents from the vector store
        docs = retriever.invoke(query)

        # --- CORRECTED LINE: Only one logging.info for the count ---
        logging.info(f"KnowledgeBaseQuery: Retrieved {len(docs)} documents for query: '{query}'")
        # --- END OF CORRECTED LINE ---

        retrieved_sources = []
        for i, doc in enumerate(docs):
            source_info = doc.metadata.get('source', 'No source info') # Assuming 'source' in metadata
            logging.info(f"  Document {i+1} Source: {source_info}")
            # If you want to see the content as well (be cautious with large documents)
            # logging.debug(f"  Document {i+1} Content Snippet: {doc.page_content[:200]}...") # Log first 200 chars
            retrieved_sources.append(source_info)
        
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt_content = f"Use the following context to answer the question:\n{context}\n\nQuestion: {query}"

        response = llm.invoke(prompt_content)
        return response.content
    except Exception as e:
        logging.error(f"LLM call failed in query_knowledge_base: {e}")
        return "Sorry, I couldn't process your knowledge base query at the moment."

# ... (rest of your code) ...

# ... (rest of your code) ...

# Pydantic Schema for SavingsRecommendation (UNCHANGED)
class SavingsRecommendationSchema(BaseModel):
    income: float = Field(..., description="The user's income.")
    spending: float = Field(..., description="The user's spending.")
    region: str = Field("global", description="The user's region (e.g., 'United States', 'UK', 'Germany').")

def recommend_savings(income: float, spending: float, region: str = "global") -> str:
    """Provide personalized saving recommendations based on income (float) and spending (float).
    Input should be a Pydantic object containing: income (float), spending (float), and optionally region (str)."""
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
            return f"Based on your income ({currency}{income:,.2f}) and spending ({currency}{spending:,.2f}), you are not currently saving. Consider reviewing your expenses to find areas where you can reduce spending and start saving."
        elif savings_amount < income * 0.1:
            return f"You are saving {currency}{savings_amount:,.2f} per period. To build a stronger financial future, aim to save at least 10-20% of your income. {savings_recommendation_text}"
        else:
            return f"You are saving {currency}{savings_amount:,.2f} per period, which is a good start. Keep up the great work! {savings_recommendation_text} Consider automating your savings."
    except Exception as e:
        logging.error(f"Error in recommend_savings: {e}")
        return "Sorry, I couldn't process savings recommendations. Please provide valid numbers for income and spending."

def budgeting_templates(query: str = "") -> str: # Added 'query: str = ""' as an optional argument
    """Provides information on various budgeting templates like 50/30/20 rule, zero-based, etc. This tool does not require specific arguments but can accept a query string."""
    # The 'query' argument is accepted but not explicitly used, preventing the TypeError.
    return (
        "Here are some popular budgeting templates:\n"
        "- **50/30/20 Rule:** 50% needs, 30% wants, 20% savings/debt repayment.\n"
        "- **Zero-Based Budgeting:** Every dollar is assigned a purpose.\n"
        "- **Envelope System:** Allocate cash into envelopes for different spending categories.\n"
        "- **Paycheck to Paycheck Budgeting:** Focuses on managing funds until the next paycheck.\n"
        "Which one would you like to know more about, or would you like to see a general template?"
    )



def credit_score_advice(query: str = "") -> str: # Added 'query: str = ""' as an optional argument
    """Give advice on improving credit score. This tool does not require specific arguments but can accept a query string."""
    # The 'query' argument is accepted but not explicitly used, preventing the TypeError.
    # You could potentially use it if you wanted to make the advice dynamic based on the query.
    return (
        "Tips to improve your credit score:\n"
        "- Pay your bills on time.\n"
        "- Keep your credit utilization below 30%.\n"
        "- Avoid opening too many new accounts quickly.\n"
        "- Regularly check your credit report for errors.\n"
        "Ask me for specific credit-related questions anytime!"
    )

def investment_advice(region: str = "global") -> str:
    """Provide investment tips based on user's region. Example regions: United States, Germany, Japan."""
    detected_region_key = detect_region(region)
    advice_text = REGION_SETTINGS.get(detected_region_key, REGION_SETTINGS["global"])["investment_tips"]
    return advice_text

# Pydantic Schema for RetirementPlanning (UNCHANGED)
class RetirementPlanningSchema(BaseModel):
    age: Optional[int] = Field(None, description="The user's current age.")
    current_savings: Optional[float] = Field(None, description="The user's current retirement savings.")
    desired_income_retirement: Optional[float] = Field(None, description="The user's desired annual income in retirement.")
    region: str = Field("global", description="The user's region (e.g., 'US', 'UK', 'Germany').")

def retirement_planning(age: Optional[int] = None, current_savings: Optional[float] = None, desired_income_retirement: Optional[float] = None, region: str = "global") -> str:
    """Advise on retirement age and savings. Can take optional parameters: age (int), current_savings (float), desired_income_retirement (float), and region (str)."""
    advice = []
    detected_region_key = detect_region(region)
    actual_region_data = REGION_SETTINGS.get(detected_region_key, REGION_SETTINGS["global"])

    retirement_age_default = actual_region_data.get("retirement_age", 65)
    savings_tip = actual_region_data.get("savings_recommendation", "Aim to save consistently for retirement.")
    currency = actual_region_data.get("currency", "$")

    if age is None:
        advice.append("Start planning for retirement as early as possible, ideally in your 20s or 30s, to maximize the benefit of compound interest.")
    elif age < 30:
        advice.append(f"At your age ({age}), you have a great head start for retirement planning. Focus on consistent contributions to a diversified portfolio.")
    elif age < 50:
        advice.append(f"At your age ({age}), it's crucial to be actively saving for retirement. Review your contributions and investment strategy regularly.")
    else:
        advice.append(f"At your age ({age}), accelerate your retirement savings. Consider catch-up contributions to retirement accounts if eligible.")

    advice.append(f"In your region ({detected_region_key.upper()}), the typical retirement age is {retirement_age_default}.")
    advice.append(savings_tip)

    if current_savings is not None and desired_income_retirement is not None:
        # A common rule of thumb for retirement income needed is 25 times your desired annual income
        estimated_needed = desired_income_retirement * 25

        if current_savings < estimated_needed * 0.2: # Less than 20% of goal
            advice.append(f"With current savings of {currency}{current_savings:,.2f} towards a goal requiring approximately {currency}{estimated_needed:,.2f}, you may need to significantly increase your savings rate.")
        elif current_savings < estimated_needed * 0.5: # Between 20% and 50% of goal
            advice.append(f"With current savings of {currency}{current_savings:,.2f} towards a goal requiring approximately {currency}{estimated_needed:,.2f}, you are on your way but consider increasing contributions to meet your desired retirement income.")
        else: # Over 50% of goal
            advice.append(f"With current savings of {currency}{current_savings:,.2f} towards a goal requiring approximately {currency}{estimated_needed:,.2f}, you are making excellent progress towards your retirement income goals!")
    else:
        advice.append("Provide your current savings and desired retirement income for a more personalized assessment.")

    return " ".join(advice)

# Pydantic Schema for CompoundInterestCalculator (UNCHANGED)
class CompoundInterestCalculatorSchema(BaseModel):
    principal: float = Field(..., description="The initial amount of money.")
    rate: float = Field(..., description="The annual interest rate as a decimal (e.g., 0.05 for 5%).")
    time: float = Field(..., description="The number of years the money is invested or borrowed for.")
    compounds_per_period: int = Field(1, description="The number of times that interest is compounded per year (e.g., 1 for annually, 12 for monthly).")

def compound_interest_calculator(principal: float, rate: float, time: float, compounds_per_period: int) -> str:
    """Calculate compound interest based on principal, annual rate (as a decimal), time (in years), and compounds per period (e.g., 1 for annually, 12 for monthly)."""
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
    except Exception as e:
        logging.error(f"Error in compound_interest_calculator: {e}")
        return "Sorry, I couldn't calculate compound interest. Please provide valid numbers for principal, rate (as a decimal, e.g., 0.05 for 5%), time, and compounds per period."

def salary_slip_analysis(salary_slip_content: str) -> str:
    """Analyze an uploaded salary slip (provided as text content) and recommend financial strategies.
    This function processes the text from a salary slip (e.g., from an OCR scan or text input).
    It looks for 'Gross Pay', 'Net Pay', and 'Deductions'.
    """
    logging.info("Attempting salary slip analysis.")
    gross_pay = None
    net_pay = None
    deductions = None

    gross_pay_match = re.search(r'Gross\s*Pay[:\s-]*\s*[\$\€£]?\s*([\d,\.]+)', salary_slip_content, re.IGNORECASE)
    net_pay_match = re.search(r'Net\s*Pay[:\s-]*\s*[\$\€£]?\s*([\d,\.]+)', salary_slip_content, re.IGNORECASE)
    deductions_match = re.search(r'Deductions[:\s-]*\s*[\$\€£]?\s*([\d,\.]+)', salary_slip_content, re.IGNORECASE)

    try:
        if gross_pay_match:
            gross_pay = float(gross_pay_match.group(1).replace(',', ''))
            logging.info(f"Found Gross Pay: {gross_pay}")
        if net_pay_match:
            net_pay = float(net_pay_match.group(1).replace(',', ''))
            logging.info(f"Found Net Pay: {net_pay}")
        if deductions_match:
            deductions = float(deductions_match.group(1).replace(',', ''))
            logging.info(f"Found Deductions: {deductions}")

        if gross_pay is None and net_pay is None and deductions is None:
            return "I couldn't find key financial figures like Gross Pay, Net Pay, or Deductions in the provided salary slip content. Please ensure the text is clear and contains these details."

        recommendations = []
        recommendations.append("Based on your salary slip:")

        if gross_pay is not None:
            recommendations.append(f"- Your Gross Pay is: ${gross_pay:,.2f}")
        if net_pay is not None:
            recommendations.append(f"- Your Net Pay is: ${net_pay:,.2f}")
        if deductions is not None:
            recommendations.append(f"- Your Total Deductions are: ${deductions:,.2f}")

        if gross_pay and net_pay:
            actual_deductions = gross_pay - net_pay
            recommendations.append(f"- Calculated deductions: ${actual_deductions:,.2f}. This accounts for approximately {(actual_deductions/gross_pay*100):.2f}% of your gross pay.")
            if actual_deductions > gross_pay * 0.3:
                recommendations.append("- Your deductions seem quite high. You might want to review your tax withholdings, retirement contributions, and other benefits to ensure they align with your financial goals.")

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
        return "I found some numbers but couldn't convert them to a valid format. Please ensure numerical values are clear (e.g., 1,000.00)."
    except Exception as e:
        logging.error(f"An unexpected error occurred during salary slip analysis: {e}")
        return "An unexpected error occurred while analyzing the salary slip. Please try again or provide content in a different format."

# --- Tool List (UNCHANGED) ---
tools = [
    Tool(name="KnowledgeBaseQuery", func=query_knowledge_base, description="Answer financial questions using knowledge base."),
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
]

# Create the agent with tools and prompt
# verbose=True is set here as you requested for debugging, but for production consider setting to False
agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)

# Create agent executor properly
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, memory=memory, verbose=True)



# --- Main Interaction Function ---
# IMPORTANT CHANGE: Changed return type to Generator and used agent_executor.stream()
async def chatbot_respond(user_input: str, uploaded_salary_slip_content: Optional[str] = None) -> Generator[str, None, None]:
    logging.info(f"Received user input: {user_input if user_input else 'No text input'} and salary slip content: {uploaded_salary_slip_content is not None}")

    if uploaded_salary_slip_content:
        # Note: salary_slip_analysis is synchronous. If it was long-running, you might need to run it in a thread pool.
        # For salary slip, it's not streaming, so yield the full response at once for simplicity
        response_text = salary_slip_analysis(uploaded_salary_slip_content)
        yield response_text
        return

    if not user_input.strip():
        yield "Please type a question or upload a salary slip."
        return

    try:
        # IMPORTANT CHANGE: Use .stream() method for the agent_executor
        # This will yield chunks of the output.
        async for chunk in agent_executor.astream({"input": user_input}):
        #async for chunk in agent_executor.stream({"input": user_input}, config=RunnableConfig(callbacks=[memory])): # Pass memory in config
            if "output" in chunk:
                # The 'output' key contains the final response from the agent
                yield chunk["output"]
                # In most cases, there's only one "output" that needs to be yielded per turn.
                # If your agent yields multiple distinct "output" chunks for a single turn,
                # you would remove this break to ensure all are yielded.
                # For standard streaming, we typically expect one final 'output'.
                # Keeping `break` for typical single-final-output streaming behavior.
                break 

    except Exception as e:
        logging.error(f"Error during agent execution: {e}")
        yield "I apologize, I encountered an error trying to answer your question. Please try again."


# --- CLI test loop (for quick local testing) ---
if __name__ == "__main__":
    print("💬 Financial Chatbot is running. Type 'exit' to quit or press Ctrl+C.")
    if not vectorstore:
        print("Warning: Pinecone vector store was not initialized. Knowledge base queries may not work.")

    print("\nSetup complete. You can now ask questions.")
    print("-" * 30)
    print("Please type your question below:")
    print("Press Ctrl+C to exit gracefully.")
    print("-" * 30)

    async def main_cli_loop():
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
                full_response = ""
                # Await the generator and print chunks
                async for response_chunk in chatbot_respond(user_input, uploaded_salary_slip_content=None):
                    print(response_chunk, end="", flush=True) # Print chunk immediately
                    full_response += response_chunk
                print("\n") # Newline after full response
            except Exception as e:
                logging.error(f"Error in main CLI loop during response generation: {e}")
                print("⚠️ An unexpected error occurred. Please check logs.")
    
    asyncio.run(main_cli_loop())        