from dotenv import load_dotenv
load_dotenv()  # Load .env file for API keys

import os
import logging
import re
from typing import List, Optional

# LangChain imports
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document

# Pinecone Imports
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize OpenAI client (for LLM and Embeddings) ---
# It's good practice to ensure the API key is available
try:
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
except ValueError as e:
    logging.error(f"Configuration Error: {e}")
    exit("Exiting: OpenAI API key is missing. Please set OPENAI_API_KEY environment variable.")
except Exception as e:
    logging.error(f"Error initializing OpenAI API key: {e}")
    exit("Exiting: Failed to set OpenAI API key.")

# --- Initialize LangChain's OpenAIEmbeddings for Pinecone retrieval ---
try:
    embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.getenv("OPENAI_API_KEY"))
    logging.info("Initialized OpenAIEmbeddings model for retrieval.")
except Exception as e:
    logging.error(f"Error initializing OpenAIEmbeddings for retrieval: {e}")
    exit("Exiting: Failed to initialize OpenAIEmbeddings. Check API key.")

# --- Pinecone Configuration for Retrieval ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = "financial-literacy-chatbot" # Must match the index name used in pinecone_data_loader.py

# ADD THESE DEBUG LINES:
logging.info(f"DEBUG app.py: PINECONE_API_KEY loaded: {'*****' if PINECONE_API_KEY else 'None'}") # Don't log full key
logging.info(f"DEBUG app.py: PINECONE_ENVIRONMENT loaded: {PINECONE_ENVIRONMENT}")
logging.info(f"DEBUG app.py: INDEX_NAME used: {INDEX_NAME}")
# END DEBUG LINES

if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    logging.error("Pinecone API key or environment not set. Please add PINECONE_API_KEY and PINECONE_ENVIRONMENT to your .env file.")
    exit("Exiting: Pinecone credentials missing for chatbot app.")

# --- Connect to Pinecone Vector Store ---
vectorstore = None
retriever = None
try:
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    logging.info("Connected to Pinecone client.")

    # Only connect to existing index, do not create or upload here
    index_names = [index["name"] for index in pc.list_indexes()]
    if INDEX_NAME not in index_names:
       logger.error(f"Pinecone index '{INDEX_NAME}' does not exist. Please run 'pinecone_data_loader.py' first to create and populate it.")
       exit()

    
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings_model)
    retriever = vectorstore.as_retriever()
    logging.info("Pinecone vector store and retriever initialized for existing index.")

except Exception as e:
    logging.error(f"Error connecting to Pinecone index for chatbot: {e}")
    exit("Exiting: Failed to connect to Pinecone. Ensure index exists and credentials are correct.")

# --- Region Settings ---
REGION_SETTINGS = {
    "us": {
        "currency": "$",
        "retirement_age": 65,
        "tax_brackets": [(0.10, 9950), (0.12, 40525), (0.22, 86375), (0.24, 164925), (0.32, 209425), (0.35, 523600), (0.37, float('inf'))],
        "investment_tips": "Consider 401(k), Roth IRA, and diverse US stock funds.",
        "savings_recommendation": "Aim to save at least 15% of your income for retirement.",
    },
    "uk": {
        "currency": "£",
        "retirement_age": 66,
        "tax_brackets": [(0.20, 12570), (0.40, 50270), (0.45, float('inf'))],
        "investment_tips": "Consider ISAs and pensions with tax relief.",
        "savings_recommendation": "Try to save at least 10-15% of your income for retirement.",
    },
    "germany": {
        "currency": "€",
        "retirement_age": 67,
        "tax_brackets": [
            (0.00, 9744),
            (0.14, 57918),
            (0.42, 274612),
            (0.45, float('inf'))
        ],
        "investment_tips": "Consider Riester pension, ETFs, and private pension plans.",
        "savings_recommendation": "Aim to save around 10-15% of your income for retirement.",
    },
}

# --- Helper function to detect region from user input ---
def detect_region(text: str) -> Optional[str]:
    text_lower = text.lower()
    if "germany" in text_lower or "deutschland" in text_lower:
        return "germany"
    elif "uk" in text_lower or "united kingdom" in text_lower:
        return "uk"
    elif "us" in text_lower or "usa" in text_lower or "united states" in text_lower:
        return "us"
    return None



def query_knowledge_base(query: str) -> str:
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Use the following context to answer the question:\n{context}\n\nQuestion: {query}"

    try:
        # REMOVED: from langchain_core.messages import HumanMessage (not needed for llm.invoke directly with a string prompt)
        # CHANGED: Use llm.invoke() directly with the string prompt
        response = llm.invoke(prompt)
        # CHANGED: Access .content directly from the response object (no [0])
        answer = response.content
    except Exception as e:
        logging.error(f"LLM call failed: {e}")
        return "Sorry, I couldn't process your query at the moment."

    return answer


# --- Savings Recommendation Tool ---
def recommend_savings(input_str: str) -> str:
    logging.info(f"Tool 'recommend_savings' called with input: {input_str}")
    income = None
    spending = None
    region = detect_region(input_str) or "us"  # Default to US if not detected
    currency = REGION_SETTINGS.get(region, REGION_SETTINGS["us"])["currency"]

    income_match = re.search(r"income[=\s]*(\d[\d,\.]*)", input_str, re.IGNORECASE)
    spending_match = re.search(r"spending[=\s]*(\d[\d,\.]*)", input_str, re.IGNORECASE)

    if income_match:
        try: income = float(income_match.group(1).replace(',', ''))
        except: pass
    if spending_match:
        try: spending = float(spending_match.group(1).replace(',', ''))
        except: pass

    if income is not None and spending is not None:
        if income < 0 or spending < 0:
            return "Income and spending must be non-negative values."
        if spending > income:
            return "Your spending exceeds your income. Consider reducing spending or increasing income first."

        recommended_savings = 0.20 * income
        needs_wants_budget = 0.80 * income
        savings_tip = REGION_SETTINGS[region]["savings_recommendation"]

        return (
            f"Based on your monthly income of {currency}{income:,.2f} and spending of {currency}{spending:,.2f}:\n"
            f"Recommended monthly savings (20% of income): {currency}{recommended_savings:,.2f}.\n"
            f"Budget for needs and wants: {currency}{needs_wants_budget:,.2f}.\n"
            f"Tip: {savings_tip}\n"
            "Remember, consistency is key!"
        )
    else:
        return (
            "To provide personalized savings advice, please share your monthly income and spending, e.g., 'income=3000, spending=2000'. "
            "Include your country or region if you want region-specific advice."
        )

# --- Budgeting Templates Tool ---
def budgeting_templates(_: str) -> str:
    logging.info("Tool 'budgeting_templates' called.")
    return (
        "Here are some popular budgeting methods:\n"
        "1. 50/30/20 Rule — 50% needs, 30% wants, 20% savings.\n"
        "2. Zero-based Budgeting — Every dollar assigned a job.\n"
        "3. Envelope System — Cash divided into labeled envelopes.\n"
        "4. Pay Yourself First — Save a set amount first.\n"
        "Let me know if you want help creating a custom budget!"
    )

# --- Credit Score Advice Tool ---
def credit_score_advice(_: str) -> str:
    logging.info("Tool 'credit_score_advice' called.")
    return (
        "Tips to improve your credit score:\n"
        "- Pay your bills on time.\n"
        "- Keep your credit utilization below 30%.\n"
        "- Avoid opening too many new accounts quickly.\n"
        "- Regularly check your credit report for errors.\n"
        "Ask me for specific credit-related questions anytime!"
    )

# --- Investment Advice Tool ---
def investment_advice(input_str: str) -> str:
    logging.info(f"Tool 'investment_advice' called with input: {input_str}")
    region = detect_region(input_str) or "us"
    tips = REGION_SETTINGS.get(region, REGION_SETTINGS["us"])["investment_tips"]
    return f"Investment advice for {region.upper()} region:\n{tips}"

# --- Retirement Planning Tool ---
def retirement_planning(input_str: str) -> str:
    logging.info(f"Tool 'retirement_planning' called with input: {input_str}")
    region = detect_region(input_str) or "us"
    retirement_age = REGION_SETTINGS.get(region, REGION_SETTINGS["us"])["retirement_age"]
    savings_tip = REGION_SETTINGS[region]["savings_recommendation"]

    # Example simple projection based on user age and savings goal
    age_match = re.search(r"age[=\s]*(\d+)", input_str, re.IGNORECASE)
    current_age = int(age_match.group(1)) if age_match else None

    if current_age is not None:
        years_left = retirement_age - current_age
        if years_left <= 0:
            return f"You have reached or passed the typical retirement age in {region.upper()} ({retirement_age}). Consider consulting a financial advisor for personalized advice."
        return (
            f"You have {years_left} years until the typical retirement age in {region.upper()} ({retirement_age}).\n"
            f"{savings_tip} The earlier you start saving, the better your retirement outlook."
        )
    else:
        return (
            f"In {region.upper()}, the typical retirement age is {retirement_age}.\n"
            f"{savings_tip} Provide your age to get a more personalized plan."
        )

# --- Compound Interest Calculator Tool ---
def compound_interest_calculator(input_str: str) -> str:
    logging.info(f"Tool 'compound_interest_calculator' called with input: {input_str}")
    # Expecting input like: principal=1000, rate=5, times=12, years=10
    try:
        principal = float(re.search(r"principal[=\s]*(\d+\.?\d*)", input_str, re.IGNORECASE).group(1))
        rate = float(re.search(r"rate[=\s]*(\d+\.?\d*)", input_str, re.IGNORECASE).group(1)) / 100
        times = int(re.search(r"times[=\s]*(\d+)", input_str, re.IGNORECASE).group(1))
        years = float(re.search(r"years[=\s]*(\d+\.?\d*)", input_str, re.IGNORECASE).group(1))
    except Exception:
        return (
            "To calculate compound interest, please provide parameters like:\n"
            "'principal=1000 rate=5 times=12 years=10'\n"
            "where rate is the annual interest rate in %, times is compounding frequency per year."
        )

    amount = principal * (1 + rate / times) ** (times * years)
    interest_earned = amount - principal
    return (
        f"Compound Interest Calculation:\n"
        f"Principal: {principal}\n"
        f"Annual Rate: {rate*100}%\n"
        f"Compounded: {times} times per year\n"
        f"Duration: {years} years\n"
        f"Total Amount: {amount:.2f}\n"
        f"Interest Earned: {interest_earned:.2f}"
    )

# --- Salary Slip Analysis Tool ---
def salary_slip_analysis(file_content: str) -> str:
    logging.info("Tool 'salary_slip_analysis' called.")
    # Placeholder parsing logic — real implementation would parse actual salary slip content
    # Here assuming input is plain text with lines like: "Basic Salary: 4000", "Tax: 500", "Deductions: 200"
    basic_salary = tax = deductions = None
    try:
        basic_salary = float(re.search(r"Basic Salary:\s*([\d,\.]+)", file_content, re.IGNORECASE).group(1).replace(',', ''))
        tax = float(re.search(r"Tax:\s*([\d,\.]+)", file_content, re.IGNORECASE).group(1).replace(',', ''))
        deductions = float(re.search(r"Deductions:\s*([\d,\.]+)", file_content, re.IGNORECASE).group(1).replace(',', ''))
    except Exception:
        return "Unable to parse salary slip details. Please ensure the format is correct."

    net_salary = basic_salary - tax - deductions
    recommendation = (
        f"Your net salary is: {net_salary:.2f}. "
        "Consider the following for your finances:\n"
        "- Emergency fund: 3-6 months of expenses.\n"
        "- Retirement savings: at least 15% of your income.\n"
        "- Diversify investments based on your risk profile.\n"
    )
    return recommendation

# --- Register all tools ---
tools = [
    Tool(name="KnowledgeBaseQuery", func=query_knowledge_base, description="Answer financial questions using knowledge base."),
    Tool(name="SavingsRecommendation", func=recommend_savings, description="Provide personalized saving recommendations based on income and spending."),
    Tool(name="BudgetingTemplates", func=budgeting_templates, description="Suggest popular budgeting methods."),
    Tool(name="CreditScoreAdvice", func=credit_score_advice, description="Give advice on improving credit score."),
    Tool(name="InvestmentAdvice", func=investment_advice, description="Provide investment tips based on user's region."),
    Tool(name="RetirementPlanning", func=retirement_planning, description="Advise on retirement age and savings."),
    Tool(name="CompoundInterestCalculator", func=compound_interest_calculator, description="Calculate compound interest based on parameters."),
    Tool(name="SalarySlipAnalysis", func=salary_slip_analysis, description="Analyze uploaded salary slip and recommend financial strategies."),
]


# --- Main Chatbot Interaction Function ---
def chatbot_respond(user_input: str, uploaded_salary_slip_content: Optional[str] = None) -> str:
    logging.info(f"Received user input: {user_input}")

    # If user uploads salary slip content, process it first
    if uploaded_salary_slip_content:
        response = salary_slip_analysis(uploaded_salary_slip_content)
        return response

    # Otherwise, process user input through the agent
    response = agent_executor.run(user_input)
    return response


# --- Setup LLM and memory ---
llm = ChatOpenAI(model="gpt-4o", temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

# Define system prompt
# prompt = ChatPromptTemplate.from_messages([
#     SystemMessage(content="You are a helpful financial literacy assistant. Use the tools as needed."),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("human", "{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad"),
# ])

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=(
        "You are a highly knowledgeable and helpful financial literacy assistant. "
        "Your primary goal is to provide accurate, factual, and concise answers related to financial topics. "
        "**Always prioritize using information retrieved from your tools, especially the 'KnowledgeBaseQuery' tool, for factual questions.** "
        "**When a tool provides relevant information, you MUST use that information to formulate a direct answer to the user's question.** "
        "**Do NOT make up information or answer outside the scope of what the tools can provide.** "
        "If, after using ALL relevant tools, you genuinely cannot find enough information to confidently and completely answer the question based on the tool outputs, "
        "then, and only then, state clearly: 'I cannot answer this question based on the information I have available.'"
    )),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


# Create the agent with tools and prompt
agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)

# Create agent executor properly
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, memory=memory, verbose=True)

# --- Chatbot respond function ---
def chatbot_respond(user_input: str) -> str:
    logging.info(f"Received user input: {user_input}")
    result = agent_executor.invoke({"input": user_input})
    return result.get("output", "No output returned.")



if __name__ == "__main__":
    print("Welcome to Financial Literacy Chatbot (LangChain Agent Demo). Type 'exit' to quit.")
    # Check if vectorstore is initialized before starting interaction
    if vectorstore is None:
        print("Chatbot cannot start: Pinecone vector store was not initialized. Check logs for errors.")
    else:
        while True:
            query = input("\nAsk your question: ")
            if query.lower() == "exit":
                break
            try:
                response = agent_executor.invoke({"input": query})
                print(f"\n💬 Chatbot:\n{response['output']}")
            except Exception as e:
                logging.error(f"Error during agent execution: {e}")
                print("I apologize, I encountered an error trying to answer your question. Please try again.")

            