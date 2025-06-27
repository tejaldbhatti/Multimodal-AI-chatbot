from dotenv import load_dotenv
load_dotenv() # This line is crucial for loading your .env file

import os
import logging
import re
import tempfile 
from typing import List, Tuple, Dict

# LangChain Imports
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document 

# Pinecone Imports
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from pinecone import PineconeApiException

# Gradio Import
import gradio as gr

# OpenAI Client for Audio APIs (Whisper and TTS)
from openai import OpenAI 

# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Variables for Chatbot Core (initialized once) ---
agent_executor = None 
retriever = None # Keep retriever for the query_knowledge_base tool (will be set in initialize_chatbot_core)
openai_client = None # Global OpenAI client for audio operations (will be set in initialize_chatbot_core)


def initialize_chatbot_core():
    """
    Initializes all the core components of the chatbot (OpenAI, Pinecone, LangChain Agent).
    This function should be called once at the start of the application.
    """
    global agent_executor, retriever, openai_client # Declare globals here

    logging.info("Initializing chatbot core components...")

    # --- Initialize OpenAI API Key (for LLM, Embeddings, and Audio) ---
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        openai_client = OpenAI(api_key=openai_api_key) # Initialize OpenAI client
        logging.info("OpenAI API Key and Client checked.")
    except ValueError as e:
        logging.error(f"Configuration Error: {e}")
        return False, f"Setup Error: OpenAI API key is missing. Please set OPENAI_API_KEY environment variable. Details: {e}"
    except Exception as e:
        logging.error(f"Error checking OpenAI API key or initializing client: {e}")
        return False, f"Setup Error: Failed to access OpenAI API key or initialize client. Details: {e}"

    # --- Initialize LangChain's OpenAIEmbeddings for Pinecone retrieval ---
    try:
        embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.getenv("OPENAI_API_KEY"))
        logging.info("Initialized OpenAIEmbeddings model for retrieval.")
    except Exception as e:
        logging.error(f"Error initializing OpenAIEmbeddings for retrieval: {e}")
        return False, f"Setup Error: Failed to initialize OpenAIEmbeddings. Check API key. Details: {e}"

    # --- Pinecone Configuration for Retrieval ---
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
    INDEX_NAME = "financial-literacy-chatbot" # Your chosen Pinecone index name

    if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
        logging.error("Pinecone API key or environment not set.")
        return False, "Setup Error: Pinecone credentials missing. Please add PINECONE_API_KEY and PINECONE_ENVIRONMENT to your .env file."

    try:
        pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        logging.info("Connected to Pinecone client.")

        existing_indexes = pc.list_indexes()
        index_exists = False
        for idx_info in existing_indexes:
            if isinstance(idx_info, dict) and idx_info.get('name') == INDEX_NAME:
                index_exists = True
                break
            elif hasattr(idx_info, 'name') and idx_info.name == INDEX_NAME:
                index_exists = True
                break
        
        if not index_exists:
            logging.error(f"Pinecone index '{INDEX_NAME}' does not exist. Please run 'pinecone_data_loader.py' first to create and populate it.")
            return False, f"Setup Error: Pinecone index '{INDEX_NAME}' not found. Please run 'pinecone_data_loader.py'."
        
        # Initialize PineconeVectorStore from existing index
        vectorstore = PineconeVectorStore.from_existing_index(INDEX_NAME, embeddings_model)
        retriever = vectorstore.as_retriever() # Set the global retriever
        logging.info("Pinecone vector store and retriever initialized from existing index.")

        # --- Knowledge Base Query Tool (defined locally within initialize_chatbot_core) ---
        def query_knowledge_base(query: str, top_k: int = 5) -> str:
            """
            Retrieves relevant documents from the Pinecone knowledge base and uses an LLM
            to answer the query. This function is designed to be called as a LangChain Tool.
            """
            logging.info(f"Tool 'query_knowledge_base' received query: '{query}'")

            if not query.strip():
                return "Please provide a non-empty question for the knowledge base tool."

            # `retriever` is accessible here due to closure
            if retriever is None: 
                logging.error("Retriever is not initialized in query_knowledge_base. This should not happen.")
                return "The knowledge base is not available right now. Please try again later."

            try:
                retrieved_docs: List[Document] = retriever.get_relevant_documents(query)

                if not retrieved_docs:
                    return "No sufficient information found in the knowledge base for that query."

                context_parts = []
                for i, doc in enumerate(retrieved_docs):
                    context_parts.append(f"<DOCUMENT_START id={i}>\n{doc.page_content}\n<DOCUMENT_END>")
                context = "\n\n".join(context_parts)

                logging.info(f"\n🔎 Retrieved top {len(retrieved_docs)} documents from Pinecone for tool:\n{context[:500]}...\n")

                tool_llm_prompt = f"""You are a financial literacy expert. Your goal is to answer questions using ONLY the information provided in the following retrieved documents.
If the answer is not directly available or cannot be reasonably inferred from the context, state that you cannot answer based on the provided information.

<RETRIEVED_DOCUMENTS>
{context}
</RETRIEVED_DOCUMENTS>

Question: {query}

Answer:
"""
                llm_for_tool = ChatOpenAI(model="gpt-4o", temperature=0.5, openai_api_key=os.getenv("OPENAI_API_KEY"))
                response = llm_for_tool.invoke(tool_llm_prompt)
                answer = response.content.strip()
                logging.info("LLM response received by tool.")
                
                if "i apologize, but i don't have enough information" in answer.lower() or \
                   "cannot answer based on the provided information" in answer.lower() or \
                   "no sufficient information found" in answer.lower():
                    return "No sufficient information found in the knowledge base to answer that."
                return answer
            except Exception as e:
                logging.error(f"Error calling LLM API or retrieving from Pinecone within tool: {e}")
                return "Internal tool error: Could not generate an answer."

        # --- Calculation Tools (defined locally within initialize_chatbot_core) ---
        def recommend_savings(input_str: str) -> str:
            logging.info(f"Tool 'recommend_savings' called with input: {input_str}")
            income = None
            spending = None
            income_match = re.search(r"income[=\s]*(\d[\d,\.]*)", input_str, re.IGNORECASE)
            spending_match = re.search(r"spending[=\s]*(\d[\d,\.]*)", input_str, re.IGNORECASE)
            if income_match:
                try: income = float(income_match.group(1).replace(',', ''))
                except ValueError: pass
            if spending_match:
                try: spending = float(spending_match.group(1).replace(',', ''))
                except ValueError: pass
            if income is not None and spending is not None:
                if income < 0 or spending < 0: return "Income and spending must be non-negative values."
                if spending > income: return "Your spending seems to exceed your income. While saving is important, focusing on reducing spending or increasing income might be your first step."
                recommended_savings = 0.20 * income
                needs_wants_budget = 0.80 * income
                return (
                    f"Based on your monthly income of ${income:,.2f} and spending of ${spending:,.2f}:\n"
                    f"Following the 50/30/20 rule, a recommended monthly savings amount (including debt repayment) is ${recommended_savings:,.2f} (20% of income).\n"
                    f"This would leave ${needs_wants_budget:,.2f} for your needs and wants.\n"
                    "Remember, consistency is key, and even small amounts add up over time."
                )
            else:
                return (
                    "To give you a personalized savings recommendation, I need your monthly income and average monthly spending. "
                    "A common guideline is the 50/30/20 rule: 50% for needs, 30% for wants, and 20% for savings and debt repayment. "
                    "Please provide your monthly income and spending, for example: 'my income is 3000 and spending is 2000'."
                )

        def get_budgeting_templates(input_str: str = "") -> str:
            logging.info(f"Tool 'get_budgeting_templates' called with input: {input_str}")
            return (
                "Budgeting templates can help you organize your finances. Common methods include:\n"
                "- **Spreadsheets:** Excel or Google Sheets offer great flexibility for custom budgets. "
                "You can find many free templates online.\n"
                "- **Budgeting Apps:** Apps like Mint, YNAB (You Need A Budget), or Personal Capital offer "
                "features like transaction tracking, goal setting, and visual reports.\n"
                "- **Pen and Paper:** A simple notebook can also work for tracking income and expenses.\n"
                "The key is to choose a method that you find easy to use and stick with."
            )

        def get_expense_tracker_info(input_str: str = "") -> str:
            logging.info(f"Tool 'get_expense_tracker_info' called with input: {input_str}")
            return (
                "An expense tracker helps you monitor where your money goes. Its benefits include:\n"
                "- **Understanding Spending Habits:** Reveals where you might be overspending.\n"
                "- **Budget Adherence:** Helps you stick to your budget and identify areas for adjustment.\n"
                "- **Financial Goal Achievement:** By seeing your spending, you can find more money for savings or debt.\n"
                "- **Tax Preparation:** Makes it easier to categorize expenses for tax purposes.\n"
                "Methods range from manual logging to using sophisticated apps."
            )

        def calculate_debt_details(input_str: str) -> str:
            logging.info(f"Tool 'calculate_debt_details' called with input: {input_str}")
            principal = None
            annual_interest_rate = None
            monthly_payment = None

            principal_match = re.search(r"principal[=\s]*(\d[\d,\.]*)", input_str, re.IGNORECASE)
            rate_match = re.search(r"interest_rate[=\s]*(\d[\d,\.]*)", input_str, re.IGNORECASE)
            payment_match = re.search(r"monthly_payment[=\s]*(\d[\d,\.]*)", input_str, re.IGNORECASE)

            if principal_match:
                try: principal = float(principal_match.group(1).replace(',', ''))
                except ValueError: pass
            if rate_match:
                try: annual_interest_rate = float(rate_match.group(1).replace(',', ''))
                except ValueError: pass
            if payment_match:
                try: monthly_payment = float(payment_match.group(1).replace(',', ''))
                except ValueError: pass

            if None in [principal, annual_interest_rate, monthly_payment]:
                return (
                    "To calculate debt details, I need the loan principal, the annual interest rate (as a percentage), "
                    "and your monthly payment. "
                    "Please provide them, for example: 'principal=10000, interest_rate=5, monthly_payment=200'."
                )

            if principal <= 0 or annual_interest_rate < 0 or monthly_payment <= 0:
                return "All input values (principal, interest rate, monthly payment) must be positive."

            monthly_interest_rate = (annual_interest_rate / 100) / 12

            if monthly_payment <= (principal * monthly_interest_rate) and annual_interest_rate > 0:
                return "Your monthly payment is too low to ever pay off the principal, or just covers interest. You might need to increase your payment to see progress."

            remaining_principal = principal
            total_interest_paid = 0
            months = 0
            max_months = 600

            while remaining_principal > 0 and months < max_months:
                interest_for_month = remaining_principal * monthly_interest_rate
                principal_paid_this_month = monthly_payment - interest_for_month

                if principal_paid_this_month <= 0 and remaining_principal > 0:
                    return "With these inputs, it seems your monthly payment is not sufficient to pay off the principal within a reasonable timeframe (e.g., it only covers interest). You may need to increase your payment."

                remaining_principal -= principal_paid_this_month
                total_interest_paid += interest_for_month
                months += 1

                if remaining_principal < 0.01:
                    principal_paid_this_month += remaining_principal
                    total_interest_paid -= remaining_principal
                    remaining_principal = 0

            if remaining_principal > 0:
                return (
                    f"It would take more than {max_months} months (50 years) to pay off a principal of ${principal:,.2f} "
                    f"with an annual interest rate of {annual_interest_rate}% and a monthly payment of ${monthly_payment:,.2f}. "
                    "You might consider increasing your monthly payment."
                )
            else:
                years = months / 12
                return (
                    f"To pay off a principal of ${principal:,.2f} with an annual interest rate of {annual_interest_rate}% "
                    f"and a monthly payment of ${monthly_payment:,.2f}:\n"
                    f"- Estimated time to pay off: {months} months ({years:.1f} years)\n"
                    f"- Estimated total interest paid: ${total_interest_paid:,.2f}"
                )

        def get_investment_planning_advice(input_str: str = "") -> str:
            logging.info(f"Tool 'get_investment_planning_advice' called with input: {input_str}")
            return (
                "Investment planning involves setting financial goals and creating a strategy to achieve them through investments. Key aspects include:\n"
                "- **Define Your Goals:** What are you saving for? (e.g., retirement, down payment, education)\n"
                "- **Assess Risk Tolerance:** How comfortable are you with market fluctuations? This influences your asset allocation.\n"
                "- **Diversification:** Spreading investments across different asset classes (stocks, bonds, real estate) to reduce risk.\n"
                "- **Long-term vs. Short-term:** Tailor investments based on your timeline.\n"
                "- **Regular Contributions:** Consistency is often key to compounding returns.\n"
                "It's often recommended to consult a financial advisor for personalized investment planning."
            )

        # --- Setup LangChain Agent ---
        logging.info("Setting up LangChain Agent...")

        llm = ChatOpenAI(model="gpt-4o", temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Define the agent's prompt for create_openai_tools_agent
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a friendly and helpful financial literacy chatbot. Your goal is to assist users with their questions about personal finance, investing, debt management, budgeting, and savings. You have several specialized tools to help you find information and provide advice. When a calculation is requested, ensure you ask for all necessary numerical inputs clearly, specifying the format (e.g., 'monthly income is 3000, spending is 2000' for savings, or 'principal=10000, interest_rate=5, monthly_payment=200' for debt calculation)."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Define LangChain Tools (referencing the locally defined functions)
        tools = [
            Tool(
                name="FinancialLiteracyRetriever",
                func=query_knowledge_base, # This tool uses the Pinecone retriever
                description="Useful for answering specific financial literacy questions by retrieving information from a comprehensive knowledge base about topics like 401k, IRA, credit scores, mortgages, etc. Input should be a concise financial literacy question.",
            ),
            Tool(
                name="SavingsAdvisor",
                func=recommend_savings,
                description="Calculates a recommended savings amount based on provided monthly income and spending (e.g., 'income=3000, spending=2000'). If numbers are not provided, it gives general savings guidelines and prompts for input. Use this when the user asks about how much they should save or for personalized savings recommendations.",
            ),
            Tool(
                name="BudgetingTemplateInfo",
                func=get_budgeting_templates,
                description="Provides information about different types of budgeting templates and methods. Use this when the user asks about budgeting templates, how to start a budget, or tools for budgeting.",
            ),
            Tool(
                name="ExpenseTrackerInfo",
                func=get_expense_tracker_info,
                description="Explains what an expense tracker is and its benefits. Use this when the user asks about tracking expenses or managing spending.",
            ),
            Tool(
                name="DebtCalculator",
                func=calculate_debt_details,
                description="Calculates the estimated time to pay off a debt and total interest paid. Requires specific inputs: 'principal=<amount>, interest_rate=<percentage>, monthly_payment=<amount>'. Use this when the user asks to calculate debt, loan payoff time, or total interest.",
            ),
            Tool(
                name="InvestmentPlanningAdvisor",
                func=get_investment_planning_advice,
                description="Offers general advice and principles for investment planning. Use this when the user asks about how to plan investments, investment strategies, or getting started with investing.",
            ),
        ]

        # Create the OpenAI Tools agent
        agent = create_openai_tools_agent(llm, tools, prompt)

        # Create the Agent Executor
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True)
        logging.info("LangChain Agent setup complete.")

    except Exception as e:
        logging.error(f"Error connecting to Pinecone index or setting up agent: {e}")
        return False, f"Setup Error: Failed to connect to Pinecone or set up agent. Details: {e}"
    
    return True, "Chatbot core initialized successfully!"

# Initialize chatbot core once when the script starts
init_success, init_message = initialize_chatbot_core()

# --- Gradio Interface Functions ---

# Predefined sample questions for each category (Comprehensive List)
SAMPLE_QUESTIONS: Dict[str, List[List[str]]] = {
    "Budgeting": [
        ["What is the 50/30/20 rule for budgeting?"],
        ["Tell me about different budgeting methods, like zero-based budgeting."],
        ["What are some common budgeting apps and their features?"],
        ["How can I create a personal budget step-by-step?"],
        ["Explain the difference between fixed and variable expenses."],
        ["What are some tips for sticking to a budget consistently?"]
    ],
    "Saving": [
        ["If my monthly income is 4000 euros and my spending is 3000 euros, how much should I save according to the 50/30/20 rule?"], # Mathematical
        ["What are some short-term savings goals and how do I achieve them?"],
        ["What's the best strategy to build an emergency fund quickly?"],
        ["How does compound interest work, and how does it affect my long-term savings?"], # Summary/Conceptual
        ["Can you recommend some effective savings strategies for young adults?"],
        ["What's the difference between saving and investing?"]
    ],
    "Retirement": [
        ["What is a 401(k) and how does it differ from a Roth 401(k)?"], # Summary/Conceptual
        ["Explain the tax implications of withdrawing from a 401(k) before age 59½."],
        ["How does inflation impact long-term retirement savings and what strategies can mitigate this risk?"],
        ["If I am 40 years old and earn $5000 a month, how much should I save monthly to retire by 65, assuming a specific return rate?"], # Mathematical (more complex, might require simplifying assumptions by agent)
        ["What are the key considerations when planning for retirement income?"],
        ["Can you summarize the concept of Social Security benefits in retirement?"]
    ],
    "Credit Score": [
        ["What is a credit score and why is it important?"],
        ["How is a credit score calculated, and what are the main factors?"], # Summary/Conceptual
        ["What are some practical ways to improve my credit score quickly?"],
        ["What is a good credit score range, and what does it mean for borrowing?"],
        ["How often should I check my credit report for errors?"],
        ["Explain the difference between soft and hard credit inquiries."]
    ],
    "Investing": [
        ["What are some basic investment planning tips for beginners?"],
        ["Explain the concept of diversification in an investment portfolio."], # Summary/Conceptual
        ["What are the different types of investments available, like stocks, bonds, and mutual funds?"],
        ["What is my risk tolerance in investing, and how do I determine it?"],
        ["If I invest $1000 per month for 20 years at an average annual return of 7%, how much will I have?"], # Mathematical (more complex, might require simplifying assumptions by agent)
        ["Can you explain dollar-cost averaging and its benefits?"]
    ]
}

def update_sample_questions_display(category_name: str) -> gr.update:
    """
    Generates a Markdown string with sample questions for the selected category
    and controls the visibility of the Markdown component.
    """
    print(f"DEBUG: update_sample_questions_display called for category: {category_name}")
    if category_name in SAMPLE_QUESTIONS:
        questions = SAMPLE_QUESTIONS[category_name]
        # Format questions as an unordered Markdown list
        markdown_content = f"### Here are some sample questions for {category_name}:\n\n"
        for q_list in questions:
            markdown_content += f"- {q_list[0]}\n"
        markdown_content += "\nFeel free to ask your own questions as well!"
        
        print(f"DEBUG: Generated Markdown content for {category_name}. Visible: True.")
        return gr.update(value=markdown_content, visible=True)
    else:
        print("DEBUG: Category not found. Hiding sample questions.")
        return gr.update(value="", visible=False)

# --- Audio Processing Functions ---
def transcribe_audio(audio_path):
    """Transcribes an audio file using OpenAI's Whisper API."""
    if audio_path is None:
        return ""
    if openai_client is None:
        return "Error: OpenAI client not initialized for transcription."
    
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        logging.info(f"Audio transcribed: {transcript.text[:50]}...")
        return transcript.text
    except Exception as e:
        logging.error(f"Error during audio transcription: {e}")
        return f"Error transcribing audio: {e}"

def synthesize_speech(text):
    """Synthesizes speech from text using OpenAI's TTS API and returns the path to the audio file."""
    if not text.strip():
        return None
    if openai_client is None:
        logging.error("OpenAI client not initialized for speech synthesis.")
        return None
        
    try:
        # Create a temporary file to save the audio
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
            speech_file_path = fp.name
            
        response = openai_client.audio.speech.create(
            model="tts-1",
            voice="alloy", # You can choose other voices like 'nova', 'onyx', 'shimmer', 'fable', 'echo'
            input=text,
        )
        response.stream_to_file(speech_file_path)
        logging.info(f"Speech synthesized to: {speech_file_path}")
        return speech_file_path
    except Exception as e:
        logging.error(f"Error during speech synthesis: {e}")
        return None # Return None if synthesis fails

# This function (formerly answer_question) now routes to the AgentExecutor
def respond_to_question(question: str, history: List[List[str]], thinking_indicator_state: gr.Markdown) -> Tuple[str, List[List[str]], str | None, gr.Markdown]:
    """
    Handles user input (text) and generates a response using the LangChain AgentExecutor,
    and synthesizes speech for the response.
    Returns (cleared_input, updated_history, audio_path, thinking_indicator_update).
    """
    # Use gr.Markdown.update for direct updates to the component
    thinking_indicator_state_update = gr.Markdown.update(value="💭 Thinking...", visible=True)

    if not init_success:
        logging.error(f"Chatbot core not initialized: {init_message}")
        error_message = f"Chatbot not initialized: {init_message}"
        new_history = list(history)
        new_history.append([question, error_message])
        # Clear thinking indicator in case of early exit
        return "", new_history, None, gr.Markdown.update(value="", visible=False)

    new_history = list(history) # Create a copy of history for modification

    chat_history_for_agent = []
    for human_msg, ai_msg in new_history:
        if human_msg:
            chat_history_for_agent.append(HumanMessage(content=human_msg))
        if ai_msg:
            chat_history_for_agent.append(AIMessage(content=ai_msg))

    text_output = ""
    audio_path = None
    try:
        logging.info(f"Invoking LangChain AgentExecutor with question: {question}")
        response = agent_executor.invoke({"input": question, "chat_history": chat_history_for_agent})
        text_output = response['output']
        logging.info(f"LangChain AgentExecutor responded with text: {text_output[:100]}...")

        audio_path = synthesize_speech(text_output) # Synthesize speech for the AI's response

        new_history.append([question, text_output]) # Append user's question and AI's text response
        # Clear thinking indicator upon successful response
        return "", new_history, audio_path, gr.Markdown.update(value="", visible=False)
    except Exception as e:
        logging.error(f"Error during agent execution in respond_to_question function: {type(e).__name__}: {e}")
        error_response_text = f"I apologize, I encountered an error trying to answer your question: {type(e).__name__}: {e}. Please try again."
        error_audio_path = synthesize_speech(error_response_text) # Attempt to synthesize error message
        new_history.append([question, error_response_text]) # Append user's question and error text
        # Clear thinking indicator upon error
        return "", new_history, error_audio_path, gr.Markdown.update(value="", visible=False)

# --- Gradio UI Layout ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 💰 Financial Literacy Chatbot")
    gr.Markdown("---")

    # Initial welcome message and category selection prompt
    with gr.Row():
        welcome_text = gr.Markdown(value="Welcome to the Financial Literacy Chatbot! I'm here to help you with your personal finance questions.\n\nChoose an area:")
    
    # Buttons for categories
    with gr.Row():
        budgeting_btn = gr.Button("Budgeting", scale=1, size="lg")
        saving_btn = gr.Button("Saving", scale=1, size="lg")
        retirement_btn = gr.Button("Retirement", scale=1, size="lg")
        credit_score_btn = gr.Button("Credit Score", scale=1, size="lg")
        investing_btn = gr.Button("Investing", scale=1, size="lg")

    # Markdown component to display sample questions
    sample_questions_display = gr.Markdown(
        value="Select a category above to see sample questions.",
        visible=False
    )

    gr.Markdown("---")
    gr.Markdown("## Ask your financial question below:")

    # Section for Audio Input (transcription)
    with gr.Row():
        with gr.Column(scale=3):
            audio_input_mic = gr.Audio(sources=["microphone"], type="filepath", label="Record your question")
        with gr.Column(scale=4):
            transcribed_text_output = gr.Textbox(label="Transcribed Text", interactive=False, placeholder="Press 'Transcribe Audio' after recording. Copy this text to the chat input below.")
            transcribe_btn = gr.Button("Transcribe Audio")
            transcribe_btn.click(transcribe_audio, inputs=audio_input_mic, outputs=transcribed_text_output)
    
    gr.Markdown("---") # Separator

    # Main Chatbot Display (gr.Chatbot) and User Input Textbox
    chatbot_display = gr.Chatbot(
        height=400,
        render_markdown=True,
        bubble_full_width=False,
        # Initial message for the chatbot
        value=[[None, "Hello! I'm your Financial Literacy Chatbot. Choose from the following categories to see sample questions, or ask your own question directly below!"]]
    )
    user_question_textbox = gr.Textbox(placeholder="Type your financial question here...", container=False, scale=7, interactive=True, label="Your Question")
    submit_button = gr.Button("Send Question")

    # This markdown will display the "Thinking..." message
    thinking_indicator = gr.Markdown("", visible=False)

    # Audio Player for Chatbot's Response
    audio_output_player = gr.Audio(label="Chatbot's Response Audio", autoplay=True)
    
    # Event for submitting question from text input
    user_question_textbox.submit(
        fn=respond_to_question,
        inputs=[user_question_textbox, chatbot_display, thinking_indicator],
        outputs=[user_question_textbox, chatbot_display, audio_output_player, thinking_indicator]
    )
    # Event for submitting question from button click
    submit_button.click(
        fn=respond_to_question,
        inputs=[user_question_textbox, chatbot_display, thinking_indicator],
        outputs=[user_question_textbox, chatbot_display, audio_output_player, thinking_indicator]
    )


    # --- Dynamic Sample Questions Logic (Direct Markdown Update) ---
    # When a category button is clicked, update the Markdown component
    # Also trigger the first sample question to be answered by the chatbot
    budgeting_btn.click(
        fn=lambda: update_sample_questions_display("Budgeting"),
        outputs=[sample_questions_display]
    ).then(
        fn=lambda history, thinking_indicator_state: respond_to_question(SAMPLE_QUESTIONS["Budgeting"][0][0], history, thinking_indicator_state),
        inputs=[chatbot_display, thinking_indicator],
        outputs=[user_question_textbox, chatbot_display, audio_output_player, thinking_indicator]
    )
    saving_btn.click(
        fn=lambda: update_sample_questions_display("Saving"),
        outputs=[sample_questions_display]
    ).then(
        fn=lambda history, thinking_indicator_state: respond_to_question(SAMPLE_QUESTIONS["Saving"][0][0], history, thinking_indicator_state),
        inputs=[chatbot_display, thinking_indicator],
        outputs=[user_question_textbox, chatbot_display, audio_output_player, thinking_indicator]
    )
    retirement_btn.click(
        fn=lambda: update_sample_questions_display("Retirement"),
        outputs=[sample_questions_display]
    ).then(
        fn=lambda history, thinking_indicator_state: respond_to_question(SAMPLE_QUESTIONS["Retirement"][0][0], history, thinking_indicator_state),
        inputs=[chatbot_display, thinking_indicator],
        outputs=[user_question_textbox, chatbot_display, audio_output_player, thinking_indicator]
    )
    credit_score_btn.click(
        fn=lambda: update_sample_questions_display("Credit Score"),
        outputs=[sample_questions_display]
    ).then(
        fn=lambda history, thinking_indicator_state: respond_to_question(SAMPLE_QUESTIONS["Credit Score"][0][0], history, thinking_indicator_state),
        inputs=[chatbot_display, thinking_indicator],
        outputs=[user_question_textbox, chatbot_display, audio_output_player, thinking_indicator]
    )
    investing_btn.click(
        fn=lambda history: update_sample_questions_display("Investing"),
        outputs=[sample_questions_display]
    ).then(
        fn=lambda history, thinking_indicator_state: respond_to_question(SAMPLE_QUESTIONS["Investing"][0][0], history, thinking_indicator_state),
        inputs=[chatbot_display, thinking_indicator],
        outputs=[user_question_textbox, chatbot_display, audio_output_player, thinking_indicator]
    )

if __name__ == "__main__":
    if init_success:
        logging.info("Starting Gradio app...")
        demo.launch(share=False)
    else:
        print(f"Gradio app could not start due to initialization errors:\n{init_message}")
