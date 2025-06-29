import os
import tempfile
from dotenv import load_dotenv
from gtts import gTTS
from faster_whisper import WhisperModel
from typing import List, Tuple, Dict # Import for type hints

# LangChain and Pinecone imports
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Gradio import
import gradio as gr

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") # e.g., "us-east-1" for AWS serverless

# Ensure all necessary environment variables are loaded
if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT]):
    print("Error: Missing one or more environment variables.")
    print("Please ensure OPENAI_API_KEY, PINECONE_API_KEY, and PINECONE_ENVIRONMENT are set in your .env file.")
    # Exit or handle gracefully if crucial variables are missing
    exit(1)

# Make sure this index exists in your Pinecone account and is populated with data!
# (Refer to the previous `ingest_knowledge.py` script for populating)
index_name = "financial-literacy-chatbot" # Use the correct index name

# --- Initialize Components ---

# 1. Embeddings
embeddings = None
try:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    print("OpenAI Embeddings initialized.")
except Exception as e:
    print(f"Error initializing OpenAI Embeddings: {e}")

# 2. Pinecone Vectorstore
vectorstore = None
if embeddings:
    try:
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        print(f"Pinecone Vectorstore initialized with index: {index_name}")
    except Exception as e:
        print(f"Error initializing Pinecone Vectorstore. Make sure the index '{index_name}' exists and API keys are correct. Error: {e}")
else:
    print("Skipping Pinecone Vectorstore initialization due to embedding error.")

# 3. LLM (Large Language Model)
llm = None
try:
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7)
    print("ChatOpenAI LLM initialized.")
except Exception as e:
    print(f"Error initializing ChatOpenAI LLM: {e}")

# 4. RetrievalQA Chain
qa_chain = None
if vectorstore and llm:
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
        print("RetrievalQA chain initialized.")
    except Exception as e:
        print(f"Error initializing RetrievalQA chain: {e}")
else:
    print("Skipping RetrievalQA chain initialization due to missing vectorstore or LLM.")

# 5. Whisper Model for ASR (Automatic Speech Recognition)
whisper_model = None
try:
    # Use a smaller model like 'tiny' or 'base' for faster transcription
    whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    print("Faster Whisper model loaded.")
except Exception as e:
    print(f"Error loading Faster Whisper model: {e}")

# Flag to indicate if core components are initialized successfully
INIT_SUCCESS = bool(vectorstore and llm and whisper_model and qa_chain)
if not INIT_SUCCESS:
    print("\n--- WARNING: Chatbot components not fully initialized. Some features may not work. ---")
    print("Please check your .env file and Pinecone index setup.")
    print("-" * 70)


# --- Functions for Audio Processing ---

def transcribe_audio(audio_path):
    """
    Transcribes an audio file using Faster Whisper.
    Returns: Transcribed text, or an error message.
    """
    if not audio_path:
        return ""
    if not whisper_model:
        return "Error: Whisper model not loaded. Cannot transcribe."
    try:
        segments, info = whisper_model.transcribe(audio_path, beam_size=5)
        transcribed_text = " ".join([segment.text for segment in segments])
        print(f"Transcribed: '{transcribed_text}' (Language: {info.language}, Probability: {info.language_probability:.2f})")
        return transcribed_text
    except Exception as e:
        print(f"Error during audio transcription: {e}")
        return f"Error transcribing audio: {e}"

def speak(text):
    """
    Converts text to speech using gTTS and saves it to a temporary MP3 file.
    Returns: Path to the temporary audio file, or None if an error occurs.
    """
    if not text.strip():
        return None
    try:
        tts = gTTS(text=text, lang='en')
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_audio_file.name)
        temp_audio_file.close() 
        print(f"Generated speech to: {temp_audio_file.name}")
        return temp_audio_file.name
    except Exception as e:
        print(f"Error during text-to-speech generation: {e}")
        return None

# --- Main Chatbot Response Function (Handles both text and audio input) ---

def chatbot_response(
    message: str | None, 
    audio_input_path: str | None, 
    history: List[List[str | None]]
) -> Tuple[str, List[List[str | None]], str | None]:
    """
    Handles incoming text or audio input, processes it, gets a response from the QA chain,
    and returns the text response, updated chat history, and synthesized audio.
    """
    if not INIT_SUCCESS:
        error_msg = "Chatbot is not fully initialized. Please check backend setup and API keys."
        history.append([(message or "Audio input"), error_msg])
        return "", history, speak(error_msg)

    user_input_text = ""
    if audio_input_path:
        # If audio is provided, transcribe it first
        transcribed = transcribe_audio(audio_input_path)
        if transcribed.startswith("Error"): # Check if transcription failed
            history.append([f"Audio Error: {transcribed}", "I could not process your audio. Please try typing your question or re-recording."])
            return "", history, None
        user_input_text = transcribed
    elif message:
        # If text message is provided (from textbox or sample question click)
        user_input_text = message.strip()

    if not user_input_text:
        return "", history, None # No input, do nothing

    response_text = ""
    audio_output_path = None
    try:
        print(f"User Query: {user_input_text}")
        if qa_chain:
            # Use the RAG chain to get the answer
            response_text = qa_chain.run(user_input_text)
            print(f"Chatbot Response (text): {response_text}")
        else:
            response_text = "The knowledge base is not initialized. Please check backend setup."
            print("QA Chain not initialized.")

        audio_output_path = speak(response_text)
        
        # Update chat history: append user's question and bot's response
        history.append([user_input_text, response_text])
        
        # Clear the user input field
        return "", history, audio_output_path

    except Exception as e:
        error_message = f"An unexpected error occurred: {e}. Please try again."
        print(f"Error in chatbot_response: {e}")
        history.append([user_input_text, error_message])
        return "", history, speak(error_message)

# --- Sample Questions Data ---
SAMPLE_QUESTIONS: Dict[str, List[str]] = {
    "Budgeting": [
        "What is the 5/30/20 rule for budgeting?",
        "Can you recommend a simple budgeting template?",
        "How do I track my expenses effectively?",
        "What are common mistakes to avoid when budgeting?"
    ],
    "Savings": [
        "What is compound interest and how does it affect savings?",
        "What are some strategies to save money quickly?",
        "How much should I save for an emergency fund?",
        "What's the difference between saving and investing?"
    ],
    "Credit Score": [
        "What is a credit score and why is it important?",
        "How can I improve my credit score?",
        "What factors affect my credit score?",
        "How often should I check my credit report?"
    ],
    "Investing": [
        "What are stocks and bonds?",
        "What is diversification in investing?",
        "How do I start investing as a beginner?",
        "Explain the concept of risk tolerance in investments."
    ],
    "Retirement Planning": [
        "What is a 401(k) and how does it work?",
        "What's the difference between a traditional IRA and a Roth IRA?",
        "How much should I save for retirement?",
        "When can I start withdrawing from my retirement accounts without penalty?"
    ]
}

import gradio as gr

# ... Your other imports and code remain unchanged ...

# --- Inside your gr.Blocks() context ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ Financial Literacy Chatbot 💰")
    gr.Markdown("Welcome! I'm here to help you with your personal finance questions. Ask me anything, or choose a category below to see some sample questions.")
    gr.Markdown("---")

    # Category Buttons
    with gr.Row():
        budgeting_btn = gr.Button("Budgeting", scale=1, size="lg")
        savings_btn = gr.Button("Savings", scale=1, size="lg")
        credit_score_btn = gr.Button("Credit Score", scale=1, size="lg")
        investing_btn = gr.Button("Investing", scale=1, size="lg")
        retirement_btn = gr.Button("Retirement Planning", scale=1, size="lg")

    gr.Markdown("---")

    # Sample Questions Row
    with gr.Row() as sample_questions_row:
        # Hidden textboxes to hold question texts for buttons (needed because buttons don't pass their labels)
        sample_question_texts = [
            gr.Textbox(value="", visible=False, elem_id=f"sample-text-{i}") for i in range(5)
        ]
        # Buttons to display sample questions (max 5)
        sample_question_btns = [
            gr.Button("", visible=False, elem_id=f"sample-btn-{i}") for i in range(5)
        ]

    # Audio Input Section (your existing code here)...
    with gr.Row():
        with gr.Column(scale=1):
            audio_input_mic = gr.Audio(sources=["microphone"], type="filepath", label="Record your question")
        with gr.Column(scale=1):
            transcribe_btn = gr.Button("Transcribe Audio (fills text box below)")

    gr.Markdown("---")

    chatbot_display = gr.Chatbot(
        height=400,
        render_markdown=True,
        bubble_full_width=False,
        value=[[None, "Hello! I'm your Financial Literacy Chatbot. Choose a category above, speak your question, or type it below!"]]
    )
    
    user_question_textbox = gr.Textbox(
        placeholder="Type your financial question here, or click a sample question...", 
        container=False, 
        scale=7, 
        interactive=True, 
        label="Your Question"
    )
    
    send_text_btn = gr.Button("Send Text Question")

    audio_output_player = gr.Audio(label="Chatbot's Spoken Response", autoplay=True, type="filepath") 

    # --- Helper functions ---

    def update_sample_questions_display(category_name: str) -> list[str]:
        """Return list of questions for category or empty list."""
        return SAMPLE_QUESTIONS.get(category_name, [])

    def show_sample_questions(category_name: str):
        questions = update_sample_questions_display(category_name)
        btn_updates = []
        text_updates = []
        for i in range(5):
            if i < len(questions):
                btn_updates.append(gr.update(value=questions[i], visible=True))
                text_updates.append(gr.update(value=questions[i]))
            else:
                btn_updates.append(gr.update(visible=False))
                text_updates.append(gr.update(value=""))
        return btn_updates + text_updates

    def on_sample_question_click(question_text, chat_history):
        # Call chatbot_response with the clicked question text, no audio input
        return chatbot_response(question_text, None, chat_history)

    # --- Events ---

    # Category buttons update sample question buttons and hidden texts
    budgeting_btn.click(fn=lambda: show_sample_questions("Budgeting"), inputs=[], outputs=sample_question_btns + sample_question_texts)
    savings_btn.click(fn=lambda: show_sample_questions("Savings"), inputs=[], outputs=sample_question_btns + sample_question_texts)
    credit_score_btn.click(fn=lambda: show_sample_questions("Credit Score"), inputs=[], outputs=sample_question_btns + sample_question_texts)
    investing_btn.click(fn=lambda: show_sample_questions("Investing"), inputs=[], outputs=sample_question_btns + sample_question_texts)
    retirement_btn.click(fn=lambda: show_sample_questions("Retirement Planning"), inputs=[], outputs=sample_question_btns + sample_question_texts)

    # Sample question buttons click - send the corresponding hidden textbox value as question
    for btn, txt in zip(sample_question_btns, sample_question_texts):
        btn.click(
            fn=on_sample_question_click,
            inputs=[txt, chatbot_display],
            outputs=[user_question_textbox, chatbot_display, audio_output_player]
        )

    # Transcribe audio fills text input
    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[audio_input_mic],
        outputs=[user_question_textbox]
    )

    # User text input (submit or send button) calls chatbot_response
    user_question_textbox.submit(
        fn=lambda msg, hist: chatbot_response(msg, None, hist),
        inputs=[user_question_textbox, chatbot_display],
        outputs=[user_question_textbox, chatbot_display, audio_output_player]
    )
    send_text_btn.click(
        fn=lambda msg, hist: chatbot_response(msg, None, hist),
        inputs=[user_question_textbox, chatbot_display],
        outputs=[user_question_textbox, chatbot_display, audio_output_player]
    )

# --- Your other code for launching etc. remains unchanged ---



    # 2. Transcribe Audio Button -> Fill Textbox
    # Note: `gr.Button.click` doesn't directly return to a textbox for `gr.Chatbot` input.
    # We will use it to *update* the `user_question_textbox`
    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[audio_input_mic],
        outputs=[user_question_textbox] # Output transcribed text directly to the user's input box
    )

    # 3. User Text Input Submission (via Enter key or Send button)
    user_question_textbox.submit(
        fn=lambda msg, hist: chatbot_response(msg, None, hist), # Pass message, no audio path
        inputs=[user_question_textbox, chatbot_display],
        outputs=[user_question_textbox, chatbot_display, audio_output_player]
    )
    send_text_btn.click(
        fn=lambda msg, hist: chatbot_response(msg, None, hist), # Pass message, no audio path
        inputs=[user_question_textbox, chatbot_display],
        outputs=[user_question_textbox, chatbot_display, audio_output_player]
    )

# --- Launch the Gradio App ---
if __name__ == "__main__":
    if not INIT_SUCCESS:
        print("\n" + "#" * 70)
        print("                 CRITICAL ERROR: CHATBOT NOT FULLY INITIALIZED")
        print("                 Please resolve the setup errors mentioned above.")
        print("                 Ensure your .env file is correct and Pinecone index is populated.")
        print("#" * 70 + "\n")
    else:
        print("\nLaunching Gradio Interface...")
        demo.launch(share=True) # share=True generates a public link (expires after 72 hours)